
import math
import os
import re
from collections import Counter
from pathlib import Path

import requests
import chromadb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

CHROMA_FOLDER = "chroma_store"
COLLECTION_NAME = "nq_chunks"

# These two are read from the environment (via .env locally) first.
# streamlit_app.py may overwrite them with Streamlit secrets when deployed.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


EMBEDDING_DIM = 1536  # must match OpenRouter text-embedding-3-small


def simple_embedding(text):
    """Create a deterministic local embedding when no API key is available."""
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    if not tokens:
        return [0.0] * EMBEDDING_DIM

    counts = Counter(tokens)
    vector = [0.0] * EMBEDDING_DIM
    for token, count in counts.items():
        index = sum(ord(ch) for ch in token) % EMBEDDING_DIM
        vector[index] += float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def get_embedding(text):
    """Ask OpenRouter to turn one piece of text into a vector of numbers, or fall back locally."""
    api_key = OPENROUTER_API_KEY
    if api_key and api_key != "your_openrouter_key_here":
        try:
            headers = {
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Natural Questions RAG Assistant",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            request_body = {"model": OPENROUTER_EMBEDDING_MODEL, "input": text}

            session = requests.Session()
            response = session.post(
                OPENROUTER_EMBEDDINGS_URL, headers=headers, json=request_body, timeout=60
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as exc:
            print(f"[get_embedding] OpenRouter API failed: {exc}  — falling back to local embedding.")

    return simple_embedding(text)


def retrieve_context(question, number_of_chunks=4):
    """Find the most relevant chunks in ChromaDB for this question."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist in ChromaDB ({CHROMA_FOLDER}). "
            "Please run the indexing pipeline (01_documents.py through 05_create_chroma_store.py) to build the vector store."
        ) from e

    question_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[question_embedding], n_results=number_of_chunks
    )

    retrieved_chunks = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({
                "chunk_text": text,
                "title": metadata["title"],
                "document_id": metadata["document_id"],
            })
    return retrieved_chunks


def build_prompt(question, retrieved_chunks):
    """Combine the question and retrieved chunks into a hybrid prompt."""
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append("[Source " + str(i) + "] " + chunk["title"])
        context_lines.append(chunk["chunk_text"])
        context_lines.append("")
    context_text = "\n".join(context_lines)

    prompt = (
        "You are a helpful assistant.\n"
        "1. If the provided context below contains the answer, answer the question and cite which source number(s) you used.\n"
        "2. If the provided context does NOT contain the answer, answer the question accurately using your general knowledge.\n\n"
        "Context:\n" + context_text + "\n"
        "Question: " + question + "\n"
        "Answer:"
    )
    return prompt


def call_llm(prompt):
    """Send the prompt to an LLM through OpenRouter and return its answer text."""
    api_key = OPENROUTER_API_KEY
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Natural Questions RAG Assistant",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    request_body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    session = requests.Session()
    response = session.post(
        OPENROUTER_CHAT_URL, headers=headers, json=request_body, timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def fallback_answer(question, retrieved_chunks):
    """Return a simple answer based on the most relevant retrieved chunk when no LLM key is available."""
    if not retrieved_chunks:
        return "No matching source found in the indexed database. (Tip: Set OPENROUTER_API_KEY in .env to allow the AI to answer any question using general knowledge!)"

    question_terms = set(re.findall(r"[a-zA-Z0-9']+", question.lower()))
    stop_words = {
        "what", "is", "the", "of", "a", "an", "in", "on", "at", "by", "for",
        "with", "about", "to", "and", "or", "who", "when", "where", "how",
        "capital", "city", "year", "person", "first", "name"
    }
    filtered_question_terms = question_terms - stop_words
    query_terms = filtered_question_terms if filtered_question_terms else question_terms

    scored_chunks = []
    for chunk in retrieved_chunks:
        chunk_terms = set(re.findall(r"[a-zA-Z0-9']+", chunk["chunk_text"].lower()))
        overlap = len(query_terms & chunk_terms)
        scored_chunks.append((overlap, chunk))

    best_score, best_chunk = max(scored_chunks, key=lambda item: item[0])
    if best_score == 0:
        return "No matching passage found in the indexed database for this question. (Tip: Set OPENROUTER_API_KEY in .env to enable AI answers for any topic via general knowledge!)"

    return best_chunk["chunk_text"]


def answer_question(question, number_of_chunks=4):
    """Full pipeline: retrieve context, build prompt, ask the LLM, return everything."""
    retrieved_chunks = retrieve_context(question, number_of_chunks=number_of_chunks)
    prompt = build_prompt(question, retrieved_chunks)

    api_key = OPENROUTER_API_KEY
    if api_key and api_key != "your_openrouter_key_here":
        try:
            answer = call_llm(prompt)
        except Exception:
            answer = fallback_answer(question, retrieved_chunks)
    else:
        answer = fallback_answer(question, retrieved_chunks)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "answer": answer,
    }
