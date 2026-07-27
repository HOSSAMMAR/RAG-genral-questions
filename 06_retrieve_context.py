

import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if one exists

CHROMA_FOLDER = "chroma_store"
COLLECTION_NAME = "nq_chunks"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


import math
import re
from collections import Counter


def simple_embedding(text):
    """Create a deterministic local embedding when no API key is available."""
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    if not tokens:
        return [0.0] * 256

    counts = Counter(tokens)
    vector = [0.0] * 256
    for token, count in counts.items():
        index = sum(ord(ch) for ch in token) % 256
        vector[index] += float(count)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def get_embedding(text):
    """Ask OpenRouter to turn one piece of text into a vector of numbers, or fall back locally."""
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_key_here":
        try:
            headers = {
                "Authorization": "Bearer " + OPENROUTER_API_KEY,
                "Content-Type": "application/json",
            }
            request_body = {
                "model": OPENROUTER_EMBEDDING_MODEL,
                "input": text,
            }
            response = requests.post(
                OPENROUTER_EMBEDDINGS_URL, headers=headers, json=request_body, timeout=60
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception:
            pass

    return simple_embedding(text)


def retrieve_context(question, number_of_chunks=4):
    """
    Return the top matching chunks for a question, as a list of dicts:
    [{"chunk_text": ..., "title": ..., "document_id": ...}, ...]
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist in ChromaDB ({CHROMA_FOLDER}). "
            "Please run 05_create_chroma_store.py to create and populate the vector store."
        ) from e

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_chunks,
    )

    retrieved_chunks = []
    chunk_texts = results["documents"][0]
    chunk_metadatas = results["metadatas"][0]

    for text, metadata in zip(chunk_texts, chunk_metadatas):
        retrieved_chunks.append({
            "chunk_text": text,
            "title": metadata["title"],
            "document_id": metadata["document_id"],
        })

    return retrieved_chunks


if __name__ == "__main__":
    sample_question = "What year was Google founded?"
    chunks = retrieve_context(sample_question, number_of_chunks=3)

    print("Question:", sample_question)
    print()
    for i, chunk in enumerate(chunks, start=1):
        print("[Source", i, "]", chunk["title"])
        print(chunk["chunk_text"])
        print()
