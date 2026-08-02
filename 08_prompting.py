# To launch the deployed app instead of running this file directly, use:
#   streamlit run streamlit_app.py

import os
from importlib import import_module
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Numbered filenames aren't valid Python identifiers, so we load them
# with import_module() instead of a plain "import 06_retrieve_context".
retrieve_module = import_module("06_retrieve_context")
context_building = import_module("07_context_building")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question, retrieved_chunks):
    """Combine the question and retrieved chunks into one grounded prompt.
    Context building itself now lives in context_building.py so it isn't
    duplicated between rag.py and 07_prompting.py."""
    prompt, _chunks_used = context_building.build_prompt(
        question, retrieved_chunks, allow_general_knowledge=False
    )
    return prompt


def call_llm(prompt):
    """Send the prompt to an LLM through OpenRouter and return its answer text."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
        raise RuntimeError("OPENROUTER_API_KEY is not configured. Set it in your .env file first.")

    headers = {
        "Authorization": "Bearer " + OPENROUTER_API_KEY,
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Natural Questions RAG Assistant",
    }
    request_body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = requests.post(
        OPENROUTER_CHAT_URL, headers=headers, json=request_body, timeout=60
    )
    response.raise_for_status()
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"].strip()


def answer_question(question, number_of_chunks=4):
    """Full pipeline: retrieve context, build prompt, ask the LLM."""
    retrieved_chunks = retrieve_module.retrieve_context(question, number_of_chunks=number_of_chunks)
    prompt = build_prompt(question, retrieved_chunks)
    try:
        answer = call_llm(prompt)
    except Exception as exc:
        answer = f"[LLM error: {exc}]"
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "answer": answer,
    }


if __name__ == "__main__":
    sample_question = "What year was Google founded?"
    result = answer_question(sample_question)

    print("QUESTION:", result["question"])
    print()
    print("ANSWER:", result["answer"])
