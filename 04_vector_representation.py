

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

INPUT_FILE = os.path.join("data", "03_chunks.json")
OUTPUT_FILE = os.path.join("data", "04_chunks_with_embeddings.json")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


EMBEDDING_DIM = 1536  # must match OpenRouter text-embedding-3-small


def simple_embedding(text):
    """Create a deterministic local embedding from the text when no API key is available."""
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    if not tokens:
        return [0.0] * EMBEDDING_DIM

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    vector = [0.0] * EMBEDDING_DIM
    for token, count in counts.items():
        index = sum(ord(ch) for ch in token) % EMBEDDING_DIM
        vector[index] += float(count)

    norm = sum(value * value for value in vector) ** 0.5
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
            response_data = response.json()
            return response_data["data"][0]["embedding"]
        except Exception as exc:
            print(f"[get_embedding] OpenRouter API failed: {exc} — falling back to local embedding.")

    return simple_embedding(text)


def main():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_key_here":
        print("OPENROUTER_API_KEY is missing or placeholder. Using local simple_embedding model.")


    print("Loading chunks from", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["chunk_text"])

        chunk_with_embedding = dict(chunk)  # copy all the existing fields
        chunk_with_embedding["embedding"] = embedding
        chunks_with_embeddings.append(chunk_with_embedding)

        print("Embedded chunk", i + 1, "/", len(chunks))
        time.sleep(0.1)  # be gentle on the API

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_with_embeddings, f)

    print("Saved", len(chunks_with_embeddings), "embedded chunks to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
