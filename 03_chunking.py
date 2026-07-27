

import json
import os

INPUT_FILE = os.path.join("data", "02_clean_documents.json")
OUTPUT_FILE = os.path.join("data", "03_chunks.json")

CHUNK_SIZE_WORDS = 120     # how many words go in one chunk
CHUNK_OVERLAP_WORDS = 20   # how many words of overlap between neighboring chunks


def split_into_chunks(text, chunk_size, overlap):
    """Split text into a list of overlapping word chunks."""
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if end >= len(words):
            break
        # Move forward, but overlap a little with the previous chunk so we
        # don't accidentally cut a sentence in half between two chunks
        start = end - overlap

    return chunks


def main():
    print("Loading clean documents from", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    for document in documents:
        text_chunks = split_into_chunks(
            document["context_text"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
        )

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_record = {
                "chunk_id": document["document_id"] + "_chunk_" + str(chunk_index),
                "document_id": document["document_id"],
                "title": document["title"],
                "chunk_text": chunk_text,
                # Keep the original question/answer attached so later steps
                # (like checking accuracy) have something to compare against.
                "source_question": document["question"],
                "source_answer": document["answer_text"],
            }
            all_chunks.append(chunk_record)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print("Created", len(all_chunks), "chunks from", len(documents), "documents")
    print("Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
