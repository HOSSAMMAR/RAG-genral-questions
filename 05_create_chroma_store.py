
import json
import os
import chromadb

INPUT_FILE = os.path.join("data", "04_chunks_with_embeddings.json")
CHROMA_FOLDER = "chroma_store"
COLLECTION_NAME = "nq_chunks"


def main():
    print("Loading embedded chunks from", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Connecting to ChromaDB at", CHROMA_FOLDER)
    chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)

    # Start fresh each time this script runs, so we don't end up with duplicates
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- that's fine

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["chunk_text"])
        metadatas.append({
            "document_id": chunk["document_id"],
            "title": chunk["title"],
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print("Added", len(ids), "chunks to the '" + COLLECTION_NAME + "' collection")
    print("Vector store saved in the '" + CHROMA_FOLDER + "' folder")


if __name__ == "__main__":
    main()
