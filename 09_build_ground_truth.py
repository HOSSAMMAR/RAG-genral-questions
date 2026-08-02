"""
09_build_ground_truth.py
-------------------------
Builds the GROUND TRUTH file used to evaluate the RAG pipeline, and saves
it to its own file so it's reproducible and reusable (you don't want your
eval set to silently change every time you run an evaluation).

WHAT IS "GROUND TRUTH" HERE?
-----------------------------
Every chunk in data/03_chunks.json already carries the original question
and correct answer it came from (source_question / source_answer), plus
which document it belongs to (document_id). That's ground truth: for a
given question, we already KNOW the correct answer and the correct source
document. This script just pulls a clean, de-duplicated sample of that
out into its own file so later scripts can grade against it.

HOW TO RUN
----------
    1. Make sure data/03_chunks.json exists (run 01 -> 03 first).
    2. Run:
         python 09_build_ground_truth.py

OUTPUT
------
    data/ground_truth.json
    A list of objects like:
        {
          "question": "...",
          "answer": "...",
          "document_id": "..."
        }
"""

import json
import os
import random
from pathlib import Path

# ---------------- Settings you can change ----------------
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "03_chunks.json"
GROUND_TRUTH_FILE = BASE_DIR / "data" / "ground_truth.json"

EVAL_SAMPLE_SIZE = 30     # how many questions to hold out for testing
RANDOM_SEED = 42          # fixed seed so the same questions get picked every time
# -----------------------------------------------------------


def load_chunks():
    """Load every chunk we indexed. Each chunk still carries the original
    question + correct answer + which document it came from."""
    print("Loading chunks from", CHUNKS_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ground_truth(chunks, sample_size, random_seed):
    """
    Pick a random sample of (question, correct_answer, correct_document_id)
    triples to use as our test set.

    We de-duplicate by document_id first, so a document that got split
    into several chunks doesn't get tested multiple times with the same
    question.
    """
    seen_documents = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id not in seen_documents:
            seen_documents[doc_id] = {
                "question": chunk["source_question"],
                "answer": chunk["source_answer"],
                "document_id": doc_id,
            }

    all_candidates = [
        item for item in seen_documents.values()
        if item["question"].strip() != "" and item["answer"].strip() != ""
    ]

    random.seed(random_seed)
    if len(all_candidates) > sample_size:
        ground_truth = random.sample(all_candidates, sample_size)
    else:
        ground_truth = all_candidates

    print("Built ground truth set with", len(ground_truth), "question/answer pairs")
    return ground_truth


def main():
    chunks = load_chunks()
    ground_truth = build_ground_truth(chunks, EVAL_SAMPLE_SIZE, RANDOM_SEED)

    GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print("Saved ground truth to", GROUND_TRUTH_FILE)
    print("Next step: run 10_evaluate_rag.py to test your RAG pipeline against it.")


if __name__ == "__main__":
    main()
