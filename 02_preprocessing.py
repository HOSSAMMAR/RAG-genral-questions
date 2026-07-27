

import json
import os
import re

INPUT_FILE = os.path.join("data", "01_raw_documents.json")
OUTPUT_FILE = os.path.join("data", "02_clean_documents.json")


def clean_text(text):
    """Basic clean-up: fix spacing, remove a few common leftover symbols."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\t", " ")
    # Collapse multiple spaces into a single space
    text = re.sub(r" +", " ", text)
    # Remove a couple of leftover Wikipedia edit-link symbols that sometimes appear
    text = text.replace("[ edit ]", "")
    text = text.strip()
    return text


def main():
    print("Loading raw documents from", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_examples = json.load(f)

    cleaned_examples = []
    for example in raw_examples:
        cleaned_example = {
            "document_id": example["document_id"],
            "title": clean_text(example["title"]),
            "question": clean_text(example["question"]),
            "context_text": clean_text(example["context_text"]),
            "answer_text": clean_text(example["answer_text"]),
        }

        # Skip anything that ended up empty after cleaning -- nothing useful to index
        if len(cleaned_example["context_text"]) == 0:
            continue

        cleaned_examples.append(cleaned_example)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_examples, f, indent=2)

    print("Cleaned", len(cleaned_examples), "documents")
    print("Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
