
import csv
import json
import os
import random
import re

# ---------- Settings you can change ----------
CSV_FOLDER = "raw_csv"   # put the uploaded CSVs in this folder
OUTPUT_FOLDER = "data"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "01_raw_documents.json")

QA_100_FILE = os.path.join(CSV_FOLDER, "100_Unique_QA_Dataset.csv")
GENERAL_KNOWLEDGE_FILE = os.path.join(CSV_FOLDER, "general_knowledge_qa.csv")
WORLD_DATES_FILE = os.path.join(CSV_FOLDER, "World_Important_Dates.csv")
JEOPARDY_FILE = os.path.join(CSV_FOLDER, "JEOPARDY_CSV.csv")
WEB_QUESTIONS_FILE = os.path.join(CSV_FOLDER, "WebQuestions_.csv")
PROGRAMMING_QA_FILE = os.path.join(CSV_FOLDER, "programming_questions_solutions.csv")

# JEOPARDY_CSV.csv has 200,000+ rows. Embedding all of them would mean
# 200,000+ API calls in 04_vector_representation.py, so we randomly sample
# a smaller number of clues instead. Raise this if you want more coverage.
MAX_JEOPARDY_EXAMPLES = 300
RANDOM_SEED = 42
# -----------------------------------------------


def clean(value):
    """Turn None/NaN-ish/empty values into a plain stripped string."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", "unknown", ""):
        return ""
    return text


def load_simple_qa_csv(file_path, source_name, id_prefix, default_title):
    """
    Handles both 100_Unique_QA_Dataset.csv (question, answer) and
    general_knowledge_qa.csv (question, answer, question_type, image).
    Both have a plain question + answer, so they're loaded the same way.
    """
    examples = []
    if not os.path.exists(file_path):
        print("Skipping", source_name, "- file not found at", file_path)
        return examples

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = clean(row.get("question"))
            answer = clean(row.get("answer"))
            if question == "" or answer == "":
                continue  # nothing useful to index

            title = clean(row.get("question_type")) or default_title

            # The "document" for a FAQ-style row is just the fact itself.
            # Writing it as Q + A (instead of only the answer) gives the
            # embedding model more context to match against future questions.
            context_text = "Question: " + question + " Answer: " + answer

            examples.append({
                "document_id": id_prefix + "_" + str(i),
                "title": title,
                "question": question,
                "context_text": context_text,
                "answer_text": answer,
            })

    print("Loaded", len(examples), "examples from", source_name)
    return examples


def load_world_dates_csv(file_path):
    """
    World_Important_Dates.csv has no question column -- it's a table of
    historical events. We build a short paragraph (context_text) out of the
    row's fields, and synthesize a question so the row still fits the
    question/context_text/answer_text schema used by the rest of the pipeline.
    """
    examples = []
    if not os.path.exists(file_path):
        print("Skipping World_Important_Dates.csv - file not found at", file_path)
        return examples

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            incident = clean(row.get("Name of Incident"))
            if incident == "":
                continue

            row_id = clean(row.get("Sl. No")) or str(len(examples))
            date = clean(row.get("Date"))
            month = clean(row.get("Month"))
            year = clean(row.get("Year"))
            country = clean(row.get("Country"))
            event_type = clean(row.get("Type of Event"))
            place = clean(row.get("Place Name"))
            impact = clean(row.get("Impact"))
            affected = clean(row.get("Affected Population"))
            responsible = clean(row.get("Important Person/Group Responsible"))
            outcome = clean(row.get("Outcome"))

            # Build a human readable date like "21 April 1526" or just "1206"
            # depending on which parts are actually known.
            date_parts = [part for part in [date, month, year] if part != ""]
            date_text = " ".join(date_parts) if date_parts else "an unknown date"

            sentences = [incident + " took place on " + date_text + "."]
            if place != "" or country != "":
                location_parts = [p for p in [place, country] if p != ""]
                # avoid "Mexico, Mexico" when place and country are identical
                location_parts = list(dict.fromkeys(location_parts))
                sentences.append("Location: " + ", ".join(location_parts) + ".")
            if event_type != "":
                sentences.append("Type of event: " + event_type + ".")
            if impact != "":
                sentences.append("Impact: " + impact + ".")
            if affected != "":
                sentences.append("Affected population: " + affected + ".")
            if responsible != "":
                sentences.append("Key person/group involved: " + responsible + ".")
            if outcome != "":
                sentences.append("Outcome: " + outcome + ".")

            context_text = " ".join(sentences)
            question = "What happened during the " + incident + "?"
            answer_text = outcome if outcome != "" else date_text

            examples.append({
                "document_id": "worlddates_" + row_id,
                "title": incident,
                "question": question,
                "context_text": context_text,
                "answer_text": answer_text,
            })

    print("Loaded", len(examples), "examples from World_Important_Dates.csv")
    return examples


def load_jeopardy_csv(file_path, max_examples, random_seed):
    """
    JEOPARDY_CSV.csv columns (note the leading spaces in the header) are:
    'Show Number', ' Air Date', ' Round', ' Category', ' Value', ' Question', ' Answer'
    Confusingly, in Jeopardy the " Question" column is actually the CLUE
    (a statement read to contestants) and " Answer" is the correct response.
    We keep that mapping: question = the clue text, answer_text = the answer.
    """
    examples = []
    if not os.path.exists(file_path):
        print("Skipping JEOPARDY_CSV.csv - file not found at", file_path)
        return examples

    all_rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clue = clean(row.get(" Question"))
            answer = clean(row.get(" Answer"))
            if clue == "" or answer == "":
                continue
            all_rows.append(row)

    print("Found", len(all_rows), "usable clues in JEOPARDY_CSV.csv")

    random.seed(random_seed)
    if len(all_rows) > max_examples:
        sampled_rows = random.sample(all_rows, max_examples)
    else:
        sampled_rows = all_rows

    for i, row in enumerate(sampled_rows):
        show_number = clean(row.get("Show Number")) or str(i)
        air_date = clean(row.get(" Air Date"))
        category = clean(row.get(" Category")) or "Jeopardy"
        clue = clean(row.get(" Question"))
        answer = clean(row.get(" Answer"))

        context_text = "Category: " + category + ". Clue: " + clue + " Answer: " + answer + "."

        examples.append({
            "document_id": "jeopardy_" + show_number + "_" + str(i),
            "title": category + (" (" + air_date + ")" if air_date != "" else ""),
            "question": clue,
            "context_text": context_text,
            "answer_text": answer,
        })

    print("Sampled", len(examples), "examples from JEOPARDY_CSV.csv")
    return examples


def parse_answers_list(raw_value):
    """
    WebQuestions_.csv stores multiple answers as a string that LOOKS like a
    Python list but is space-separated instead of comma-separated, e.g.:
    "['Jamaican Creole English Language' 'Jamaican English']"
    A regex pulls out each single-quoted item regardless of that formatting.
    """
    if raw_value is None:
        return []
    items = re.findall(r"'([^']*)'", str(raw_value))
    return [item.strip() for item in items if item.strip() != ""]


def load_web_questions_csv(file_path):
    """
    WebQuestions_.csv columns: url, question, answers
    'answers' can contain more than one correct answer, so we join them
    with '; ' to build a single answer_text.
    """
    examples = []
    if not os.path.exists(file_path):
        print("Skipping WebQuestions_.csv - file not found at", file_path)
        return examples

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = clean(row.get("question"))
            answers = parse_answers_list(row.get("answers"))
            if question == "" or len(answers) == 0:
                continue

            answer_text = "; ".join(answers)
            url = clean(row.get("url"))
            # The url is a Freebase entity link like ".../en/jamaica" -- use
            # the last part as a readable title.
            title = url.rstrip("/").split("/")[-1].replace("_", " ") if url != "" else "Web Question"

            context_text = "Question: " + question + " Answer: " + answer_text

            examples.append({
                "document_id": "webq_" + str(i),
                "title": title,
                "question": question,
                "context_text": context_text,
                "answer_text": answer_text,
            })

    print("Loaded", len(examples), "examples from WebQuestions_.csv")
    return examples


def load_programming_qa_csv(file_path):
    """
    programming_questions_solutions.csv columns:
    Question, Difficulty Level, Programming Language, AI-Generated Solution,
    Time Complexity, Explanation, Topic
    Each row is a coding problem + its (synthetic) solution snippet. We fold
    the solution code, complexity, and explanation into one context_text so
    retrieval can answer both "what's the solution" and "why" questions.
    """
    examples = []
    if not os.path.exists(file_path):
        print("Skipping programming_questions_solutions.csv - file not found at", file_path)
        return examples

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = clean(row.get("Question"))
            solution = clean(row.get("AI-Generated Solution"))
            if question == "" or solution == "":
                continue

            difficulty = clean(row.get("Difficulty Level"))
            language = clean(row.get("Programming Language"))
            complexity = clean(row.get("Time Complexity"))
            explanation = clean(row.get("Explanation"))
            topic = clean(row.get("Topic")) or "Programming"

            context_parts = [question + "."]
            if language != "":
                context_parts.append("Language: " + language + ".")
            if difficulty != "":
                context_parts.append("Difficulty: " + difficulty + ".")
            context_parts.append("Solution:\n" + solution)
            if complexity != "":
                context_parts.append("Time complexity: " + complexity + ".")
            if explanation != "":
                context_parts.append(explanation)

            context_text = " ".join(context_parts)
            # The solution snippet itself is the most useful "answer"
            answer_text = solution

            examples.append({
                "document_id": "progqa_" + str(i),
                "title": topic,
                "question": question,
                "context_text": context_text,
                "answer_text": answer_text,
            })

    print("Loaded", len(examples), "examples from programming_questions_solutions.csv")
    return examples


def main():
    all_examples = []

    all_examples.extend(
        load_simple_qa_csv(
            QA_100_FILE,
            "100_Unique_QA_Dataset.csv",
            id_prefix="qa100",
            default_title="General Knowledge Q&A",
        )
    )
    all_examples.extend(
        load_simple_qa_csv(
            GENERAL_KNOWLEDGE_FILE,
            "general_knowledge_qa.csv",
            id_prefix="genqa",
            default_title="General Knowledge Q&A",
        )
    )
    all_examples.extend(load_world_dates_csv(WORLD_DATES_FILE))
    all_examples.extend(
        load_jeopardy_csv(JEOPARDY_FILE, MAX_JEOPARDY_EXAMPLES, RANDOM_SEED)
    )
    all_examples.extend(load_web_questions_csv(WEB_QUESTIONS_FILE))
    all_examples.extend(load_programming_qa_csv(PROGRAMMING_QA_FILE))

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2)

    print("Saved", len(all_examples), "total examples to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
