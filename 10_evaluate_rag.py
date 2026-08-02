"""
10_evaluate_rag.py
--------------------
Evaluates your RAG pipeline against the ground truth file built by
09_build_ground_truth.py, and visualizes the results.

WHAT THIS MEASURES
-------------------
    a) RETRIEVAL quality  -> Hit Rate@k, MRR
       ("did we find the right source document?")
    b) ANSWER quality     -> Exact Match, F1 score
       ("was the final answer actually correct?")
    c) CONTEXT BUILDING   -> logs exactly which chunks were retrieved and
       fed into the prompt for every question, so you can inspect it
       instead of just trusting it happened.

HOW TO RUN
----------
    1. Run 09_build_ground_truth.py first (creates data/ground_truth.json).
    2. Make sure your Chroma vector store exists (run 01 -> 05 first).
    3. Make sure rag.py is in the same folder as this file.
    4. Run:
         python 10_evaluate_rag.py

OUTPUT
------
    data/eval_results/eval_details.csv   -> one row per test question
    data/eval_results/eval_summary.png   -> bar charts + histogram
"""

import csv
import json
import os
import re
import string
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # so this works even without a display (e.g. on a server)
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

import rag  # your existing rag.py -- we reuse retrieve_context() and answer_question()

# ---------------- Settings you can change ----------------
BASE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_FILE = BASE_DIR / "data" / "ground_truth.json"
RESULTS_FOLDER = BASE_DIR / "data" / "eval_results"
DETAILS_CSV = RESULTS_FOLDER / "eval_details.csv"
SUMMARY_PNG = RESULTS_FOLDER / "eval_summary.png"

TOP_K = 4   # how many chunks to retrieve per question (same as your app uses)
# -----------------------------------------------------------


def load_ground_truth():
    """Load the ground truth file built by 08_build_ground_truth.py."""
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise FileNotFoundError(
            f"{GROUND_TRUTH_FILE} not found. Run 09_build_ground_truth.py first."
        )
    print("Loading ground truth from", GROUND_TRUTH_FILE)
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# RETRIEVAL EVALUATION  ("did we find the right chunk?")
# =========================================================
def evaluate_retrieval(eval_item, top_k):
    """
    Ask the retriever for the top_k chunks for this question, then check:
      - did ANY of them come from the correct document?  (a "hit")
      - at what rank (1st, 2nd, 3rd...) did the correct document appear?

    Also returns the retrieved chunks so we can inspect the "context"
    that would have been fed to the LLM (this makes context building
    visible and logged, instead of a black box).
    """
    retrieved_chunks = rag.retrieve_context(eval_item["question"], number_of_chunks=top_k)

    rank = None
    for position, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["document_id"] == eval_item["document_id"]:
            rank = position
            break

    return {
        "hit": rank is not None,
        "rank": rank,
        "retrieved_chunks": retrieved_chunks,
    }


# =========================================================
# ANSWER EVALUATION ("was the final answer correct?")
# =========================================================
def normalize_text(text):
    """Lowercase, strip punctuation/extra spaces -- makes comparisons fair.
    e.g. "Paris." and "paris" should count as the same answer."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction, gold_answer):
    """1.0 if the normalized correct answer text appears in the model's
    answer, else 0.0."""
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold_answer)
    if gold_norm == "":
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def f1_score(prediction, gold_answer):
    """
    Token-overlap F1, the standard beginner-friendly QA metric.
    It rewards partial credit: if the answer shares some words with the
    correct answer but isn't a perfect match, it still scores > 0.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    common = {}
    for token in pred_tokens:
        if token in gold_tokens:
            common[token] = common.get(token, 0) + 1

    num_common = sum(min(common.get(t, 0), gold_tokens.count(t)) for t in set(pred_tokens))
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(eval_item):
    """Run the FULL rag pipeline (retrieve + prompt + LLM) and grade the answer."""
    try:
        result = rag.answer_question(eval_item["question"], number_of_chunks=TOP_K)
        predicted_answer = result.get("answer", "")
    except Exception as exc:
        predicted_answer = f"[Evaluation error: {exc}]"

    return {
        "predicted_answer": predicted_answer,
        "exact_match": exact_match(predicted_answer, eval_item["answer"]),
        "f1": f1_score(predicted_answer, eval_item["answer"]),
    }


# =========================================================
# RUN EVERYTHING AND COLLECT RESULTS
# =========================================================
def run_evaluation(ground_truth):
    rows = []
    for i, eval_item in enumerate(ground_truth, start=1):
        print(f"Evaluating {i}/{len(ground_truth)}: {eval_item['question'][:60]}...")

        retrieval_result = evaluate_retrieval(eval_item, TOP_K)
        answer_result = evaluate_answer(eval_item)

        context_preview = " | ".join(
            c["title"] for c in retrieval_result["retrieved_chunks"]
        )

        rows.append({
            "question": eval_item["question"],
            "correct_answer": eval_item["answer"],
            "correct_document_id": eval_item["document_id"],
            "retrieval_hit": retrieval_result["hit"],
            "retrieval_rank": retrieval_result["rank"],
            "retrieved_context_titles": context_preview,
            "predicted_answer": answer_result["predicted_answer"],
            "exact_match": answer_result["exact_match"],
            "f1_score": round(answer_result["f1"], 3),
        })

    return rows


# =========================================================
# SUMMARIZE METRICS
# =========================================================
def summarize(rows):
    n = len(rows)
    hits = [r for r in rows if r["retrieval_hit"]]

    hit_rate = len(hits) / n if n else 0.0
    # Mean Reciprocal Rank: 1/rank if found, 0 if not found, averaged.
    reciprocal_ranks = [1.0 / r["retrieval_rank"] if r["retrieval_hit"] else 0.0 for r in rows]
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    exact_match_rate = sum(r["exact_match"] for r in rows) / n if n else 0.0
    avg_f1 = sum(r["f1_score"] for r in rows) / n if n else 0.0

    return {
        "n_questions": n,
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "exact_match_rate": exact_match_rate,
        "avg_f1": avg_f1,
    }


def print_report(summary):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY  (plain-English explanation included)")
    print("=" * 60)
    print(f"Questions tested:        {summary['n_questions']}")
    print()
    print(f"RETRIEVAL - Hit Rate@{TOP_K}:   {summary['hit_rate_at_k']:.1%}")
    print("  -> % of questions where the correct source document was")
    print(f"     somewhere in the top {TOP_K} retrieved chunks.")
    print()
    print(f"RETRIEVAL - MRR:              {summary['mrr']:.3f}")
    print("  -> Rewards finding the right chunk EARLY (rank 1 is best).")
    print("     1.0 = always found at rank 1. 0.0 = never found.")
    print()
    print(f"ANSWER - Exact Match rate:    {summary['exact_match_rate']:.1%}")
    print("  -> % of answers where the correct answer text appears")
    print("     in the model's response.")
    print()
    print(f"ANSWER - Average F1 score:    {summary['avg_f1']:.3f}")
    print("  -> Partial-credit word-overlap score between the model's")
    print("     answer and the correct answer (0 = no overlap, 1 = perfect).")
    print("=" * 60 + "\n")


# =========================================================
# VISUALIZE
# =========================================================
def make_charts(rows, summary, output_path):
    if plt is None:
        print("matplotlib is not installed; skipping chart generation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("RAG Pipeline Evaluation", fontsize=14, fontweight="bold")

    # --- Chart 1: Retrieval metrics ---
    ax1 = axes[0]
    labels1 = [f"Hit Rate@{TOP_K}", "MRR"]
    values1 = [summary["hit_rate_at_k"], summary["mrr"]]
    bars1 = ax1.bar(labels1, values1, color=["#4C72B0", "#55A868"])
    ax1.set_ylim(0, 1)
    ax1.set_title("Retrieval Quality")
    ax1.set_ylabel("Score (0-1)")
    for bar, value in zip(bars1, values1):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                  ha="center", fontweight="bold")

    # --- Chart 2: Answer quality metrics ---
    ax2 = axes[1]
    labels2 = ["Exact Match", "Avg F1"]
    values2 = [summary["exact_match_rate"], summary["avg_f1"]]
    bars2 = ax2.bar(labels2, values2, color=["#C44E52", "#8172B2"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Answer Quality")
    ax2.set_ylabel("Score (0-1)")
    for bar, value in zip(bars2, values2):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                  ha="center", fontweight="bold")

    # --- Chart 3: distribution of retrieval ranks (where the answer was found) ---
    ax3 = axes[2]
    ranks = [r["retrieval_rank"] for r in rows if r["retrieval_rank"] is not None]
    misses = len(rows) - len(ranks)
    rank_labels = [str(r) for r in range(1, TOP_K + 1)] + ["Not found"]
    rank_counts = [ranks.count(r) for r in range(1, TOP_K + 1)] + [misses]
    ax3.bar(rank_labels, rank_counts, color="#DD8452")
    ax3.set_title("Where Was the Right Chunk Found?")
    ax3.set_xlabel("Rank position")
    ax3.set_ylabel("Number of questions")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print("Saved chart to", output_path)


def save_details_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved per-question details to", path)


# =========================================================
# MAIN
# =========================================================
def main():
    ground_truth = load_ground_truth()

    rows = run_evaluation(ground_truth)
    summary = summarize(rows)

    print_report(summary)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    save_details_csv(rows, DETAILS_CSV)
    make_charts(rows, summary, SUMMARY_PNG)

    print("\nDone! Open", SUMMARY_PNG, "to see the charts,")
    print("and", DETAILS_CSV, "to inspect exactly what was retrieved")
    print("and answered for every single test question.")


if __name__ == "__main__":
    main()
