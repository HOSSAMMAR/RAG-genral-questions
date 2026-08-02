"""
context_building.py
---------------------
This is the "context building" step of the RAG pipeline: turning the raw
chunks that 06_retrieve_context.py found into the actual text block that
gets handed to the LLM.

WHY IS THIS ITS OWN FILE?
--------------------------
Before this split, the exact same build_prompt() function was copy-pasted
in both rag.py and 07_prompting.py. That's a problem for two reasons:
    1. If you improve context building (e.g. add a relevance cutoff, or
       de-duplicate near-identical chunks), you'd have to remember to
       change it in TWO places or they silently drift apart.
    2. It hides an important pipeline step inside files that are really
       about something else (running the app / running a one-off script).

Putting it here makes context building a visible, testable step of the
pipeline that sits between:
    06_retrieve_context.py  (finds the chunks)
    08_prompting.py         (asks the LLM using the context built here)

WHAT THIS FILE DOES
--------------------
    1. Takes the list of retrieved chunks (from retrieve_context()).
    2. Optionally filters out low-quality matches (score cutoff).
    3. Optionally removes near-duplicate chunks.
    4. Formats everything into a labeled [Source N] context block.
    5. Wraps that context block + the question into the final prompt text.
"""

import re


# ---------------- Settings you can change ----------------
# If a chunk's retrieval distance is above this, we drop it instead of
# stuffing it into the prompt. Chroma's default distance is smaller = more
# similar, so a HIGH number here means "let almost everything through".
# Lower this once you've checked what typical distances look like for
# good vs. bad matches in your own data.
MAX_DISTANCE = None  # e.g. set to 0.8 to start filtering; None = no filtering

# If two retrieved chunks overlap this much in wording, treat the second
# one as a near-duplicate and drop it (keeps context focused, not repetitive).
DUPLICATE_WORD_OVERLAP_THRESHOLD = 0.9
# -----------------------------------------------------------


def _word_set(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def filter_by_relevance(retrieved_chunks, max_distance=MAX_DISTANCE):
    """
    Drop chunks whose retrieval distance is worse than max_distance.
    Skipped automatically if a chunk has no 'distance' field, or if
    max_distance is None (filtering disabled).
    """
    if max_distance is None:
        return retrieved_chunks

    filtered = []
    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue
        distance = chunk.get("distance")
        if distance is None or distance <= max_distance:
            filtered.append(chunk)
    return filtered


def remove_near_duplicates(retrieved_chunks, overlap_threshold=DUPLICATE_WORD_OVERLAP_THRESHOLD):
    """
    Walk through the retrieved chunks in order and drop any chunk whose
    words overlap too heavily with a chunk we've already kept. This stops
    the same passage (e.g. two overlapping chunk windows from one document)
    from taking up two of your limited context slots.
    """
    kept_chunks = []
    kept_word_sets = []

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue
        words = _word_set(chunk.get("chunk_text", ""))
        is_duplicate = False

        for existing_words in kept_word_sets:
            if len(words) == 0 or len(existing_words) == 0:
                continue
            overlap = len(words & existing_words) / len(words | existing_words)
            if overlap >= overlap_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_chunks.append(chunk)
            kept_word_sets.append(words)

    return kept_chunks


def format_context_block(retrieved_chunks):
    """Turn the list of chunks into the labeled '[Source N] ...' text block
    that gets shown to the LLM."""
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("title", "Untitled source")
        chunk_text = chunk.get("chunk_text", "")
        context_lines.append("[Source " + str(i) + "] " + title)
        context_lines.append(chunk_text)
        context_lines.append("")
    return "\n".join(context_lines)


def build_context(retrieved_chunks):
    """
    Full context-building pipeline: filter low-relevance chunks, remove
    near-duplicates, then format into the final context text block.
    Returns (context_text, chunks_actually_used) so callers can log/inspect
    exactly what made it into the prompt.
    """
    relevant_chunks = filter_by_relevance(retrieved_chunks)
    deduped_chunks = remove_near_duplicates(relevant_chunks)
    context_text = format_context_block(deduped_chunks)
    return context_text, deduped_chunks


def build_prompt(question, retrieved_chunks, allow_general_knowledge=False):
    """
    Combine the question and retrieved chunks into the final prompt text
    sent to the LLM.

    allow_general_knowledge:
        False -> strict mode (matches 07_prompting.py): answer ONLY from
                 context, say "I don't know" if it's not there.
        True  -> hybrid mode (matches rag.py): use context when it has the
                 answer, otherwise fall back to general knowledge.
    """
    context_text, chunks_used = build_context(retrieved_chunks)

    if allow_general_knowledge:
        instructions = (
            "You are a helpful assistant.\n"
            "1. If the provided context below contains the answer, answer the "
            "question and cite which source number(s) you used.\n"
            "2. If the provided context does NOT contain the answer, answer "
            "the question accurately using your general knowledge.\n\n"
        )
    else:
        instructions = (
            "You are a helpful assistant that answers questions using ONLY "
            "the context below.\n"
            "If the context doesn't contain the answer, say 'I don't know "
            "based on the given context.'\n"
            "Always mention which source number(s) you used.\n\n"
        )

    prompt = (
        instructions
        + "Context:\n" + context_text + "\n"
        + "Question: " + question + "\n"
        + "Answer:"
    )
    return prompt, chunks_used
