streamlit run streamlit_app.py

import os
import requests
from importlib import import_module
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if one exists

retrieve_module = import_module("06_retrieve_context")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question, retrieved_chunks):
    """Combine the question and retrieved chunks into one grounded prompt."""
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append("[Source " + str(i) + "] " + chunk["title"])
        context_lines.append(chunk["chunk_text"])
        context_lines.append("")

    context_text = "\n".join(context_lines)

    prompt = "You are a helpful assistant that answers questions using ONLY the context below.\n"
    prompt += "If the context doesn't contain the answer, say 'I don't know based on the given context.'\n"
    prompt += "Always mention which source number(s) you used.\n\n"
    prompt += "Context:\n" + context_text + "\n"
    prompt += "Question: " + question + "\n"
    prompt += "Answer:"
    return prompt


def call_llm(prompt):
    """Send the prompt to an LLM through OpenRouter and return its answer text."""
    headers = {
        "Authorization": "Bearer " + OPENROUTER_API_KEY,
        "Content-Type": "application/json",
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
    answer = call_llm(prompt)
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
