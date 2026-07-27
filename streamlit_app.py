"""
streamlit_app.py
-------------------
The final deployed app. Type a general-knowledge question and get an answer
that's grounded in the Natural Questions Wikipedia passages we indexed in
Steps 1-5.

Run locally:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing rag so rag.py picks up the variables on import
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import streamlit as st
import rag

# ---- Always ensure rag has the API key ----
# Force-set from environment (loaded from .env above) or Streamlit secrets.
_key = os.environ.get("OPENROUTER_API_KEY", "")
_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Try Streamlit secrets as a fallback (for Streamlit Cloud deployment)
if not _key or _key == "your_openrouter_key_here":
    try:
        _key = st.secrets.get("OPENROUTER_API_KEY", _key)
        _model = st.secrets.get("OPENROUTER_MODEL", _model)
    except Exception:
        pass

rag.OPENROUTER_API_KEY = _key
rag.OPENROUTER_MODEL = _model

st.set_page_config(page_title="Natural Questions RAG Assistant", page_icon="🔎")

st.title("🔎 Natural Questions")
st.write(
    "Ask a general-knowledge question. "
)

if not rag.OPENROUTER_API_KEY:
    st.warning(
        "No OpenRouter API key found. Set OPENROUTER_API_KEY in a local .env "
        "file, or add it to Streamlit secrets when deployed."
    )

question = st.text_input("Your question:")

if st.button("Ask") and question.strip() != "":
    with st.spinner("Searching and thinking..."):
        result = None
        try:
            result = rag.answer_question(question, number_of_chunks=4)
        except Exception as e:
            st.error("Something went wrong: " + str(e))

    if result is not None:
        st.subheader("Answer")
        st.write(result["answer"])
