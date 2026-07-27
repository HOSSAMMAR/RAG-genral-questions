# Natural Questions RAG Assistant

A simple Retrieval-Augmented Generation (RAG) project that answers general
knowledge questions, grounded in real Wikipedia passages from Google's
[Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions)
dataset.

## Pipeline

```
01_documents.py            -> download raw NQ question/document pairs
02_preprocessing.py        -> clean the text
03_chunking.py              -> split long passages into smaller chunks
04_vector_representation.py -> turn chunks into embeddings (OpenRouter)
05_create_chroma_store.py   -> save embeddings into a local ChromaDB store
06_retrieve_context.py      -> given a question, find the best-matching chunks
07_prompting.py              -> build a grounded prompt and ask the LLM (OpenRouter)
rag.py                       -> combines steps 6 + 7 for the live app
streamlit_app.py             -> the deployed chat UI
```

## 1. Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your real OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Get a key at [openrouter.ai](https://openrouter.ai/). **Never commit your real
`.env` file or put your key directly in any `.py` file.**

## 2. Build the knowledge base (run once, in order)

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
```

Each script prints its progress and saves its output into the `data/` folder
(or, for step 5, into a `chroma_store/` folder) so you can check the results
of each stage before moving to the next one.

## 3. Try retrieval and prompting on their own (optional, but useful for debugging)

```bash
python 06_retrieve_context.py
python 07_prompting.py
```

## 4. Run the app locally

```bash
streamlit run streamlit_app.py
```

## 5. Deploy to Streamlit Cloud

1. Push this project to a GitHub repository (make sure `.env` is **not**
   included -- check `.gitignore`).
2. Also push the `data/` and `chroma_store/` folders you generated in step 2
   above, since the deployed app needs the vector store to already exist.
3. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at your repo and `streamlit_app.py`.
4. In your app's **Settings -> Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```

## Final submission checklist

- [ ] All required files exist (see the pipeline list above)
- [ ] `requirements.txt` exists
- [ ] Your real API key is **not** in the ZIP or the GitHub repo
- [ ] Streamlit secrets are set up in valid TOML format
- [ ] The deployed Streamlit app runs successfully
- [ ] Answers use retrieved context
- [ ] Answers cite sources
