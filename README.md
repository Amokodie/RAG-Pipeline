# RAG Pipeline (Alignment Audit Demo)

Streamlit app for Assignment 3: TF‑IDF / semantic / hybrid retrieval over `session7_alignment_audit_dataset.csv`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional semantic-only UI: `streamlit run rag_demo.py`

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already connected if you use `origin`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → **Deploy an app**.
4. Select repository **`Amokodie/RAG-Pipeline`**, branch **`main`**.
5. **Main file path:** `app.py`
6. Click **Deploy**.

First boot installs dependencies (`torch`, `sentence-transformers`, etc.) and may take several minutes. The **Advanced** tab downloads an embedding model on first use; if the app restarts or hits memory limits on the free tier, try again after cold start finishes.

### Optional: OpenAI (Advanced tab)

In the Cloud app → **Settings** → **Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
```

The app reads this for the optional LLM button (same as local `.env`).
