# RAG Pipeline (Alignment Audit Demo)

Streamlit app for Assignment 3: TF‑IDF / semantic / hybrid retrieval over `session7_alignment_audit_dataset.csv`, plus optional **IT knowledge base** (SQLite) in **Ask AI**, and a separate **AeroFleet X200** FAISS RAG demo (Session 8 corpus).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Other entry points:

- **Alignment-only mini UI:** `streamlit run rag_demo.py`
- **AeroFleet X200 (D01–D10, FAISS + authority D02 vs D03):**
  ```bash
  cd rag_session8_classroom_exercise
  streamlit run aerofleet_rag_app.py
  ```
  Or from repo root: `streamlit run rag_session8_classroom_exercise/aerofleet_rag_app.py`

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already connected if you use `origin`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → **Deploy an app**.
4. Select repository **`Amokodie/RAG-Pipeline`**, branch **`main`**.
5. **Main file path:** `app.py` (main alignment-audit demo)
6. Click **Deploy**.

**Optional second app (AeroFleet demo):** create another deployment with **Main file path:**  
`rag_session8_classroom_exercise/aerofleet_rag_app.py`

First boot installs dependencies (`torch`, `sentence-transformers`, `faiss-cpu`, etc.) and may take several minutes. The **Advanced** tab downloads an embedding model on first use; the AeroFleet app also downloads `all-MiniLM-L6-v2` on first query. If the app restarts or hits memory limits on the free tier, try again after cold start finishes.

### Optional: OpenAI (Advanced tab)

In the Cloud app → **Settings** → **Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
```

The app reads this for the optional LLM button (same as local `.env`).
