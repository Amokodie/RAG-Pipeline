# Starter Files

## Minimum Setup
Install:
```bash
pip install -r requirements.txt
```

## Run Python Starter
```bash
python rag_lab_starter.py
```

## What this starter does
- loads the corpus
- chunks documents by paragraph
- builds a lightweight TF-IDF retriever
- prints top-3 results for selected queries

## Why this is only a baseline
This starter is intentionally simple so it can run quickly in class.

It does **not** yet:
- use semantic embeddings
- filter outdated documents
- rerank by authority
- force evidence extraction

Your job is to inspect the retrieval and decide where it fails.
