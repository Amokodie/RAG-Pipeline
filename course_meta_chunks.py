"""
Course / site FAQ chunks — prepended to the IT KB so retrieval can answer
“what is this site”, AeroFleet vs Session 7, RAG overview, etc.
"""

from __future__ import annotations

# (title, category, body, source)
COURSE_META_DOCUMENTS: list[tuple[str, str, str, str]] = [
    (
        "What this Streamlit application is (Session 7)",
        "Course_meta",
        "This **main** Streamlit app (`app.py`) is the **Session 6–8 / Assignment 3 RAG concept demo**. "
        "It loads a small **alignment audit CSV** (cases like H01, O01, S04, B01) about helpfulness, safety, honesty, and bias. "
        "It is **not** the AeroFleet drone manual. Retrieval uses TF‑IDF and optional hybrid BM25+dense embeddings; tabs show analysis, live retrieval, Ask AI, and Advanced settings. "
        "If you ask about **AeroFleet** here, retrieval may still pick an unrelated audit row unless you use the dedicated AeroFleet app (see separate entry).",
        "curated",
    ),
    (
        "AeroFleet X200 — where that content lives",
        "Course_meta",
        "**AeroFleet X200** is a **fictional** battery cooling / maintenance **engineering corpus** (documents D01–D10) used in **Session 8** exercises. "
        "It lives in this repo under `rag_session8_classroom_exercise/` and has its **own** Streamlit entry: run `streamlit run rag_session8_classroom_exercise/aerofleet_rag_app.py`. "
        "That app uses **FAISS** + sentence-transformers over markdown manuals — not the Session 7 CSV. "
        "Do not confuse AeroFleet procedures with the alignment-audit case IDs in this app.",
        "curated",
    ),
    (
        "What is RAG (in this course)",
        "Course_meta",
        "**Retrieval-Augmented Generation (RAG)** means: (1) **chunk** documents, (2) **embed** chunks into vectors, (3) **retrieve** top‑k chunks for a user query, (4) **condition** an LLM (or show grounded text) so answers cite retrieved facts instead of guessing. "
        "This reduces **hallucination** compared to using the model alone. This demo compares TF‑IDF, dense/hybrid retrieval, and optional OpenAI with strict grounding.",
        "curated",
    ),
    (
        "Are you a bot",
        "Course_meta",
        "The **Ask AI** area is a **retrieval + template / API** interface: it is not a human. "
        "Offline mode stitches instructor notes and a local SQLite **IT knowledge base**; with an OpenAI key it calls the API with **grounding rules**. "
        "There is no persistent consciousness — only your query, retrieved rows, and the prompt.",
        "curated",
    ),
    (
        "Session 7 vs Session 8 in this repository",
        "Course_meta",
        "**Session 7** material is the alignment-audit dataset and this **RAG Concept Demo** (`app.py`). "
        "**Session 8** adds the **AeroFleet** classroom exercise folder with corpus D01–D10, eval queries, and `aerofleet_rag_app.py`. "
        "Use the correct app for each assignment part.",
        "curated",
    ),
    (
        "Why a wrong audit row sometimes appears",
        "Course_meta",
        "If your question does not match any row in the CSV well, **semantic retrieval** still returns the *least bad* chunk — which can look unrelated (e.g. a Safety case when you asked about AeroFleet). "
        "Check **audit relevance score** and the **Course_meta** / IT KB passages. For AeroFleet-specific answers, open the **AeroFleet** Streamlit app instead.",
        "curated",
    ),
    (
        "3D plots and LSA in this demo",
        "Course_meta",
        "The **Analysis & 3D** tab projects TF‑IDF chunks with **LSA** (truncated SVD) for visualization. "
        "**Ask AI** can show a compact **3D LSA** view of the query vs chunks after each question. "
        "3D is for intuition — actual retrieval still uses full TF‑IDF cosine (or hybrid) in high dimensions, not 3D distance.",
        "curated",
    ),
]
