"""
Conversational Q&A over the alignment audit: retrieval + grounded context (+ optional OpenAI).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def resolve_row_index(df: pd.DataFrame, case_id: str) -> int:
    m = df.index[df["case_id"].astype(str) == str(case_id)].tolist()
    return int(m[0]) if m else 0


def retrieve_best_case(
    df: pd.DataFrame,
    retriever: Any,
    hybrid: Any | None,
    query: str,
) -> tuple[int, str, str]:
    """
    Returns (row_index, case_id, method_label).
    Prefers hybrid BM25+dense when `hybrid` is available; else TF‑IDF.
    """
    q = query.strip()
    if not q:
        return 0, str(df.iloc[0]["case_id"]), "default"

    if hybrid is not None:
        try:
            hits = hybrid.search(q, top_k=1, alpha=0.55, fusion="weighted")
            if hits:
                h0 = hits[0]
                return h0.row_index, h0.case_id, "hybrid (BM25 + dense)"
        except Exception:
            pass

    hits = retriever.query(q, top_k=1)
    if hits:
        h0 = hits[0]
        pos = int(h0.chunk_index)
        if 0 <= pos < len(df):
            return pos, h0.case_id, "TF‑IDF"
        pos = resolve_row_index(df, h0.case_id)
        return pos, h0.case_id, "TF‑IDF"
    return 0, str(df.iloc[0]["case_id"]), "default (no match)"


def offline_markdown_answer(
    df: pd.DataFrame,
    row_index: int,
    case_id: str,
    instructor_text: str,
    method: str,
) -> str:
    row = df.iloc[row_index]
    return (
        f"**Retrieved row:** `{case_id}` · *{row['category']}* · {row['subcategory']}\n\n"
        f"*Retrieval:* {method}\n\n"
        f"**Grounded answer (course-aligned):**\n\n{instructor_text}\n\n"
        "---\n"
        f"*Original audit prompt:* {row['user_prompt'][:400]}{'…' if len(str(row['user_prompt'])) > 400 else ''}"
    )


def openai_conversational_answer(
    *,
    api_key: str,
    model: str,
    user_query: str,
    context_block: str,
) -> str:
    """Short conversational reply; still grounded on context only."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    system = (
        "You are a friendly teaching assistant for a **foundation models & alignment** lab. "
        "The student is looking at a small **alignment audit** CSV with case IDs (H01, O01, B01, …).\n\n"
        "Rules:\n"
        "- Use ONLY the RETRIEVED_AUDIT_ROW below. Do not invent papers, venues, or facts.\n"
        "- Answer in 2–6 short paragraphs or bullets. You may restate the instructor notes clearly.\n"
        "- If the question is only loosely related, say what the retrieved case covers and what it does not.\n\n"
        "RETRIEVED_AUDIT_ROW:\n"
        f"{context_block}"
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ],
        temperature=0.35,
        max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()
