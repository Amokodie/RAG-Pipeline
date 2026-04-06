"""
Streamlit RAG demo: AeroFleet X200 battery cooling — FAISS + authority-aware retrieval.
Run: streamlit run aerofleet_rag_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from aerofleet_pipeline import (
    AeroFleetIndex,
    apply_authority_filter,
    build_context_for_llm,
    catalog_path,
    load_all_chunks,
    load_catalog,
)

STRICT_SYSTEM = """You are a technical assistant for AeroFleet X200 battery cooling maintenance.

Rules:
- Answer ONLY using the CONTEXT passages below. Each passage is labeled with document id and status.
- After factual statements, add a citation like [Source: D02] using the document id from the passage header.
- If the CONTEXT does not contain enough information to answer, reply with exactly:
  I cannot find this in the technical manual.
- Never invent service bulletin numbers, dates, or thresholds not present in CONTEXT.
- If CONTEXT includes both current (D02) and outdated (D03) material, IGNORE any numbers from D03; D02 is the maintenance authority for intervals.
"""


def _llm_answer(api_key: str, model: str, user_query: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": STRICT_SYSTEM + "\n\nCONTEXT:\n" + context},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()


def _offline_fallback(trace_rows: list[tuple]) -> str:
    """No API key: show grounded excerpts from top chunks."""
    lines = [
        "*OpenAI API key not set — showing **extractive** grounded excerpts from retrieved chunks.*\n",
        "**Top passages (by similarity):**",
    ]
    for c, sc in trace_rows[:3]:
        lines.append(
            f"- **[{c.doc_id}]** (score {sc:.3f}) — {c.text[:500]}{'…' if len(c.text) > 500 else ''}"
        )
    lines.append(
        "\nAdd `OPENAI_API_KEY` in the sidebar (or environment) for a synthesized answer with citations."
    )
    return "\n\n".join(lines)


@st.cache_resource
def build_faiss_index():
    chunks, _cat = load_all_chunks()
    idx = AeroFleetIndex()
    idx.fit(chunks)
    return idx


def main() -> None:
    st.set_page_config(page_title="AeroFleet X200 RAG Demo", page_icon="🔋", layout="wide")
    st.title("AeroFleet X200 — Battery cooling RAG demo")
    st.markdown(
        "**Retrieval-Augmented Generation** over the D01–D10 markdown corpus with **FAISS** vector search, "
        "**authority rules** (D02 overrides D03), and optional **OpenAI** generation."
    )

    try:
        catalog = load_catalog()
        index = build_faiss_index()
    except Exception as e:
        st.error(f"Failed to load corpus or build index: {e}")
        st.stop()

    with st.sidebar:
        st.subheader("OpenAI (optional)")
        api_key = st.text_input("API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        model = st.text_input("Model", value="gpt-4o-mini")
        st.caption("Without a key, answers show retrieved excerpts only.")
        st.divider()
        st.subheader("Corpus")
        st.dataframe(
            catalog[["doc_id", "title", "status", "effective_date", "authority_level"]],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Catalog: `{catalog_path().name}` · Chunks use ~{200} words with ~{50}-word overlap.")

    q = st.text_input(
        "Maintenance / operations question",
        placeholder='e.g. What is the first troubleshooting step for uneven fan draw?',
    )
    run = st.button("Run RAG query", type="primary")

    if run and q.strip():
        query = q.strip()
        with st.spinner("Embedding query + searching FAISS…"):
            raw_hits = index.search(query, k=20)
            context_rows, auth_notes = apply_authority_filter(raw_hits, retrieval_k=5, pool_size=20)
            context_block = build_context_for_llm(context_rows)
            trace_display = raw_hits[:3]

        st.subheader("1) Answer")
        for note in auth_notes:
            st.warning(note)

        if api_key.strip():
            try:
                ans = _llm_answer(api_key.strip(), model.strip() or "gpt-4o-mini", query, context_block)
                st.success(ans)
            except Exception as ex:
                st.error(str(ex))
                st.markdown(_offline_fallback(trace_display))
        else:
            st.info(_offline_fallback(trace_display))

        st.subheader("2) Retrieval trace (Top-3 raw hits, before authority filter)")
        trace_data = []
        for i, (c, sc) in enumerate(trace_display, 1):
            trace_data.append(
                {
                    "rank": i,
                    "doc_id": c.doc_id,
                    "similarity": round(sc, 5),
                    "status": c.status,
                    "chunk_excerpt": c.text[:320] + ("…" if len(c.text) > 320 else ""),
                }
            )
        st.dataframe(pd.DataFrame(trace_data), hide_index=True, use_container_width=True)

        st.subheader("3) Chunks used for generation (after authority filter)")
        used_data = []
        for i, (c, sc) in enumerate(context_rows, 1):
            used_data.append(
                {
                    "rank": i,
                    "doc_id": c.doc_id,
                    "similarity": round(sc, 5),
                    "effective_date": c.effective_date,
                    "authority_level": c.authority_level,
                    "excerpt": c.text[:280] + ("…" if len(c.text) > 280 else ""),
                }
            )
        st.dataframe(pd.DataFrame(used_data), hide_index=True, use_container_width=True)

        st.subheader("4) Source metadata (documents touched)")
        seen = {}
        for c, _ in context_rows:
            if c.doc_id not in seen:
                row = catalog[catalog["doc_id"] == c.doc_id].iloc[0]
                seen[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "title": row["title"],
                    "status": row["status"],
                    "effective_date": row["effective_date"],
                    "authority_level": row["authority_level"],
                }
        st.dataframe(pd.DataFrame(list(seen.values())), hide_index=True, use_container_width=True)

    elif run:
        st.warning("Enter a non-empty question.")

    with st.expander("Demo test ideas (from assignment brief)"):
        st.markdown(
            """
- **Q:** *What is the first troubleshooting step for uneven fan draw?*  
  Expect D06 (+ D09 glossary); first step: inspect connector before replacing fan.

- **Q:** *What is the fan inspection interval?*  
  Expect **D02**: 80 cycles / 30 days — not D03’s 100 cycles.

- **Q:** *What does D10 say about fast charging?*  
  Expect lab summary: stable below 45 °C core, etc.
"""
        )


if __name__ == "__main__":
    main()
