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
    merge_retrieval_with_safety_priority,
)

# AeroFleet Technical Guardrail — strict non-parametric use of D01–D10 only
AEROFLEET_TECHNICAL_GUARDRAIL = """You are the **AeroFleet X200 Technical Support Engine**. Your sole purpose is to provide maintenance and operational guidance based **exclusively** on the provided technical documentation passages (D01–D10) in CONTEXT below.

**Core instruction (anti-hallucination):**
1. **Strict grounding:** Do **not** use internal knowledge about drones, batteries, or engineering from your pre-training. If the answer is not contained within the CONTEXT passages, state exactly:
   I am sorry, but the provided technical manuals do not contain information to answer this specific query.
2. **Authority hierarchy (always respect document status in CONTEXT headers):**
   - **D02** (Maintenance Manual v2.1) is **CURRENT AUTHORITY** for inspection intervals and maintenance thresholds.
   - **D03** is **OUTDATED ARCHIVE** — never use D03 to override **D02**; if both appear, ignore superseded numbers and intervals from D03; follow **D02**.
   - **D08** is **informational only** — it does **not** change inspection intervals, thresholds, or maintenance rules from D02 or service bulletins.
3. **Synonym mapping:** If the user uses informal language, align terms using **Glossary (D09)** when it appears in CONTEXT (e.g. map phrases like "uneven fan draw" to concepts defined there such as fan current imbalance / Pack-ΔT when CONTEXT supports it).
4. **Safety first:** If the user mentions smoke, odor, visible vapor/fumes, or similar during charging, **prioritize procedures from D07 (Emergency Safety Protocol)** in CONTEXT before routine maintenance tips.

**Retrieval & response strategy (already applied upstream; reflect in your answer):**
- Use only the Top passages provided in CONTEXT.
- If CONTEXT includes superseded D03 material alongside D02, **prioritize D02** for any conflicting numbers.
- For every factual claim, include the source document id, e.g. [Source: D02] or [Source: D07].

**Output format (use these exact section headings):**
**Answer:** [Precise technical response grounded only in CONTEXT]

**Citations:** [Comma-separated list of document IDs used, e.g. D02, D06, D09]

**Confidence Level:** [High / Medium / Low — High when passages directly answer the question; Medium when inference from glossary/synonym mapping is needed; Low when CONTEXT is thin or ambiguous]
"""


def _llm_answer(api_key: str, model: str, user_query: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": AEROFLEET_TECHNICAL_GUARDRAIL + "\n\n--- CONTEXT (retrieved passages) ---\n" + context,
            },
            {"role": "user", "content": user_query},
        ],
        temperature=0.15,
        max_tokens=1000,
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
    st.set_page_config(
        page_title="AeroFleet X200 | BCU Technical Databank",
        page_icon="🔋",
        layout="wide",
    )
    st.title("AeroFleet X200 — Battery Cooling Technical Databank")
    st.caption(
        "Manual-grounded RAG assistant · Corpus D01–D10 · FAISS semantic search · Authority-aware (D02/D03/D08) · "
        "Safety-prioritized retrieval (D07) · Strict technical guardrails when using OpenAI."
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
        with st.expander("Technical Guardrail (system prompt)"):
            st.markdown(
                "The LLM uses **CONTEXT-only** answers, refusal if missing, **D02>D03**, **D08** non-binding, "
                "**D09** synonym bridge, **D07** priority when charging + smoke/odor cues fire (retrieval merge)."
            )

    q = st.text_input(
        "Maintenance / operations question",
        placeholder='e.g. What is the first troubleshooting step for uneven fan draw?',
    )
    run = st.button("Run RAG query", type="primary")

    if run and q.strip():
        query = q.strip()
        with st.spinner("Embedding query + searching FAISS…"):
            raw_hits = index.search(query, k=20)
            raw_hits, safety_notes = merge_retrieval_with_safety_priority(index, query, raw_hits)
            context_rows, auth_notes = apply_authority_filter(raw_hits, retrieval_k=5, pool_size=20)
            context_block = build_context_for_llm(context_rows)
            trace_display = raw_hits[:3]

        st.subheader("1) Answer")
        for note in safety_notes:
            st.info(note)
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
