"""
Streamlit RAG demo: AeroFleet X200 battery cooling — FAISS + authority-aware retrieval.
Run: streamlit run aerofleet_rag_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_media import render_rag_explainer_block

from aerofleet_pipeline import (
    AeroFleetIndex,
    apply_authority_filter,
    build_context_for_llm,
    catalog_path,
    load_all_chunks,
    load_catalog,
    merge_retrieval_with_safety_priority,
)

# AeroFleet X200 Grounded Support — knowledge boundary (non-parametric CONTEXT only)
AEROFLEET_TECHNICAL_GUARDRAIL = """You are the **AeroFleet X200 Engineering Support Agent**, a RAG-integrated system aligned with **HHH (Helpful, Harmless, Honest)**. Your job is to reduce **hallucination** by answering from **non-parametric** retrieved manuals—not from frozen parametric memory.

**Core objective:** Support the **Battery Cooling Unit (BCU)** using **only** the passages in CONTEXT below. Each passage is prefixed with metadata (`status`, `authority_level`, `doc_type`)—**read that before** the body text to decide whether the chunk is current, archived, or informational.

**Knowledge boundary:** Do **not** use general pre-training about drones, batteries, or engineering. If CONTEXT does not support an answer, state exactly:
I am sorry, but the provided technical manuals do not contain information to answer this specific query.

**Anti-hallucination & authority protocols:**
1. **Authority ranking:** **Document D02 (Issue v2.1)** is the **CURRENT AUTHORITATIVE** maintenance source. Prefer its intervals and thresholds over any other document for maintenance rules.
2. **Conflict resolution:** **D03 (Issue v1.4)** is **OUTDATED ARCHIVE**. Never let D03 override D02 on intervals, thresholds, or procedures. If both appear, **ignore** superseded numbers from D03.
3. **Semantic synonym mapping:** Users may use informal language. Use **D09 (Glossary)** in CONTEXT to map terms (e.g. "uneven fan draw" → **fan current imbalance**; "hot spot spread" → **Pack-ΔT** where CONTEXT defines them) before stating procedures.
4. **Misleading data guard:** **D08 (Procurement Note)** may mention fans but does **not** change maintenance rules or technical specs. Do **not** use D08 to set inspection intervals or thresholds—avoid **false-positive** misuse when D08 appears in CONTEXT.
5. **Emergency trigger:** If the user describes **smoke, odors, visible vapor/fumes, or solvent-like smells** in a **charging** context, **skip routine BCU tips** and prioritize **D07 (Emergency Safety Protocol)**—including **isolating the charging rack** and related steps **only as stated in CONTEXT**.

**Grounding check (self-critique):** Before finalizing, ask: *Is each factual claim directly supported by a cited passage?* If not, remove or qualify it.

**Output requirements:**
- Every factual sentence ends with a source tag such as [Source: D02].
- **Confidence:** label **High** if evidence directly answers the query; **Medium** if glossary/interpretation bridges informal terms; **Low** if CONTEXT is thin or ambiguous.

**Output format (use these exact section headings):**
**Answer:** …

**Citations:** …

**Confidence Level:** High / Medium / Low — (one line rationale)

**Grounding note:** One sentence on how CONTEXT supported the answer (or why confidence is Low).
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
        "Grounded engineering support · Corpus D01–D10 · FAISS + sentence embeddings · Metadata-aware CONTEXT "
        "(status / authority) · D02>D03 · D08 non-binding · D07 safety merge · HHH-aligned guardrail when using OpenAI."
    )

    _base = Path(__file__).resolve().parent
    _root = _base.parent
    render_rag_explainer_block(
        _base,
        _root,
        caption=(
            "This explainer supports the Assignment 3 **RAG concept demo**: **Index → Retrieve → Generate** — "
            "documents are chunked and embedded so retrieval supplies **grounded** context, shrinking reliance on the "
            "model’s parametric memory alone (the **knowledge boundary** problem)."
        ),
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
