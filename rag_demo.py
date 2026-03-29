"""
RAG Alignment Guard — semantic retrieval + strict grounding (Assignment 3).

Run: streamlit run rag_demo.py

Uses sentence-transformers on category|subcategory|user_prompt; shows CSV vs grounded answer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from branding import render_authors_banner
from grounded_responses import REVISED_RESPONSES, STRICT_SYSTEM_PROMPT
from rag_pipeline import load_audit_dataset
from semantic_retrieval import SemanticRetriever, build_index_text
from ui_theme import hero_engineering_ribbon, inject_engineering_theme

DATA_PATH = Path(__file__).resolve().parent / "session7_alignment_audit_package" / "data" / "session7_alignment_audit_dataset.csv"

# Normalized cosine similarity; below this, Strict Mode refuses to answer from context.
STRICT_MIN_SIMILARITY = 0.18


@st.cache_data
def load_audit_data() -> pd.DataFrame:
    return load_audit_dataset(DATA_PATH)


@st.cache_resource
def semantic_engine() -> tuple[SemanticRetriever, pd.DataFrame]:
    df = load_audit_data()
    texts = build_index_text(df)
    eng = SemanticRetriever(model_name="all-MiniLM-L6-v2")
    eng.fit(df, texts)
    return eng, df


def grounded_reply(case_id: str) -> str:
    return REVISED_RESPONSES.get(
        case_id,
        "Retrieved grounding: follow safety, honesty, and fairness policies from the alignment audit row.",
    )


def main() -> None:
    st.set_page_config(
        page_title="RAG Alignment Guard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"

    try:
        df = load_audit_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    eng, _ = semantic_engine()

    with st.sidebar:
        st.markdown("### Appearance")
        st.radio("Theme", ["light", "dark"], horizontal=True, key="ui_theme", label_visibility="collapsed")
        inject_engineering_theme(st.session_state.ui_theme)

        st.header("RAG workflow")
        st.markdown(
            "**1. Index:** `category` · `subcategory` · `user_prompt` → one embedding per row.\n\n"
            "**2. Semantic search:** `all-MiniLM-L6-v2` cosine similarity (normalized).\n\n"
            "**3. Grounding:** inject `REVISED_RESPONSES[case_id]` — arXiv not Nature (O01), RLHF≠DPO (H02), fair hiring (B*).\n\n"
            "**4. Strict mode:** refuse if similarity < {:.2f} or insist on context-only.".format(STRICT_MIN_SIMILARITY)
        )

    st.title("RAG Alignment Guard")
    hero_engineering_ribbon(st.session_state.ui_theme)
    render_authors_banner()

    st.markdown("### Topic: Foundation models, hallucination, and retrieval-grounded alignment")
    st.caption(f"Data: `{DATA_PATH.name}` · **{len(df)}** cases")

    strict = st.toggle(
        "Strict Mode",
        value=False,
        help="Applies a context-only system policy and refuses weak retrieval matches.",
    )
    if strict:
        st.info(f"**System (Strict):** {STRICT_SYSTEM_PROMPT}")

    query = st.text_input(
        "Ask a technical question (paraphrases OK)",
        placeholder="e.g. Did Anthropic publish Constitutional AI in Nature?  ·  RLHF vs DPO  ·  fair hiring criteria",
    )

    top_k = st.slider("Show top‑k semantic neighbors", min_value=1, max_value=min(8, len(df)), value=5)

    if not query.strip():
        st.info("Enter a query to run semantic retrieval and compare hallucinated vs grounded answers.")
        st.dataframe(df[["case_id", "category", "user_prompt"]].head(8), use_container_width=True, hide_index=True)
        return

    hits = eng.search(query.strip(), top_k=top_k)
    if not hits:
        st.warning("Semantic engine not ready.")
        return

    best = hits[0]
    row = df.iloc[best.row_index]
    cid = best.case_id
    score = best.score

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Hallucinated / brittle response (from dataset)")
        st.caption(f"**{row['category']}** · {row['subcategory']}")
        st.markdown(f"**Matched case:** `{cid}` · semantic score **{score:.3f}**")
        st.markdown("**User prompt (indexed row)**")
        st.info(row["user_prompt"])
        st.markdown("**`model_response` (may be misaligned)**")
        st.text(str(row["model_response"]))

    with col2:
        st.markdown("### RAG‑augmented response (grounded)")
        refuse = strict and score < STRICT_MIN_SIMILARITY
        if refuse:
            st.error(
                "**Strict Mode — refusal:** Retrieved context does not match this query strongly enough "
                f"(similarity {score:.3f} < {STRICT_MIN_SIMILARITY}). I will not invent an answer."
            )
        else:
            st.success("**Grounded on retrieved `case_id` + course-aligned revision text**")
            st.markdown(grounded_reply(cid))

    st.divider()
    st.subheader("Semantic neighborhood (all scores cosine similarity)")
    rows = []
    for h in hits:
        r = df.iloc[h.row_index]
        rows.append(
            {
                "rank": len(rows) + 1,
                "case_id": h.case_id,
                "score": round(h.score, 4),
                "category": r["category"],
                "subcategory": (str(r["subcategory"])[:60] + "…") if len(str(r["subcategory"])) > 60 else r["subcategory"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Indexed text (what was embedded for the top hit)", expanded=False):
        st.code(build_index_text(df.iloc[[best.row_index]])[0], language="text")

    st.markdown(
        "**Spotlight logic:** O01 → arXiv citation · H02 → RLHF vs DPO distinct · B01–B04 → fair screening · "
        "H04 → push back on false premises when that row is retrieved."
    )


if __name__ == "__main__":
    main()
