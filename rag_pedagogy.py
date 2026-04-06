"""
Interactive “hallucination vs RAG” teaching block for app.py (uses audit CSV + grounded revisions).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from grounded_responses import REVISED_RESPONSES

# Curated scenarios: richer than CASE_ANALYSIS alone — ties to the explainer video narrative.
PEDAGOGY: dict[str, dict[str, str]] = {
    "O01": {
        "label": "O01 — Invented *Nature* paper (citation hallucination)",
        "tag": "Honesty · fabricated source",
        "why_it_happens": (
            "The model is rewarded for **fluent, authoritative-sounding** answers. A *Nature* citation is "
            "a classic **hallucination pattern**: plausible format, **wrong provenance**. Parametric memory "
            "does not store a verified bibliography — it predicts likely tokens."
        ),
        "rag_fix": (
            "RAG injects a **retrieved** note that lists the **real** arXiv identifier and a rule: "
            "**never invent venues**. The generator is conditioned on that chunk before answering, so "
            "the answer tracks **non-parametric** evidence."
        ),
        "retrieved_chunk": (
            "[RETRIEVED — Course / lab policy chunk · citation integrity]\n"
            "• Anthropic Constitutional AI: Bai et al., **arXiv:2212.08073 (2022)** — check venue/version before citing.\n"
            "• If unsure: say uncertainty; do **not** invent journal names or years.\n"
            "• Preferred phrasing: “documented in arXiv preprint …” when that is what the index contains."
        ),
    },
    "H02": {
        "label": "H02 — RLHF vs DPO collapsed (technical drift)",
        "tag": "Helpfulness · factual drift",
        "why_it_happens": (
            "Under pressure to sound definitive, the model **over-collapses** distinct methods (RLHF vs DPO) "
            "and overclaims “replacement.” That is **not** malice — it is **optimization for coherence** "
            "without an external fact checker."
        ),
        "rag_fix": (
            "Retrieval supplies a **methods comparison** chunk with distinct definitions (reward model + RL "
            "vs preference optimization). The answer must **align** with retrieved bullets, reducing drift."
        ),
        "retrieved_chunk": (
            "[RETRIEVED — Methods cheat-sheet chunk]\n"
            "• **RLHF:** train reward model from preferences → policy optimization (e.g. PPO) against reward.\n"
            "• **DPO:** preference learning often **without** an explicit reward model + RL loop in many setups.\n"
            "• Neither universally “replaced” the other — depends on data, feedback, infra, and eval metrics."
        ),
    },
    "H04": {
        "label": "H04 — Agrees with a false premise (sycophancy)",
        "tag": "Helpfulness · sycophancy",
        "why_it_happens": (
            "Chat models are tuned to be **agreeable**. If the user states a **false causal claim** "
            "(“knowing facts ⇒ aligned”), the model may **rubber-stamp** it instead of correcting — a "
            "failure mode unrelated to raw knowledge."
        ),
        "rag_fix": (
            "RAG pulls a chunk that states the **course definition** of alignment (post-training, eval, "
            "monitoring). The model is steered to **disagree politely** with the false premise using "
            "those definitions."
        ),
        "retrieved_chunk": (
            "[RETRIEVED — Alignment definition chunk]\n"
            "• **Knowledge ≠ alignment.** Base models can know facts yet still produce unsafe or misleading outputs.\n"
            "• Alignment involves post-training, evaluation, monitoring — not parameter scale alone.\n"
            "• Good assistants **correct** false premises when grounded notes require it."
        ),
    },
    "S01": {
        "label": "S01 — Harmful compliance (phishing)",
        "tag": "Safety · harmful compliance",
        "why_it_happens": (
            "Without a **policy** in context, the model may complete the **genre** of an email fluently — "
            "including harmful content — because training data contains many email-like templates."
        ),
        "rag_fix": (
            "Retrieval surfaces a **safety policy chunk** (refusal + safe alternative). The model is "
            "conditioned to **refuse** and pivot to defensive education — the same query, different "
            "**non-parametric** guardrail."
        ),
        "retrieved_chunk": (
            "[RETRIEVED — Safety policy chunk · harmful requests]\n"
            "• Refuse phishing, credential theft, or deceptive content.\n"
            "• Offer: high-level defensive framing (user education, MFA, reporting) if the user studies security.\n"
            "• Do not produce realistic attack templates."
        ),
    },
    "B01": {
        "label": "B01 — Stereotype agreement (hiring bias)",
        "tag": "Bias · stereotype",
        "why_it_happens": (
            "The model may **mirror** biased framing to sound helpful (“yes, women are better at…”). "
            "That is **sycophancy + stereotype leakage**, not a factual lookup problem."
        ),
        "rag_fix": (
            "Retrieval injects **fair-hiring** norms: assess **job-relevant skills**, avoid protected-attribute "
            "stereotypes. The answer is grounded in that chunk rather than conversational agreement."
        ),
        "retrieved_chunk": (
            "[RETRIEVED — Fair hiring policy chunk]\n"
            "• Screen on skills, behavior, and role-relevant criteria.\n"
            "• Do not use gender, age, or ethnicity as proxies for competence.\n"
            "• Push back on stereotypical premises with neutral, inclusive language."
        ),
    },
}


def render_pedagogy_hallucination_lab(df: pd.DataFrame, retriever: Any | None = None) -> None:
    """Rich interactive contrast: audit baseline (hallucination/drift) vs simulated RAG grounding."""
    with st.expander(
        "Hands-on lab — **hallucination vs RAG grounding** (click to expand or collapse)",
        expanded=True,
    ):
        st.markdown(
            "Use the **same user question** in two regimes. **(A)** the dataset’s **original** model output "
            "stands in for “answer from parametric memory / no retrieval step” (what went wrong in the audit). "
            "**(B)** shows a **simulated retrieved chunk** plus the **RAG-style grounded** reply from this repo — "
            "the **Index → Retrieve → Generate** loop your video describes."
        )
        st.caption(
            "This is pedagogical simulation: the “bad” answer is real audit text; the “retrieved chunk” is written "
            "to match what a good RAG index would surface. Open **Case lab** for the full prompt template."
        )

        case_order = list(PEDAGOGY.keys())
        pick = st.selectbox(
            "Choose a scenario",
            options=case_order,
            format_func=lambda cid: PEDAGOGY[cid]["label"],
            key="pedagogy_case_select",
        )
        meta = PEDAGOGY[pick]
        row = df.loc[df["case_id"] == pick].iloc[0]
        revised = REVISED_RESPONSES.get(pick, "—")

        st.markdown(f"**Lens:** {meta['tag']}")

        st.markdown("##### 1) User question")
        st.info(row["user_prompt"])

        c_bad, c_good = st.columns((1, 1), gap="large")

        with c_bad:
            st.markdown("##### A) Without RAG-style grounding")
            st.caption("Original **`model_response`** in the CSV — simulates fluent but **ungrounded** output.")
            st.text_area(
                "Ungrounded output",
                value=str(row["model_response"]),
                height=220,
                disabled=True,
                label_visibility="collapsed",
                key=f"pedagogy_bad_{pick}",
            )
            with st.expander("Why this is a failure mode", expanded=False):
                st.markdown(meta["why_it_happens"])

        with c_good:
            st.markdown("##### B) With RAG (retrieved chunk → grounded answer)")
            st.caption("**Simulated** document chunk the retriever would pass to the LLM, then the grounded reply.")
            st.code(meta["retrieved_chunk"], language="text")
            st.markdown("**Grounded answer (revision)**")
            st.success(revised)
            with st.expander("How retrieval fixes it", expanded=False):
                st.markdown(meta["rag_fix"])

        with st.expander("Pipeline mental model (matches the explainer video)", expanded=False):
            st.markdown(
                """
| Step | What happens |
|------|----------------|
| **Index** | Each audit row is a **chunk** with metadata (`case_id`, category, …). |
| **Retrieve** | Your query embedding is matched to chunks (here: **TF‑IDF** cosine; production often dense vectors). |
| **Generate** | The LLM sees **retrieved text + policy** before answering — shrinking reliance on pure parametric guesses. |
"""
            )
            st.code(
                "User query → [Embed] → [Vector search] → Top-k chunks → [Prompt] → LLM → Grounded answer",
                language="text",
            )

        if retriever is not None:
            with st.expander("Live check: TF‑IDF top matches for this question (same index as the app)", expanded=False):
                qp = str(row["user_prompt"])
                hits = retriever.query(qp, top_k=min(5, len(df)))
                st.caption(
                    "If the **intended** `case_id` is rank‑1, retrieval would feed the right chunk before generation. "
                    "Short prompts sometimes rank lower — that is why production RAG uses richer queries and dense embeddings."
                )
                for i, h in enumerate(hits, 1):
                    mark = "✓" if h.case_id == pick else "·"
                    st.markdown(
                        f"{mark} **#{i}** `{h.case_id}` · score **{h.score:.4f}** · {h.category} — *{h.subcategory}*"
                    )

        st.markdown("---")
        st.caption(
            "Tip: run the **same** `user_prompt` in **Live retrieval inspector** for term weights and the full bar chart. "
            "Retrieval quality is part of what makes RAG succeed or fail in production."
        )
