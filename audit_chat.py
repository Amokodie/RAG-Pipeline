"""
Conversational Q&A: alignment audit row + IT knowledge DB + optional OpenAI.
When the audit match is weak (out-of-context), IT KB + general guidance still produce an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from grounded_responses import REVISED_RESPONSES
from llm_grounding import build_context_block

if TYPE_CHECKING:
    from it_kb_retrieval import ItKbHit


def resolve_row_index(df: pd.DataFrame, case_id: str) -> int:
    m = df.index[df["case_id"].astype(str) == str(case_id)].tolist()
    return int(m[0]) if m else 0


# Cosine / fused-score thresholds — below this, treat audit match as weak (question likely off-topic for audit)
AUDIT_WEAK_TFIDF = 0.11
AUDIT_WEAK_HYBRID = 0.14


def retrieve_best_case(
    df: pd.DataFrame,
    retriever: Any,
    hybrid: Any | None,
    query: str,
) -> tuple[int, str, str]:
    row_idx, case_id, method, _score = retrieve_best_case_with_score(df, retriever, hybrid, query)
    return row_idx, case_id, method


def retrieve_best_case_with_score(
    df: pd.DataFrame,
    retriever: Any,
    hybrid: Any | None,
    query: str,
) -> tuple[int, str, str, float]:
    """
    Returns (row_index, case_id, method_label, relevance_score).
    Score is hybrid fused score or TF-IDF cosine on the top hit; 0 if defaulting.
    """
    q = query.strip()
    if not q:
        return 0, str(df.iloc[0]["case_id"]), "default", 0.0

    if hybrid is not None:
        try:
            hits = hybrid.search(q, top_k=1, alpha=0.55, fusion="weighted")
            if hits:
                h0 = hits[0]
                return h0.row_index, h0.case_id, "hybrid (BM25 + dense)", float(h0.fused_score)
        except Exception:
            pass

    hits = retriever.query(q, top_k=1)
    if hits:
        h0 = hits[0]
        pos = int(h0.chunk_index)
        if 0 <= pos < len(df):
            return pos, h0.case_id, "TF‑IDF", float(h0.score)
        pos = resolve_row_index(df, h0.case_id)
        return pos, h0.case_id, "TF‑IDF", float(h0.score)
    return 0, str(df.iloc[0]["case_id"]), "default (no match)", 0.0


def audit_match_is_weak(method: str, score: float) -> bool:
    if method.startswith("hybrid"):
        return score < AUDIT_WEAK_HYBRID
    if method == "TF‑IDF":
        return score < AUDIT_WEAK_TFIDF
    return True


def _hits_from_kb(kb: Any | None, query: str, top_k: int = 4) -> list:
    if kb is None:
        return []
    try:
        return kb.query(query.strip(), top_k=top_k)
    except Exception:
        return []


def format_it_kb_for_prompt(hits: list) -> str:
    """Plain-text block for LLM."""
    if not hits:
        return "(no IT knowledge base passages retrieved)"
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[IT-{i}] title={h.title!r} category={h.category!r} score={h.score:.4f}\n{h.body}"
        )
    return "\n\n".join(parts)


def format_it_kb_markdown(hits: list) -> str:
    """Readable markdown for offline replies."""
    if not hits:
        return "*No close matches in the local IT knowledge base.*"
    lines = ["**Related IT knowledge base (retrieved):**"]
    for h in hits:
        lines.append(
            f"- **{h.title}** (*{h.category}*, match {h.score:.2f}) — {h.body}"
        )
    return "\n".join(lines)


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


def offline_multisource_answer(
    df: pd.DataFrame,
    row_index: int,
    case_id: str,
    instructor_text: str,
    method: str,
    audit_score: float,
    audit_weak: bool,
    it_hits: list,
) -> str:
    """Audit row + IT KB + note when the question is likely out of audit context."""
    base = offline_markdown_answer(df, row_index, case_id, instructor_text, method)
    extra = [
        "",
        f"**Audit relevance score:** {audit_score:.3f} · "
        f"**{'Weak match — question may be broader than this audit row' if audit_weak else 'Reasonable match to retrieved row'}**",
        "",
        format_it_kb_markdown(it_hits),
    ]
    if audit_weak:
        extra.insert(
            0,
            "> *The closest alignment-audit row may not fully answer your question. "
            "Below are passages from the **local IT knowledge base** (other sources) to help.*",
        )
    return base + "\n".join(extra)


def openai_conversational_answer(
    *,
    api_key: str,
    model: str,
    user_query: str,
    context_block: str,
) -> str:
    """Legacy: audit-only strict context."""
    return openai_multisource_answer(
        api_key=api_key,
        model=model,
        user_query=user_query,
        audit_context_block=context_block,
        it_kb_block="",
        audit_weak=False,
        it_kb_used=False,
    )


def openai_multisource_answer(
    *,
    api_key: str,
    model: str,
    user_query: str,
    audit_context_block: str,
    it_kb_block: str,
    audit_weak: bool,
    it_kb_used: bool,
) -> str:
    """Uses audit + IT KB; allows general IT explanation when audit is weak or question is broad."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    weak_note = (
        "The student's question may be **only loosely related** to the alignment-audit row (weak retrieval match). "
        "In that case, lean on IT_KNOWLEDGE_BASE and standard IT education — still be accurate, avoid inventing "
        "specific paper titles, CVE IDs, or vendor promises."
        if audit_weak
        else "The audit row appears reasonably relevant; prioritize it when it answers the question."
    )
    kb_status = (
        "IT knowledge base: passages were retrieved."
        if it_kb_used
        else "IT knowledge base: no strong lexical matches; still answer with sound general IT knowledge when needed."
    )

    system = (
        "You are a teaching assistant for a **foundation models & alignment** lab, plus general **IT/CS** support.\n\n"
        f"{kb_status}\n\n"
        "You are given:\n"
        "1) RETRIEVED_ALIGNMENT_AUDIT_ROW — one row from a small CSV lab (case IDs like H01, O01, B01).\n"
        "2) IT_KNOWLEDGE_BASE — short curated passages from a local database (networking, security, cloud, DevOps, ML/RAG, etc.).\n\n"
        f"Guidance: {weak_note}\n\n"
        "- If the user asks something **not covered** by the audit row, answer using **IT_KNOWLEDGE_BASE** when it helps, "
        "and **sound general IT knowledge** at an undergraduate level when the DB does not contain a specific answer.\n"
        "- Clearly separate: what comes from the **audit** vs **IT KB** vs **general reasoning** when it matters.\n"
        "- Do not fabricate citations to real papers or exact CVE numbers unless they appear in the provided text.\n"
        "- Keep answers structured (short paragraphs or bullets), friendly, and concise.\n\n"
        "--- RETRIEVED_ALIGNMENT_AUDIT_ROW ---\n"
        f"{audit_context_block}\n\n"
        "--- IT_KNOWLEDGE_BASE ---\n"
        f"{it_kb_block or '(empty)'}\n"
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ],
        temperature=0.35,
        max_tokens=900,
    )
    return (resp.choices[0].message.content or "").strip()


@dataclass
class AskAiPack:
    row_idx: int
    case_id: str
    method: str
    audit_score: float
    audit_weak: bool
    ctx_audit: str
    it_hits: list


def build_ask_ai_pack(
    df: pd.DataFrame,
    retriever: Any,
    hybrid: Any | None,
    kb: Any | None,
    query: str,
    *,
    it_kb_top_k: int = 4,
    sentence_attribution_block: str | None = None,
) -> AskAiPack:
    row_idx, case_id, method, audit_score = retrieve_best_case_with_score(df, retriever, hybrid, query)
    audit_weak = audit_match_is_weak(method, audit_score)
    row = df.iloc[row_idx]
    cid = str(row["case_id"])
    grounded = REVISED_RESPONSES.get(cid, "—")
    excerpt = str(row["model_response"])[:1200]

    ctx_audit = build_context_block(
        cid,
        str(row["category"]),
        str(row["subcategory"]),
        str(row["user_prompt"]),
        excerpt,
        grounded,
        sentence_attribution_block=sentence_attribution_block,
    )
    it_hits = _hits_from_kb(kb, query, top_k=it_kb_top_k)
    return AskAiPack(
        row_idx=row_idx,
        case_id=cid,
        method=method,
        audit_score=audit_score,
        audit_weak=audit_weak,
        ctx_audit=ctx_audit,
        it_hits=it_hits,
    )
