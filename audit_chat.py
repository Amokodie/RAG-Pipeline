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


def is_site_meta_query(q: str) -> bool:
    """Questions about the app itself, AeroFleet vs Session 7, RAG, or the assistant identity."""
    low = q.strip().lower()
    if len(low) < 2:
        return False
    needles = (
        "what is this site",
        "what is this app",
        "what is this demo",
        "what is this about",
        "aerofleet",
        "aero fleet",
        "are you a bot",
        "are you a robot",
        "are you an ai",
        "who are you",
        "what is rag",
        "what's rag",
        "session 7",
        "session 8",
        "this streamlit",
        "alignment audit",
        "how do i run",
        "streamlit run",
        "where is the video",
        "explainer video",
    )
    return any(n in low for n in needles)


def is_app_navigation_query(q: str) -> bool:
    """Questions about tabs, sidebar, what each page does, or how to use this Streamlit UI."""
    low = q.strip().lower()
    if len(low) < 3:
        return False
    needles = (
        "navigation",
        "navigate",
        "sidebar",
        "which tab",
        "what tab",
        "tabs ",
        " tab ",
        "main page",
        "overview &",
        "overview tab",
        "corpus tab",
        "analysis &",
        "analysis tab",
        "what is the analysis",
        "meaning of the analysis",
        "what does analysis",
        "3d embedding",
        "3d tab",
        "lsa ",
        "case lab",
        "live retrieval",
        "retrieval inspector",
        "ask ai tab",
        "advanced tab",
        "advanced:",
        "concepts &",
        "concepts tab",
        "heatmap",
        "silhouette",
        "margin bar",
        "sankey",
        "how do i use",
        "where do i find",
        "where is the",
        "how does this app",
        "how does the app",
        "pages in the app",
        "appearance",
        "light mode",
        "dark mode",
        "theme",
        "engineering storyboard",
        "spotlight cases",
        "hybrid +",
        "bm25",
        "sentence attribution",
        "failure modes",
        "assignment mapping",
        "vertical block",
        "streamlit cloud",
        "manage app",
        "ask ai",
        "where is ask",
        "which page has",
        "go to the",
    )
    return any(n in low for n in needles)


def is_student_concern_query(q: str) -> bool:
    """Broad student questions: stress, integrity, privacy, teams, grades anxiety — boost KB retrieval."""
    low = q.strip().lower()
    if len(low) < 3:
        return False
    needles = (
        "worried",
        "worry",
        "stress",
        "stressed",
        "anxiety",
        "anxious",
        "overwhelm",
        "nervous",
        "scared",
        "afraid",
        "panic",
        "depress",
        "lonely",
        "help me",
        "i need help",
        "struggling",
        "struggle",
        "failing",
        "fail the",
        "grade",
        "grades",
        "gpa",
        "exam",
        "deadline",
        "extension",
        "late submission",
        "plagiarism",
        "plagiarize",
        "cheat",
        "cheating",
        "academic integrity",
        "cite",
        "citation",
        "reference",
        "privacy",
        "api key",
        "openai key",
        "secret",
        "password",
        "cost",
        "pay for",
        "team",
        "group project",
        "partner",
        "conflict",
        "not working",
        "doesn't work",
        "error",
        "broken",
        "deploy",
        "streamlit cloud",
        "english",
        "esl",
        "language barrier",
        "accessibility",
        "accommodation",
        "disability",
        "career",
        "job",
        "interview",
        "internship",
        "imposter",
        "burnout",
        "sleep",
        "mental health",
        "counseling",
        "counselling",
        "family pressure",
        "disappoint",
        "ethical",
        "ethics",
        "is it ok to",
        "allowed to use",
        "chatgpt",
        "integrity",
    )
    return any(n in low for n in needles)


def is_glossary_concept_query(q: str) -> bool:
    """
    Definition-style questions (what is / define …) about course concepts.
    These should lean on the KB + definitions, not a tangentially similar audit row (e.g. H04 vs hallucination).
    """
    low = q.strip().lower()
    if len(low) < 6:
        return False
    asking = any(
        p in low
        for p in (
            "what is ",
            "what are ",
            "what's ",
            "define ",
            "definition of ",
            "meaning of ",
            "explain ",
            "what does ",
            "tell me about",
        )
    )
    if not asking:
        return False
    terms = (
        "hallucination",
        "hallucinate",
        "rag",
        "retrieval",
        "embedding",
        "grounding",
        "alignment",
        "rlhf",
        "dpo",
        "fine-tun",
        "sycophancy",
        "bias",
        "token",
        "parametric",
        "confabulation",
        "llm",
        "temperature",
        "overfitting",
        "softmax",
        "vector",
        "attention",
        "prompt",
    )
    return any(t in low for t in terms)


def should_boost_kb_recall(query: str) -> bool:
    """Widen IT KB merge for site FAQ, app navigation, student concerns, or glossary questions."""
    return (
        is_site_meta_query(query)
        or is_app_navigation_query(query)
        or is_student_concern_query(query)
        or is_glossary_concept_query(query)
    )


def merge_meta_kb_hits(kb: Any | None, query: str, it_hits: list, *, top_k: int) -> list:
    """
    When the user asks meta, navigation, student-concern, or glossary questions, merge in a high-recall retrieval so
    Course_meta / IT chunks surface even if the user query is sparse.
    """
    if kb is None or not should_boost_kb_recall(query):
        return it_hits[:top_k]
    if is_glossary_concept_query(query):
        boost_q = (
            f"{query.strip()} definition explain LLM hallucination false fluent confabulation "
            "RAG retrieval grounding citation alignment parametric memory IT_KNOWLEDGE_BASE course"
        )
    elif is_app_navigation_query(query):
        boost_q = (
            f"{query.strip()} Streamlit tabs Overview corpus Analysis 3D LSA Case lab Live retrieval Ask AI Advanced Concepts "
            "sidebar Appearance theme heatmap silhouette margin cosine TF-IDF sparse matrix hybrid BM25 sentence attribution "
            "navigation Engineering storyboard explainer video pedagogy hands-on lab"
        )
    else:
        boost_q = (
            "Session 7 alignment audit CSV RAG demo AeroFleet X200 Session 8 separate app "
            "streamlit battery cooling what is this course bot assistant "
            "OpenAI API key privacy academic integrity plagiarism citation worried stress exam grade deadline "
            "Student_support counseling team group extension mental health imposter career English accessibility"
        )
    try:
        boost = kb.query(boost_q, top_k=8)
    except Exception:
        return it_hits[:top_k]
    by_title: dict[str, Any] = {}
    for h in it_hits:
        by_title[h.title] = h
    for h in boost:
        if h.title not in by_title or h.score > by_title[h.title].score:
            by_title[h.title] = h
    concern = is_student_concern_query(query)
    site = is_site_meta_query(query)
    gloss = is_glossary_concept_query(query)
    nav = is_app_navigation_query(query)

    def _cat_rank(cat: str) -> int:
        # Pure wellbeing / study questions: surface Student_support before generic Course_meta.
        if concern and not site:
            if cat == "Student_support":
                return 0
            if cat == "Course_meta":
                return 1
            return 2
        # Definitions: Course_meta + AI/ML before Student_support.
        if gloss and not site and not concern:
            if cat == "Course_meta":
                return 0
            if cat == "AI/ML":
                return 1
            if cat == "Student_support":
                return 2
            return 3
        # App structure / tabs: Course_meta guides first.
        if nav and not concern:
            if cat == "Course_meta":
                return 0
            if cat == "AI/ML":
                return 1
            if cat == "Student_support":
                return 2
            return 3
        if cat == "Course_meta":
            return 0
        if cat == "Student_support":
            return 1
        return 2

    merged = sorted(
        by_title.values(),
        key=lambda h: (_cat_rank(getattr(h, "category", "")), -h.score),
    )
    return merged[:top_k]


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
    lines = ["**Knowledge base (IT + course FAQ):**"]
    for h in hits:
        cat = getattr(h, "category", "")
        if cat == "Course_meta":
            tag = "course / site FAQ"
        elif cat == "Student_support":
            tag = "student support (general guidance)"
        else:
            tag = cat
        lines.append(
            f"- **{h.title}** (*{tag}*, match {h.score:.2f}) — {h.body}"
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
    *,
    user_query: str = "",
) -> str:
    """Audit row + IT KB + note when the question is likely out of audit context."""
    row = df.iloc[row_index]
    kb_md = format_it_kb_markdown(it_hits)

    if is_glossary_concept_query(user_query) and it_hits:
        audit_block = (
            f"**Retrieved row:** `{case_id}` · *{row['category']}* · {row['subcategory']}\n\n"
            f"*Retrieval:* {method}\n\n"
            f"**Note from that audit case (illustrative — not always a literal definition):**\n\n{instructor_text}\n\n"
            f"---\n*Original audit prompt:* {str(row['user_prompt'])[:400]}"
            f"{'…' if len(str(row['user_prompt'])) > 400 else ''}"
        )
        return (
            "> **Definition-style question:** The **knowledge base** below answers *what the term means*. "
            "The audit row is a **related teaching example** from the CSV (e.g. H04 is about **sycophancy**, "
            "not the textbook definition of **hallucination**).\n\n"
            + kb_md
            + "\n\n---\n\n**Related alignment-audit row (lab example):**\n\n"
            + audit_block
            + "\n\n"
            + f"**Audit relevance score:** {audit_score:.3f} · "
            + (
                "*Prefer the passages above for definitions.*"
                if audit_weak
                else "*Audit match is moderate — still use KB above for definitions.*"
            )
        )

    if is_app_navigation_query(user_query) and it_hits and not is_glossary_concept_query(user_query):
        audit_block = (
            f"**Retrieved row:** `{case_id}` · *{row['category']}* · {row['subcategory']}\n\n"
            f"*Retrieval:* {method}\n\n"
            f"**Optional lab context:**\n\n{instructor_text}\n\n"
            f"---\n*Original audit prompt:* {str(row['user_prompt'])[:400]}"
            f"{'…' if len(str(row['user_prompt'])) > 400 else ''}"
        )
        return (
            "> **App / navigation question:** The **knowledge base** below describes **tabs, sidebar, and what each area does**. "
            "The alignment-audit row at the end is **optional** CSV context—not a substitute for the UI guides.\n\n"
            + kb_md
            + "\n\n---\n\n**Optional — related alignment-audit row:**\n\n"
            + audit_block
            + "\n\n"
            + f"**Audit relevance score:** {audit_score:.3f}"
        )

    base = offline_markdown_answer(df, row_index, case_id, instructor_text, method)
    extra = [
        "",
        f"**Audit relevance score:** {audit_score:.3f} · "
        f"**{'Weak match — question may be broader than this audit row' if audit_weak else 'Reasonable match to retrieved row'}**",
        "",
        kb_md,
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
    if is_glossary_concept_query(user_query):
        weak_note = (
            "**The user asked for a DEFINITION or concept explanation.** "
            "Answer using **IT_KNOWLEDGE_BASE first** (definitions, RAG, hallucination, etc.). "
            "Use the alignment-audit row **only as an optional example** if it truly illustrates the term; "
            "many rows (e.g. H04 sycophancy) are **not** the definition of hallucination. "
            "Lead with a clear definition in your own words grounded in the KB."
        )
    elif is_app_navigation_query(user_query):
        weak_note = (
            "**The user is asking about this Streamlit app: navigation, tabs, sidebar, or what a page/section means.** "
            "Answer primarily from **IT_KNOWLEDGE_BASE** passages whose titles mention **tabs**, **Analysis**, **Overview**, etc. "
            "List tab names accurately: **Overview & corpus**, **Analysis & 3D embedding**, **Case lab (failure vs RAG)**, "
            "**Live retrieval inspector**, **Ask AI**, **Advanced: hybrid + LLM**, **Concepts & checklist**. "
            "The alignment-audit row is **secondary** unless it illustrates an alignment concept relevant to the question."
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
    if should_boost_kb_recall(query):
        audit_weak = True  # prefer Course_meta / navigation / glossary / student KB over a tangential audit row
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
    tk = max(it_kb_top_k, 10) if should_boost_kb_recall(query) else it_kb_top_k
    it_hits = _hits_from_kb(kb, query, top_k=tk)
    it_hits = merge_meta_kb_hits(kb, query, it_hits, top_k=tk)
    return AskAiPack(
        row_idx=row_idx,
        case_id=cid,
        method=method,
        audit_score=audit_score,
        audit_weak=audit_weak,
        ctx_audit=ctx_audit,
        it_hits=it_hits,
    )
