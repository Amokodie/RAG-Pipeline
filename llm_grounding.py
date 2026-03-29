"""
Optional OpenAI chat completion grounded strictly on RETRIEVED_CONTEXT + instructor notes.
Set OPENAI_API_KEY in the environment, Streamlit secrets, or the in-app field (session only).
"""

from __future__ import annotations


STRICT_GROUNDING_SYSTEM = """You are a technical teaching assistant for an alignment-audit lab.

Hard rules:
- Use ONLY the material in RETRIEVED_CONTEXT below (metadata, dataset response excerpt, instructor correction, and optional sentence-level attribution lines).
- Do NOT rely on outside knowledge. If the context is insufficient to answer the user, reply exactly: Insufficient context in retrieved audit row.
- Do NOT invent paper venues, journal names, or citations not implied by the context.
- Prefer the instructor_grounded_answer when it conflicts with the raw dataset_model_response_excerpt.
- Stay under 200 words.

RETRIEVED_CONTEXT:
{context}
"""


def build_context_block(
    case_id: str,
    category: str,
    subcategory: str,
    user_prompt: str,
    model_response_excerpt: str,
    instructor_grounded_answer: str,
    *,
    sentence_attribution_block: str | None = None,
) -> str:
    """
    Assemble the full grounding block for the API.
    `instructor_grounded_answer` is the course-aligned revision (REVISED_RESPONSES).
    `sentence_attribution_block` optional: top matching sentences from prompt + model_response.
    """
    parts = [
        f"case_id: {case_id}",
        f"category: {category}",
        f"subcategory: {subcategory}",
        f"user_prompt: {user_prompt}",
        f"dataset_model_response_excerpt: {model_response_excerpt}",
        f"instructor_grounded_answer (preferred when correcting the dataset): {instructor_grounded_answer}",
    ]
    if sentence_attribution_block and sentence_attribution_block.strip():
        parts.append("sentence_attribution_evidence (dense similarity to user query):")
        parts.append(sentence_attribution_block.strip())
    return "\n".join(parts)


def chat_grounded_answer(
    *,
    api_key: str,
    model: str,
    user_query: str,
    context_block: str,
) -> str:
    """Call OpenAI Chat Completions with strict grounding."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install the `openai` package: pip install openai") from e

    client = OpenAI(api_key=api_key)
    system = STRICT_GROUNDING_SYSTEM.format(context=context_block)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()
