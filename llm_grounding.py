"""
Optional OpenAI chat completion grounded strictly on RETRIEVED_CONTEXT.
Set OPENAI_API_KEY in the environment or pass the key from the UI (session only).
"""

from __future__ import annotations


STRICT_GROUNDING_SYSTEM = """You are a technical teaching assistant for an alignment-audit lab.

Rules:
- Answer ONLY using the RETRIEVED_CONTEXT block below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say exactly: "Insufficient context in retrieved audit row."
- Do not invent citations, venues, or paper titles.
- Keep answers concise (under 200 words).

RETRIEVED_CONTEXT:
{context}
"""


def build_context_block(
    case_id: str,
    category: str,
    subcategory: str,
    user_prompt: str,
    model_response_excerpt: str,
    grounded_notes: str,
) -> str:
    return (
        f"case_id: {case_id}\n"
        f"category: {category}\n"
        f"subcategory: {subcategory}\n"
        f"user_prompt: {user_prompt}\n"
        f"dataset_model_response_excerpt: {model_response_excerpt}\n"
        f"instructor_grounded_answer: {grounded_notes}"
    )


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
