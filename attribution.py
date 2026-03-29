"""
Sentence-level attribution: which sentences in prompt vs model_response match the query (dense similarity).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence split (good enough for audit rows)."""
    if not text or not str(text).strip():
        return []
    t = str(text).replace("\r\n", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n+", t.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class SentenceAttribution:
    text: str
    score: float
    source: str  # "user_prompt" | "model_response"


def attribute_sentences(
    model: SentenceTransformer,
    query: str,
    document: str,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """
    Return up to `top_n` sentences ranked by cosine similarity to the query embedding.
    """
    sents = split_sentences(document)
    if not sents:
        return []
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    emb = model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q, emb)[0].detach().cpu().numpy()
    order = np.argsort(-sims)
    out: list[tuple[str, float]] = []
    for i in order[:top_n]:
        i = int(i)
        out.append((sents[i], float(sims[i])))
    return out


def attribute_prompt_and_response(
    model: SentenceTransformer,
    query: str,
    user_prompt: str,
    model_response: str,
    top_n: int = 8,
) -> list[SentenceAttribution]:
    """
    Score every sentence from `user_prompt` and `model_response` separately, tag source, return global top‑N.
    """
    items: list[tuple[str, str]] = []
    for s in split_sentences(user_prompt):
        items.append((s, "user_prompt"))
    for s in split_sentences(model_response):
        items.append((s, "model_response"))
    if not items:
        return []
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    texts = [t for t, _ in items]
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q, emb)[0].detach().cpu().numpy()
    order = np.argsort(-sims)
    out: list[SentenceAttribution] = []
    for j in order[:top_n]:
        j = int(j)
        out.append(SentenceAttribution(text=texts[j], score=float(sims[j]), source=items[j][1]))
    return out


def format_attribution_for_llm_context(rows: list[SentenceAttribution]) -> str:
    """Compact lines for RETRIEVED_CONTEXT (evidence sentences)."""
    lines = []
    for i, r in enumerate(rows, 1):
        src = "prompt" if r.source == "user_prompt" else "model_response"
        lines.append(f"{i}. [{src}] (sim={r.score:.3f}) {r.text}")
    return "\n".join(lines)
