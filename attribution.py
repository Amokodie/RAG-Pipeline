"""
Sentence-level attribution: which sentence in a chunk best matches the query (dense similarity).
"""

from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer, util


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence split (good enough for audit rows)."""
    if not text or not str(text).strip():
        return []
    t = str(text).replace("\r\n", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n+", t.strip())
    return [p.strip() for p in parts if p.strip()]


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
