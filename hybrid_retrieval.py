"""
Hybrid retrieval: BM25 (lexical) + dense sentence-transformers + score fusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from semantic_retrieval import SemanticRetriever


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def weighted_fusion(dense: np.ndarray, bm25: np.ndarray, alpha: float) -> np.ndarray:
    """alpha in [0,1]: weight on dense; (1-alpha) on BM25 after per-channel min-max."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * _minmax(dense) + (1.0 - a) * _minmax(bm25)


def rrf_fusion(rankings: list[list[int]], k: int = 60) -> np.ndarray:
    """
    Reciprocal Rank Fusion across ordered row-index lists (each list is best-first).
    Returns a score vector aligned with row indices 0..n-1.
    """
    if not rankings:
        return np.array([])
    n = max(max(r) for r in rankings if r) + 1 if any(rankings) else 0
    scores = np.zeros(n, dtype=np.float64)
    for rlist in rankings:
        for rank, idx in enumerate(rlist):
            if 0 <= idx < n:
                scores[idx] += 1.0 / (k + rank + 1)
    return scores


@dataclass
class HybridHit:
    row_index: int
    case_id: str
    fused_score: float
    dense_score: float
    bm25_score: float


class HybridRetriever:
    """BM25 + dense embeddings; supports weighted fusion or RRF."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._dense = SemanticRetriever(model_name)
        self._bm25: BM25Okapi | None = None
        self._df: pd.DataFrame | None = None
        self._dense_is_bm25_fallback: bool = False

    @property
    def dense(self) -> SemanticRetriever:
        return self._dense

    def fit(self, df: pd.DataFrame, corpus_texts: list[str]) -> None:
        self._df = df.reset_index(drop=True)
        self._dense.fit(self._df, corpus_texts)
        self._dense_is_bm25_fallback = not self._dense.available
        tokenized = [simple_tokenize(t) for t in corpus_texts]
        # rank_bm25 tolerates empty docs if we add a dummy token
        tokenized = [t if t else ["empty"] for t in tokenized]
        self._bm25 = BM25Okapi(tokenized)

    def bm25_distribution(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.array([])
        q = simple_tokenize(query)
        if not q:
            q = ["query"]
        return np.array(self._bm25.get_scores(q), dtype=np.float64)

    def dense_distribution(self, query: str) -> np.ndarray:
        d = self._dense.similarity_distribution(query)
        if d.size > 0:
            return d
        # SentenceTransformer failed (network timeout, etc.): use BM25 as the dense signal so hybrid still runs
        if self._bm25 is not None:
            b = self.bm25_distribution(query)
            return _minmax(b)
        return np.array([])

    def fused_distribution(self, query: str, alpha: float, mode: str = "weighted") -> np.ndarray:
        d = self.dense_distribution(query)
        b = self.bm25_distribution(query)
        if d.size == 0 or b.size == 0:
            return np.array([])
        if mode == "weighted":
            return weighted_fusion(d, b, alpha)
        if mode == "rrf":
            order_d = np.argsort(-d).tolist()
            order_b = np.argsort(-b).tolist()
            return rrf_fusion([order_d, order_b])
        return weighted_fusion(d, b, alpha)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        alpha: float = 0.5,
        fusion: str = "weighted",
    ) -> list[HybridHit]:
        if self._df is None:
            return []
        d = self.dense_distribution(query)
        b = self.bm25_distribution(query)
        if d.size == 0 or b.size == 0:
            return []
        if fusion == "rrf":
            fused = self.fused_distribution(query, alpha, mode="rrf")
        else:
            fused = weighted_fusion(d, b, alpha)
        k = min(top_k, len(fused))
        idxs = np.argsort(-fused)[:k]
        out: list[HybridHit] = []
        for idx in idxs:
            idx = int(idx)
            out.append(
                HybridHit(
                    row_index=idx,
                    case_id=str(self._df.iloc[idx]["case_id"]),
                    fused_score=float(fused[idx]),
                    dense_score=float(d[idx]),
                    bm25_score=float(b[idx]),
                )
            )
        return out
