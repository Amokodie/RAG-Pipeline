"""
TF-IDF retrieval over the IT knowledge SQLite corpus.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItKbHit:
    doc_id: int
    title: str
    category: str
    body: str
    source: str
    score: float


def _index_text(title: str, category: str, body: str) -> str:
    return f"{category} {title} {body}"


class ItKbRetriever:
    """Lightweight TF-IDF index over `it_docs` rows."""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._rows: list[tuple[int, str, str, str, str]] = []

    def fit(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "SELECT id, title, category, body, source FROM it_docs ORDER BY id"
            )
            self._rows = [tuple(r) for r in cur.fetchall()]
        finally:
            conn.close()

        if not self._rows:
            self._vectorizer = None
            self._matrix = None
            return

        texts = [_index_text(t, c, b) for (i, t, c, b, s) in self._rows]
        self._vectorizer = TfidfVectorizer(
            max_features=12_000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(self, text: str, top_k: int = 4) -> list[ItKbHit]:
        if not self._rows or self._vectorizer is None or self._matrix is None:
            return []
        q = self._vectorizer.transform([text])
        sims = cosine_similarity(q, self._matrix).ravel()
        k = min(top_k, len(sims))
        order = np.argsort(-sims)[:k]
        out: list[ItKbHit] = []
        for idx in order:
            idx = int(idx)
            did, title, cat, body, src = self._rows[idx]
            out.append(
                ItKbHit(
                    doc_id=did,
                    title=title,
                    category=cat,
                    body=body,
                    source=src,
                    score=float(sims[idx]),
                )
            )
        return out
