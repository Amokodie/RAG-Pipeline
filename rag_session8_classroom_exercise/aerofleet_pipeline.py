"""
AeroFleet X200 RAG pipeline: load catalog + markdown corpus, chunk, embed, FAISS index, authority rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate token budget with words (~0.75 words per token for English technical text)
CHUNK_WORDS = 200
OVERLAP_WORDS = 50


@dataclass
class ChunkRecord:
    chunk_id: int
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_date: str
    authority_level: str
    filename: str
    text: str
    chunk_index: int


def base_dir() -> Path:
    return Path(__file__).resolve().parent


def corpus_dir() -> Path:
    return base_dir() / "dataset" / "corpus"


def catalog_path() -> Path:
    return base_dir() / "dataset" / "corpus_catalog.csv"


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(catalog_path())


def chunk_text_words(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Fixed-size overlapping word windows (proxy for token chunking)."""
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap_words, 1)
    chunks: list[str] = []
    i = 0
    while i < len(words):
        piece = words[i : i + chunk_words]
        chunks.append(" ".join(piece))
        i += step
    return chunks


def load_all_chunks() -> tuple[list[ChunkRecord], pd.DataFrame]:
    catalog = load_catalog()
    records: list[ChunkRecord] = []
    cid = 0
    for _, row in catalog.iterrows():
        fn = row["filename"]
        path = corpus_dir() / fn
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        parts = chunk_text_words(raw)
        for j, part in enumerate(parts):
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    doc_type=str(row["doc_type"]),
                    status=str(row["status"]),
                    effective_date=str(row["effective_date"]),
                    authority_level=str(row["authority_level"]),
                    filename=str(fn),
                    text=part.strip(),
                    chunk_index=j,
                )
            )
            cid += 1
    return records, catalog


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


class AeroFleetIndex:
    """Dense embeddings + FAISS inner-product search on normalized vectors (= cosine similarity)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._chunks: list[ChunkRecord] = []
        self._dim: int | None = None

    def fit(self, chunks: list[ChunkRecord]) -> None:
        from sentence_transformers import SentenceTransformer
        import faiss

        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._model = SentenceTransformer(self.model_name)
        emb = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype(np.float32)
        emb = l2_normalize(emb)
        self._dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(emb)

    def search(self, query: str, k: int = 10) -> list[tuple[ChunkRecord, float]]:
        if self._model is None or self._index is None or not self._chunks:
            return []
        q = self._model.encode([query], convert_to_numpy=True).astype(np.float32)
        q = l2_normalize(q)
        scores, idxs = self._index.search(q, min(k, len(self._chunks)))
        out: list[tuple[ChunkRecord, float]] = []
        for sc, ix in zip(scores[0], idxs[0]):
            if ix < 0:
                continue
            out.append((self._chunks[int(ix)], float(sc)))
        return out


def apply_authority_filter(
    ranked: list[tuple[ChunkRecord, float]],
    *,
    retrieval_k: int = 5,
    pool_size: int = 20,
) -> tuple[list[tuple[ChunkRecord, float]], list[str]]:
    """
    If D02 appears anywhere in the top `pool_size` hits, drop D03 chunks when building the answer context
    (scan full ranked list in order until `retrieval_k` chunks are kept).
    """
    notes: list[str] = []
    pool = ranked[:pool_size]
    d02_present = any(c.doc_id == "D02" for c, _ in pool)
    d03_in_raw_top = any(c.doc_id == "D03" for c, _ in ranked[:retrieval_k])

    if d03_in_raw_top and d02_present:
        notes.append(
            "Authority rule: **D03** (outdated archive) was skipped for generation because **D02** "
            "(current maintenance manual v2.1) is in the retrieval pool—intervals/thresholds must follow **D02**."
        )
    elif d03_in_raw_top and not d02_present:
        notes.append(
            "Warning: **D03** is outdated. Prefer **D02** Maintenance Manual v2.1 for current procedures."
        )

    selected: list[tuple[ChunkRecord, float]] = []
    for c, s in ranked:
        if len(selected) >= retrieval_k:
            break
        if c.doc_id == "D03" and d02_present:
            continue
        selected.append((c, s))
    return selected, notes


def build_context_for_llm(chunks: list[tuple[ChunkRecord, float]]) -> str:
    parts = []
    for c, score in chunks:
        header = (
            f"[{c.doc_id} | {c.status} | effective {c.effective_date} | similarity {score:.4f}]\n"
            f"{c.text}"
        )
        parts.append(header)
    return "\n\n---\n\n".join(parts)
