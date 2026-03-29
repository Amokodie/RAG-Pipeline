"""
Simulated RAG pipeline: chunking, TF-IDF vectors (embedding proxy), cosine retrieval.
No external APIs — suitable for classroom demos and reproducible runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    case_id: str
    category: str
    subcategory: str
    user_prompt: str
    score: float
    chunk_index: int
    model_response_excerpt: str = ""


def default_data_path() -> Path:
    base = Path(__file__).resolve().parent
    return base / "session7_alignment_audit_package" / "data" / "session7_alignment_audit_dataset.csv"


def load_audit_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or default_data_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place session7_alignment_audit_dataset.csv under "
            "session7_alignment_audit_package/data/ or pass csv_path=."
        )
    df = pd.read_csv(path)
    required = {"case_id", "category", "subcategory", "user_prompt", "model_response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


CHUNK_RESPONSE_CHARS = 500


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Each row is one chunk. Text combines prompt + category signals for retrieval.
    """
    chunks: list[str] = []
    meta: list[dict] = []
    for i, row in df.iterrows():
        resp = str(row["model_response"])
        excerpt = resp[:CHUNK_RESPONSE_CHARS]
        text = " ".join(
            [
                str(row["category"]),
                str(row["subcategory"]),
                str(row["user_prompt"]),
                excerpt,
            ]
        )
        chunks.append(text)
        meta.append(
            {
                "chunk_index": len(chunks) - 1,
                "case_id": row["case_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "user_prompt": row["user_prompt"],
                "model_response": row["model_response"],
                "response_excerpt": excerpt,
                "df_index": i,
            }
        )
    return chunks, meta


class TfidfRetriever:
    """TF-IDF acts as a transparent stand-in for dense embedding retrieval in demos."""

    def __init__(self, max_features: int = 4096, ngram_range: tuple[int, int] = (1, 2)):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            strip_accents="unicode",
        )
        self._chunk_matrix = None
        self._meta: list[dict] = []
        self._chunks_fitted: list[str] = []
        self._svd_3d: TruncatedSVD | None = None
        self._chunk_xyz: np.ndarray | None = None

    def fit(self, chunks: list[str], meta: list[dict]) -> None:
        self._meta = meta
        self._chunks_fitted = list(chunks)
        self._chunk_matrix = self._vectorizer.fit_transform(chunks)

    def corpus_stats(self) -> dict:
        if self._chunk_matrix is None:
            return {"n_chunks": 0, "vocabulary_size": 0, "max_features_cap": self._vectorizer.max_features}
        vocab = getattr(self._vectorizer, "vocabulary_", None) or {}
        n, d = self._chunk_matrix.shape
        nnz = int(self._chunk_matrix.nnz)
        density = nnz / float(max(n * d, 1))
        return {
            "n_chunks": n,
            "vocabulary_size": len(vocab),
            "max_features_cap": self._vectorizer.max_features,
            "matrix_nonzero": nnz,
            "matrix_density": density,
            "avg_chunk_chars": float(np.mean([len(c) for c in self._chunks_fitted])) if self._chunks_fitted else 0.0,
        }

    def chunk_pairwise_cosine(self) -> np.ndarray:
        """Full n×n cosine similarity between chunk TF‑IDF rows."""
        if self._chunk_matrix is None:
            return np.array([])
        return cosine_similarity(self._chunk_matrix, self._chunk_matrix)

    def _fit_lsa_3d(self) -> tuple[TruncatedSVD, np.ndarray]:
        """Latent Semantic Analysis: first 3 factors for visualization (sparse-safe)."""
        if self._chunk_matrix is None:
            raise RuntimeError("Retriever not fitted.")
        if self._svd_3d is not None and self._chunk_xyz is not None:
            return self._svd_3d, self._chunk_xyz
        n_samples, n_features = self._chunk_matrix.shape
        n_comp = min(3, n_samples - 1, max(1, n_features - 1))
        n_comp = max(1, n_comp)
        self._svd_3d = TruncatedSVD(n_components=n_comp, random_state=42)
        self._chunk_xyz = self._svd_3d.fit_transform(self._chunk_matrix)
        return self._svd_3d, self._chunk_xyz

    def lsa_3d_layout(
        self, query: str | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, TruncatedSVD, np.ndarray]:
        """
        Returns (chunk_coords [n, n_comp], query_coords [1, n_comp] or None, svd_model, explained_variance_ratio).
        Pads to 3 columns for Plotly when n_comp < 3 (small corpora).
        """
        svd, chunk = self._fit_lsa_3d()
        ev = svd.explained_variance_ratio_
        q_coords = None
        if query and query.strip():
            q = self._vectorizer.transform([query.strip()])
            q_coords = svd.transform(q)
        chunk_p = _pad_xyz(chunk)
        q_p = _pad_xyz(q_coords) if q_coords is not None else None
        return chunk_p, q_p, svd, ev

    def retrieval_margin(self, query: str) -> dict:
        """Top-1 vs runner-up cosine and a softmax concentration proxy."""
        sims = self.similarity_distribution(query)
        if sims.size == 0:
            return {"top1": 0.0, "top2": 0.0, "margin": 0.0, "n": 0}
        sorted_s = np.sort(sims)[::-1]
        top1 = float(sorted_s[0])
        top2 = float(sorted_s[1]) if len(sorted_s) > 1 else 0.0
        z = sims - np.max(sims)
        ez = np.exp(np.clip(z, -50, 50))
        p = ez / np.maximum(ez.sum(), 1e-12)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        return {
            "top1": top1,
            "top2": top2,
            "margin": top1 - top2,
            "entropy_bits": ent / np.log(2),
            "n": int(sims.size),
        }

    def similarity_distribution(self, text: str) -> np.ndarray:
        """Cosine similarity of query against every chunk (for charts / analysis)."""
        if self._chunk_matrix is None:
            return np.array([])
        q = self._vectorizer.transform([text])
        return cosine_similarity(q, self._chunk_matrix).ravel()

    def top_weighted_terms(self, text: str, top_n: int = 18) -> list[tuple[str, float]]:
        """Non-zero TF-IDF terms for `text`, sorted by weight (explains the query vector)."""
        if self._chunk_matrix is None:
            return []
        q = self._vectorizer.transform([text])
        arr = q.toarray().ravel()
        names = self._vectorizer.get_feature_names_out()
        idx = np.argsort(-arr)
        out: list[tuple[str, float]] = []
        for i in idx:
            if arr[i] <= 0:
                break
            out.append((str(names[i]), float(arr[i])))
            if len(out) >= top_n:
                break
        return out

    def top_chunk_terms(self, chunk_index: int, top_n: int = 18) -> list[tuple[str, float]]:
        """Dominant TF-IDF terms for an indexed chunk (post-fit)."""
        if self._chunk_matrix is None or chunk_index < 0 or chunk_index >= self._chunk_matrix.shape[0]:
            return []
        row = self._chunk_matrix.getrow(chunk_index)
        arr = row.toarray().ravel()
        names = self._vectorizer.get_feature_names_out()
        idx = np.argsort(-arr)
        out: list[tuple[str, float]] = []
        for i in idx:
            if arr[i] <= 0 or len(out) >= top_n:
                break
            out.append((str(names[i]), float(arr[i])))
        return out[:top_n]

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._chunk_matrix is None or not self._meta:
            return []
        q = self._vectorizer.transform([text])
        sims = cosine_similarity(q, self._chunk_matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        out: list[RetrievalResult] = []
        for idx in order:
            m = self._meta[int(idx)]
            excerpt = m.get("response_excerpt", str(m.get("model_response", ""))[:CHUNK_RESPONSE_CHARS])
            out.append(
                RetrievalResult(
                    case_id=m["case_id"],
                    category=m["category"],
                    subcategory=m["subcategory"],
                    user_prompt=m["user_prompt"],
                    score=float(sims[idx]),
                    chunk_index=m["chunk_index"],
                    model_response_excerpt=excerpt,
                )
            )
        return out


def _pad_xyz(arr: np.ndarray) -> np.ndarray:
    """Ensure [*, 3] for 3D plotting when TruncatedSVD returns 1–2 components."""
    if arr.size == 0:
        return arr.reshape(0, 3)
    a = np.atleast_2d(arr)
    n = a.shape[1]
    if n >= 3:
        return a[:, :3]
    pad = np.zeros((a.shape[0], 3 - n))
    return np.hstack([a, pad])


def keyword_fallback(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Simple substring match on user_prompt (baseline before semantic layer)."""
    q = query.strip()
    if not q:
        return df.iloc[0:0]
    mask = df["user_prompt"].str.contains(q, case=False, na=False, regex=False)
    if not mask.any():
        mask = df["category"].str.contains(q, case=False, na=False, regex=False) | df[
            "subcategory"
        ].str.contains(q, case=False, na=False, regex=False)
    return df[mask]
