"""
Quantitative helpers for the alignment-audit RAG demo (TF‑IDF space).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from rag_pipeline import TfidfRetriever


def intra_inter_category_cosine(df: pd.DataFrame, sim_matrix: np.ndarray) -> dict:
    """
    Mean cosine between chunk pairs that share a `category` vs pairs that do not.
    Higher separation (intra − inter) suggests lexical clustering by HHH axis label.
    """
    cats = df["category"].astype(str).values
    n = len(cats)
    intra: list[float] = []
    inter: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if cats[i] == cats[j]:
                intra.append(s)
            else:
                inter.append(s)
    mean_intra = float(np.mean(intra)) if intra else 0.0
    mean_inter = float(np.mean(inter)) if inter else 0.0
    return {
        "mean_intra_category_cosine": mean_intra,
        "mean_inter_category_cosine": mean_inter,
        "separation_intra_minus_inter": mean_intra - mean_inter,
        "pairs_intra": len(intra),
        "pairs_inter": len(inter),
    }


def silhouette_on_lsa(chunk_xyz: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Silhouette score on 3D LSA coordinates using category as pseudo-labels.
    Interpret with caution: labels are coarse (4 categories × 4 rows).
    """
    if chunk_xyz.shape[0] < 4:
        return {"silhouette": None, "note": "Too few points for a stable silhouette."}
    le = LabelEncoder()
    y = le.fit_transform(df["category"].astype(str))
    n_labels = len(np.unique(y))
    if n_labels < 2:
        return {"silhouette": None, "note": "Need at least 2 categories."}
    sil = float(silhouette_score(chunk_xyz[:, :3], y, metric="euclidean"))
    return {"silhouette": sil, "n_labels": n_labels, "note": "Euclidean distance in LSA 3D space; category as cluster label."}


def sparsity_report(stats: dict) -> dict:
    """Density of the document-term matrix (fraction of nonzeros)."""
    n = stats.get("n_chunks", 0)
    v = stats.get("vocabulary_size", 0)
    nnz = stats.get("matrix_nonzero", 0)
    denom = max(n * v, 1)
    density = nnz / denom
    return {
        "tfidf_matrix_density": float(density),
        "tfidf_matrix_sparsity": float(1.0 - density),
        "interpretation": "Sparse rows are normal: each chunk uses only a fraction of the vocabulary.",
    }


def rank_distribution_report(retriever: TfidfRetriever, df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, use its `user_prompt` as the query and record cosine rank of that row’s chunk.
    Summarizes how often lexical self-queries recover the intended chunk at rank 1.
    """
    rows = []
    case_order = df["case_id"].tolist()
    for pos, cid in enumerate(case_order):
        q = str(df.loc[df["case_id"] == cid, "user_prompt"].iloc[0])
        sims = retriever.similarity_distribution(q)
        order = np.argsort(-sims)
        rank = int(np.where(order == pos)[0][0]) + 1
        top_sim = float(sims[pos])
        rows.append({"case_id": cid, "self_query_rank": rank, "self_cosine": top_sim})
    return pd.DataFrame(rows)


def softmax_distribution(sims: np.ndarray) -> np.ndarray:
    """Full-corpus softmax over cosine scores (same as retrieval_margin in rag_pipeline)."""
    z = sims - np.max(sims)
    ez = np.exp(np.clip(z, -50, 50))
    return ez / np.maximum(ez.sum(), 1e-12)


def max_cosine_per_category(df: pd.DataFrame, sims: np.ndarray) -> dict[str, float]:
    """For one query, best-matching chunk score inside each alignment `category`."""
    sims = np.asarray(sims).ravel()
    cats = df["category"].astype(str).values
    out: dict[str, float] = {}
    for cat in sorted(set(cats.tolist())):
        mask = cats == cat
        out[cat] = float(np.max(sims[mask])) if np.any(mask) else 0.0
    return out


def chunk_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Token-ish proxy: character counts for prompts and responses."""
    pr = df["user_prompt"].astype(str).str.len()
    mr = df["model_response"].astype(str).str.len()
    return pd.DataFrame(
        {
            "case_id": df["case_id"],
            "category": df["category"],
            "prompt_chars": pr,
            "response_chars": mr,
            "total_chars": pr + mr,
        }
    )
