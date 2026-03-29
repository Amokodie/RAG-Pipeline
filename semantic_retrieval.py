"""
Semantic retrieval over alignment-audit rows using sentence-transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


@dataclass
class SemanticHit:
    row_index: int
    case_id: str
    score: float


def build_index_text(df: pd.DataFrame) -> list[str]:
    """Index: category + subcategory + user_prompt (searchable semantics)."""
    return [
        f"{row['category']} | {row['subcategory']} | {row['user_prompt']}"
        for _, row in df.iterrows()
    ]


class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._embeddings: torch.Tensor | None = None
        self._df: pd.DataFrame | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            raise RuntimeError("SemanticRetriever not fitted.")
        return self._model

    def fit(self, df: pd.DataFrame, index_texts: list[str]) -> None:
        self._df = df.reset_index(drop=True)
        self._model = SentenceTransformer(self._model_name)
        self._embeddings = self._model.encode(
            index_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def similarity_distribution(self, query: str) -> np.ndarray:
        """Dense cosine similarity vs every chunk (same order as fitted dataframe rows)."""
        if self._model is None or self._embeddings is None:
            return np.array([])
        q = self._model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sims = util.cos_sim(q, self._embeddings)[0]
        return sims.detach().cpu().numpy()

    def search(self, query: str, top_k: int = 5) -> list[SemanticHit]:
        if self._model is None or self._embeddings is None or self._df is None:
            return []
        sims = self.similarity_distribution(query)
        k = min(top_k, int(sims.shape[0]))
        idxs = np.argsort(-sims)[:k]
        out: list[SemanticHit] = []
        for idx in idxs:
            idx = int(idx)
            out.append(
                SemanticHit(
                    row_index=idx,
                    case_id=str(self._df.iloc[idx]["case_id"]),
                    score=float(sims[idx]),
                )
            )
        return out


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)
