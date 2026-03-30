"""
Semantic retrieval over alignment-audit rows using sentence-transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util

from hf_hub_config import configure_hf_hub, load_sentence_transformer


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
        self._model = None
        self._embeddings: torch.Tensor | None = None
        self._df: pd.DataFrame | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._model is not None and self._embeddings is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(
                "SentenceTransformer not loaded. Check network / Hugging Face access, or set HF_ENDPOINT mirror."
            )
        return self._model

    def fit(self, df: pd.DataFrame, index_texts: list[str]) -> None:
        self._df = df.reset_index(drop=True)
        self._load_error = None
        configure_hf_hub()
        try:
            self._model = load_sentence_transformer(self._model_name)
            self._embeddings = self._model.encode(
                index_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            self._model = None
            self._embeddings = None
            self._load_error = f"{type(e).__name__}: {e}"

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


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    configure_hf_hub()
    return load_sentence_transformer(model_name)
