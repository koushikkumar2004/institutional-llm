"""Wrappers around the two small retrieval models (both run fine on a laptop CPU).

The heavy library (sentence-transformers / PyTorch) is only imported when a model
is actually created, so tests can swap in lightweight fakes.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EmbedderProtocol(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class RerankerProtocol(Protocol):
    name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class Embedder:
    """Bi-encoder: turns each text into a normalised vector (MiniLM, 384 numbers)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,  # makes inner product == cosine similarity
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


class Reranker:
    """Cross-encoder: reads the question and one FAQ together and scores how well they match."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        scores = self._model.predict([(query, d) for d in documents], show_progress_bar=False)
        return [float(s) for s in np.atleast_1d(scores)]
