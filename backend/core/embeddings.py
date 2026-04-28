"""
core/embeddings.py — Embedding model wrapper.

Uses sentence-transformers (all-MiniLM-L6-v2 by default).
  - 22M parameters, 384-dimensional embeddings
  - ~2500 sentences/sec on CPU
  - Cosine similarity friendly (L2-normalised output)

The EmbeddingModel is a singleton to avoid reloading weights
on every request.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.config import settings


class EmbeddingModel:
    """
    Thread-safe singleton wrapper around SentenceTransformer.

    Usage
    -----
    model = EmbeddingModel.get_instance()
    vectors = model.embed(["text one", "text two"])
    """

    _instance: Optional["EmbeddingModel"] = None
    _lock = threading.Lock()

    def __init__(self):
        logger.info(
            f"[Embeddings] Loading '{settings.EMBEDDING_MODEL}' "
            f"on device='{settings.EMBEDDING_DEVICE}' …"
        )
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"[Embeddings] Loaded. Dimension={self._dim}")

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Encode texts into normalised embedding vectors.

        Returns a list of float lists (not numpy arrays) for
        compatibility with ChromaDB's API.
        """
        if not texts:
            return []

        # sentence-transformers returns numpy arrays, normalised by default
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # Ensures cosine sim == dot product
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Convenience method for a single query string."""
        return self.embed([text])[0]

    def cosine_similarity(
        self, vec_a: List[float], vec_b: List[float]
    ) -> float:
        """Compute cosine similarity between two already-normalised vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        # Both are L2-normalised so dot product == cosine similarity
        return float(np.dot(a, b))
