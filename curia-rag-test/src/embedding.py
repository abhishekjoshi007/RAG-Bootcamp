from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_BATCH_SIZE


TERM_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#]*")


def normalize_terms(text: str) -> list[str]:
    return [term.lower() for term in TERM_RE.findall(text)]


class LocalTfidfEmbedder:
    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        doc_count = max(len(texts), 1)
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(normalize_terms(text)))
        self.idf = {
            term: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for term, freq in document_frequency.items()
        }

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(normalize_terms(text))
        if not counts:
            return {}
        vector = {
            term: (1.0 + math.log(count)) * self.idf.get(term, 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return vector
        return {term: value / norm for term, value in vector.items()}

    def embed_many(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.embed(text) for text in texts]


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


class DenseEmbedder:
    """Dense embedder backed by a sentence-transformers model (all-mpnet-base-v2 default).

    The model is lazy-loaded on first call so importing this module doesn't
    trigger a large download.  Vectors are L2-normalised so dot-product equals
    cosine similarity — compatible with FAISS IndexFlatIP.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def _loaded(self) -> SentenceTransformer:
        if self._model is None:
            import torch
            # Force single-threaded CPU: MPS and multi-threaded BLAS both
            # cause segfaults on macOS when encoding large batches.
            torch.set_num_threads(1)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        return self._loaded.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def embed_many(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        return self._loaded.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        ).astype(np.float32)
