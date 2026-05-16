from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from .embedding import DenseEmbedder, LocalTfidfEmbedder, cosine_sparse
from .models import Chunk, SearchResult


class InMemoryIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[dict[str, float]],
        embedder: LocalTfidfEmbedder,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "InMemoryIndex":
        embedder = LocalTfidfEmbedder()
        embedder.fit([chunk.text for chunk in chunks])
        return cls(chunks, embedder.embed_many([chunk.text for chunk in chunks]), embedder)

    def search(self, query: str, k: int = 8) -> list[SearchResult]:
        query_vector = self.embedder.embed(query)
        scored = [
            SearchResult(chunk=chunk, similarity=cosine_sparse(query_vector, vector), score=0.0)
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda result: result.similarity, reverse=True)
        return scored[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "InMemoryIndex":
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, InMemoryIndex):
            raise TypeError(f"{path} does not contain an InMemoryIndex")
        return loaded


class FaissIndex:
    """Dense-vector index backed by FAISS IndexFlatIP (exact cosine search).

    Vectors are produced by DenseEmbedder (all-mpnet-base-v2 default) which
    L2-normalises embeddings, so dot-product == cosine similarity.

    Serialisation: a single pickle file containing the FAISS binary, all
    Chunk objects, and the embedder model name so the file is self-contained.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: DenseEmbedder,
        _index: faiss.IndexFlatIP,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self._index = _index

    @classmethod
    def build(cls, chunks: list[Chunk], model_name: str = "all-mpnet-base-v2") -> "FaissIndex":
        embedder = DenseEmbedder(model_name)
        vectors: np.ndarray = embedder.embed_many([c.text for c in chunks])
        index: faiss.IndexFlatIP = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(chunks, embedder, index)

    def search(self, query: str, k: int = 8) -> list[SearchResult]:
        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            return []
        q_vec: np.ndarray = self.embedder.embed(query).reshape(1, -1)
        scores, indices = self._index.search(q_vec, actual_k)
        return [
            SearchResult(chunk=self.chunks[idx], similarity=float(score), score=0.0)
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "faiss_arr": faiss.serialize_index(self._index),
            "chunks": self.chunks,
            "model_name": self.embedder.model_name,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    @staticmethod
    def load(path: Path) -> "FaissIndex":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        index: faiss.IndexFlatIP = faiss.deserialize_index(payload["faiss_arr"])
        embedder = DenseEmbedder(payload["model_name"])
        return FaissIndex(payload["chunks"], embedder, index)
