from __future__ import annotations

import pickle
from pathlib import Path

from .embedding import LocalTfidfEmbedder, cosine_sparse
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
