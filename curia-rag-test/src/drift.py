from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .indexing import FaissIndex, InMemoryIndex
from .models import Chunk


_SOURCE_LABELS: dict[str, str] = {
    "job_posting":   "Industry (jobs)",
    "arxiv":         "Research (arXiv)",
    "stackoverflow": "Practitioners (SO)",
    "github_readme": "Open-source (GH)",
}


def _source_label(source: str) -> str:
    return _SOURCE_LABELS.get(source, source)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftBucket:
    label: str
    source: str
    centroid: tuple[float, ...]
    chunk_count: int


@dataclass(frozen=True)
class DriftPair:
    from_label: str
    to_label: str
    cosine_distance: float


@dataclass(frozen=True)
class DriftResult:
    skill: str
    mode: str                              # "cross_source" or "temporal"
    buckets: tuple[DriftBucket, ...]
    pairs: tuple[DriftPair, ...]
    max_drift: float
    mean_drift: float
    drifted: bool
    drift_threshold: float
    chunk_count_total: int


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        return np.zeros(0, dtype=np.float32)
    stack = np.stack(embeddings)
    return _l2_normalize(stack.mean(axis=0)).astype(np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    return float(1.0 - np.dot(a, b))


def _quarter_label(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


class SemanticDriftDetector:

    def __init__(
        self,
        index: FaissIndex | InMemoryIndex,
        chunks_per_skill: int = 40,
        min_chunks_per_bucket: int = 2,
        drift_threshold: float = 0.15,
        mode: str = "cross_source",   # or "temporal"
    ) -> None:
        if mode not in ("cross_source", "temporal"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.index = index
        self.chunks_per_skill = chunks_per_skill
        self.min_chunks_per_bucket = min_chunks_per_bucket
        self.drift_threshold = drift_threshold
        self.mode = mode

    def _retrieve_chunks(self, skill: str) -> list[Chunk]:
        results = self.index.search(skill, k=self.chunks_per_skill)
        return [r.chunk for r in results if r.similarity > 0.05]

    def _embed_chunks(self, chunks: list[Chunk]) -> list[np.ndarray]:
        if isinstance(self.index, FaissIndex):
            chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.index.chunks)}
            embeddings: list[np.ndarray] = []
            need_embed: list[tuple[int, str]] = []   # (output_idx, text)
            for out_idx, c in enumerate(chunks):
                idx = chunk_id_to_idx.get(c.chunk_id)
                if idx is not None:
                    embeddings.append(self.index._index.reconstruct(idx))
                else:
                    embeddings.append(np.zeros(0, dtype=np.float32))
                    need_embed.append((out_idx, c.text))
            if need_embed:
                fresh = self.index.embedder.embed_many([t for _, t in need_embed])
                for (out_idx, _), vec in zip(need_embed, fresh):
                    embeddings[out_idx] = vec
            return embeddings
        return [self.index.embedder.embed(c.text) for c in chunks]

    def _bucket_key(self, chunk: Chunk) -> str:
        if self.mode == "temporal":
            return _quarter_label(chunk.date)
        return chunk.source

    def analyze_skill(self, skill: str) -> DriftResult | None:
        chunks = self._retrieve_chunks(skill)
        if len(chunks) < max(4, 2 * self.min_chunks_per_bucket):
            return None

        embeddings = self._embed_chunks(chunks)

        groups: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            groups.setdefault(self._bucket_key(c), []).append(i)

        kept = {k: idxs for k, idxs in groups.items() if len(idxs) >= self.min_chunks_per_bucket}
        if len(kept) < 2:
            return None

        buckets: list[DriftBucket] = []
        centroids: dict[str, np.ndarray] = {}
        for key, idxs in kept.items():
            c = _centroid([embeddings[i] for i in idxs])
            centroids[key] = c
            pretty = _source_label(key) if self.mode == "cross_source" else key
            buckets.append(DriftBucket(
                label=pretty,
                source=key,
                centroid=tuple(float(x) for x in c.tolist()),
                chunk_count=len(idxs),
            ))

        keys = list(kept.keys())
        pairs: list[DriftPair] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                pretty_a = _source_label(ka) if self.mode == "cross_source" else ka
                pretty_b = _source_label(kb) if self.mode == "cross_source" else kb
                dist = _cosine_distance(centroids[ka], centroids[kb])
                pairs.append(DriftPair(
                    from_label=pretty_a,
                    to_label=pretty_b,
                    cosine_distance=round(dist, 4),
                ))

        if not pairs:
            return None

        distances = [p.cosine_distance for p in pairs]
        max_drift  = max(distances)
        mean_drift = sum(distances) / len(distances)

        return DriftResult(
            skill=skill,
            mode=self.mode,
            buckets=tuple(buckets),
            pairs=tuple(sorted(pairs, key=lambda p: p.cosine_distance, reverse=True)),
            max_drift=round(max_drift, 4),
            mean_drift=round(mean_drift, 4),
            drifted=max_drift > self.drift_threshold,
            drift_threshold=self.drift_threshold,
            chunk_count_total=sum(b.chunk_count for b in buckets),
        )

    def analyze_skills(self, skills: list[str]) -> list[DriftResult]:
        out: list[DriftResult] = []
        for s in skills:
            r = self.analyze_skill(s)
            if r is not None:
                out.append(r)
        return out

    @staticmethod
    def top_drifters(results: list[DriftResult], n: int = 5) -> list[DriftResult]:
        return sorted(
            [r for r in results if r.drifted],
            key=lambda r: r.max_drift,
            reverse=True,
        )[:n]
