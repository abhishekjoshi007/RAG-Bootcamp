from __future__ import annotations

from collections import defaultdict
from datetime import date

from .indexing import InMemoryIndex
from .models import SearchResult


def recency_multiplier(
    doc_date: date,
    today: date | None = None,
    half_life_days: int = 365,
) -> float:
    today = today or date.today()
    age_days = max((today - doc_date).days, 0)
    recency_weight = 0.5 ** (age_days / half_life_days)
    return 0.7 + 0.3 * recency_weight


class Retriever:
    def __init__(
        self,
        index: InMemoryIndex,
        recency_half_life_days: int = 365,
    ) -> None:
        self.index = index
        self.recency_half_life_days = recency_half_life_days

    def retrieve(
        self,
        query: str,
        k: int = 8,
        candidate_k: int = 50,
        source_quotas: dict[str, int] | None = None,
        today: date | None = None,
    ) -> list[SearchResult]:
        candidates = self.index.search(query, k=max(candidate_k, k))
        rescored = [
            SearchResult(
                chunk=result.chunk,
                similarity=result.similarity,
                score=result.similarity
                * recency_multiplier(
                    result.chunk.date,
                    today=today,
                    half_life_days=self.recency_half_life_days,
                ),
            )
            for result in candidates
            if result.similarity > 0
        ]

        if source_quotas:
            grouped: dict[str, list[SearchResult]] = defaultdict(list)
            for result in rescored:
                grouped[result.chunk.source].append(result)
            selected: list[SearchResult] = []
            for source, quota in source_quotas.items():
                selected.extend(
                    sorted(grouped.get(source, []), key=lambda item: item.score, reverse=True)[
                        :quota
                    ]
                )
            selected.sort(key=lambda item: item.score, reverse=True)
            return selected[:k]

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:k]
