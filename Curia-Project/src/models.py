from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str
    date: date
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_id: str
    title: str
    source: str
    date: date
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    similarity: float
    score: float


@dataclass(frozen=True)
class Recommendation:
    signal_strength: str
    summary: str
    emerging_topics: list[str]
    evidence_ids: list[str]
