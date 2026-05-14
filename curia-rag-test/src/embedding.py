from __future__ import annotations

import math
import re
from collections import Counter


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
