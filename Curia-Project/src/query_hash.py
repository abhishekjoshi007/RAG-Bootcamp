"""Canonical query representation and hashing.

A LearnerQuery is the input to the per-query pipeline. Two queries that should
yield the same recommendation produce the same hash: text fields are lowercased
and whitespace-collapsed, list fields are de-duplicated and sorted, then the
canonical dict is JSON-serialized with sorted keys and SHA-256'd (16 hex chars).

v1 requires exact text match for query_text and goal; semantic equivalence is
deferred to v2.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s.strip().lower())


@dataclass(frozen=True)
class LearnerQuery:
    program: str
    curriculum_unit_ids: tuple[str, ...] = field(default_factory=tuple)
    goal: str = ""
    completed_skills: tuple[str, ...] = field(default_factory=tuple)
    query_text: str = ""

    def normalized(self) -> dict:
        return {
            "program": _normalize_text(self.program),
            "curriculum_unit_ids": sorted(set(self.curriculum_unit_ids)),
            "goal": _normalize_text(self.goal),
            "completed_skills": sorted(set(self.completed_skills)),
            "query_text": _normalize_text(self.query_text),
        }

    def hash(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
