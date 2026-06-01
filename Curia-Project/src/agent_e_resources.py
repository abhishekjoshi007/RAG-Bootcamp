"""Agent E — resource catalog matcher.

Maps each tracked skill to open-courseware resources and scores them
demand- and drift-aware, per the proposal:

    match_score = skill_match * demand * (1 - drift_risk)

where demand is the mean Agent A skill-intensity and drift_risk is the Agent C
drift score, both read from the cache (defaults applied on miss). v1 uses a
small synthetic catalog; v2 ingests MIT OCW / edX / freeCodeCamp / GitHub
awesome-lists and tags them via the shared skill-extraction pipeline.
"""
from __future__ import annotations

import hashlib

from .forecasting import tracked_skills

_PROVIDERS = (("mit_ocw", "course"), ("edx", "course"), ("fcc", "tutorial"))

_DEFAULT_DEMAND = 0.5
_DEFAULT_DRIFT_RISK = 0.0


def _stable_int(*parts: str) -> int:
    digest = hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class ResourceMatcher:
    def __init__(self, cache: object | None = None, skills=None) -> None:
        self.cache = cache
        self.skills = list(skills) if skills else tracked_skills()

    def _demand(self, skill_id: str) -> float:
        if self.cache is None:
            return _DEFAULT_DEMAND
        entries = self.cache.get_agent_a(skill_id)
        if not entries:
            return _DEFAULT_DEMAND
        mean_intensity = sum(e.value["intensity"] for e in entries) / len(entries)
        return round(max(0.0, min(1.0, mean_intensity)), 4)

    def _drift_risk(self, skill_id: str) -> float:
        if self.cache is None:
            return _DEFAULT_DRIFT_RISK
        row = self.cache.get_agent_c(skill_id)
        if not row:
            return _DEFAULT_DRIFT_RISK
        return round(max(0.0, min(1.0, float(row.get("drift_score", 0.0)))), 4)

    def match_all_skills(self) -> list[dict]:
        rows: list[dict] = []
        for skill in self.skills:
            demand = self._demand(skill)
            drift_risk = self._drift_risk(skill)
            for depth, (prefix, fmt) in enumerate(_PROVIDERS):
                h = _stable_int(skill, prefix)
                skill_match = round(0.6 + (h % 40) / 100.0, 4)
                match_score = round(
                    max(0.0, min(1.0, skill_match * demand * (1.0 - drift_risk))), 4
                )
                rows.append({
                    "skill_id": skill,
                    "resource_id": f"{prefix}_{h % 100000}",
                    "match_score": match_score,
                    "prerequisite_depth": depth,
                    "estimated_hours": 10 + (h % 40),
                    "meta": {
                        "title": f"{skill.title()} ({fmt})",
                        "url": f"https://example.org/{prefix}/{h % 100000}",
                        "source": prefix,
                        "format": fmt,
                        "level": depth,
                        "skill_match": skill_match,
                        "demand": demand,
                        "drift_risk": drift_risk,
                    },
                })
        return rows
