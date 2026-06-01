"""Agent D — Curriculum & Roadmap (live, query-specific).

Overlays the pre-materialized agent outputs (A demand, B forecasts, C drift,
E resources) onto the learner's goal to produce an ordered roadmap. v1 is a
deterministic assembler; the LLM still synthesizes the final narrative.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class RoadmapAgent:
    def build_roadmap(
        self,
        learner_query,
        evidence: Sequence[Any],
        agent_a: Mapping[str, Any],
        agent_b: Mapping[str, Any],
        agent_c: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> dict:
        completed = set(learner_query.completed_skills)
        steps: list[dict] = []
        for skill in sorted(agent_a.keys()):
            if skill in completed:
                continue
            forecasts = agent_b.get(skill, {}) or {}
            slope = self._dominant_slope(forecasts)
            steps.append({
                "skill": skill,
                "demand_entries": len(agent_a.get(skill, []) or []),
                "trend": "rising" if slope > 0 else "declining" if slope < 0 else "stable",
                "drift": (agent_c.get(skill) or {}).get("direction"),
                "resources": [r.get("resource_id") for r in (resources.get(skill) or [])][:3],
            })
        return {
            "goal": learner_query.goal,
            "program": learner_query.program,
            "completed_skills": sorted(completed),
            "steps": steps,
        }

    @staticmethod
    def _dominant_slope(forecasts: Mapping[int, Any]) -> float:
        for horizon in (12, 24, 6, 3):
            row = forecasts.get(horizon)
            if row:
                return float(row.get("slope", 0.0))
        return 0.0
