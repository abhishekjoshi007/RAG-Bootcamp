"""Agent A — fused skill-intensity signal (v1 weighted aggregator).

Counts skill mentions per (skill, source, ISO week) across the corpus, weights
by source quality, and normalizes to a 0..1 intensity per skill. v2 swaps in the
contrastive alignment encoder behind the same compute_for_window() API.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import CORPUS_DIR
from .forecasting import tracked_skills

_SOURCE_QUALITY = {
    "job_posting": 1.0,
    "arxiv": 0.8,
    "stackoverflow": 0.7,
    "github_readme": 0.6,
    "hackernews": 0.7,
}


def _iso_week(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


class FusionAgent:
    def __init__(self, corpus_dir: Path | str = CORPUS_DIR, skills=None) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.skills = list(skills) if skills else tracked_skills()

    def _load_corpus(self) -> list[dict]:
        docs: list[dict] = []
        for path in sorted(self.corpus_dir.glob("*.json")):
            try:
                docs.append(json.loads(path.read_text()))
            except Exception:
                continue
        return docs

    def compute_for_window(self, weeks: int = 13) -> list[dict]:
        cutoff = datetime.now(timezone.utc).date() - timedelta(weeks=weeks)
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for doc in self._load_corpus():
            try:
                d = date.fromisoformat(doc["date"])
            except Exception:
                continue
            if d < cutoff:
                continue
            source = doc.get("source", "")
            week = _iso_week(d)
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            for skill in self.skills:
                if skill in text:
                    counts[(skill, source, week)] += 1

        per_skill_max: dict[str, int] = defaultdict(int)
        for (skill, _source, _week), n in counts.items():
            per_skill_max[skill] = max(per_skill_max[skill], n)

        rows: list[dict] = []
        for (skill, source, week), n in counts.items():
            quality = _SOURCE_QUALITY.get(source, 0.5)
            denom = per_skill_max[skill] or 1
            rows.append({
                "skill_id": skill,
                "source": source,
                "week_iso": week,
                "intensity": round(min(1.0, (n / denom) * quality), 4),
                "attribution": {source: round(quality, 3)},
                "n_mentions": n,
            })
        return rows
