"""
Answer-relevance evaluation — C9 Eval Set 3.

Measures whether the recommendation actually addresses the CS2023 unit's
intent, using human-annotated 1–5 Likert ratings in relevance_ratings.jsonl.

Reports mean rating, distribution, and whether the 3.5 minimum target is met.

Usage
    python3 eval/run_relevance_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RATINGS_PATH = ROOT / "data" / "eval" / "relevance_ratings.jsonl"
LABELS_PATH = ROOT / "data" / "eval" / "faithfulness_labels.jsonl"


def main() -> None:
    ratings = [
        json.loads(line)
        for line in RATINGS_PATH.read_text().splitlines()
        if line.strip()
    ]

    labels_by_id = {}
    for line in LABELS_PATH.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            labels_by_id[row["recommendation_id"]] = row

    per_unit: dict[str, list[int]] = {}
    results: list[dict] = []

    for row in ratings:
        rec_id = row["recommendation_id"]
        unit_id = row["unit_id"]
        rating = int(row["rating"])

        per_unit.setdefault(unit_id, []).append(rating)
        results.append({
            "recommendation_id": rec_id,
            "unit_id": unit_id,
            "rating": rating,
            "notes": row.get("notes", ""),
        })
        print(json.dumps({"recommendation_id": rec_id, "unit_id": unit_id, "rating": rating}))

    all_ratings = [r["rating"] for r in results]
    mean_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0
    distribution = dict(sorted(Counter(all_ratings).items()))

    per_unit_means = {
        uid: round(sum(rs) / len(rs), 2) for uid, rs in per_unit.items()
    }

    summary = {
        "mean_rating": round(mean_rating, 3),
        "distribution": distribution,
        "per_unit_means": per_unit_means,
        "n": len(all_ratings),
        "target_mean": 3.5,
        "passed_target": mean_rating >= 3.5,
    }
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
