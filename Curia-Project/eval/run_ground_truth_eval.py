"""
External ground-truth evaluation (RQ1 + curriculum anchoring).

Two checks against real reference data:

  1. Agent A vs BLS  — Spearman rank correlation between CURIA's fused
     skill-demand ranking and BLS occupational-growth ranking. Validates that
     the demand signal tracks the real labor market (RQ1).

  2. CS2023 coverage — fraction of each curriculum unit's topics that CURIA
     tracks and that show a demand signal. Validates the curriculum anchor.

BLS data is read from data/eval/bls_oes_sample.json (a SAMPLE — swap in the
official BLS export via --bls). Spearman is computed without scipy.

Usage
    python3 eval/run_ground_truth_eval.py
    python3 eval/run_ground_truth_eval.py --bls path/to/official_bls.json --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_a_fusion import FusionAgent
from src.config import EVAL_DIR, UNITS_FILE
from src.forecasting import _canonical_skill, tracked_skills
from src.storage import load_units

_DEFAULT_BLS = EVAL_DIR / "bls_oes_sample.json"


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _spearman(a: list[float], b: list[float]) -> float:
    return round(_pearson(_ranks(a), _ranks(b)), 4)


def _agent_a_demand(skills: list[str]) -> dict[str, float]:
    rows = FusionAgent(skills=skills).compute_for_window(weeks=10_000)
    by_skill: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_skill[r["skill_id"]].append(r["intensity"])
    return {s: (sum(v) / len(v) if v else 0.0) for s, v in by_skill.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bls", default=str(_DEFAULT_BLS))
    parser.add_argument("--out", help="Directory to also write a JSON report into")
    args = parser.parse_args()

    bls_doc = json.loads(Path(args.bls).read_text())
    bls = {_canonical_skill(o["skill"]): float(o["growth_pct"]) for o in bls_doc["occupations"]}
    is_sample = "SAMPLE" in bls_doc.get("source", "")

    # ---- 1. Agent A vs BLS ----
    bls_skills = list(bls.keys())
    demand = _agent_a_demand(bls_skills)
    overlap = [s for s in bls_skills if demand.get(s, 0.0) > 0.0]
    if len(overlap) >= 2:
        rho = _spearman([demand[s] for s in overlap], [bls[s] for s in overlap])
    else:
        rho = None
    bls_block = {
        "spearman_rho": rho,
        "n_skills_overlap": len(overlap),
        "n_bls_skills": len(bls_skills),
        "bls_source_is_sample": is_sample,
        "note": ("Demand signal sparse on current corpus; scale ingest for a real RQ1 result"
                 if (rho is None or len(overlap) < 5) else "OK"),
    }

    # ---- 2. CS2023 coverage ----
    tracked = set(tracked_skills())
    units = load_units(UNITS_FILE)
    all_demand = _agent_a_demand(sorted(tracked))
    per_unit = []
    cov_sum = surf_sum = 0.0
    for u in units:
        topics = [_canonical_skill(t) for t in u.get("current_topics", [])]
        if not topics:
            continue
        covered = [t for t in topics if t in tracked]
        surfaced = [t for t in topics if all_demand.get(t, 0.0) > 0.0]
        cov = len(covered) / len(topics)
        surf = len(surfaced) / len(topics)
        cov_sum += cov
        surf_sum += surf
        per_unit.append({
            "unit_id": u["id"], "n_topics": len(topics),
            "framework_coverage": round(cov, 4),
            "demand_surfaced": round(surf, 4),
        })
    n = max(len(per_unit), 1)
    cs2023_block = {
        "mean_framework_coverage": round(cov_sum / n, 4),
        "mean_demand_surfaced": round(surf_sum / n, 4),
        "per_unit": per_unit,
    }

    report = {
        "agent_a_vs_bls": bls_block,
        "cs2023_coverage": cs2023_block,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        path = out_dir / f"ground_truth_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
