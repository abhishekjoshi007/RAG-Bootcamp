"""
Faithfulness evaluation — C8 Citation Grounding.

Measures two metrics on the faithfulness_labels.jsonl ground-truth set:

  citation_precision   fraction of cited IDs that exist in the retrieved set
  claim_grounding      fraction of atomic claims rated as supported AND
                       correctly cited by human annotators

Usage
    python3 eval/run_faithfulness_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS_PATH = ROOT / "data" / "eval" / "faithfulness_labels.jsonl"


def citation_precision(cited_ids: list[str], evidence_ids: list[str]) -> float:
    """Fraction of LLM-cited IDs that appear in the retrieved evidence set."""
    if not cited_ids:
        return 1.0
    retrieved = set(evidence_ids)
    return sum(1 for c in cited_ids if c in retrieved) / len(cited_ids)


def claim_grounding(claims: list[dict]) -> float:
    """Fraction of atomic claims that are both supported and correctly cited."""
    if not claims:
        return 0.0
    passing = sum(
        1 for c in claims if c.get("supported") and c.get("cited_correctly")
    )
    return passing / len(claims)


def extract_inline_ids(summary: str) -> list[str]:
    """Pull document IDs cited inline in summary text e.g. (jp_llm_engineer)."""
    import re
    return re.findall(r"\b[a-z]{2,}_[a-z0-9_]+\b", summary)


def main() -> None:
    rows = [
        json.loads(line)
        for line in LABELS_PATH.read_text().splitlines()
        if line.strip()
    ]

    totals = {"citation_precision": 0.0, "claim_grounding": 0.0}
    results: list[dict] = []

    for row in rows:
        inline = extract_inline_ids(row["summary"])
        explicit = row.get("evidence_ids", [])
        all_cited = list(dict.fromkeys(inline + explicit))

        cp = citation_precision(all_cited, row["evidence_ids"])
        cg = claim_grounding(row["claims"])
        totals["citation_precision"] += cp
        totals["claim_grounding"] += cg

        entry = {
            "recommendation_id": row["recommendation_id"],
            "unit_id": row["unit_id"],
            "citation_precision": round(cp, 4),
            "claim_grounding": round(cg, 4),
            "num_claims": len(row["claims"]),
            "num_cited": len(all_cited),
        }
        results.append(entry)
        print(json.dumps(entry))

    n = max(len(rows), 1)
    aggregate = {
        "aggregate": {
            k: round(v / n, 4) for k, v in totals.items()
        },
        "n": n,
        "target_citation_precision": 0.95,
        "target_claim_grounding": 0.85,
    }
    passed = (
        aggregate["aggregate"]["citation_precision"] >= aggregate["target_citation_precision"]
        and aggregate["aggregate"]["claim_grounding"] >= aggregate["target_claim_grounding"]
    )
    aggregate["passed_target"] = passed
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
