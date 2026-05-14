from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing import InMemoryIndex
from src.pipeline import CuriaRagPipeline
from src.storage import get_unit, load_units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", default="cs_ai_01")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    index_path = ROOT / "audit" / "local_index.pkl"
    if not index_path.exists():
        raise SystemExit("Index not found. Run: python3 scripts/build_index.py")

    units = load_units(ROOT / "data" / "cs2023_units.json")
    unit = get_unit(units, args.unit_id)
    index = InMemoryIndex.load(index_path)
    pipeline = CuriaRagPipeline(
        index,
        audit_path=ROOT / "audit" / "audit_log.db",
        source_quotas={
            "job_posting": 2,
            "arxiv": 2,
            "stackoverflow": 1,
            "github_readme": 1,
        },
    )
    result = pipeline.run(unit, k=args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
