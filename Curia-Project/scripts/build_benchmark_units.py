from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_units import collect_curriculum_units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default=str(ROOT / "data" / "universities"))
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "benchmark_units_tamu_50.json"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    paths = sorted(src_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No curriculum JSON files found in {src_dir}")

    limit = None if args.limit < 0 else args.limit
    units, duplicates = collect_curriculum_units(paths, limit=limit)
    if not units:
        raise SystemExit(f"No units found in {src_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(units, indent=2) + "\n")

    print(json.dumps({
        "source_dir": str(src_dir),
        "out": str(out_path),
        "units": len(units),
        "duplicate_ids_skipped": len(duplicates),
        "duplicate_ids_sample": duplicates[:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
