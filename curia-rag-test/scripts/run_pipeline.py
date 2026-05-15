from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.config import AUDIT_DB_PATH, INDEX_PATH, SOURCE_QUOTAS, UNITS_FILE
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus, get_unit, load_units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", default="cs_ai_01")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--local", action="store_true", help="Force LocalGroundedGenerator (no LLM)")
    args = parser.parse_args()

    if INDEX_PATH.exists():
        index = FaissIndex.load(INDEX_PATH)
    else:
        print("Index not found — building now (this downloads all-mpnet-base-v2 once) …")
        index = build_index_from_corpus(ROOT / "data" / "corpus")
        index.save(INDEX_PATH)

    units = load_units(UNITS_FILE)
    unit = get_unit(units, args.unit_id)

    generator = None
    if args.local:
        from src.llm import LocalGroundedGenerator
        generator = LocalGroundedGenerator()

    pipeline = CuriaRagPipeline(
        index,
        audit_path=AUDIT_DB_PATH,
        source_quotas=SOURCE_QUOTAS,
        generator=generator,
    )
    result = pipeline.run(unit, k=args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
