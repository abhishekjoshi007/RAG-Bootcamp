#!/usr/bin/env python3
"""Weekly batch refresh — invoked by cron.

Usage:
    python scripts/batch_refresh.py                 # A + B + resources
    python scripts/batch_refresh.py --include-drift # quarterly C run + cascade
    python scripts/batch_refresh.py --skip-ingest   # for testing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import AuditLog
from src.batch import BatchRunner
from src.cache import CacheLayer
from src.config import AUDIT_DB_PATH, BATCH_LOG_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-drift", action="store_true",
                        help="Run Agent C drift detection (quarterly)")
    parser.add_argument("--skip-forecast", action="store_true",
                        help="Skip Agent B forecasting")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip new-document ingest + reindex")
    args = parser.parse_args()

    log_path = Path(BATCH_LOG_PATH)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parents[1] / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    cache = CacheLayer(AUDIT_DB_PATH)
    audit = AuditLog(AUDIT_DB_PATH)
    runner = BatchRunner(
        cache=cache, audit=audit,
        run_drift=args.include_drift,
        run_forecast=not args.skip_forecast,
        skip_ingest=args.skip_ingest,
    )
    result = runner.run_full_refresh()
    print(json.dumps({
        "status": result.status,
        "n_docs_added": result.n_docs_added,
        "n_agent_a": result.n_agent_a,
        "n_agent_b": result.n_agent_b,
        "n_agent_c": result.n_agent_c,
        "n_resources": result.n_resources,
        "errors": result.errors,
    }, indent=2))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
