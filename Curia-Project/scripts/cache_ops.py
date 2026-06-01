#!/usr/bin/env python3
"""Operator CLI for cache maintenance.

    python scripts/cache_ops.py stats
    python scripts/cache_ops.py invalidate --skill rag --layers a,b
    python scripts/cache_ops.py invalidate --query-hash <hash> --reason user_feedback
    python scripts/cache_ops.py purge-stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import CacheLayer
from src.config import AUDIT_DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Print cache statistics")

    p_inv = sub.add_parser("invalidate", help="Invalidate cache entries")
    p_inv.add_argument("--skill", help="Skill ID")
    p_inv.add_argument("--query-hash", dest="query_hash", help="Recommendation query hash")
    p_inv.add_argument("--layers", default="a,b,c,resources",
                       help="Comma-separated: a,b,c,resources")
    p_inv.add_argument("--reason", default="manual")

    sub.add_parser("purge-stale", help="Delete all expired rows")

    args = parser.parse_args()
    cache = CacheLayer(AUDIT_DB_PATH)

    if args.cmd == "stats":
        print(json.dumps(cache.stats(), indent=2))
    elif args.cmd == "invalidate":
        if args.query_hash:
            n = cache.invalidate_recommendation(args.query_hash, args.reason)
            print(json.dumps({"recommendation": n}, indent=2))
        elif args.skill:
            layers = tuple(layer for layer in args.layers.split(",") if layer)
            print(json.dumps(cache.invalidate_skill(args.skill, layers, args.reason), indent=2))
        else:
            parser.error("invalidate requires --skill or --query-hash")
    elif args.cmd == "purge-stale":
        print(json.dumps(cache.purge_stale(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
