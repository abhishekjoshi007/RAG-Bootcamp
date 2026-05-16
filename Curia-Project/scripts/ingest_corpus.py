"""
CLI for CURIA corpus ingestion.

Fetches job postings and technical documents from all enabled sources
and writes normalised JSON files to data/corpus/.

Usage
    python3 scripts/ingest_corpus.py            # all sources
    python3 scripts/ingest_corpus.py --sources remoteok arxiv github
    python3 scripts/ingest_corpus.py --no-rebuild-index

After ingestion, rebuild the FAISS index so the new docs are searchable:
    python3 scripts/build_index.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.ingest import (
    fetch_arbeitnow,
    fetch_arxiv,
    fetch_github_repos,
    fetch_greenhouse,
    fetch_hn_hiring,
    fetch_lever,
    fetch_remoteok,
    fetch_remotive,
    fetch_stackoverflow,
    fetch_themuse,
    fetch_usajobs,
    fetch_weworkremotely,
    ingest_all,
)
from src.storage import build_index_from_corpus

CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_PATH = ROOT / "audit" / "faiss_index.pkl"

ALL_SOURCE_NAMES = [
    "greenhouse",
    "lever",
    "weworkremotely",
    "remoteok",
    "remotive",
    "arbeitnow",
    "themuse",
    "hn_hiring",
    "usajobs",
    "arxiv",
    "stackoverflow",
    "github",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus documents from all sources.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCE_NAMES,
        default=None,
        metavar="SOURCE",
        help=f"Limit to these sources. Choices: {', '.join(ALL_SOURCE_NAMES)}",
    )
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="Skip rebuilding the FAISS index after ingestion.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite docs that already exist in the corpus.",
    )
    args = parser.parse_args()

    if args.sources:
        _run_selected(args.sources, skip_existing=not args.force)
    else:
        print(f"Ingesting from all {len(ALL_SOURCE_NAMES)} sources …")
        written = ingest_all(CORPUS_DIR, skip_existing=not args.force)
        print(f"\nDone. {written} new document(s) written to {CORPUS_DIR}")

    if not args.no_rebuild_index:
        print("\nRebuilding FAISS index (this may take a moment) …")
        index = build_index_from_corpus(CORPUS_DIR)
        index.save(INDEX_PATH)
        print(f"Index saved → {INDEX_PATH}  ({len(index.chunks)} chunks)")


def _run_selected(sources: list[str], skip_existing: bool) -> None:
    import json

    fetcher_map = {
        "greenhouse": fetch_greenhouse,
        "lever": fetch_lever,
        "weworkremotely": fetch_weworkremotely,
        "remoteok": fetch_remoteok,
        "remotive": fetch_remotive,
        "arbeitnow": fetch_arbeitnow,
        "themuse": fetch_themuse,
        "hn_hiring": fetch_hn_hiring,
        "usajobs": fetch_usajobs,
        "arxiv": fetch_arxiv,
        "stackoverflow": fetch_stackoverflow,
        "github": fetch_github_repos,
    }

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if skip_existing:
        for p in CORPUS_DIR.glob("*.json"):
            try:
                existing_ids.add(json.loads(p.read_text())["id"])
            except Exception:
                pass

    total = 0
    for name in sources:
        fetcher = fetcher_map[name]
        print(f"Fetching from {name} …")
        try:
            docs = fetcher()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        written = 0
        for doc in docs:
            if doc.id in existing_ids:
                continue
            out = CORPUS_DIR / f"{doc.id}.json"
            payload = {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "date": doc.date.isoformat(),
                "text": doc.text,
                "metadata": doc.metadata,
            }
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            existing_ids.add(doc.id)
            written += 1
            total += 1
        print(f"  → {written} new docs (fetched {len(docs)} total)")

    print(f"\nDone. {total} new document(s) written to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
