#!/usr/bin/env python3
"""Ingest historical arXiv CS papers into a CURIA corpus directory.

This is intentionally separate from scripts/ingest_corpus.py so a large,
dated research corpus can be built in data/corpus_large without changing the
small smoke-test corpus.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARXIV_CATEGORIES


ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
API_URL = "https://export.arxiv.org/api/query"


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path, ATOM_NS)
    return (found.text or "").strip() if found is not None else ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _paper_id(entry: ET.Element) -> str:
    url = _text(entry, "atom:id")
    return url.rstrip("/").rsplit("/", 1)[-1]


def _published_date(entry: ET.Element) -> str:
    published = _text(entry, "atom:published") or _text(entry, "atom:updated")
    return datetime.fromisoformat(published.replace("Z", "+00:00")).date().isoformat()


def _doc_from_entry(entry: ET.Element) -> dict:
    paper_id = _paper_id(entry)
    doc_id = f"axhist_{_safe_id(paper_id)}"
    title = _clean(_text(entry, "atom:title"))
    summary = _clean(_text(entry, "atom:summary"))
    authors = [
        _clean(_text(author, "atom:name"))
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [
        category.attrib.get("term", "")
        for category in entry.findall("atom:category", ATOM_NS)
        if category.attrib.get("term")
    ]
    text = "\n".join(
        part for part in [
            title,
            summary,
            f"Categories: {', '.join(categories)}" if categories else "",
            f"Authors: {', '.join(authors[:12])}" if authors else "",
        ]
        if part
    )
    return {
        "id": doc_id,
        "title": title or paper_id,
        "source": "arxiv",
        "date": _published_date(entry),
        "text": text[:4000],
        "metadata": {
            "arxiv_id": paper_id,
            "url": _text(entry, "atom:id"),
            "updated": _text(entry, "atom:updated"),
            "categories": categories,
            "authors": authors,
            "historical_ingest": True,
        },
    }


def _fetch_entries(query: str, start: int, max_results: int, timeout: int) -> list[ET.Element]:
    params = urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    request = Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "curia-research-ingest/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        root = ET.fromstring(response.read())
    return root.findall("atom:entry", ATOM_NS)


def _year_end(year: int, final_date: date) -> date:
    return final_date if year == final_date.year else date(year, 12, 31)


def _query(category: str, year: int, final_date: date) -> str:
    start = f"{year}01010000"
    end_date = _year_end(year, final_date)
    end = f"{end_date:%Y%m%d}2359"
    return f"cat:{category} AND submittedDate:[{start} TO {end}]"


def _write_doc(doc: dict, out_dir: Path, force: bool) -> bool:
    out_path = out_dir / f"{doc['id']}.json"
    if out_path.exists() and not force:
        return False
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=True))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "corpus_large"))
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-date", default=date.today().isoformat(),
                        help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--categories", nargs="+", default=ARXIV_CATEGORIES)
    parser.add_argument("--max-per-category-year", type=int, default=25)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Delay between arXiv API calls; arXiv asks for polite pacing")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    final_date = date.fromisoformat(args.end_date)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_seen = 0
    for year in range(args.start_year, final_date.year + 1):
        for category in args.categories:
            query = _query(category, year, final_date)
            fetched_for_window = 0
            start = 0
            while fetched_for_window < args.max_per_category_year:
                remaining = args.max_per_category_year - fetched_for_window
                page_size = min(args.page_size, remaining)
                try:
                    entries = _fetch_entries(query, start, page_size, args.timeout)
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"ERROR arxiv {category} {year}: {exc}", file=sys.stderr)
                    break
                if not entries:
                    break
                written = 0
                for entry in entries:
                    doc = _doc_from_entry(entry)
                    total_seen += 1
                    if _write_doc(doc, out_dir, args.force):
                        written += 1
                        total_written += 1
                fetched_for_window += len(entries)
                print(
                    json.dumps(
                        {
                            "source": "arxiv",
                            "category": category,
                            "year": year,
                            "fetched_window": fetched_for_window,
                            "written_page": written,
                            "written_total": total_written,
                        }
                    )
                )
                start += len(entries)
                if len(entries) < page_size:
                    break
                time.sleep(args.sleep)

    print(json.dumps({"done": True, "seen": total_seen, "written": total_written, "out": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
