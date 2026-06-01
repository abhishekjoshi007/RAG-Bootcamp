#!/usr/bin/env python3
"""Ingest historical Stack Overflow questions via the Stack Exchange API."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.config import SO_TAGS


API_URL = "https://api.stackexchange.com/2.3/questions"


def _timestamp(day: date, end_of_day: bool = False) -> int:
    hour = 23 if end_of_day else 0
    minute = 59 if end_of_day else 0
    second = 59 if end_of_day else 0
    return int(datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=timezone.utc).timestamp())


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _fetch_page(
    tag: str,
    from_day: date,
    to_day: date,
    page: int,
    page_size: int,
    timeout: int,
) -> dict:
    params = {
        "site": "stackoverflow",
        "tagged": tag,
        "fromdate": _timestamp(from_day),
        "todate": _timestamp(to_day, end_of_day=True),
        "page": page,
        "pagesize": page_size,
        "order": "desc",
        "sort": "creation",
        "filter": "withbody",
    }
    key = os.environ.get("STACKEXCHANGE_KEY")
    if key:
        params["key"] = key
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": "curia-research-ingest/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _doc_from_item(item: dict, tag: str) -> dict:
    question_id = str(item["question_id"])
    title = _strip_html(item.get("title", ""))
    body = _strip_html(item.get("body", ""))
    tags = item.get("tags", [])
    created = datetime.fromtimestamp(item["creation_date"], tz=timezone.utc).date().isoformat()
    text = "\n".join(
        part for part in [
            title,
            body,
            f"Tags: {', '.join(tags)}" if tags else "",
        ]
        if part
    )
    return {
        "id": f"sohist_{_safe_id(question_id)}",
        "title": title or f"Stack Overflow question {question_id}",
        "source": "stackoverflow",
        "date": created,
        "text": text[:4000],
        "metadata": {
            "question_id": int(question_id),
            "url": item.get("link"),
            "tags": tags,
            "score": item.get("score"),
            "answer_count": item.get("answer_count"),
            "historical_tag_query": tag,
            "historical_ingest": True,
        },
    }


def _write_doc(doc: dict, out_dir: Path, force: bool) -> bool:
    out_path = out_dir / f"{doc['id']}.json"
    if out_path.exists() and not force:
        return False
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=True))
    return True


def _year_end(year: int, final_date: date) -> date:
    return final_date if year == final_date.year else date(year, 12, 31)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "corpus_large"))
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-date", default=date.today().isoformat(),
                        help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--tags", nargs="+", default=SO_TAGS)
    parser.add_argument("--pages-per-tag-year", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    final_date = date.fromisoformat(args.end_date)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_seen = 0
    for year in range(args.start_year, final_date.year + 1):
        from_day = date(year, 1, 1)
        to_day = _year_end(year, final_date)
        for tag in args.tags:
            for page in range(1, args.pages_per_tag_year + 1):
                try:
                    payload = _fetch_page(tag, from_day, to_day, page, args.page_size, args.timeout)
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"ERROR stackoverflow {tag} {year} page {page}: {exc}", file=sys.stderr)
                    break
                backoff = payload.get("backoff")
                items = payload.get("items", [])
                written = 0
                for item in items:
                    doc = _doc_from_item(item, tag)
                    total_seen += 1
                    if _write_doc(doc, out_dir, args.force):
                        written += 1
                        total_written += 1
                print(
                    json.dumps(
                        {
                            "source": "stackoverflow",
                            "tag": tag,
                            "year": year,
                            "page": page,
                            "items": len(items),
                            "written_page": written,
                            "written_total": total_written,
                            "quota_remaining": payload.get("quota_remaining"),
                        }
                    )
                )
                if backoff:
                    time.sleep(float(backoff))
                else:
                    time.sleep(args.sleep)
                if not payload.get("has_more"):
                    break

    print(json.dumps({"done": True, "seen": total_seen, "written": total_written, "out": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
