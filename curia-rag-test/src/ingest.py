"""
Multi-source corpus ingestion for CURIA Agent A.

Fetches job postings, academic papers, technical Q&A, and code-repository
data from eight free, legally-accessible APIs and saves normalised Document
JSON files to the corpus directory.

Job posting sources (6)
  RemoteOK    https://remoteok.com/api           free JSON, no auth
  Remotive    https://remotive.com/api           free JSON, no auth
  Arbeitnow   https://www.arbeitnow.com/api      free JSON, no auth
  HN Hiring   HN Firebase + Algolia              free, no auth
  The Muse    https://www.themuse.com/api        free JSON, no auth
  USAJOBS     https://data.usajobs.gov/api       needs USAJOBS_API_KEY + USAJOBS_USER_AGENT

Other sources
  arXiv       http://export.arxiv.org/api        free Atom, no auth
  Stack Exch  https://api.stackexchange.com/2.3  free JSON, gzip, no auth
  GitHub      https://api.github.com             free JSON; GITHUB_TOKEN raises rate limit
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from .models import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CS-relevant search terms per source
# ---------------------------------------------------------------------------
_JOB_TAGS_REMOTEOK = ["machine-learning", "devops", "backend", "cloud", "security", "python"]
_JOB_QUERIES_REMOTIVE = ["machine learning", "LLM", "devops", "cloud native", "cybersecurity"]
_JOB_QUERIES_ARBEITNOW = ["machine learning", "kubernetes", "devsecops", "cloud engineer"]
_JOB_QUERIES_MUSE = ["Software Engineer", "Machine Learning", "DevOps", "Cloud", "Security"]
_USAJOBS_KEYWORDS = ["machine learning", "software engineer", "cloud", "cybersecurity"]
_ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.SE", "cs.CR", "cs.DC"]
_SO_TAGS = ["machine-learning", "kubernetes", "large-language-model", "devops", "security"]
_GITHUB_TOPICS = ["machine-learning", "llm", "rag", "cloud-native", "devsecops"]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, headers: dict[str, str] | None = None) -> dict | list:
    req_headers: dict[str, str] = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw: bytes = resp.read()
        if resp.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:64]


def _truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _parse_date(raw: str | None) -> date:
    try:
        return date.fromisoformat((raw or "")[:10])
    except (ValueError, TypeError):
        return date.today()


# ---------------------------------------------------------------------------
# Job posting sources
# ---------------------------------------------------------------------------

def fetch_remoteok(max_per_tag: int = 5) -> list[Document]:
    """RemoteOK free JSON API — no auth required."""
    docs: list[Document] = []
    seen: set[str] = set()
    for tag in _JOB_TAGS_REMOTEOK:
        url = f"https://remoteok.com/api?tags={urllib.parse.quote(tag)}"
        try:
            items = _get_json(url)
            if not isinstance(items, list):
                continue
            count = 0
            for item in items:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                doc_id = f"rk_{item['id']}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = _truncate(_strip_html(item.get("description", "")))
                if not text:
                    continue
                title = f"{item.get('position', 'Remote Job')} — {item.get('company', '')}".strip(" —")
                docs.append(Document(
                    id=doc_id,
                    title=title,
                    source="job_posting",
                    date=_parse_date(item.get("date")),
                    text=text,
                    metadata={"origin": "remoteok", "url": item.get("url", ""), "tag": tag},
                ))
                count += 1
                if count >= max_per_tag:
                    break
        except Exception as exc:
            logger.warning("RemoteOK tag=%s: %s", tag, exc)
        time.sleep(0.5)
    return docs


def fetch_remotive(max_per_query: int = 5) -> list[Document]:
    """Remotive free JSON API — no auth required."""
    docs: list[Document] = []
    seen: set[str] = set()
    for query in _JOB_QUERIES_REMOTIVE:
        url = (
            f"https://remotive.com/api/remote-jobs?"
            f"search={urllib.parse.quote(query)}&limit={max_per_query}"
        )
        try:
            payload = _get_json(url)
            if not isinstance(payload, dict):
                continue
            for item in payload.get("jobs", []):
                doc_id = f"rt_{item['id']}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = _truncate(_strip_html(item.get("description", "")))
                if not text:
                    continue
                docs.append(Document(
                    id=doc_id,
                    title=f"{item.get('title', '')} — {item.get('company_name', '')}".strip(" —"),
                    source="job_posting",
                    date=_parse_date(item.get("publication_date")),
                    text=text,
                    metadata={"origin": "remotive", "url": item.get("url", ""), "query": query},
                ))
        except Exception as exc:
            logger.warning("Remotive query=%r: %s", query, exc)
        time.sleep(0.4)
    return docs


def fetch_arbeitnow(max_per_query: int = 5) -> list[Document]:
    """Arbeitnow free job board API — no auth required."""
    docs: list[Document] = []
    seen: set[str] = set()
    for query in _JOB_QUERIES_ARBEITNOW:
        url = (
            f"https://www.arbeitnow.com/api/job-board-api?"
            f"search={urllib.parse.quote(query)}"
        )
        try:
            payload = _get_json(url)
            if not isinstance(payload, dict):
                continue
            for item in list(payload.get("data", []))[:max_per_query]:
                doc_id = f"an_{_slug(item.get('slug', '') or str(hash(item.get('title', ''))))}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = _truncate(_strip_html(item.get("description", "")))
                if not text:
                    continue
                docs.append(Document(
                    id=doc_id,
                    title=f"{item.get('title', '')} — {item.get('company_name', '')}".strip(" —"),
                    source="job_posting",
                    date=_parse_date(item.get("created_at")),
                    text=text,
                    metadata={"origin": "arbeitnow", "url": item.get("url", ""), "query": query},
                ))
        except Exception as exc:
            logger.warning("Arbeitnow query=%r: %s", query, exc)
        time.sleep(0.4)
    return docs


def fetch_themuse(max_per_query: int = 5) -> list[Document]:
    """The Muse jobs API — no auth required for public access."""
    docs: list[Document] = []
    seen: set[str] = set()
    for query in _JOB_QUERIES_MUSE:
        url = (
            f"https://www.themuse.com/api/public/jobs?"
            f"category={urllib.parse.quote(query)}&page=1&descending=true"
        )
        try:
            payload = _get_json(url)
            if not isinstance(payload, dict):
                continue
            for item in list(payload.get("results", []))[:max_per_query]:
                doc_id = f"tm_{item.get('id', '')}"
                if doc_id in seen or not doc_id.strip("tm_"):
                    continue
                seen.add(doc_id)
                contents = item.get("contents", "") or ""
                text = _truncate(_strip_html(contents))
                if not text:
                    continue
                company = item.get("company", {}).get("name", "") if isinstance(item.get("company"), dict) else ""
                docs.append(Document(
                    id=doc_id,
                    title=f"{item.get('name', 'Job')} — {company}".strip(" —"),
                    source="job_posting",
                    date=_parse_date(item.get("publication_date")),
                    text=text,
                    metadata={"origin": "themuse", "url": item.get("refs", {}).get("landing_page", ""), "query": query},
                ))
        except Exception as exc:
            logger.warning("TheMuse query=%r: %s", query, exc)
        time.sleep(0.4)
    return docs


def fetch_hn_hiring(max_postings: int = 25) -> list[Document]:
    """
    Hacker News 'Ask HN: Who is hiring?' monthly thread.
    Uses Algolia to locate the latest post, HN Firebase API for top-level comments.
    """
    docs: list[Document] = []
    try:
        search_url = (
            "https://hn.algolia.com/api/v1/search?"
            "query=Ask+HN+Who+is+hiring&tags=story,ask_hn&hitsPerPage=1"
        )
        search_data = _get_json(search_url)
        hits = search_data.get("hits", [])
        if not hits:
            logger.warning("HN hiring: no 'Who is hiring' post found via Algolia")
            return docs

        post_id = hits[0]["objectID"]
        post_item = _get_json(f"https://hacker-news.firebaseio.com/v0/item/{post_id}.json")
        post_date = _parse_date(None)
        if isinstance(post_item.get("time"), (int, float)):
            import datetime as _dt
            post_date = _dt.date.fromtimestamp(post_item["time"])

        kids: list[int] = post_item.get("kids", [])[:max_postings]
        for kid_id in kids:
            try:
                comment = _get_json(f"https://hacker-news.firebaseio.com/v0/item/{kid_id}.json")
                if not isinstance(comment, dict):
                    continue
                if comment.get("dead") or comment.get("deleted"):
                    continue
                text = _strip_html(comment.get("text", ""))
                if not text or len(text) < 100:
                    continue
                docs.append(Document(
                    id=f"hn_{kid_id}",
                    title=f"HN Hiring: {text[:80].strip()}",
                    source="job_posting",
                    date=post_date,
                    text=_truncate(text),
                    metadata={"origin": "hn_hiring", "parent_post": post_id},
                ))
                time.sleep(0.1)
            except Exception as exc:
                logger.debug("HN item %s: %s", kid_id, exc)
    except Exception as exc:
        logger.warning("HN hiring fetch failed: %s", exc)
    return docs


def fetch_usajobs(max_per_keyword: int = 5) -> list[Document]:
    """
    USAJOBS API — requires env vars USAJOBS_API_KEY and USAJOBS_USER_AGENT.
    Register (free) at https://developer.usajobs.gov/
    """
    api_key = os.environ.get("USAJOBS_API_KEY", "")
    user_agent = os.environ.get("USAJOBS_USER_AGENT", "")
    if not api_key or not user_agent:
        logger.info("USAJOBS_API_KEY / USAJOBS_USER_AGENT not set — skipping USAJOBS")
        return []

    docs: list[Document] = []
    seen: set[str] = set()
    headers = {"Authorization-Key": api_key, "User-Agent": user_agent}

    for keyword in _USAJOBS_KEYWORDS:
        url = (
            "https://data.usajobs.gov/api/search?"
            f"Keyword={urllib.parse.quote(keyword)}&ResultsPerPage={max_per_keyword}"
        )
        try:
            payload = _get_json(url, headers=headers)
            items = payload.get("SearchResult", {}).get("SearchResultItems", [])
            for item in items:
                matched = item.get("MatchedObjectDescriptor", {})
                doc_id = f"uj_{_slug(matched.get('PositionID', ''))}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = _truncate(
                    matched.get("UserArea", {}).get("Details", {}).get("JobSummary", "")
                    or matched.get("QualificationSummary", "")
                )
                if not text:
                    continue
                docs.append(Document(
                    id=doc_id,
                    title=matched.get("PositionTitle", "Federal Position"),
                    source="job_posting",
                    date=_parse_date(matched.get("PublicationStartDate")),
                    text=text,
                    metadata={
                        "origin": "usajobs",
                        "url": matched.get("PositionURI", ""),
                        "keyword": keyword,
                    },
                ))
        except Exception as exc:
            logger.warning("USAJOBS keyword=%r: %s", keyword, exc)
        time.sleep(0.5)
    return docs


# ---------------------------------------------------------------------------
# Academic / technical sources
# ---------------------------------------------------------------------------

_ARXIV_NS = "http://www.w3.org/2005/Atom"


def fetch_arxiv(max_per_category: int = 8) -> list[Document]:
    """arXiv cs.* papers via Atom search API — no auth required."""
    docs: list[Document] = []
    seen: set[str] = set()
    for cat in _ARXIV_CATEGORIES:
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=cat:{cat}&max_results={max_per_category}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CURIA/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                tree = ET.parse(resp)
            root = tree.getroot()
            for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
                raw_id = (entry.findtext(f"{{{_ARXIV_NS}}}id") or "").strip()
                arxiv_id = raw_id.rsplit("/", 1)[-1].replace(".", "_").replace("/", "_")
                doc_id = f"ax_{arxiv_id}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                title = (entry.findtext(f"{{{_ARXIV_NS}}}title") or "").strip()
                summary = (entry.findtext(f"{{{_ARXIV_NS}}}summary") or "").strip()
                published = (entry.findtext(f"{{{_ARXIV_NS}}}published") or "")[:10]
                if not summary:
                    continue
                docs.append(Document(
                    id=doc_id,
                    title=title,
                    source="arxiv",
                    date=_parse_date(published),
                    text=_truncate(summary, max_chars=1200),
                    metadata={"origin": "arxiv", "arxiv_id": arxiv_id, "category": cat},
                ))
        except Exception as exc:
            logger.warning("arXiv category=%s: %s", cat, exc)
        time.sleep(1.0)
    return docs


def fetch_stackoverflow(max_per_tag: int = 5) -> list[Document]:
    """Stack Exchange API v2.3 — no auth; responses are gzip-compressed."""
    docs: list[Document] = []
    seen: set[str] = set()
    import datetime as _dt

    for tag in _SO_TAGS:
        url = (
            f"https://api.stackexchange.com/2.3/questions?"
            f"tagged={urllib.parse.quote(tag)}&site=stackoverflow"
            f"&sort=votes&pagesize={max_per_tag}&filter=withbody"
        )
        try:
            payload = _get_json(url)
            if not isinstance(payload, dict):
                continue
            for item in payload.get("items", []):
                doc_id = f"so_{item['question_id']}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                body = _truncate(_strip_html(item.get("body", "")))
                if not body:
                    continue
                epoch = item.get("creation_date", 0)
                doc_date = _dt.date.fromtimestamp(epoch) if epoch else date.today()
                docs.append(Document(
                    id=doc_id,
                    title=item.get("title", "Stack Overflow Question"),
                    source="stackoverflow",
                    date=doc_date,
                    text=body,
                    metadata={
                        "origin": "stackoverflow",
                        "url": item.get("link", ""),
                        "score": item.get("score", 0),
                        "tag": tag,
                    },
                ))
        except Exception as exc:
            logger.warning("Stack Exchange tag=%s: %s", tag, exc)
        time.sleep(0.5)
    return docs


def fetch_github_repos(max_per_topic: int = 4) -> list[Document]:
    """
    GitHub repo READMEs via search API.
    Set GITHUB_TOKEN in env to raise rate limit from 60 to 5 000 req/hr.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {"User-Agent": "CURIA/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for topic in _GITHUB_TOPICS:
        search_url = (
            f"https://api.github.com/search/repositories?"
            f"q=topic:{urllib.parse.quote(topic)}+language:python"
            f"&sort=stars&order=desc&per_page={max_per_topic}"
        )
        try:
            result = _get_json(search_url, headers=headers)
            if not isinstance(result, dict):
                continue
            for repo in result.get("items", []):
                owner = repo.get("owner", {}).get("login", "")
                name = repo.get("name", "")
                if not owner or not name:
                    continue
                doc_id = f"gh_{_slug(owner)}_{_slug(name)}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                readme_url = f"https://api.github.com/repos/{owner}/{name}/readme"
                try:
                    readme_data = _get_json(readme_url, headers=headers)
                    if readme_data.get("encoding") != "base64":
                        continue
                    readme_text = base64.b64decode(readme_data["content"]).decode(
                        "utf-8", errors="replace"
                    )
                    readme_text = _strip_html(readme_text)
                except Exception:
                    readme_text = repo.get("description", "") or ""
                if not readme_text:
                    continue
                pushed_at = repo.get("pushed_at", "")[:10]
                desc = repo.get("description", "") or ""
                docs.append(Document(
                    id=doc_id,
                    title=f"{owner}/{name}: {desc[:80]}".rstrip(": "),
                    source="github_readme",
                    date=_parse_date(pushed_at),
                    text=_truncate(readme_text, max_chars=1500),
                    metadata={
                        "origin": "github",
                        "url": repo.get("html_url", ""),
                        "stars": repo.get("stargazers_count", 0),
                        "topic": topic,
                    },
                ))
                time.sleep(0.3)
        except Exception as exc:
            logger.warning("GitHub topic=%s: %s", topic, exc)
        time.sleep(0.5)
    return docs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_all(corpus_dir: Path, skip_existing: bool = True) -> int:
    """
    Run all fetchers and write new Document JSON files to corpus_dir.

    Skips documents whose IDs already exist in corpus_dir (skip_existing=True).
    Returns the count of newly written files.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if skip_existing:
        for p in corpus_dir.glob("*.json"):
            try:
                existing_ids.add(json.loads(p.read_text())["id"])
            except Exception:
                pass

    fetchers = [
        ("remoteok", fetch_remoteok),
        ("remotive", fetch_remotive),
        ("arbeitnow", fetch_arbeitnow),
        ("themuse", fetch_themuse),
        ("hn_hiring", fetch_hn_hiring),
        ("usajobs", fetch_usajobs),
        ("arxiv", fetch_arxiv),
        ("stackoverflow", fetch_stackoverflow),
        ("github", fetch_github_repos),
    ]

    written = 0
    for name, fetcher in fetchers:
        logger.info("Fetching from %s …", name)
        try:
            docs = fetcher()
        except Exception as exc:
            logger.warning("Fetcher %s raised unexpectedly: %s", name, exc)
            docs = []

        new_for_source = 0
        for doc in docs:
            if doc.id in existing_ids:
                continue
            out_path = corpus_dir / f"{doc.id}.json"
            payload = {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "date": doc.date.isoformat(),
                "text": doc.text,
                "metadata": doc.metadata,
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            existing_ids.add(doc.id)
            new_for_source += 1
            written += 1

        logger.info("  %s → %d new docs (fetched %d)", name, new_for_source, len(docs))

    return written
