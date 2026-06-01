"""
Multi-source corpus ingestion for CURIA Agent A.

Fetches job postings, academic papers, technical Q&A, and code-repository
data from eleven free, legally-accessible APIs and saves normalised Document
JSON files to the corpus directory.

Job posting sources (9)
  Greenhouse    boards-api.greenhouse.io         public ATS board API, no auth
                ~4 000 tech companies (Stripe, Databricks, Snowflake, Figma …)
  Lever         api.lever.co/v0/postings         public ATS board API, no auth
                top tech companies (Netflix, Twitch, Qualtrics …)
  We Work Rem.  weworkremotely.com RSS           free RSS feed, no auth
  RemoteOK      remoteok.com/api                 free JSON, no auth
  Remotive      remotive.com/api                 free JSON, no auth
  Arbeitnow     arbeitnow.com/api                free JSON, no auth
  HN Hiring     HN Firebase + Algolia            free, no auth
  The Muse      themuse.com/api                  free JSON, no auth
  USAJOBS       data.usajobs.gov/api             needs USAJOBS_API_KEY + USAJOBS_USER_AGENT

  NOTE — Indeed / Glassdoor / ZipRecruiter are intentionally excluded.
  Indeed terminated its Publisher API in 2022 and prohibits scraping via ToS.
  IEEE publication requires provenance-clean data; unofficial Indeed wrappers
  (e.g. RapidAPI resellers) would fail a reviewer's data audit.
  Greenhouse + Lever together cover ~10 000 tech companies through their
  officially-public job board APIs and are the standard approach for
  academic labor-market research.

Other sources
  arXiv       export.arxiv.org/api             free Atom, no auth
  Stack Exch  api.stackexchange.com/2.3        free JSON, gzip, no auth
  GitHub      api.github.com                   free JSON; GITHUB_TOKEN raises rate limit
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
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .config import (
    ARXIV_CATEGORIES,
    GITHUB_TOPICS,
    GREENHOUSE_COMPANIES,
    INGEST_HTTP_TIMEOUT,
    INGEST_MAX_ARXIV,
    INGEST_MAX_CHARS_ARXIV,
    INGEST_MAX_CHARS_DEFAULT,
    INGEST_MAX_CHARS_GITHUB,
    INGEST_MAX_HN_POSTINGS,
    INGEST_MAX_PER_COMPANY,
    INGEST_MAX_PER_TAG,
    INGEST_MAX_WWR,
    JOB_QUERIES_ARBEITNOW,
    JOB_QUERIES_MUSE,
    JOB_QUERIES_REMOTIVE,
    JOB_TAGS_REMOTEOK,
    LEVER_COMPANIES,
    SO_TAGS,
    USAJOBS_KEYWORDS,
)
from .models import Document

logger = logging.getLogger(__name__)

_JOB_TAGS_REMOTEOK    = JOB_TAGS_REMOTEOK
_JOB_QUERIES_REMOTIVE = JOB_QUERIES_REMOTIVE
_JOB_QUERIES_ARBEITNOW = JOB_QUERIES_ARBEITNOW
_JOB_QUERIES_MUSE     = JOB_QUERIES_MUSE
_USAJOBS_KEYWORDS     = USAJOBS_KEYWORDS
_ARXIV_CATEGORIES     = ARXIV_CATEGORIES
_SO_TAGS              = SO_TAGS
_GITHUB_TOPICS        = GITHUB_TOPICS
_GREENHOUSE_COMPANIES = GREENHOUSE_COMPANIES
_LEVER_COMPANIES      = LEVER_COMPANIES


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict | list:
    req_headers: dict[str, str] = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=INGEST_HTTP_TIMEOUT) as resp:
        raw: bytes = resp.read()
        if resp.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:64]


def _truncate(text: str, max_chars: int = INGEST_MAX_CHARS_DEFAULT) -> str:
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


def fetch_remoteok(max_per_tag: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_remotive(max_per_query: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_arbeitnow(max_per_query: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_themuse(max_per_query: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_hn_hiring(max_postings: int = INGEST_MAX_HN_POSTINGS) -> list[Document]:
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


def fetch_usajobs(max_per_keyword: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_greenhouse(max_per_company: int = INGEST_MAX_PER_COMPANY) -> list[Document]:
    """
    Greenhouse public Job Board API — no auth required.

    Companies using Greenhouse explicitly make their postings available via
    boards-api.greenhouse.io.  This is the officially documented public
    endpoint designed for aggregators and job boards.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    for company in _GREENHOUSE_COMPANIES:
        url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
        try:
            payload = _get_json(url)
            if not isinstance(payload, dict):
                continue
            jobs = payload.get("jobs", [])[:max_per_company]
            for job in jobs:
                doc_id = f"gh_ats_{company}_{job.get('id', '')}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                content_html = job.get("content", "") or ""
                text = _truncate(_strip_html(content_html))
                if not text:
                    text = job.get("title", "")
                if not text:
                    continue
                location = job.get("location", {}).get("name", "") if isinstance(job.get("location"), dict) else ""
                updated = (job.get("updated_at") or "")[:10]
                docs.append(Document(
                    id=doc_id,
                    title=f"{job.get('title', 'Role')} — {company.title()} ({location})".strip(" ()"),
                    source="job_posting",
                    date=_parse_date(updated),
                    text=text,
                    metadata={
                        "origin": "greenhouse",
                        "company": company,
                        "url": job.get("absolute_url", ""),
                    },
                ))
        except Exception as exc:
            logger.debug("Greenhouse company=%s: %s", company, exc)
        time.sleep(0.3)
    return docs


def fetch_lever(max_per_company: int = INGEST_MAX_PER_COMPANY) -> list[Document]:
    """
    Lever public Job Board API — no auth required.

    Companies using Lever make their postings available via
    api.lever.co/v0/postings/{company}.  This is Lever's officially
    documented public endpoint intended for job aggregators.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    for company in _LEVER_COMPANIES:
        url = f"https://api.lever.co/v0/postings/{company}?mode=json&limit={max_per_company}"
        try:
            items = _get_json(url)
            if not isinstance(items, list):
                continue
            for item in items[:max_per_company]:
                doc_id = f"lv_{company}_{item.get('id', '')}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                plain = item.get("descriptionPlain", "") or _strip_html(item.get("description", ""))
                lists_html = " ".join(
                    lst.get("content", "")
                    for lst in item.get("lists", [])
                    if isinstance(lst, dict)
                )
                text = _truncate(_strip_html(plain + " " + lists_html))
                if not text:
                    continue
                categories = item.get("categories", {}) or {}
                location = categories.get("location", "")
                created_ms = item.get("createdAt", 0) or 0
                import datetime as _dt
                doc_date = _dt.date.fromtimestamp(created_ms / 1000) if created_ms else date.today()
                docs.append(Document(
                    id=doc_id,
                    title=f"{item.get('text', 'Role')} — {company.title()} ({location})".strip(" ()"),
                    source="job_posting",
                    date=doc_date,
                    text=text,
                    metadata={
                        "origin": "lever",
                        "company": company,
                        "url": item.get("hostedUrl", ""),
                        "team": categories.get("team", ""),
                    },
                ))
        except Exception as exc:
            logger.debug("Lever company=%s: %s", company, exc)
        time.sleep(0.3)
    return docs


def fetch_weworkremotely(max_results: int = INGEST_MAX_WWR) -> list[Document]:
    """
    We Work Remotely — RSS feed, no auth required.
    Covers programming, devops, and design remote roles globally.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    rss_url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "CURIA/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            tree = ET.parse(resp)
        root = tree.getroot()
        channel = root.find("channel")
        if channel is None:
            return docs
        for item in list(channel.findall("item"))[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            description = _strip_html(item.findtext("description") or "")
            pub_date = (item.findtext("pubDate") or "")[:16]
            if not description or not title:
                continue
            doc_id = f"wwr_{_slug(title + link)}"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                from email.utils import parsedate
                import datetime as _dt
                parsed = parsedate(pub_date)
                doc_date = _dt.date(*parsed[:3]) if parsed else date.today()
            except Exception:
                doc_date = date.today()
            docs.append(Document(
                id=doc_id,
                title=title,
                source="job_posting",
                date=doc_date,
                text=_truncate(description),
                metadata={"origin": "weworkremotely", "url": link},
            ))
    except Exception as exc:
        logger.warning("WeWorkRemotely RSS: %s", exc)
    return docs


_ARXIV_NS = "http://www.w3.org/2005/Atom"


def fetch_arxiv(max_per_category: int = INGEST_MAX_ARXIV) -> list[Document]:
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
                    text=_truncate(summary, max_chars=INGEST_MAX_CHARS_ARXIV),
                    metadata={"origin": "arxiv", "arxiv_id": arxiv_id, "category": cat},
                ))
        except Exception as exc:
            logger.warning("arXiv category=%s: %s", cat, exc)
        time.sleep(1.0)
    return docs


def fetch_stackoverflow(max_per_tag: int = INGEST_MAX_PER_TAG) -> list[Document]:
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


def fetch_github_repos(max_per_topic: int = INGEST_MAX_PER_COMPANY) -> list[Document]:
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
                    text=_truncate(readme_text, max_chars=INGEST_MAX_CHARS_GITHUB),
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


def ingest_all(
    corpus_dir: Path,
    skip_existing: bool = True,
    since: Optional[datetime] = None,
) -> int:
    """
    Run all fetchers and write new Document JSON files to corpus_dir.

    Skips documents whose IDs already exist in corpus_dir (skip_existing=True)
    and, when `since` is given, documents dated before it (makes batch ingest
    incremental). Returns the count of newly written files.
    """
    since_date = since.date() if since is not None else None
    corpus_dir.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if skip_existing:
        for p in corpus_dir.glob("*.json"):
            try:
                existing_ids.add(json.loads(p.read_text())["id"])
            except Exception:
                pass

    fetchers = [
        ("greenhouse", fetch_greenhouse),
        ("lever", fetch_lever),
        ("weworkremotely", fetch_weworkremotely),
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
            if since_date is not None and doc.date < since_date:
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
