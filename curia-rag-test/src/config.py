"""
Central configuration for CURIA RAG.

Every tunable parameter lives here. Override any value via environment
variable (e.g. CURIA_CHUNK_MAX_TOKENS=400) without touching source code.

Naming convention: CURIA_<SECTION>_<PARAM>
"""

from __future__ import annotations

import os
from pathlib import Path


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "")
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else default


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

CORPUS_DIR       = ROOT / "data" / "corpus"
UNITS_FILE       = ROOT / "data" / "cs2023_units.json"
EVAL_DIR         = ROOT / "data" / "eval"
AUDIT_DIR        = ROOT / "audit"
INDEX_PATH       = AUDIT_DIR / "faiss_index.pkl"
AUDIT_DB_PATH    = AUDIT_DIR / "audit_log.db"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS  = _int("CURIA_CHUNK_MAX_TOKENS", 160)
CHUNK_OVERLAP     = _int("CURIA_CHUNK_OVERLAP", 30)

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL       = _str("CURIA_EMBED_MODEL", "all-mpnet-base-v2")
EMBED_BATCH_SIZE  = _int("CURIA_EMBED_BATCH_SIZE", 8)

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K             = _int("CURIA_RETRIEVAL_K", 8)
RETRIEVAL_CANDIDATE_K   = _int("CURIA_RETRIEVAL_CANDIDATE_K", 50)
RECENCY_HALF_LIFE_DAYS  = _int("CURIA_RECENCY_HALF_LIFE_DAYS", 365)
RECENCY_BASE_WEIGHT     = _float("CURIA_RECENCY_BASE_WEIGHT", 0.7)
RECENCY_BONUS_WEIGHT    = _float("CURIA_RECENCY_BONUS_WEIGHT", 0.3)

# Source quotas: max documents per source type returned by retriever
SOURCE_QUOTAS: dict[str, int] = {
    "job_posting":   _int("CURIA_QUOTA_JOB_POSTING", 3),
    "arxiv":         _int("CURIA_QUOTA_ARXIV", 2),
    "stackoverflow": _int("CURIA_QUOTA_STACKOVERFLOW", 2),
    "github_readme": _int("CURIA_QUOTA_GITHUB_README", 1),
}

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL         = _str("CURIA_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE   = _float("CURIA_LLM_TEMPERATURE", 0.0)
LLM_MAX_TOKENS    = _int("CURIA_LLM_MAX_TOKENS", 1024)
LLM_MAX_RETRIES   = _int("CURIA_LLM_MAX_RETRIES", 3)

# Local generator signal thresholds (used when no API key is set)
LOCAL_SIGNAL_HIGH   = _float("CURIA_LOCAL_SIGNAL_HIGH", 0.28)
LOCAL_SIGNAL_MEDIUM = _float("CURIA_LOCAL_SIGNAL_MEDIUM", 0.14)

# ── Ingestion ─────────────────────────────────────────────────────────────────
INGEST_MAX_CHARS_DEFAULT = _int("CURIA_INGEST_MAX_CHARS", 2000)
INGEST_MAX_CHARS_ARXIV   = _int("CURIA_INGEST_MAX_CHARS_ARXIV", 1200)
INGEST_MAX_CHARS_GITHUB  = _int("CURIA_INGEST_MAX_CHARS_GITHUB", 1500)
INGEST_HTTP_TIMEOUT      = _int("CURIA_INGEST_HTTP_TIMEOUT", 20)

INGEST_MAX_PER_TAG       = _int("CURIA_INGEST_MAX_PER_TAG", 5)
INGEST_MAX_PER_COMPANY   = _int("CURIA_INGEST_MAX_PER_COMPANY", 3)
INGEST_MAX_HN_POSTINGS   = _int("CURIA_INGEST_MAX_HN_POSTINGS", 25)
INGEST_MAX_WWR           = _int("CURIA_INGEST_MAX_WWR", 20)
INGEST_MAX_ARXIV         = _int("CURIA_INGEST_MAX_ARXIV", 8)

# ── Search keywords / topics (override via env as comma-separated strings) ───
JOB_TAGS_REMOTEOK = _list("CURIA_TAGS_REMOTEOK",
    ["machine-learning", "devops", "backend", "cloud", "security", "python"])

JOB_QUERIES_REMOTIVE = _list("CURIA_QUERIES_REMOTIVE",
    ["machine learning", "LLM", "devops", "cloud native", "cybersecurity"])

JOB_QUERIES_ARBEITNOW = _list("CURIA_QUERIES_ARBEITNOW",
    ["machine learning", "kubernetes", "devsecops", "cloud engineer"])

JOB_QUERIES_MUSE = _list("CURIA_QUERIES_MUSE",
    ["Software Engineer", "Machine Learning", "DevOps", "Cloud", "Security"])

USAJOBS_KEYWORDS = _list("CURIA_USAJOBS_KEYWORDS",
    ["machine learning", "software engineer", "cloud", "cybersecurity"])

ARXIV_CATEGORIES = _list("CURIA_ARXIV_CATEGORIES",
    ["cs.AI", "cs.LG", "cs.SE", "cs.CR", "cs.DC"])

SO_TAGS = _list("CURIA_SO_TAGS",
    ["machine-learning", "kubernetes", "large-language-model", "devops", "security"])

GITHUB_TOPICS = _list("CURIA_GITHUB_TOPICS",
    ["machine-learning", "llm", "rag", "cloud-native", "devsecops"])

GREENHOUSE_COMPANIES = _list("CURIA_GREENHOUSE_COMPANIES", [
    "stripe", "airbnb", "lyft", "snowflake", "databricks", "figma", "notion",
    "vercel", "plaid", "brex", "retool", "cohere", "huggingface", "scaleai",
    "anthropic", "openai", "mistral", "modal", "weaviate", "pinecone",
    "confluent", "dbt-labs", "airbyte", "dagster-labs", "prefecthq",
])

LEVER_COMPANIES = _list("CURIA_LEVER_COMPANIES", [
    "netflix", "twitch", "qualtrics", "gitlab", "hashicorp", "elastic",
    "cloudflare", "fastly", "datadog", "newrelic", "pagerduty",
    "grammarly", "canva", "miro", "linear", "loom",
])

# ── Evaluation targets ────────────────────────────────────────────────────────
EVAL_TARGET_RECALL_8          = _float("CURIA_EVAL_RECALL_8", 0.70)
EVAL_TARGET_CITATION_PRECISION = _float("CURIA_EVAL_CITATION_PRECISION", 0.95)
EVAL_TARGET_CLAIM_GROUNDING    = _float("CURIA_EVAL_CLAIM_GROUNDING", 0.85)
EVAL_TARGET_RELEVANCE_MEAN     = _float("CURIA_EVAL_RELEVANCE_MEAN", 3.5)
EVAL_TARGET_ADVERSARIAL_DROP   = _float("CURIA_EVAL_ADVERSARIAL_DROP", 0.30)
