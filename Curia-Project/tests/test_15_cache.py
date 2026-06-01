"""Step 15 — CacheLayer: SQLite-backed multi-layer cache, tmp_path isolated."""
import sqlite3

import pytest

from src.cache import CacheLayer, _iso, _now
from datetime import timedelta


@pytest.fixture
def cache(tmp_path):
    return CacheLayer(tmp_path / "test_cache.db")


# ---------- Recommendation cache ----------
def test_recommendation_miss_returns_none(cache):
    assert cache.get_recommendation("doesnotexist") is None


def test_recommendation_set_then_get(cache):
    cache.set_recommendation(
        query_hash="abc123",
        normalized_query={"program": "tamu_cs"},
        recommendation={"summary": "Learn RAG"},
        evidence_ids=["ax_001", "gh_002"],
        llm_model="gpt-4o-mini",
        citation_check_ok=True,
    )
    result = cache.get_recommendation("abc123")
    assert result is not None
    assert result["recommendation"] == {"summary": "Learn RAG"}
    assert result["evidence_ids"] == ["ax_001", "gh_002"]


def test_expired_recommendation_returns_none(cache):
    cache.set_recommendation(
        query_hash="abc123",
        normalized_query={},
        recommendation={},
        evidence_ids=[],
        llm_model="x",
        citation_check_ok=True,
        ttl_days=-1,
    )
    assert cache.get_recommendation("abc123") is None


def test_recommendation_hit_count_increments(cache):
    cache.set_recommendation(
        query_hash="h", normalized_query={}, recommendation={"s": 1},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    cache.get_recommendation("h")
    cache.get_recommendation("h")
    cache.get_recommendation("h")
    stats = cache.stats()
    assert stats["recommendation_total_hits"] == 3


def test_set_recommendation_preserves_hit_count_on_replace(cache):
    cache.set_recommendation(
        query_hash="h", normalized_query={}, recommendation={"s": 1},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    cache.get_recommendation("h")
    cache.get_recommendation("h")
    cache.set_recommendation(
        query_hash="h", normalized_query={}, recommendation={"s": 2},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    assert cache.get_recommendation("h")["recommendation"] == {"s": 2}
    assert cache.stats()["recommendation_total_hits"] == 3


# ---------- Agent A ----------
def _agent_a_row(skill="rag", source="arxiv", week="2026-W10"):
    return {
        "skill_id": skill, "source": source, "week_iso": week,
        "intensity": 0.5, "attribution": {"arxiv": 1.0}, "n_mentions": 4,
    }


def test_agent_a_set_then_get(cache):
    assert cache.set_agent_a([_agent_a_row(), _agent_a_row(source="github_readme")]) == 2
    entries = cache.get_agent_a("rag")
    assert len(entries) == 2
    assert {e.value["source"] for e in entries} == {"arxiv", "github_readme"}
    assert all(e.value["attribution"] == {"arxiv": 1.0} for e in entries)


def test_agent_a_source_filter(cache):
    cache.set_agent_a([_agent_a_row(source="arxiv"), _agent_a_row(source="github_readme")])
    entries = cache.get_agent_a("rag", sources=["arxiv"])
    assert len(entries) == 1
    assert entries[0].value["source"] == "arxiv"


def test_agent_a_bulk_set_idempotent(cache):
    cache.set_agent_a([_agent_a_row()])
    cache.set_agent_a([_agent_a_row()])
    assert len(cache.get_agent_a("rag")) == 1


# ---------- Agent B ----------
def test_agent_b_set_then_get(cache):
    cache.set_agent_b([{
        "skill_id": "rag", "horizon_months": 12, "forecast_value": 0.8,
        "ci_lower": 0.7, "ci_upper": 0.9, "slope": 0.01,
        "model_name": "linear", "backtest_mape": 0.12,
    }])
    row = cache.get_agent_b("rag", 12)
    assert row is not None
    assert row["forecast_value"] == 0.8
    assert row["model_name"] == "linear"


def test_agent_b_miss_returns_none(cache):
    assert cache.get_agent_b("rag", 12) is None


def test_agent_b_optional_mape(cache):
    cache.set_agent_b([{
        "skill_id": "rag", "horizon_months": 3, "forecast_value": 0.5,
        "ci_lower": 0.4, "ci_upper": 0.6, "slope": 0.0, "model_name": "linear",
    }])
    assert cache.get_agent_b("rag", 3)["backtest_mape"] is None


# ---------- Agent C ----------
def test_agent_c_set_then_get(cache):
    cache.set_agent_c([{
        "skill_id": "rag", "drift_score": 0.42, "drift_p_value": 0.01,
        "direction": "expanding", "evidence_blob": {"before": ["x"], "after": ["y"]},
        "window_start": "2025-01", "window_end": "2026-01",
    }])
    row = cache.get_agent_c("rag")
    assert row is not None
    assert row["drift_score"] == 0.42
    assert row["evidence_blob"] == {"before": ["x"], "after": ["y"]}


def test_agent_c_miss_returns_none(cache):
    assert cache.get_agent_c("nope") is None


# ---------- Resources ----------
def test_resources_set_then_get_ordered_by_score(cache):
    cache.set_resources([
        {"skill_id": "rag", "resource_id": "mit_ocw_1", "match_score": 0.5,
         "prerequisite_depth": 0, "estimated_hours": 10, "meta": {"title": "A"}},
        {"skill_id": "rag", "resource_id": "edx_2", "match_score": 0.9,
         "prerequisite_depth": 1, "estimated_hours": 20, "meta": {"title": "B"}},
    ])
    rows = cache.get_resources_for_skill("rag", top_k=5)
    assert [r["resource_id"] for r in rows] == ["edx_2", "mit_ocw_1"]
    assert rows[0]["meta"] == {"title": "B"}


def test_resources_respects_top_k(cache):
    cache.set_resources([
        {"skill_id": "rag", "resource_id": f"r{i}", "match_score": float(i),
         "prerequisite_depth": 0, "estimated_hours": None, "meta": {}}
        for i in range(5)
    ])
    assert len(cache.get_resources_for_skill("rag", top_k=2)) == 2


# ---------- Invalidation ----------
def test_invalidate_recommendation_logs(cache):
    cache.set_recommendation(
        query_hash="h", normalized_query={}, recommendation={},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    assert cache.invalidate_recommendation("h", "user_feedback") == 1
    assert cache.get_recommendation("h") is None
    with cache._conn() as conn:
        row = conn.execute(
            "SELECT cache_table, reason, rows_affected FROM cache_invalidations"
        ).fetchone()
    assert row["cache_table"] == "recommendation_cache"
    assert row["reason"] == "user_feedback"
    assert row["rows_affected"] == 1


def test_invalidate_skill_cascades_multiple_tables(cache):
    cache.set_agent_a([_agent_a_row()])
    cache.set_agent_b([{
        "skill_id": "rag", "horizon_months": 12, "forecast_value": 0.8,
        "ci_lower": 0.7, "ci_upper": 0.9, "slope": 0.01, "model_name": "linear",
    }])
    affected = cache.invalidate_skill("rag", ("a", "b"), "drift_detected")
    assert affected == {"a": 1, "b": 1}
    assert cache.get_agent_a("rag") == []
    assert cache.get_agent_b("rag", 12) is None
    with cache._conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM cache_invalidations").fetchone()[0]
    assert n == 2


# ---------- Purge + schema ----------
def test_purge_stale_removes_only_expired(cache):
    cache.set_recommendation(
        query_hash="fresh", normalized_query={}, recommendation={},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    cache.set_recommendation(
        query_hash="stale", normalized_query={}, recommendation={},
        evidence_ids=[], llm_model="m", citation_check_ok=True, ttl_days=-1,
    )
    purged = cache.purge_stale()
    assert purged["recommendations"] == 1
    assert cache.get_recommendation("fresh") is not None


def test_cache_layer_initializes_schema_without_audit_log(cache):
    with cache._conn() as conn:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {
        "agent_a_cache", "agent_b_cache", "agent_c_cache",
        "resource_cache", "recommendation_cache", "batch_runs",
        "cache_invalidations", "recommendation_skill_links",
    }.issubset(tables)


def test_empty_stats(cache):
    stats = cache.stats()
    assert stats["counts"]["agent_a"] == 0
    assert stats["recommendation_total_hits"] == 0
    assert stats["recommendation_entries"] == 0
