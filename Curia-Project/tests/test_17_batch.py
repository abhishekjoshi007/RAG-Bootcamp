"""Step 17 — BatchRunner: writes batch_runs, fills cache tables, cascades on drift.

Mocks the heavy agents so the runner logic is tested in isolation.
"""
import sqlite3

import pytest

import src.batch as batch_mod
from src.audit import AuditLog
from src.batch import BatchRunner
from src.cache import CacheLayer


class _FakeFusion:
    def __init__(self, *a, **k):
        pass

    def compute_for_window(self, weeks=13):
        return [{
            "skill_id": "rag", "source": "arxiv", "week_iso": "2026-W10",
            "intensity": 0.6, "attribution": {"arxiv": 0.8}, "n_mentions": 5,
        }]


class _FakeForecaster:
    def __init__(self, cache=None, *a, **k):
        self.cache = cache

    def forecast_all_skills(self, horizons=(3, 6, 12, 24), skills=None):
        return [{
            "skill_id": "rag", "horizon_months": 12, "forecast_value": 0.8,
            "ci_lower": 0.7, "ci_upper": 0.9, "slope": 0.01,
            "model_name": "linear", "backtest_mape": 0.1,
        }]


class _FakeDrift:
    def __init__(self, cache=None, *a, **k):
        self.cache = cache

    def detect_all_skills(self, skills=None):
        return [{
            "skill_id": "rag", "drift_score": 0.5, "drift_p_value": None,
            "direction": "expanding", "evidence_blob": {"note": "x"},
            "window_start": "2025-01", "window_end": "2026-01",
        }]


class _FakeResources:
    def __init__(self, cache=None, *a, **k):
        self.cache = cache

    def match_all_skills(self):
        return [{
            "skill_id": "rag", "resource_id": "mit_ocw_1", "match_score": 0.9,
            "prerequisite_depth": 0, "estimated_hours": 20, "meta": {"title": "RAG"},
        }]


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "FusionAgent", _FakeFusion)
    monkeypatch.setattr(batch_mod, "SkillForecaster", _FakeForecaster)
    monkeypatch.setattr(batch_mod, "SemanticDriftDetector", _FakeDrift)
    monkeypatch.setattr(batch_mod, "ResourceMatcher", _FakeResources)
    db = tmp_path / "audit.db"
    audit = AuditLog(db)
    cache = CacheLayer(db)
    return cache, audit


def test_batch_success_fills_cache(wired):
    cache, audit = wired
    runner = BatchRunner(cache, audit, run_drift=False, run_forecast=True, skip_ingest=True)
    result = runner.run_full_refresh()
    assert result.status == "success"
    assert result.n_agent_a == 1
    assert result.n_agent_b == 1
    assert result.n_resources == 1
    assert cache.get_agent_a("rag")
    assert cache.get_agent_b("rag", 12) is not None
    assert cache.get_resources_for_skill("rag")


def test_batch_logs_batch_run(wired):
    cache, audit = wired
    runner = BatchRunner(cache, audit, skip_ingest=True)
    runner.run_full_refresh()
    with sqlite3.connect(audit.db_path) as conn:
        row = conn.execute(
            "SELECT status, n_agent_a FROM batch_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row[0] == "success"
    assert row[1] == 1


def test_drift_cascade_invalidates_downstream(wired):
    cache, audit = wired
    cache.set_recommendation(
        query_hash="qh", normalized_query={}, recommendation={"s": 1},
        evidence_ids=[], llm_model="m", citation_check_ok=True,
    )
    cache.link_recommendation_skills("qh", ["rag"])
    runner = BatchRunner(cache, audit, run_drift=True, run_forecast=True, skip_ingest=True)
    result = runner.run_full_refresh()
    assert result.status == "success"
    assert cache.get_agent_c("rag") is not None
    # drift_score 0.5 > DRIFT_INVALIDATION_THRESHOLD (0.35) → cascade
    assert cache.get_agent_b("rag", 12) is None
    assert cache.get_recommendation("qh") is None
    with sqlite3.connect(audit.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM cache_invalidations WHERE reason='drift_detected'"
        ).fetchone()[0]
    assert n >= 2
