"""Step 12 — Pipeline: full end-to-end with real corpus, FAISS, and OpenAI."""
import os
import tempfile
import pytest
from pathlib import Path
from src.config import CORPUS_DIR, SOURCE_QUOTAS, UNITS_FILE
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus, get_unit, load_units

UNITS = load_units(UNITS_FILE)
REQUIRED_OUTPUT_KEYS = {"query", "prompt", "evidence", "recommendation",
                        "citation_check", "audit_id"}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("pipeline")
    index = build_index_from_corpus(CORPUS_DIR)
    return CuriaRagPipeline(
        index,
        audit_path=tmpdir / "audit.db",
        source_quotas=SOURCE_QUOTAS,
    )


def test_pipeline_returns_all_keys(pipeline):
    unit = get_unit(UNITS, "cs_ai_01")
    result = pipeline.run(unit)
    assert REQUIRED_OUTPUT_KEYS.issubset(result.keys())


def test_pipeline_evidence_is_non_empty(pipeline):
    unit = get_unit(UNITS, "cs_ai_01")
    result = pipeline.run(unit)
    assert len(result["evidence"]) > 0


def test_pipeline_evidence_has_correct_structure(pipeline):
    unit = get_unit(UNITS, "cs_ai_01")
    result = pipeline.run(unit)
    for ev in result["evidence"]:
        assert "parent_id" in ev
        assert "similarity" in ev
        assert "score" in ev
        assert "text" in ev


def test_pipeline_recommendation_has_valid_signal(pipeline):
    unit = get_unit(UNITS, "cs_ai_01")
    result = pipeline.run(unit)
    assert result["recommendation"]["signal_strength"] in ("high", "medium", "low")


def test_pipeline_citation_check_field_present(pipeline):
    unit = get_unit(UNITS, "cs_sec_01")
    result = pipeline.run(unit)
    cc = result["citation_check"]
    assert "passed" in cc
    assert "cited_ids" in cc
    assert "retrieved_ids" in cc
    assert "missing_ids" in cc


def test_pipeline_audit_id_is_integer(pipeline):
    unit = get_unit(UNITS, "cs_cloud_01")
    result = pipeline.run(unit)
    assert isinstance(result["audit_id"], int)


def test_pipeline_no_hallucinated_citations(pipeline):
    """Every cited ID must be in the retrieved set."""
    for unit_id in ("cs_ai_01", "cs_sec_01", "cs_cloud_01"):
        unit = get_unit(UNITS, unit_id)
        result = pipeline.run(unit)
        assert result["citation_check"]["missing_ids"] == [], \
            f"Hallucinated citations for {unit_id}: {result['citation_check']['missing_ids']}"


def test_pipeline_all_three_units(pipeline):
    for uid in ("cs_ai_01", "cs_sec_01", "cs_cloud_01"):
        unit = get_unit(UNITS, uid)
        result = pipeline.run(unit)
        assert result["recommendation"]["signal_strength"] in ("high", "medium", "low"), \
            f"Bad signal for {uid}"


# ---------- Cache hit/miss/disabled flows (no real corpus needed) ----------
from datetime import date
from src.models import Chunk, Recommendation, SearchResult
from src.query_hash import LearnerQuery


class _FakeIndex:
    def search(self, query, k=8):
        chunk = Chunk(chunk_id="jp_001_c0", parent_id="jp_001", title="ML role",
                      source="job_posting", date=date(2025, 6, 1),
                      text="machine learning evidence", chunk_index=0)
        return [SearchResult(chunk=chunk, similarity=0.9, score=0.9)]


class _CountingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, unit, evidence):
        self.calls += 1
        return Recommendation(signal_strength="high", summary="Learn (jp_001).",
                              emerging_topics=["ml"], evidence_ids=["jp_001"])


def _learner_pipeline(tmp_path):
    gen = _CountingGenerator()
    pipe = CuriaRagPipeline(_FakeIndex(), audit_path=tmp_path / "audit.db", generator=gen)
    return pipe, gen


QUERY = LearnerQuery(program="tamu_cs", goal="learn machine learning", query_text="ml roadmap")


def test_cache_miss_then_hit_skips_llm(tmp_path):
    pipe, gen = _learner_pipeline(tmp_path)
    first = pipe.run(QUERY)
    assert first["cache_status"] == "miss"
    assert gen.calls == 1
    second = pipe.run(QUERY)
    assert second["cache_status"] == "hit"
    assert gen.calls == 1
    assert second["recommendation"] == first["recommendation"]
    assert second["evidence_ids"] == first["evidence_ids"]


def test_cache_miss_writes_to_cache(tmp_path):
    pipe, _ = _learner_pipeline(tmp_path)
    pipe.run(QUERY)
    assert pipe.cache.get_recommendation(QUERY.hash()) is not None


def test_use_cache_false_bypasses_cache(tmp_path, monkeypatch):
    import src.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "USE_CACHE", False)
    pipe, gen = _learner_pipeline(tmp_path)
    pipe.run(QUERY)
    pipe.run(QUERY)
    assert gen.calls == 2
    assert pipe.cache.get_recommendation(QUERY.hash()) is None
