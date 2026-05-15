"""Step 1 — Config: every parameter loads, types are correct, env overrides work."""
import os
import importlib
from pathlib import Path
import pytest


def test_paths_are_path_objects():
    from src.config import CORPUS_DIR, UNITS_FILE, INDEX_PATH, AUDIT_DB_PATH, EVAL_DIR
    assert isinstance(CORPUS_DIR, Path)
    assert isinstance(UNITS_FILE, Path)
    assert isinstance(INDEX_PATH, Path)
    assert isinstance(AUDIT_DB_PATH, Path)
    assert isinstance(EVAL_DIR, Path)


def test_numeric_params_are_correct_types():
    from src.config import (
        CHUNK_MAX_TOKENS, CHUNK_OVERLAP,
        RETRIEVAL_K, RETRIEVAL_CANDIDATE_K, RECENCY_HALF_LIFE_DAYS,
        LLM_MAX_TOKENS, LLM_MAX_RETRIES, EMBED_BATCH_SIZE,
    )
    assert isinstance(CHUNK_MAX_TOKENS, int) and CHUNK_MAX_TOKENS > 0
    assert isinstance(CHUNK_OVERLAP, int) and 0 < CHUNK_OVERLAP < CHUNK_MAX_TOKENS
    assert isinstance(RETRIEVAL_K, int) and RETRIEVAL_K > 0
    assert isinstance(RETRIEVAL_CANDIDATE_K, int) and RETRIEVAL_CANDIDATE_K >= RETRIEVAL_K
    assert isinstance(RECENCY_HALF_LIFE_DAYS, int) and RECENCY_HALF_LIFE_DAYS > 0
    assert isinstance(LLM_MAX_TOKENS, int) and LLM_MAX_TOKENS > 0
    assert isinstance(LLM_MAX_RETRIES, int) and LLM_MAX_RETRIES > 0
    assert isinstance(EMBED_BATCH_SIZE, int) and EMBED_BATCH_SIZE > 0


def test_float_params_are_correct_types():
    from src.config import (
        RECENCY_BASE_WEIGHT, RECENCY_BONUS_WEIGHT,
        LLM_TEMPERATURE, LOCAL_SIGNAL_HIGH, LOCAL_SIGNAL_MEDIUM,
    )
    assert isinstance(RECENCY_BASE_WEIGHT, float)
    assert isinstance(RECENCY_BONUS_WEIGHT, float)
    assert abs(RECENCY_BASE_WEIGHT + RECENCY_BONUS_WEIGHT - 1.0) < 0.01, \
        "Base + bonus weights must sum to 1.0"
    assert 0.0 <= LLM_TEMPERATURE <= 1.0
    assert 0 < LOCAL_SIGNAL_MEDIUM < LOCAL_SIGNAL_HIGH


def test_string_params():
    from src.config import LLM_MODEL, EMBED_MODEL
    assert isinstance(LLM_MODEL, str) and len(LLM_MODEL) > 0
    assert isinstance(EMBED_MODEL, str) and len(EMBED_MODEL) > 0


def test_source_quotas_complete():
    from src.config import SOURCE_QUOTAS
    required_sources = {"job_posting", "arxiv", "stackoverflow", "github_readme"}
    assert required_sources.issubset(SOURCE_QUOTAS.keys())
    for k, v in SOURCE_QUOTAS.items():
        assert isinstance(v, int) and v > 0, f"Quota for {k} must be positive int"


def test_company_lists_non_empty():
    from src.config import GREENHOUSE_COMPANIES, LEVER_COMPANIES
    assert len(GREENHOUSE_COMPANIES) >= 10, "Need at least 10 Greenhouse companies"
    assert len(LEVER_COMPANIES) >= 5, "Need at least 5 Lever companies"
    assert len(set(GREENHOUSE_COMPANIES)) == len(GREENHOUSE_COMPANIES), "Duplicates in Greenhouse list"
    assert len(set(LEVER_COMPANIES)) == len(LEVER_COMPANIES), "Duplicates in Lever list"


def test_env_override(monkeypatch):
    monkeypatch.setenv("CURIA_RETRIEVAL_K", "42")
    import src.config as cfg
    importlib.reload(cfg)
    assert cfg.RETRIEVAL_K == 42
    # Cleanup — reload with original env
    monkeypatch.delenv("CURIA_RETRIEVAL_K", raising=False)
    importlib.reload(cfg)


def test_eval_targets_in_range():
    from src.config import (
        EVAL_TARGET_RECALL_8, EVAL_TARGET_CITATION_PRECISION,
        EVAL_TARGET_CLAIM_GROUNDING, EVAL_TARGET_RELEVANCE_MEAN,
        EVAL_TARGET_ADVERSARIAL_DROP,
    )
    assert 0 < EVAL_TARGET_RECALL_8 <= 1
    assert 0 < EVAL_TARGET_CITATION_PRECISION <= 1
    assert 0 < EVAL_TARGET_CLAIM_GROUNDING <= 1
    assert 1 <= EVAL_TARGET_RELEVANCE_MEAN <= 5
    assert 0 < EVAL_TARGET_ADVERSARIAL_DROP <= 1
