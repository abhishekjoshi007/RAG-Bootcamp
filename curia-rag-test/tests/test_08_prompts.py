"""Step 8 — Prompts: evidence block has doc IDs, prompt has JSON instructions."""
import pytest
from datetime import date
from src.models import Chunk, SearchResult
from src.prompts import format_evidence, build_recommendation_prompt

UNIT = {"id": "cs_ai_01", "title": "Generative AI", "description": "LLMs.",
        "current_topics": ["transformers"]}


def _result(doc_id: str, sim: float = 0.8) -> SearchResult:
    chunk = Chunk(chunk_id=f"{doc_id}_c0", parent_id=doc_id, title="T",
                  source="job_posting", date=date(2025, 1, 1),
                  text=f"Content about {doc_id}", chunk_index=0)
    return SearchResult(chunk=chunk, similarity=sim, score=sim * 0.9)


RESULTS = [_result("jp_001", 0.9), _result("ax_002", 0.7), _result("so_003", 0.5)]


def test_format_evidence_contains_doc_ids():
    block = format_evidence(RESULTS)
    assert "jp_001" in block
    assert "ax_002" in block
    assert "so_003" in block


def test_format_evidence_numbered():
    block = format_evidence(RESULTS)
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block


def test_format_evidence_contains_source_and_date():
    block = format_evidence(RESULTS)
    assert "job_posting" in block
    assert "2025-01-01" in block


def test_build_prompt_contains_unit_title():
    prompt = build_recommendation_prompt(UNIT, RESULTS)
    assert "Generative AI" in prompt


def test_build_prompt_contains_evidence_ids():
    prompt = build_recommendation_prompt(UNIT, RESULTS)
    assert "jp_001" in prompt


def test_build_prompt_has_json_instruction():
    prompt = build_recommendation_prompt(UNIT, RESULTS)
    assert "JSON" in prompt or "json" in prompt
    assert "signal_strength" in prompt
    assert "summary" in prompt
    assert "emerging_topics" in prompt
    assert "evidence_ids" in prompt


def test_build_prompt_has_no_hallucination_instruction():
    prompt = build_recommendation_prompt(UNIT, RESULTS)
    assert "not" in prompt.lower() and "evidence" in prompt.lower()


def test_empty_evidence_returns_prompt():
    prompt = build_recommendation_prompt(UNIT, [])
    assert isinstance(prompt, str) and len(prompt) > 50
