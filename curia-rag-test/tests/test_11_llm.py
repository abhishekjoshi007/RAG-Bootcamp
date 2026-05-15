"""Step 11 — LLM: real OpenAI call, structured output parsed, citation present."""
import os
import pytest
from datetime import date
from src.models import Chunk, Recommendation, SearchResult
from src.llm import OpenAIGenerator, parse_json_response


def _result(doc_id: str, text: str) -> SearchResult:
    chunk = Chunk(chunk_id=f"{doc_id}_c0", parent_id=doc_id, title="Test Doc",
                  source="job_posting", date=date(2025, 1, 1),
                  text=text, chunk_index=0)
    return SearchResult(chunk=chunk, similarity=0.85, score=0.80)


UNIT = {
    "id": "cs_ai_01",
    "title": "Generative AI and Large Language Models",
    "description": "Study of transformer-based generative systems, prompt engineering.",
    "current_topics": ["transformers", "RAG", "hallucination mitigation"],
}

EVIDENCE = [
    _result("jp_001", "Senior ML Engineer role requires hands-on experience with "
            "retrieval-augmented generation, vector databases, and LLM deployment."),
    _result("ax_001", "Research on claim-level faithfulness and citation precision "
            "in RAG systems shows hallucination rates drop with grounding checks."),
]


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_generator_returns_recommendation():
    gen = OpenAIGenerator()
    rec = gen.generate(UNIT, EVIDENCE)
    assert isinstance(rec, Recommendation)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_signal_strength_is_valid():
    gen = OpenAIGenerator()
    rec = gen.generate(UNIT, EVIDENCE)
    assert rec.signal_strength in ("high", "medium", "low")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_summary_is_non_empty():
    gen = OpenAIGenerator()
    rec = gen.generate(UNIT, EVIDENCE)
    assert isinstance(rec.summary, str) and len(rec.summary) > 20


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_evidence_ids_are_subset_of_retrieved():
    gen = OpenAIGenerator()
    rec = gen.generate(UNIT, EVIDENCE)
    retrieved_ids = {"jp_001", "ax_001"}
    for eid in rec.evidence_ids:
        assert eid in retrieved_ids, f"LLM cited hallucinated ID: {eid}"


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_emerging_topics_is_list():
    gen = OpenAIGenerator()
    rec = gen.generate(UNIT, EVIDENCE)
    assert isinstance(rec.emerging_topics, list)


def test_parse_json_response_strips_markdown_fence():
    raw = '```json\n{"signal_strength": "high", "summary": "ok"}\n```'
    data = parse_json_response(raw)
    assert data["signal_strength"] == "high"


def test_parse_json_response_plain_json():
    raw = '{"signal_strength": "low", "summary": "test", "emerging_topics": [], "evidence_ids": []}'
    data = parse_json_response(raw)
    assert data["signal_strength"] == "low"


def test_missing_api_key_raises_environment_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = OpenAIGenerator()
    gen._client = None  # force re-init
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        _ = gen._openai
