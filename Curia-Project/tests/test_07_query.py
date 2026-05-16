"""Step 7 — Query construction: topics included/excluded, HyDE prompt non-empty."""
from src.query import build_query, build_hyde_prompt

UNIT = {
    "id": "cs_ai_01",
    "title": "Generative AI and Large Language Models",
    "description": "Covers transformer-based systems and prompt engineering.",
    "current_topics": ["transformers", "RAG", "hallucination mitigation"],
}


def test_query_includes_title():
    q = build_query(UNIT)
    assert "Generative AI" in q


def test_query_includes_description():
    q = build_query(UNIT)
    assert "transformer" in q.lower()


def test_query_includes_topics_by_default():
    q = build_query(UNIT)
    assert "RAG" in q or "transformers" in q


def test_query_excludes_topics_when_flag_false():
    q = build_query(UNIT, include_topics=False)
    assert "RAG" not in q
    assert "transformers" not in q


def test_query_no_empty_pieces():
    unit_no_desc = {"id": "x", "title": "My Unit", "description": "", "current_topics": []}
    q = build_query(unit_no_desc)
    assert q.strip() == "My Unit"
    assert ".." not in q


def test_hyde_prompt_is_non_empty():
    prompt = build_hyde_prompt(UNIT)
    assert isinstance(prompt, str) and len(prompt) > 20


def test_hyde_prompt_contains_topic_title():
    prompt = build_hyde_prompt(UNIT)
    assert "Generative AI" in prompt or "Large Language" in prompt
