"""Step 16 — LearnerQuery hashing: deterministic, order/case-insensitive, distinct."""
from src.query_hash import LearnerQuery, _normalize_text


def test_hash_is_deterministic():
    q = LearnerQuery(program="TAMU CS", query_text="learn rag")
    assert q.hash() == q.hash()
    assert len(q.hash()) == 16


def test_hash_order_insensitive_on_list_fields():
    a = LearnerQuery(program="p", curriculum_unit_ids=("cs_ai_01", "cs_sec_01"),
                     completed_skills=("python", "sql"))
    b = LearnerQuery(program="p", curriculum_unit_ids=("cs_sec_01", "cs_ai_01"),
                     completed_skills=("sql", "python"))
    assert a.hash() == b.hash()


def test_hash_dedupes_list_fields():
    a = LearnerQuery(program="p", completed_skills=("python", "python", "sql"))
    b = LearnerQuery(program="p", completed_skills=("python", "sql"))
    assert a.hash() == b.hash()


def test_hash_case_and_whitespace_insensitive_on_text():
    a = LearnerQuery(program="TAMU  CS", goal="Build A  RAG System")
    b = LearnerQuery(program="tamu cs", goal="build a rag system")
    assert a.hash() == b.hash()


def test_hash_differs_for_material_changes():
    base = LearnerQuery(program="p", query_text="learn rag")
    assert base.hash() != LearnerQuery(program="p", query_text="learn drift").hash()
    assert base.hash() != LearnerQuery(program="q", query_text="learn rag").hash()
    assert base.hash() != LearnerQuery(
        program="p", query_text="learn rag", goal="ship it"
    ).hash()


def test_normalized_shape():
    q = LearnerQuery(program=" P ", curriculum_unit_ids=("b", "a"),
                     goal="G", completed_skills=("y", "x"), query_text="Q")
    norm = q.normalized()
    assert norm == {
        "program": "p",
        "curriculum_unit_ids": ["a", "b"],
        "goal": "g",
        "completed_skills": ["x", "y"],
        "query_text": "q",
    }


def test_normalize_text_collapses_whitespace():
    assert _normalize_text("  Hello   World\n") == "hello world"
