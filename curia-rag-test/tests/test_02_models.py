"""Step 2 — Models: dataclasses are frozen, fields typed, equality works."""
import pytest
from datetime import date
from dataclasses import FrozenInstanceError


def _sample_doc():
    from src.models import Document
    return Document(id="doc_1", title="Test", source="job_posting",
                    date=date(2025, 1, 1), text="hello world")


def _sample_chunk():
    from src.models import Chunk
    return Chunk(chunk_id="doc_1_c0", parent_id="doc_1", title="Test",
                 source="job_posting", date=date(2025, 1, 1),
                 text="hello world", chunk_index=0)


def test_document_is_frozen():
    doc = _sample_doc()
    with pytest.raises(FrozenInstanceError):
        doc.title = "mutated"  # type: ignore


def test_chunk_is_frozen():
    chunk = _sample_chunk()
    with pytest.raises(FrozenInstanceError):
        chunk.text = "mutated"  # type: ignore


def test_search_result_is_frozen():
    from src.models import SearchResult
    chunk = _sample_chunk()
    result = SearchResult(chunk=chunk, similarity=0.9, score=0.85)
    with pytest.raises(FrozenInstanceError):
        result.similarity = 0.1  # type: ignore


def test_recommendation_is_frozen():
    from src.models import Recommendation
    rec = Recommendation(signal_strength="high", summary="Good.",
                         emerging_topics=["llm"], evidence_ids=["doc_1"])
    with pytest.raises(FrozenInstanceError):
        rec.signal_strength = "low"  # type: ignore


def test_document_equality():
    from src.models import Document
    d1 = Document(id="x", title="T", source="arxiv",
                  date=date(2025, 1, 1), text="abc")
    d2 = Document(id="x", title="T", source="arxiv",
                  date=date(2025, 1, 1), text="abc")
    assert d1 == d2


def test_document_fields_correct():
    doc = _sample_doc()
    assert doc.id == "doc_1"
    assert doc.source == "job_posting"
    assert isinstance(doc.date, date)
    assert isinstance(doc.metadata, dict)


def test_chunk_parent_id_linkage():
    chunk = _sample_chunk()
    assert chunk.parent_id == "doc_1"
    assert chunk.chunk_id.startswith("doc_1")


def test_recommendation_signal_values():
    from src.models import Recommendation
    for sig in ("high", "medium", "low"):
        rec = Recommendation(signal_strength=sig, summary=".",
                             emerging_topics=[], evidence_ids=[])
        assert rec.signal_strength == sig
