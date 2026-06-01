"""Step 9 — Grounding: citation check passes/fails, hallucination detected."""
from datetime import date
from src.models import Chunk, Recommendation, SearchResult
from src.grounding import check_citations, CitationCheck


def _result(doc_id: str) -> SearchResult:
    chunk = Chunk(chunk_id=f"{doc_id}_c0", parent_id=doc_id, title="T",
                  source="job_posting", date=date(2025, 1, 1),
                  text="some text", chunk_index=0)
    return SearchResult(chunk=chunk, similarity=0.8, score=0.75)


EVIDENCE = [_result("jp_001"), _result("ax_002"), _result("so_003")]


def test_all_citations_valid_passes():
    rec = Recommendation(signal_strength="high",
                         summary="Good evidence (jp_001, ax_002).",
                         emerging_topics=["llm"],
                         evidence_ids=["jp_001", "ax_002"])
    check = check_citations(rec, EVIDENCE)
    assert check.passed is True
    assert check.missing_ids == []


def test_hallucinated_citation_fails():
    rec = Recommendation(signal_strength="high",
                         summary="Uses jp_999 which was not retrieved.",
                         emerging_topics=[],
                         evidence_ids=["jp_999"])
    check = check_citations(rec, EVIDENCE)
    assert check.passed is False
    assert "jp_999" in check.missing_ids


def test_inline_citations_extracted_from_summary():
    rec = Recommendation(signal_strength="medium",
                         summary="Based on ax_002 and so_003, we see trends.",
                         emerging_topics=[],
                         evidence_ids=[])
    check = check_citations(rec, EVIDENCE)
    assert "ax_002" in check.cited_ids
    assert "so_003" in check.cited_ids


def test_dotted_source_id_is_not_truncated():
    evidence = [_result("axhist_2601.00150v3")]
    rec = Recommendation(signal_strength="medium",
                         summary="The evidence supports this update (axhist_2601.00150v3).",
                         emerging_topics=[],
                         evidence_ids=["axhist_2601.00150v3"])
    check = check_citations(rec, evidence)
    assert check.passed is True
    assert check.cited_ids == ["axhist_2601.00150v3"]
    assert check.missing_ids == []


def test_hyphenated_source_id_is_extracted():
    evidence = [_result("gh_repo-name_001")]
    rec = Recommendation(signal_strength="medium",
                         summary="The repository evidence is relevant (gh_repo-name_001).",
                         emerging_topics=[],
                         evidence_ids=[])
    check = check_citations(rec, evidence)
    assert check.passed is True
    assert check.cited_ids == ["gh_repo-name_001"]


def test_arxiv_version_mismatch_canonicalizes():
    # Real-world case from the multi-LLM benchmark: a model cited v1 of an
    # arXiv paper while the corpus indexed v4. Same paper, different version.
    evidence = [_result("axhist_2501.00701v4")]
    rec = Recommendation(signal_strength="medium",
                         summary="The trend is supported (axhist_2501.00701v1).",
                         emerging_topics=[],
                         evidence_ids=["axhist_2501.00701v1"])
    check = check_citations(rec, evidence)
    assert check.passed is True
    assert check.missing_ids == []


def test_arxiv_version_mismatch_for_modern_ax_prefix_too():
    evidence = [_result("ax_2501.00701v2")]
    rec = Recommendation(signal_strength="medium",
                         summary="Cited (ax_2501.00701v5).",
                         emerging_topics=[],
                         evidence_ids=["ax_2501.00701v5"])
    check = check_citations(rec, evidence)
    assert check.passed is True


def test_non_arxiv_id_version_suffix_is_not_stripped():
    # Only arXiv-style YYMM.NNNNN IDs get version canonicalization; an
    # accidental "v1" on a job-posting ID is still a real mismatch.
    evidence = [_result("jp_001v2")]
    rec = Recommendation(signal_strength="medium",
                         summary="Cited (jp_001v1).",
                         emerging_topics=[],
                         evidence_ids=["jp_001v1"])
    check = check_citations(rec, evidence)
    assert check.passed is False


def test_retrieved_ids_match_evidence():
    check = check_citations(
        Recommendation(signal_strength="low", summary=".", emerging_topics=[], evidence_ids=[]),
        EVIDENCE
    )
    assert set(check.retrieved_ids) == {"jp_001", "ax_002", "so_003"}


def test_empty_recommendation_passes():
    rec = Recommendation(signal_strength="low", summary="No citations.",
                         emerging_topics=[], evidence_ids=[])
    check = check_citations(rec, EVIDENCE)
    assert check.passed is True


def test_citation_check_is_frozen():
    from dataclasses import FrozenInstanceError
    rec = Recommendation(signal_strength="low", summary=".", emerging_topics=[], evidence_ids=[])
    check = check_citations(rec, EVIDENCE)
    try:
        check.passed = False
        assert False, "Should have raised FrozenInstanceError"
    except (FrozenInstanceError, AttributeError):
        pass
