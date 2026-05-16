"""Step 6 — Retrieval: recency scoring, source quotas, top-k ordering."""
import pytest
from datetime import date, timedelta
from src.models import Chunk, SearchResult
from src.indexing import FaissIndex
from src.retrieval import Retriever, recency_multiplier


def _chunk(text, doc_id, source="job_posting", days_ago=0):
    d = date.today() - timedelta(days=days_ago)
    return Chunk(chunk_id=f"{doc_id}_c0", parent_id=doc_id, title="T",
                 source=source, date=d, text=text, chunk_index=0)


CHUNKS = [
    _chunk("machine learning llm transformer rag vector database", "ml_new", "job_posting", 10),
    _chunk("machine learning llm transformer rag vector database", "ml_old", "job_posting", 730),
    _chunk("kubernetes cloud native containers microservices", "cloud1", "arxiv", 30),
    _chunk("kubernetes service mesh observability tracing", "cloud2", "arxiv", 60),
    _chunk("stack overflow kubernetes slo burn rate", "so1", "stackoverflow", 90),
    _chunk("github devops ci cd pipeline automation", "gh1", "github_readme", 120),
]


@pytest.fixture(scope="module")
def retriever():
    index = FaissIndex.build(CHUNKS)
    return Retriever(index)


def test_recency_multiplier_recent_is_higher():
    today = date.today()
    recent = recency_multiplier(today - timedelta(days=1))
    old = recency_multiplier(today - timedelta(days=700))
    assert recent > old


def test_recency_multiplier_bounds():
    today = date.today()
    score = recency_multiplier(today)
    assert 0.7 <= score <= 1.0
    old_score = recency_multiplier(date(2000, 1, 1))
    assert old_score >= 0.7


def test_recent_doc_scores_higher_than_old(retriever):
    """ml_new and ml_old have identical text; ml_new (10 days ago) should win."""
    results = retriever.retrieve("machine learning llm transformer", k=6)
    ids = [r.chunk.parent_id for r in results]
    assert ids.index("ml_new") < ids.index("ml_old")


def test_returns_at_most_k_results(retriever):
    results = retriever.retrieve("machine learning", k=3)
    assert len(results) <= 3


def test_source_quotas_enforced(retriever):
    quotas = {"job_posting": 1, "arxiv": 1, "stackoverflow": 1, "github_readme": 1}
    results = retriever.retrieve("cloud kubernetes machine learning", k=8,
                                  source_quotas=quotas)
    from collections import Counter
    counts = Counter(r.chunk.source for r in results)
    for source, max_count in quotas.items():
        assert counts.get(source, 0) <= max_count, \
            f"Source {source} exceeded quota: {counts.get(source, 0)} > {max_count}"


def test_scores_descending(retriever):
    results = retriever.retrieve("machine learning", k=6)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_zero_similarity_results_excluded(retriever):
    """Results with similarity=0 should not appear."""
    results = retriever.retrieve("machine learning kubernetes", k=10)
    for r in results:
        assert r.similarity > 0
