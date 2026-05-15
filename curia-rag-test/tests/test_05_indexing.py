"""Step 5 — FaissIndex: build, search, save, load round-trip."""
import pytest
import tempfile
from datetime import date
from pathlib import Path
from src.models import Chunk, SearchResult
from src.indexing import FaissIndex


def _chunk(text: str, doc_id: str = "d1", idx: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}_c{idx}", parent_id=doc_id,
                 title="T", source="job_posting",
                 date=date(2025, 1, 1), text=text, chunk_index=idx)


SAMPLE_CHUNKS = [
    _chunk("machine learning transformer retrieval augmented generation", "ml_doc", 0),
    _chunk("kubernetes cloud native containers microservices orchestration", "cloud_doc", 0),
    _chunk("supply chain security sbom vulnerability scanning devops", "sec_doc", 0),
    _chunk("serverless functions aws lambda event driven architecture", "serverless_doc", 0),
    _chunk("deep learning neural network gradient descent backpropagation", "dl_doc", 0),
]


@pytest.fixture(scope="module")
def index():
    return FaissIndex.build(SAMPLE_CHUNKS)


def test_build_creates_index(index):
    assert index._index.ntotal == len(SAMPLE_CHUNKS)
    assert len(index.chunks) == len(SAMPLE_CHUNKS)


def test_search_returns_correct_count(index):
    results = index.search("machine learning", k=3)
    assert len(results) == 3


def test_search_results_are_search_result_objects(index):
    results = index.search("kubernetes", k=2)
    for r in results:
        assert isinstance(r, SearchResult)
        assert isinstance(r.similarity, float)
        assert isinstance(r.chunk, Chunk)


def test_search_top_result_is_most_relevant(index):
    """Searching for ML terms should return the ML doc first."""
    results = index.search("retrieval augmented generation llm", k=5)
    assert results[0].chunk.parent_id == "ml_doc"


def test_search_similarities_are_descending(index):
    results = index.search("cloud kubernetes", k=5)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_perfect_match_similarity_near_one(index):
    """Searching with the exact text of a chunk should return sim ≈ 1.0"""
    exact_text = SAMPLE_CHUNKS[0].text
    results = index.search(exact_text, k=1)
    assert results[0].similarity > 0.98


def test_search_k_larger_than_corpus(index):
    """k > number of docs should return all docs without error."""
    results = index.search("anything", k=100)
    assert len(results) == len(SAMPLE_CHUNKS)


def test_save_and_load_round_trip(index):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_index.pkl"
        index.save(path)
        assert path.exists()

        loaded = FaissIndex.load(path)
        assert loaded._index.ntotal == index._index.ntotal
        assert len(loaded.chunks) == len(index.chunks)

        # Same query should return same top result
        original_top = index.search("machine learning", k=1)[0].chunk.parent_id
        loaded_top = loaded.search("machine learning", k=1)[0].chunk.parent_id
        assert original_top == loaded_top


def test_loaded_index_embedder_model_name(index):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.pkl"
        index.save(path)
        loaded = FaissIndex.load(path)
        assert loaded.embedder.model_name == "all-mpnet-base-v2"
