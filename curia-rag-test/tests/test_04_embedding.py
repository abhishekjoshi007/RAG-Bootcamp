"""Step 4 — Embedding: correct shape, L2 normalised, semantic similarity works."""
import numpy as np
import pytest
from src.embedding import DenseEmbedder


@pytest.fixture(scope="module")
def embedder():
    return DenseEmbedder()


def test_embed_returns_float32_array(embedder):
    vec = embedder.embed("machine learning engineer")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32


def test_embed_correct_shape(embedder):
    vec = embedder.embed("test sentence")
    assert vec.ndim == 1
    assert vec.shape[0] == 768  # all-mpnet-base-v2 dimension


def test_embed_is_l2_normalised(embedder):
    vec = embedder.embed("normalised vector test")
    magnitude = float(np.linalg.norm(vec))
    assert abs(magnitude - 1.0) < 1e-5, f"Expected norm≈1.0, got {magnitude}"


def test_embed_many_returns_2d_array(embedder):
    texts = ["first sentence", "second sentence", "third sentence"]
    vecs = embedder.embed_many(texts)
    assert vecs.ndim == 2
    assert vecs.shape == (3, 768)
    assert vecs.dtype == np.float32


def test_embed_many_all_normalised(embedder):
    texts = [f"sentence number {i}" for i in range(5)]
    vecs = embedder.embed_many(texts)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_identical_texts_produce_identical_vectors(embedder):
    text = "retrieval augmented generation"
    v1 = embedder.embed(text)
    v2 = embedder.embed(text)
    assert np.allclose(v1, v2)


def test_similar_texts_have_higher_cosine_than_dissimilar(embedder):
    """'machine learning engineer' should be closer to 'deep learning developer'
    than to 'software supply chain security'."""
    query = embedder.embed("machine learning engineer")
    similar = embedder.embed("deep learning developer neural networks")
    dissimilar = embedder.embed("software supply chain security sbom")

    sim_score = float(np.dot(query, similar))
    dis_score = float(np.dot(query, dissimilar))
    assert sim_score > dis_score, (
        f"Semantic similarity failed: similar={sim_score:.3f} dissimilar={dis_score:.3f}"
    )


def test_embed_single_vs_batch_consistent(embedder):
    texts = ["kubernetes cluster monitoring", "prompt engineering llm"]
    single = np.stack([embedder.embed(t) for t in texts])
    batch = embedder.embed_many(texts)
    assert np.allclose(single, batch, atol=1e-4)


def test_model_name_stored(embedder):
    assert embedder.model_name == "all-mpnet-base-v2"
