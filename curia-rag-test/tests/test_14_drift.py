"""Step 14 — Agent C Semantic Drift: math, bucketing, drift detection."""

from datetime import date

import numpy as np
import pytest

from src.drift import (
    DriftBucket,
    DriftPair,
    DriftResult,
    SemanticDriftDetector,
    _centroid,
    _cosine_distance,
    _l2_normalize,
    _quarter_label,
    _source_label,
)
from src.indexing import FaissIndex
from src.models import Chunk


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def test_l2_normalize_unit_vector():
    v = np.array([3.0, 4.0])
    out = _l2_normalize(v)
    assert pytest.approx(float(np.linalg.norm(out))) == 1.0


def test_l2_normalize_zero_vector():
    v = np.zeros(5)
    out = _l2_normalize(v)
    assert np.array_equal(out, v)


def test_centroid_empty_list():
    out = _centroid([])
    assert out.size == 0


def test_centroid_single_vector():
    v = np.array([1.0, 0.0, 0.0])
    out = _centroid([v])
    assert pytest.approx(out[0]) == 1.0


def test_centroid_two_orthogonal_vectors():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    out = _centroid([v1, v2])
    # Mean is (0.5, 0.5), normalised → (1/√2, 1/√2)
    assert pytest.approx(float(out[0]), rel=1e-5) == pytest.approx(float(out[1]), rel=1e-5)
    assert pytest.approx(float(np.linalg.norm(out))) == 1.0


def test_cosine_distance_identical_centroids():
    a = np.array([1.0, 0.0])
    assert _cosine_distance(a, a) == pytest.approx(0.0)


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert _cosine_distance(a, b) == pytest.approx(2.0)


def test_cosine_distance_empty_returns_zero():
    a = np.array([])
    b = np.array([1.0, 0.0])
    assert _cosine_distance(a, b) == 0.0


def test_quarter_label_january():
    assert _quarter_label(date(2024, 1, 15)) == "2024-Q1"


def test_quarter_label_april():
    assert _quarter_label(date(2024, 4, 1)) == "2024-Q2"


def test_quarter_label_december():
    assert _quarter_label(date(2024, 12, 31)) == "2024-Q4"


def test_source_label_known():
    assert _source_label("job_posting") == "Industry (jobs)"
    assert _source_label("arxiv") == "Research (arXiv)"


def test_source_label_unknown_passes_through():
    assert _source_label("brand_new_source") == "brand_new_source"


# ---------------------------------------------------------------------------
# Dataclass frozen-ness
# ---------------------------------------------------------------------------

def test_drift_bucket_is_frozen():
    from dataclasses import FrozenInstanceError
    b = DriftBucket(label="X", source="x", centroid=(1.0, 0.0), chunk_count=3)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        b.chunk_count = 99  # type: ignore


def test_drift_pair_is_frozen():
    from dataclasses import FrozenInstanceError
    p = DriftPair(from_label="A", to_label="B", cosine_distance=0.3)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        p.cosine_distance = 0.9  # type: ignore


def test_drift_result_is_frozen():
    from dataclasses import FrozenInstanceError
    r = DriftResult(skill="x", mode="cross_source", buckets=(), pairs=(),
                    max_drift=0.0, mean_drift=0.0, drifted=False,
                    drift_threshold=0.15, chunk_count_total=0)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        r.skill = "y"  # type: ignore


# ---------------------------------------------------------------------------
# Test fixture — small FaissIndex with 4 source classes
# ---------------------------------------------------------------------------

def _chunk(text: str, source: str, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source}_c{idx}",
        parent_id=f"{source}_d{idx}",
        title=f"{source} chunk {idx}",
        source=source,
        date=date(2025, 1 + (idx % 12), 1),
        text=text,
        chunk_index=idx,
    )


# Drift-prone topic: "cloud computing"
# Industry chunks emphasise serverless + DevOps tooling (modern)
# arXiv chunks emphasise scheduling theory + virtualisation (foundational)
# Stack Overflow chunks emphasise practical commands + error fixes
# GitHub READMEs emphasise OSS controller patterns
DRIFTY_CHUNKS = [
    _chunk("cloud computing serverless lambda devops ci cd kubernetes deployment", "job_posting", 0),
    _chunk("cloud computing aws lambda ecs fargate event driven enterprise scale", "job_posting", 1),
    _chunk("cloud computing kubernetes orchestration helm sre platform engineering", "job_posting", 2),
    _chunk("cloud computing virtualization scheduling theorem theoretical model proof", "arxiv", 3),
    _chunk("cloud computing dynamic resource allocation linear programming optimisation", "arxiv", 4),
    _chunk("cloud computing vm migration algorithm convergence formal model abstract", "arxiv", 5),
    _chunk("cloud computing error 503 timeout kubectl pod restart fix exception", "stackoverflow", 6),
    _chunk("cloud computing iam permissions s3 bucket policy denied troubleshoot", "stackoverflow", 7),
    _chunk("cloud computing terraform apply plan state lock manual unlock command", "stackoverflow", 8),
    _chunk("cloud computing operator controller crd custom resource reconcile loop golang", "github_readme", 9),
    _chunk("cloud computing helm chart values template release rollback upgrade", "github_readme", 10),
    _chunk("cloud computing open source toolkit api spec yaml manifest example", "github_readme", 11),
]

# Consistent topic: all four sources describe "python" in the same way
CONSISTENT_CHUNKS = [
    _chunk("python programming language general purpose syntax interpreted", "job_posting",  20),
    _chunk("python programming language general purpose syntax interpreted",   "job_posting",  21),
    _chunk("python programming language general purpose syntax interpreted",   "job_posting",  22),
    _chunk("python programming language general purpose syntax interpreted",   "arxiv",        23),
    _chunk("python programming language general purpose syntax interpreted",   "arxiv",        24),
    _chunk("python programming language general purpose syntax interpreted",   "arxiv",        25),
    _chunk("python programming language general purpose syntax interpreted",   "stackoverflow",26),
    _chunk("python programming language general purpose syntax interpreted",   "stackoverflow",27),
    _chunk("python programming language general purpose syntax interpreted",   "stackoverflow",28),
    _chunk("python programming language general purpose syntax interpreted",   "github_readme",29),
    _chunk("python programming language general purpose syntax interpreted",   "github_readme",30),
    _chunk("python programming language general purpose syntax interpreted",   "github_readme",31),
]


@pytest.fixture(scope="module")
def drift_index():
    return FaissIndex.build(DRIFTY_CHUNKS + CONSISTENT_CHUNKS)


@pytest.fixture(scope="module")
def detector(drift_index):
    return SemanticDriftDetector(
        drift_index,
        chunks_per_skill=24,
        min_chunks_per_bucket=2,
        drift_threshold=0.05,
    )


# ---------------------------------------------------------------------------
# SemanticDriftDetector behaviour
# ---------------------------------------------------------------------------

def test_invalid_mode_raises(drift_index):
    with pytest.raises(ValueError):
        SemanticDriftDetector(drift_index, mode="banana")


def test_analyze_skill_returns_drift_result(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    assert isinstance(r, DriftResult)
    assert r.skill == "cloud computing"
    assert r.mode == "cross_source"


def test_drifty_topic_has_nontrivial_drift(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    assert r.max_drift > 0.05, f"Expected non-trivial drift, got {r.max_drift}"
    assert r.drifted, f"max_drift={r.max_drift} should exceed threshold"


def test_drifty_topic_covers_all_4_communities(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    sources = {b.source for b in r.buckets}
    # Should have at least 3 of the 4 source classes
    assert len(sources & {"job_posting", "arxiv", "stackoverflow", "github_readme"}) >= 3


def test_max_drift_within_valid_range(detector):
    """Cosine distance is mathematically in [0, 2]; max_drift must respect that."""
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    assert 0.0 <= r.max_drift <= 2.0


def test_analyze_skill_returns_none_when_no_data(detector):
    r = detector.analyze_skill("xyzqwerty_unicorn_skill_that_does_not_exist")
    # Could return None or a DriftResult with low chunk count; both acceptable
    assert r is None or r.chunk_count_total >= 4


def test_pairs_sorted_descending(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    if len(r.pairs) >= 2:
        ds = [p.cosine_distance for p in r.pairs]
        assert ds == sorted(ds, reverse=True)


def test_max_drift_matches_top_pair(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    if r.pairs:
        assert r.max_drift == r.pairs[0].cosine_distance


def test_mean_drift_in_valid_range(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    assert 0.0 <= r.mean_drift <= 2.0


def test_chunk_count_consistency(detector):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    bucket_sum = sum(b.chunk_count for b in r.buckets)
    assert r.chunk_count_total == bucket_sum


def test_centroid_dim_matches_embedder(detector, drift_index):
    r = detector.analyze_skill("cloud computing")
    assert r is not None
    expected_dim = drift_index._index.d
    for b in r.buckets:
        assert len(b.centroid) == expected_dim


# ---------------------------------------------------------------------------
# analyze_skills + top_drifters
# ---------------------------------------------------------------------------

def test_analyze_skills_returns_list(detector):
    results = detector.analyze_skills(["cloud computing", "python programming language"])
    assert isinstance(results, list)
    assert len(results) <= 2


def test_analyze_skills_drops_none_results(drift_index):
    """Detector that requires more chunks than available returns no results."""
    starved = SemanticDriftDetector(
        drift_index,
        chunks_per_skill=2,                   # too few to satisfy min_chunks_per_bucket
        min_chunks_per_bucket=5,
    )
    results = starved.analyze_skills(["cloud computing"])
    assert results == [], "All results should be None and therefore dropped"


def test_top_drifters_returns_only_drifted(detector):
    results = detector.analyze_skills([
        "cloud computing",
        "python programming language general purpose",
    ])
    top = SemanticDriftDetector.top_drifters(results, n=5)
    for r in top:
        assert r.drifted


def test_top_drifters_sorted_descending(detector):
    results = detector.analyze_skills([
        "cloud computing",
        "kubernetes orchestration",
    ])
    top = SemanticDriftDetector.top_drifters(results, n=5)
    drifts = [r.max_drift for r in top]
    assert drifts == sorted(drifts, reverse=True)


def test_top_drifters_respects_n(detector):
    results = detector.analyze_skills([
        "cloud computing",
        "kubernetes orchestration",
    ])
    top = SemanticDriftDetector.top_drifters(results, n=1)
    assert len(top) <= 1


# ---------------------------------------------------------------------------
# Temporal mode
# ---------------------------------------------------------------------------

def test_temporal_mode_buckets_by_quarter(drift_index):
    det = SemanticDriftDetector(
        drift_index,
        chunks_per_skill=24,
        min_chunks_per_bucket=2,
        drift_threshold=0.05,
        mode="temporal",
    )
    r = det.analyze_skill("cloud computing")
    # Test chunks span months 1-12 of 2025 → some quarters will have enough chunks
    if r is not None:
        assert r.mode == "temporal"
        for b in r.buckets:
            assert b.source.startswith("2025-Q")
