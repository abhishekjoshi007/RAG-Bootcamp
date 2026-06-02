"""Cache ablation benchmark contracts."""

from scripts.bench_cache_ablation import render_paper_section, run_ablation


def test_cache_ablation_modes_have_expected_call_counts():
    report = run_ablation(
        n_unique=24,
        repeats=6,
        n_skills=8,
        seed=11,
        miss_latency_ms=1.0,
        llm_cost_usd=0.25,
        skills_per_drift_event=3,
    )

    no_cache = report["modes"]["no_cache"]
    ttl_only = report["modes"]["ttl_only"]
    drift = report["modes"]["drift_cascade"]

    assert no_cache["requests"] == 144
    assert no_cache["total_llm_calls"] == 144
    assert no_cache["hit_rate"] == 0.0

    assert ttl_only["total_llm_calls"] == 24
    assert ttl_only["hits"] == 120
    assert ttl_only["hit_rate"] == 0.8333

    assert 24 < drift["total_llm_calls"] < 144
    assert drift["drift_invalidations_triggered"] == 3
    assert drift["drift_invalidations"]["recommendation_rows_invalidated"] > 0
    assert drift["drift_invalidations"]["agent_b_rows_invalidated"] > 0


def test_cache_ablation_cost_and_paper_section_are_generated():
    report = run_ablation(
        n_unique=10,
        repeats=3,
        n_skills=5,
        seed=7,
        miss_latency_ms=1.0,
        llm_cost_usd=0.5,
        skills_per_drift_event=2,
    )

    assert report["benchmark"]["paid_llm_calls_executed"] is False
    assert report["modes"]["no_cache"]["cost_estimate_usd"] == 15.0
    assert report["modes"]["ttl_only"]["cost_estimate_usd"] == 5.0

    section = render_paper_section(report)
    assert "Drift-Cascaded Recommendation Cache" in section
    assert "TTL-only cache" in section
    assert "No paid LLM calls were executed" in section
    assert "Served-staleness rate" in section


def test_drift_cascade_serves_zero_stale_hits_by_construction():
    # Because drift-cascade invalidates entries linked to drifted skills,
    # any later hit on that query hash is on a post-drift re-cache, so
    # served-staleness must be 0. TTL-only, lacking invalidation, must
    # serve at least one stale hit when drift events fire on cached skills.
    report = run_ablation(
        n_unique=24,
        repeats=6,
        n_skills=8,
        seed=11,
        miss_latency_ms=1.0,
        llm_cost_usd=0.25,
        skills_per_drift_event=3,
    )

    ttl = report["modes"]["ttl_only"]["freshness"]
    drift = report["modes"]["drift_cascade"]["freshness"]

    assert drift["served_stale_hits"] == 0
    assert drift["served_staleness_rate"] == 0.0
    assert ttl["served_stale_hits"] > 0
    assert ttl["served_staleness_rate"] > 0.0
    assert ttl["fresh_hits"] + ttl["served_stale_hits"] == report["modes"]["ttl_only"]["hits"]

    # The delta block exposes the freshness trade-off for the paper section.
    delta = report["deltas"]["drift_cascade_vs_ttl_only"]
    assert delta["served_staleness_rate_delta"] < 0.0
