#!/usr/bin/env python3
"""Cache & batch benchmark — produces the Velocity metrics for the paper.

Default mode is dependency-free (no faiss/torch/openai): it times the light
batch agents (A/B/E) over the real corpus, measures recommendation-cache read
latency, and simulates hit-rate / LLM-call reduction on a defined workload.

    python scripts/bench_cache.py                  # deps-free metrics
    python scripts/bench_cache.py --full           # + pipeline + batch (needs venv)
    python scripts/bench_cache.py --out results/   # also write JSON report

Reported metrics map to the spec success criteria (90%+ hit rate, <=200ms hit
latency) and the §11 Velocity argument.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return round(s[idx], 3)


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def bench_deps_free(n_rec: int = 500, n_reads: int = 2000,
                    n_unique: int = 100, repeats: int = 10) -> dict:
    from src.agent_a_fusion import FusionAgent
    from src.agent_e_resources import ResourceMatcher
    from src.cache import CacheLayer
    from src.forecasting import SkillForecaster

    db = Path(tempfile.mkdtemp()) / "bench.db"
    cache = CacheLayer(db)

    # 1. Light batch agent materialization timing (real corpus for Agent A)
    t0 = time.perf_counter()
    a_rows = FusionAgent().compute_for_window(weeks=520)
    t_a = time.perf_counter() - t0
    cache.set_agent_a(a_rows)

    t0 = time.perf_counter()
    b_rows = SkillForecaster(cache=cache).forecast_all_skills(horizons=(3, 6, 12, 24))
    t_b = time.perf_counter() - t0
    cache.set_agent_b(b_rows)

    t0 = time.perf_counter()
    e_rows = ResourceMatcher(cache=cache).match_all_skills()
    t_e = time.perf_counter() - t0
    cache.set_resources(e_rows)

    materialization = {
        "agent_a": {"rows": len(a_rows), "seconds": round(t_a, 4)},
        "agent_b": {"rows": len(b_rows), "seconds": round(t_b, 4)},
        "agent_e": {"rows": len(e_rows), "seconds": round(t_e, 4)},
    }

    # 2. Recommendation-cache read latency (hit vs miss)
    for i in range(n_rec):
        cache.set_recommendation(
            query_hash=f"q{i}", normalized_query={"i": i},
            recommendation={"summary": f"rec {i}", "evidence_ids": ["ax_1"]},
            evidence_ids=["ax_1", "gh_2"], llm_model="local", citation_check_ok=True,
        )
    hit_lat: list[float] = []
    for i in range(n_reads):
        key = f"q{i % n_rec}"
        t0 = time.perf_counter()
        cache.get_recommendation(key)
        hit_lat.append(_ms(time.perf_counter() - t0))
    miss_lat: list[float] = []
    for i in range(min(n_reads, 1000)):
        t0 = time.perf_counter()
        cache.get_recommendation(f"absent_{i}")
        miss_lat.append(_ms(time.perf_counter() - t0))

    read_latency_ms = {
        "hit_mean": round(sum(hit_lat) / len(hit_lat), 3),
        "hit_p50": _pct(hit_lat, 0.50),
        "hit_p95": _pct(hit_lat, 0.95),
        "miss_mean": round(sum(miss_lat) / len(miss_lat), 3),
        "meets_200ms_target": _pct(hit_lat, 0.95) <= 200.0,
    }

    # 3. Hit-rate + LLM-call reduction on a defined workload
    rng = random.Random(42)
    workload = [f"w{i}" for i in range(n_unique) for _ in range(repeats)]
    rng.shuffle(workload)
    seen: set[str] = set()
    hits = misses = 0
    for key in workload:
        if key in seen:
            hits += 1
        else:
            misses += 1
            seen.add(key)
    total = len(workload)
    workload_metrics = {
        "workload": f"{n_unique} unique x {repeats} repeats = {total} queries (shuffled)",
        "hit_rate": round(hits / total, 4),
        "llm_calls_without_cache": total,
        "llm_calls_with_cache": misses,
        "llm_calls_avoided": hits,
        "llm_cost_reduction_pct": round(100.0 * hits / total, 2),
    }

    return {
        "mode": "deps_free",
        "batch_materialization_light_agents": materialization,
        "recommendation_read_latency_ms": read_latency_ms,
        "workload_hit_rate": workload_metrics,
        "db_size_bytes": db.stat().st_size,
    }


def bench_full(n_unique: int = 6, repeats: int = 5) -> dict:
    from src.config import AUDIT_DB_PATH, CORPUS_DIR, INDEX_PATH, SOURCE_QUOTAS, UNITS_FILE
    from src.audit import AuditLog
    from src.batch import BatchRunner
    from src.cache import CacheLayer
    from src.indexing import FaissIndex
    from src.llm import LocalGroundedGenerator
    from src.pipeline import CuriaRagPipeline
    from src.query_hash import LearnerQuery
    from src.storage import build_index_from_corpus, load_units

    if INDEX_PATH.exists():
        index = FaissIndex.load(INDEX_PATH)
    else:
        index = build_index_from_corpus(CORPUS_DIR)

    class _CountingGen(LocalGroundedGenerator):
        calls = 0

        def generate(self, unit, evidence):
            type(self).calls += 1
            return super().generate(unit, evidence)

    db = Path(tempfile.mkdtemp()) / "bench_full.db"
    gen = _CountingGen()
    pipe = CuriaRagPipeline(index, audit_path=db, source_quotas=SOURCE_QUOTAS, generator=gen)

    units = load_units(UNITS_FILE)
    base = [
        LearnerQuery(program="tamu_cs", goal=u.get("title", ""),
                     query_text=u.get("description", ""),
                     curriculum_unit_ids=(u["id"],))
        for u in units
    ][:n_unique]
    while len(base) < n_unique:
        base.append(LearnerQuery(program="tamu_cs", query_text=f"topic {len(base)}"))

    rng = random.Random(7)
    workload = [q for q in base for _ in range(repeats)]
    rng.shuffle(workload)

    hit_lat: list[float] = []
    miss_lat: list[float] = []
    hits = misses = 0
    _CountingGen.calls = 0
    t_start = time.perf_counter()
    for q in workload:
        t0 = time.perf_counter()
        result = pipe.run(q)
        dt = _ms(time.perf_counter() - t0)
        if result.get("cache_status") == "hit":
            hits += 1
            hit_lat.append(dt)
        else:
            misses += 1
            miss_lat.append(dt)
    total_wall = time.perf_counter() - t_start
    total = len(workload)

    batch_audit = AuditLog(db)
    batch_cache = CacheLayer(db)
    t0 = time.perf_counter()
    br = BatchRunner(batch_cache, batch_audit, run_drift=True, run_forecast=True, skip_ingest=True)
    batch_result = br.run_full_refresh()
    batch_seconds = round(time.perf_counter() - t0, 3)

    speedup = (round((sum(miss_lat) / len(miss_lat)) / (sum(hit_lat) / len(hit_lat)), 1)
               if hit_lat and miss_lat else None)

    return {
        "mode": "full",
        "pipeline_workload": {
            "workload": f"{n_unique} unique x {repeats} repeats = {total} queries",
            "hit_rate": round(hits / total, 4),
            "generator_calls": _CountingGen.calls,
            "generator_calls_without_cache_would_be": total,
            "hit_latency_ms": {"mean": round(sum(hit_lat) / len(hit_lat), 2) if hit_lat else None,
                               "p95": _pct(hit_lat, 0.95)},
            "miss_latency_ms": {"mean": round(sum(miss_lat) / len(miss_lat), 2) if miss_lat else None,
                                "p95": _pct(miss_lat, 0.95)},
            "hit_vs_miss_speedup_x": speedup,
            "total_wall_seconds": round(total_wall, 3),
        },
        "batch_run": {
            "status": batch_result.status,
            "seconds": batch_seconds,
            "n_agent_a": batch_result.n_agent_a,
            "n_agent_b": batch_result.n_agent_b,
            "n_agent_c": batch_result.n_agent_c,
            "n_resources": batch_result.n_resources,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also run pipeline + batch metrics (needs full venv)")
    parser.add_argument("--out", help="Directory to also write a JSON report into")
    args = parser.parse_args()

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    report["deps_free"] = bench_deps_free()
    if args.full:
        report["full"] = bench_full()

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cache_metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(text)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
