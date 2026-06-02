#!/usr/bin/env python3
"""Cache-policy ablation for the Path C velocity paper.

The benchmark isolates cache policy behavior without paid LLM calls. It uses
the real SQLite-backed CacheLayer for recommendation storage, skill links, and
drift-triggered invalidation logs, while treating every cache miss as one
logical retrieval/generation/grounding call. The miss latency and per-call cost
are explicit assumptions in the output JSON.

Default workload:

    100 unique recommendation queries x 10 repeats = 1,000 shuffled requests

Modes:
    no_cache       Every request is a miss.
    ttl_only       Recommendation cache with TTL, no drift invalidation.
    drift_cascade  TTL cache plus skill-drift invalidation cascade.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class WorkloadItem:
    query_hash: str
    skill_ids: tuple[str, ...]
    ordinal: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[idx], 4)


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": round(sum(values) / len(values), 4),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "p99": _pct(values, 0.99),
    }


def _build_workload(
    n_unique: int,
    repeats: int,
    n_skills: int,
    seed: int,
) -> tuple[list[WorkloadItem], list[str]]:
    if n_unique <= 0:
        raise ValueError("n_unique must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if n_skills <= 0:
        raise ValueError("n_skills must be positive")

    skills = [f"skill_{i:02d}" for i in range(n_skills)]
    unique: list[WorkloadItem] = []
    for i in range(n_unique):
        primary = skills[i % n_skills]
        secondary = skills[(i * 7 + 3) % n_skills]
        skill_ids = (primary,) if secondary == primary else (primary, secondary)
        unique.append(WorkloadItem(query_hash=f"q{i:03d}", skill_ids=skill_ids, ordinal=i))

    workload = [item for item in unique for _ in range(repeats)]
    random.Random(seed).shuffle(workload)
    return workload, skills


def _recommendation_payload(item: WorkloadItem) -> dict[str, Any]:
    return {
        "summary": f"Recommendation for {item.query_hash}",
        "skills": list(item.skill_ids),
        "evidence_ids": [f"evidence_{item.ordinal:03d}_a", f"evidence_{item.ordinal:03d}_b"],
    }


def _seed_agent_b(cache: Any, skill_ids: Iterable[str]) -> int:
    rows = []
    for skill in skill_ids:
        for horizon in (3, 6, 12, 24):
            rows.append(
                {
                    "skill_id": skill,
                    "horizon_months": horizon,
                    "forecast_value": 0.5,
                    "ci_lower": 0.4,
                    "ci_upper": 0.6,
                    "slope": 0.0,
                    "model_name": "ablation_seed",
                    "backtest_mape": None,
                }
            )
    return cache.set_agent_b(rows)


def _future_query_hashes_by_skill(
    workload: list[WorkloadItem],
    start_idx: int,
) -> dict[str, set[str]]:
    future: dict[str, set[str]] = {}
    for item in workload[start_idx:]:
        for skill in item.skill_ids:
            future.setdefault(skill, set()).add(item.query_hash)
    return future


def _cached_query_hashes_by_skill(seen: dict[str, WorkloadItem]) -> dict[str, set[str]]:
    cached: dict[str, set[str]] = {}
    for item in seen.values():
        for skill in item.skill_ids:
            cached.setdefault(skill, set()).add(item.query_hash)
    return cached


def _choose_drift_skills(
    workload: list[WorkloadItem],
    position: int,
    seen: dict[str, WorkloadItem],
    skills_per_event: int,
) -> list[str]:
    cached = _cached_query_hashes_by_skill(seen)
    future = _future_query_hashes_by_skill(workload, position)
    scored: list[tuple[int, int, str]] = []
    for skill, cached_hashes in cached.items():
        future_reuses = len(cached_hashes & future.get(skill, set()))
        if future_reuses:
            scored.append((future_reuses, len(cached_hashes), skill))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [skill for _, _, skill in scored[:skills_per_event]]


def _run_drift_event(cache: Any, skills: list[str], event_index: int) -> dict[str, Any]:
    reason = f"ablation_drift_event_{event_index}"
    affected_agent_b = 0
    affected_recommendations = 0
    for skill in skills:
        affected_agent_b += cache.invalidate_skill(skill, ("b",), reason=reason).get("b", 0)
        affected_recommendations += cache.invalidate_recommendations_touching_skill(
            skill, reason=reason
        )
    return {
        "event_index": event_index,
        "skills": skills,
        "agent_b_rows_invalidated": affected_agent_b,
        "recommendation_rows_invalidated": affected_recommendations,
        "total_rows_invalidated": affected_agent_b + affected_recommendations,
    }


def _invalidation_log_summary(cache: Any) -> dict[str, Any]:
    with cache._conn() as conn:
        rows = conn.execute(
            "SELECT cache_table, COUNT(*) AS log_rows, "
            "SUM(rows_affected) AS rows_affected "
            "FROM cache_invalidations GROUP BY cache_table"
        ).fetchall()
    by_table = {
        row["cache_table"]: {
            "log_rows": row["log_rows"],
            "rows_affected": row["rows_affected"] or 0,
        }
        for row in rows
    }
    return {
        "log_rows": sum(v["log_rows"] for v in by_table.values()),
        "rows_affected": sum(v["rows_affected"] for v in by_table.values()),
        "by_table": by_table,
    }


def _run_no_cache(
    workload: list[WorkloadItem],
    miss_latency_ms: float,
    llm_cost_usd: float,
) -> dict[str, Any]:
    total = len(workload)
    latencies = [float(miss_latency_ms)] * total
    return {
        "description": "No recommendation cache; every request recomputes retrieval, generation, and grounding.",
        "requests": total,
        "hits": 0,
        "misses": total,
        "hit_rate": 0.0,
        "miss_rate": 1.0,
        "total_llm_calls": total,
        "llm_calls_avoided": 0,
        "cost_estimate_usd": round(total * llm_cost_usd, 6),
        "latency_ms": _latency_summary(latencies),
        "cache_hit_latency_ms": _latency_summary([]),
        "cache_miss_read_latency_ms": _latency_summary([]),
        "cache_write_latency_ms": _latency_summary([]),
        "drift_invalidations_triggered": 0,
        "drift_invalidations": {
            "events": 0,
            "recommendation_rows_invalidated": 0,
            "agent_b_rows_invalidated": 0,
            "total_rows_invalidated": 0,
            "log_summary": {"log_rows": 0, "rows_affected": 0, "by_table": {}},
        },
        "active_recommendations_end": 0,
        "freshness": {
            "would_have_fired_drift_events": 0,
            "would_have_fired_event_details": [],
            "fresh_hits": 0,
            "served_stale_hits": 0,
            "served_staleness_rate": 0.0,
        },
    }


def _run_cache_mode(
    mode: str,
    workload: list[WorkloadItem],
    skill_ids: list[str],
    miss_latency_ms: float,
    llm_cost_usd: float,
    skills_per_drift_event: int,
) -> dict[str, Any]:
    from src.cache import CacheLayer

    db = Path(tempfile.mkdtemp(prefix=f"curia_cache_{mode}_")) / "cache.db"
    cache = CacheLayer(db)
    seeded_agent_b = _seed_agent_b(cache, skill_ids)

    total = len(workload)
    event_positions = {
        int(total * fraction)
        for fraction in (0.25, 0.50, 0.75)
        if 0 < int(total * fraction) < total
    }
    hits = 0
    misses = 0
    seen: dict[str, WorkloadItem] = {}
    latencies: list[float] = []
    hit_read_latencies: list[float] = []
    miss_read_latencies: list[float] = []
    write_latencies: list[float] = []
    drift_events: list[dict[str, Any]] = []

    # Served-staleness tracking — for every mode, record:
    #   - when each entry was (re-)cached (position in the workload),
    #   - which skills any would-have-fired drift event has touched since.
    # A cache hit is "served stale" if at least one of its skills was touched
    # by a would-have-fired drift event AFTER the entry was last cached.
    # For drift_cascade this should be 0 by construction (invalidation forces
    # re-cache post-drift); for ttl_only it quantifies the freshness cost the
    # cascade is paying for.
    entry_cached_at: dict[str, int] = {}
    skill_last_drifted_at: dict[str, int] = {}
    would_have_fired_events: list[dict[str, Any]] = []
    served_stale_hits = 0
    fresh_hits = 0

    for idx, item in enumerate(workload):
        if idx in event_positions:
            event_skills = _choose_drift_skills(
                workload, idx, seen, skills_per_event=skills_per_drift_event
            )
            if event_skills:
                event_index = len(would_have_fired_events) + 1
                would_have_fired_events.append({
                    "event_index": event_index,
                    "position": idx,
                    "skills": event_skills,
                })
                for skill in event_skills:
                    skill_last_drifted_at[skill] = idx
                if mode == "drift_cascade":
                    drift_events.append(_run_drift_event(cache, event_skills, event_index))

        read_start = time.perf_counter()
        cached = cache.get_recommendation(item.query_hash)
        read_ms = _ms(time.perf_counter() - read_start)

        if cached is not None:
            hits += 1
            hit_read_latencies.append(read_ms)
            latencies.append(read_ms)
            entry_pos = entry_cached_at.get(item.query_hash, idx)
            served_stale = any(
                skill_last_drifted_at.get(skill, -1) > entry_pos
                for skill in item.skill_ids
            )
            if served_stale:
                served_stale_hits += 1
            else:
                fresh_hits += 1
            continue

        misses += 1
        miss_read_latencies.append(read_ms)
        payload = _recommendation_payload(item)
        write_start = time.perf_counter()
        cache.set_recommendation(
            query_hash=item.query_hash,
            normalized_query={"query_hash": item.query_hash, "skills": list(item.skill_ids)},
            recommendation=payload,
            evidence_ids=payload["evidence_ids"],
            llm_model="logical-cache-ablation",
            citation_check_ok=True,
        )
        cache.link_recommendation_skills(item.query_hash, item.skill_ids)
        write_ms = _ms(time.perf_counter() - write_start)
        write_latencies.append(write_ms)
        latencies.append(read_ms + write_ms + miss_latency_ms)
        seen[item.query_hash] = item
        entry_cached_at[item.query_hash] = idx

    stats = cache.stats()
    log_summary = _invalidation_log_summary(cache)
    total_llm_calls = misses
    rec_invalidated = sum(e["recommendation_rows_invalidated"] for e in drift_events)
    agent_b_invalidated = sum(e["agent_b_rows_invalidated"] for e in drift_events)
    served_staleness_rate = round(served_stale_hits / hits, 4) if hits else 0.0
    return {
        "description": (
            "TTL recommendation cache only; entries expire by age, with no drift invalidation."
            if mode == "ttl_only"
            else "TTL recommendation cache plus skill-drift invalidation cascade."
        ),
        "db_size_bytes": db.stat().st_size,
        "seeded_agent_b_rows": seeded_agent_b,
        "requests": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4),
        "miss_rate": round(misses / total, 4),
        "total_llm_calls": total_llm_calls,
        "llm_calls_avoided": total - total_llm_calls,
        "cost_estimate_usd": round(total_llm_calls * llm_cost_usd, 6),
        "latency_ms": _latency_summary(latencies),
        "cache_hit_latency_ms": _latency_summary(hit_read_latencies),
        "cache_miss_read_latency_ms": _latency_summary(miss_read_latencies),
        "cache_write_latency_ms": _latency_summary(write_latencies),
        "drift_invalidations_triggered": len(drift_events),
        "drift_invalidations": {
            "events": len(drift_events),
            "event_details": drift_events,
            "recommendation_rows_invalidated": rec_invalidated,
            "agent_b_rows_invalidated": agent_b_invalidated,
            "total_rows_invalidated": rec_invalidated + agent_b_invalidated,
            "log_summary": log_summary,
        },
        "active_recommendations_end": stats["recommendation_entries"],
        "recommendation_total_hits_recorded": stats["recommendation_total_hits"],
        "freshness": {
            "would_have_fired_drift_events": len(would_have_fired_events),
            "would_have_fired_event_details": would_have_fired_events,
            "fresh_hits": fresh_hits,
            "served_stale_hits": served_stale_hits,
            "served_staleness_rate": served_staleness_rate,
        },
    }


def _delta(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_calls = base["total_llm_calls"]
    candidate_calls = candidate["total_llm_calls"]
    base_cost = base["cost_estimate_usd"]
    candidate_cost = candidate["cost_estimate_usd"]
    base_stale = base.get("freshness", {}).get("served_staleness_rate", 0.0)
    candidate_stale = candidate.get("freshness", {}).get("served_staleness_rate", 0.0)
    return {
        "llm_call_delta": candidate_calls - base_calls,
        "llm_call_reduction_pct": (
            round(100.0 * (base_calls - candidate_calls) / base_calls, 2)
            if base_calls
            else None
        ),
        "cost_delta_usd": round(candidate_cost - base_cost, 6),
        "cost_reduction_pct": (
            round(100.0 * (base_cost - candidate_cost) / base_cost, 2)
            if base_cost
            else None
        ),
        "hit_rate_delta": round(candidate["hit_rate"] - base["hit_rate"], 4),
        "served_staleness_rate_delta": round(candidate_stale - base_stale, 4),
    }


def run_ablation(
    n_unique: int = 100,
    repeats: int = 10,
    n_skills: int = 20,
    seed: int = 42,
    miss_latency_ms: float = 250.0,
    llm_cost_usd: float = 0.01,
    skills_per_drift_event: int = 5,
) -> dict[str, Any]:
    workload, skills = _build_workload(n_unique, repeats, n_skills, seed)
    modes = {
        "no_cache": _run_no_cache(workload, miss_latency_ms, llm_cost_usd),
        "ttl_only": _run_cache_mode(
            "ttl_only", workload, skills, miss_latency_ms, llm_cost_usd, skills_per_drift_event
        ),
        "drift_cascade": _run_cache_mode(
            "drift_cascade",
            workload,
            skills,
            miss_latency_ms,
            llm_cost_usd,
            skills_per_drift_event,
        ),
    }
    return {
        "generated_at": _now_iso(),
        "benchmark": {
            "name": "cache_policy_ablation",
            "workload": f"{n_unique} unique x {repeats} repeats = {len(workload)} queries (shuffled)",
            "n_unique": n_unique,
            "repeats": repeats,
            "n_skills": n_skills,
            "rng_seed": seed,
            "cache_backend": "src.cache.CacheLayer SQLite",
            "paid_llm_calls_executed": False,
            "miss_latency_ms_assumption": miss_latency_ms,
            "llm_cost_usd_per_miss_assumption": llm_cost_usd,
            "drift_events_planned": 3,
            "skills_per_drift_event": skills_per_drift_event,
        },
        "modes": modes,
        "deltas": {
            "ttl_only_vs_no_cache": _delta(modes["no_cache"], modes["ttl_only"]),
            "drift_cascade_vs_no_cache": _delta(modes["no_cache"], modes["drift_cascade"]),
            "drift_cascade_vs_ttl_only": _delta(modes["ttl_only"], modes["drift_cascade"]),
        },
    }


def _fmt_ms(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _fmt_cost(value: float) -> str:
    return f"${value:.2f}"


def render_paper_section(report: dict[str, Any]) -> str:
    modes = report["modes"]
    rows = []
    for mode_id, label in (
        ("no_cache", "No cache"),
        ("ttl_only", "TTL-only cache"),
        ("drift_cascade", "Drift-cascaded cache"),
    ):
        mode = modes[mode_id]
        rows.append(
            "| {label} | {hit_rate:.3f} | {calls} | {avoided} | {hit_p95} | "
            "{e2e_p95} | {cost} | {drift_rows} | {stale_rate:.3f} |".format(
                label=label,
                hit_rate=mode["hit_rate"],
                calls=mode["total_llm_calls"],
                avoided=mode["llm_calls_avoided"],
                hit_p95=_fmt_ms(mode["cache_hit_latency_ms"]["p95"]),
                e2e_p95=_fmt_ms(mode["latency_ms"]["p95"]),
                cost=_fmt_cost(mode["cost_estimate_usd"]),
                drift_rows=mode["drift_invalidations"]["total_rows_invalidated"],
                stale_rate=mode["freshness"]["served_staleness_rate"],
            )
        )

    workload = report["benchmark"]["workload"]
    ttl_delta = report["deltas"]["ttl_only_vs_no_cache"]
    drift_delta = report["deltas"]["drift_cascade_vs_no_cache"]
    miss_latency = report["benchmark"]["miss_latency_ms_assumption"]
    cost = report["benchmark"]["llm_cost_usd_per_miss_assumption"]

    return f"""# Drift-Cascaded Recommendation Cache

This section is generated from `results/headline_cache_ablation.json`.

## Algorithm

The system maintains a recommendation cache keyed by a normalized learner query.
Each recommendation is stored only after citation validation succeeds, and each
entry is linked to the skill identifiers used to produce it. A detected skill
drift event invalidates both the skill forecast layer and all recommendations
that depend on the drifted skill.

```text
Algorithm: Drift-Cascaded Recommendation Cache
Input: learner query q, extracted skills S(q), evidence retriever R, generator G
State: recommendation cache C, skill forecast cache F, query-skill links L

1. h <- stable_hash(normalize(q))
2. if C[h] exists and C[h].expires_at > now:
       return C[h]
3. E <- R(q)
4. y <- G(q, E)
5. if citation_check(y, E) fails:
       return y without caching
6. C[h] <- y with TTL
7. for each skill s in S(q):
       L.add(h, s)
8. return y

OnDrift(skills D):
1. for each skill s in D:
       delete F[s, *]
       for each query hash h in L where L[h] contains s:
           delete C[h]
```

## Invariants

- Citation invariant: a generated recommendation is cached only when the
  citation check passes.
- TTL invariant: expired recommendations are never returned as hits.
- Drift cascade invariant: when a skill drifts, stale downstream forecasts and
  recommendations linked to that skill are invalidated before reuse.
- Cost invariant: only cache misses invoke retrieval, generation, and grounding.

## Complexity

Expected recommendation lookup is O(1) by primary-key query hash. Cache insertion
is O(s), where s is the number of linked skills for the query. Drift invalidation
is O(k + r), where k is the number of drifted skills and r is the number of
cached recommendations linked to those skills.

## Ablation

The ablation used `{workload}`. No paid LLM calls were executed during the
ablation; each cache miss is counted as one logical retrieval/generation/
grounding call with an explicit latency assumption of {miss_latency:.1f} ms and
a cost assumption of ${cost:.4f} per miss.

| Policy | Hit rate | LLM calls | Calls avoided | Hit p95 ms | End-to-end p95 ms | Estimated cost | Drift rows invalidated | Served-staleness rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

TTL-only caching reduced logical LLM calls by {ttl_delta["llm_call_reduction_pct"]:.2f}%
relative to no cache, but served {modes["ttl_only"]["freshness"]["served_stale_hits"]}
recommendations from entries whose dependent skills had drifted
(served-staleness rate {modes["ttl_only"]["freshness"]["served_staleness_rate"]:.3f}).
The drift-cascaded cache still reduced calls by
{drift_delta["llm_call_reduction_pct"]:.2f}% relative to no cache while explicitly
invalidating stale skill-dependent recommendations and forecasts; its
served-staleness rate is {modes["drift_cascade"]["freshness"]["served_staleness_rate"]:.3f}
by construction.

The drift-cascade trade-off relative to TTL-only is therefore explicit:
{abs(report["deltas"]["drift_cascade_vs_ttl_only"]["hit_rate_delta"]) * 100:.1f}pp
lower hit rate and ${abs(report["deltas"]["drift_cascade_vs_ttl_only"]["cost_delta_usd"]):.2f}
higher cost per 1{','}000 queries, in exchange for eliminating served staleness on
skill-drifted recommendations.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/headline_cache_ablation.json")
    parser.add_argument("--paper-out", default="results/cache_velocity_algorithm_section.md")
    parser.add_argument("--n-unique", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-skills", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--miss-latency-ms", type=float, default=250.0)
    parser.add_argument("--llm-cost-usd", type=float, default=0.01)
    parser.add_argument("--skills-per-drift-event", type=int, default=5)
    args = parser.parse_args()

    report = run_ablation(
        n_unique=args.n_unique,
        repeats=args.repeats,
        n_skills=args.n_skills,
        seed=args.seed,
        miss_latency_ms=args.miss_latency_ms,
        llm_cost_usd=args.llm_cost_usd,
        skills_per_drift_event=args.skills_per_drift_event,
    )
    text = json.dumps(report, indent=2)
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")

    paper_out = Path(args.paper_out)
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    paper_out.write_text(render_paper_section(report), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"wrote {paper_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
