# Drift-Cascaded Recommendation Cache

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

The ablation used `100 unique x 10 repeats = 1000 queries (shuffled)`. No paid LLM calls were executed during the
ablation; each cache miss is counted as one logical retrieval/generation/
grounding call with an explicit latency assumption of 250.0 ms and
a cost assumption of $0.0100 per miss.

| Policy | Hit rate | LLM calls | Calls avoided | Hit p95 ms | End-to-end p95 ms | Estimated cost | Drift rows invalidated | Served-staleness rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No cache | 0.000 | 1000 | 0 | NA | 250.000 | $10.00 | 0 | 0.000 |
| TTL-only cache | 0.900 | 100 | 900 | 3.406 | 251.367 | $1.00 | 0 | 0.416 |
| Drift-cascaded cache | 0.783 | 217 | 783 | 1.842 | 252.559 | $2.17 | 145 | 0.000 |

TTL-only caching reduced logical LLM calls by 90.00%
relative to no cache, but served 374
recommendations from entries whose dependent skills had drifted
(served-staleness rate 0.416).
The drift-cascaded cache still reduced calls by
78.30% relative to no cache while explicitly
invalidating stale skill-dependent recommendations and forecasts; its
served-staleness rate is 0.000
by construction.

The drift-cascade trade-off relative to TTL-only is therefore explicit:
11.7pp
lower hit rate and $1.17
higher cost per 1,000 queries, in exchange for eliminating served staleness on
skill-drifted recommendations.
