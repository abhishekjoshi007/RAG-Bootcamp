# Paper 1 Draft Scaffold: Drift-Cascaded Recommendation Caching for Auditable Curriculum RAG

Status: working outline for Path C Paper 1. This draft is shaped around the
evidence currently in:

- `results/headline_cache_ablation.json`
- `results/cache_velocity_algorithm_section.md`
- `results/headline_llamaindex_baseline.json`
- `results/headline_multi_llm_50q_17models_rechecked.json`
- `results/multi_llm_50q_17models_ieee_table.tex`
- `results/headline_scorecard.json`

Submission note: the current LlamaIndex artifact records `n_corpus_docs: 182`.
Do not claim the 3,375-document large-corpus LlamaIndex baseline unless that
baseline is rerun with `--corpus-dir data/corpus_large` or the artifact is
otherwise corrected.

## Candidate Title

Drift-Cascaded Recommendation Caching for Fast, Auditable Curriculum RAG

Alternative:

Freshness-Aware Recommendation Caching for Citation-Audited Curriculum RAG

## Abstract Draft

Retrieval-augmented generation systems are increasingly used to summarize
technical evidence into recommendations, but production use exposes two
undermeasured system problems: repeated queries make uncached generation
needlessly slow and expensive, while cached recommendations can become stale
when the underlying skill landscape drifts. We present CURIA, a curriculum RAG
system with a drift-cascaded recommendation cache that links generated
recommendations to the skills and evidence that produced them. When a skill
drift event fires, CURIA invalidates both downstream forecasts and all cached
recommendations linked to the drifted skill. We evaluate the design on a
1,000-query workload and introduce served-staleness, the fraction of cache hits
that would return a recommendation generated before a relevant drift event.
TTL-only caching reaches a 0.900 hit rate and reduces projected cost to $1.00
per 1,000 queries, but serves stale recommendations on 41.6% of hits. The
drift-cascaded cache lowers the hit rate to 0.783 and raises projected cost to
$2.17, but eliminates served-staleness. In a system-level comparison against
LlamaIndex using the same generator, LlamaIndex produces no inline source-ID
citations across two embedding configurations, while CURIA's structured output
contract and post-hoc citation checks produce auditable recommendations. These
results show that citation-audited recommendation systems need freshness-aware
cache invalidation, not only faster retrieval or stronger LLMs.

## One-Sentence Claim

CURIA shows that the useful systems contribution is not caching alone; it is a
freshness-aware cache contract that trades modest hit-rate loss for zero
served-staleness while preserving citation-auditable outputs.

## Contributions

1. We define served-staleness as a cache freshness metric for recommendation
   RAG: the fraction of cache hits generated before a subsequent relevant skill
   drift event.
2. We introduce the Drift-Cascaded Recommendation Cache, which links cached
   recommendations to skill IDs and invalidates forecasts plus recommendations
   when skill drift is detected.
3. We provide a cache-policy ablation over a 1,000-query workload showing that
   TTL-only caching reaches 0.900 hit rate but 0.416 served-staleness, while
   drift-cascade reaches 0.783 hit rate and 0.000 served-staleness.
4. We compare CURIA against a LlamaIndex RAG baseline and show that off-the-shelf
   framework RAG does not produce auditable inline source-ID citations under the
   tested prompt, whereas CURIA enforces citations structurally.
5. We release reproducible scripts and paper artifacts for the ablation,
   LlamaIndex baseline, and 17-model faithfulness table.

## Introduction Draft

Curriculum recommendations require a stricter form of RAG than ordinary
question answering. A system must identify which skills are gaining or losing
importance, retrieve evidence about those skills, explain the recommended
curriculum change, and provide citations that can be audited by instructors or
program committees. This makes the system problem different from generic RAG:
latency and cost matter because curriculum planners may repeatedly query the
same program areas, but freshness also matters because cached recommendations
can become stale after skill-demand drift.

The obvious engineering answer is to cache generated recommendations. However,
ordinary TTL caching only answers one question: "is this entry old?" It does not
answer the curriculum-specific question: "has the skill evidence that justified
this recommendation changed since the recommendation was generated?" In our
setting, this distinction matters. A cached answer can still be inside its TTL
while serving obsolete skill semantics after a drift event.

We address this with a drift-cascaded recommendation cache. CURIA stores each
generated recommendation under a normalized query hash, but it also records the
skills linked to that recommendation. When the drift detector fires for a skill,
the system invalidates the affected forecast rows and all cached
recommendations linked to that skill. This design turns freshness from an
implicit age heuristic into an explicit dependency contract.

We evaluate the contract with a cache ablation across three policies:
no-cache, TTL-only cache, and drift-cascaded cache. The result is not that
drift-cascade maximizes hit rate. It does not. TTL-only caching reaches a 0.900
hit rate and costs $1.00 per 1,000 queries under the miss-cost assumption. The
drift-cascaded policy reaches a lower 0.783 hit rate and costs $2.17 because it
deliberately recomputes recommendations after drift. The key result is
freshness: TTL-only caching serves stale recommendations on 41.6% of hits, while
drift-cascade reduces served-staleness to 0.0%.

We also compare CURIA with LlamaIndex as a framework-level baseline. Using the
same generator and benchmark units, the tested LlamaIndex configurations
produce no inline source-ID citations, making their apparent 1.000 citation
precision vacuous: there are no citations to audit. CURIA instead forces a
structured JSON output with mandatory `SOURCE_ID` fields and rejects uncited or
unsupported outputs through post-hoc citation validation. This finding shifts
the story away from "which LLM is best?" and toward the system architecture
needed for auditable recommendation RAG.

## Research Questions

RQ1: Does a drift-cascaded recommendation cache reduce served-staleness relative
to TTL-only caching?

RQ2: What latency and cost tradeoff does drift-cascade impose relative to
TTL-only caching and no-cache generation?

RQ3: Do off-the-shelf RAG framework outputs satisfy the citation audit contract
needed for curriculum recommendations?

RQ4: When all models receive retrieved evidence, what remains as the useful
quality discriminator: citation precision or evidence coverage?

## Method Section Outline

### System

- Ingestion and indexing: curriculum and technical evidence are chunked,
  embedded, and retrieved for each curriculum unit.
- Generation: recommendations are generated as structured JSON with mandatory
  evidence IDs.
- Grounding: every cited ID must be a subset of the retrieved evidence IDs.
- Caching: recommendation cache uses normalized query hashes and stores
  query-to-skill links.
- Drift cascade: skill drift invalidates forecast rows and linked
  recommendations.

### Algorithm

Use `results/cache_velocity_algorithm_section.md` as the source for the
algorithm block, invariants, and complexity section.

### Metric: Served-Staleness

Define a cache hit as stale if the returned recommendation was cached before a
drift event touching any skill linked to that recommendation.

Formula:

```text
served_staleness = served_stale_hits / all_cache_hits
```

Interpretation:

- TTL-only can have high hit rate and high served-staleness.
- Drift-cascade should have zero served-staleness by construction, assuming
  drift events are detected and propagated before subsequent reads.

## Results Tables To Include

### Table 1: Cache Policy Ablation

Source: `results/headline_cache_ablation.json`

| Policy | Hit rate | Cost / 1k | Served-staleness | LLM calls | Drift rows |
|---|---:|---:|---:|---:|---:|
| No cache | 0.000 | $10.00 | 0.000 | 1,000 | 0 |
| TTL-only | 0.900 | $1.00 | 0.416 | 100 | 0 |
| Drift-cascade | 0.783 | $2.17 | 0.000 | 217 | 145 |

Main takeaway: drift-cascade spends 117 extra recomputes per 1,000 queries
relative to TTL-only, but eliminates a 41.6% served-staleness rate.

### Table 2: LlamaIndex Baseline

Source: `results/headline_llamaindex_baseline.json`

Current artifact reports `n_corpus_docs: 182`; update this table if rerun on
the large corpus.

| System | Embedding | Evidence coverage | Mean latency | Projected cost / 1k |
|---|---|---:|---:|---:|
| LlamaIndex default | `text-embedding-3-small` | 0.000 | 3.476 s | $0.235 |
| LlamaIndex matched | `all-mpnet-base-v2` | 0.000 | 4.109 s | $0.281 |
| CURIA local baseline | `all-mpnet-base-v2` | 0.535 | ~0.5 ms hit path | ~$0.06 at full-hit projection |

Main takeaway: the tested LlamaIndex path produces no auditable inline source-ID
citations, so citation precision is not a meaningful win condition for it.

### Table 3: Multi-LLM Faithfulness Summary

Source:

- `results/headline_multi_llm_50q_17models_rechecked.json`
- `results/multi_llm_50q_17models_ieee_table.tex`

Use this table to support the claim that citation precision saturates once
structural citation checking is enforced. Evidence coverage, not citation
precision, discriminates model quality.

## Discussion Points

- TTL-only caching is cheaper and faster on stable workloads, but it is not
  freshness-aware.
- Drift-cascade is the right policy when recommendations depend on dynamic skill
  evidence.
- The local extractive baseline is a speed and faithfulness ceiling, but a
  quality/coverage floor.
- LlamaIndex is a strong baseline for building RAG quickly, but the tested
  configuration does not enforce an auditable citation contract.
- The contribution should be framed as a systems contract: dependency tracking,
  invalidation, and citation validation.

## Limitations

- The cache ablation uses simulated miss latency and miss cost rather than
  executing paid LLM calls on every miss.
- Served-staleness assumes drift events are correctly detected; the metric
  evaluates propagation and cache behavior, not drift detector recall.
- The current LlamaIndex headline artifact records 182 indexed documents. Rerun
  on `data/corpus_large` before claiming the large-corpus system baseline.
- LlamaIndex should be rerun with a stricter JSON citation prompt for a
  camera-ready version.
- This short paper should not headline BLS correlation, forecasting, or human
  relevance results; those belong to Paper 2 after data and label scale-up.

## Venue Framing

Best fit:

- CIKM short paper: systems contribution, cache freshness metric, auditable RAG.
- IEEE BigData applied research: reproducible RAG system with operational
  latency/cost/freshness measurements.
- Workshop version: emphasize practical deployment lessons for curriculum RAG.

Avoid framing this as a generic multi-agent RAG paper. The novelty should be:
served-staleness plus drift-cascaded invalidation for citation-audited
recommendation RAG.

## Camera-Ready Checklist

- Rerun LlamaIndex baseline on `data/corpus_large` if the paper claims the
  3,375-document corpus.
- Add a stricter JSON-output LlamaIndex condition.
- Optionally rerun cache ablation with real miss-path LLM calls on a smaller
  workload to validate the simulated miss assumptions.
- Convert Table 1 and Table 2 into IEEE LaTeX.
- Move Gemini failure rows from the 17-model table to appendix or footnote them
  clearly.
- Keep forecasting and BLS results out of the main Path C Paper 1 claim.
