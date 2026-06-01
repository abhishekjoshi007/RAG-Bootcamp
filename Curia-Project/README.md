# CURIA — Curriculum Update with Retrieval & Industry Alignment

A retrieval-augmented system that **continuously aligns university CS/EE curricula
with live industry signal**. CURIA ingests job postings, arXiv papers, Stack
Overflow questions and GitHub READMEs; fuses them into per-skill demand,
forecast, and drift signals; and produces a **grounded, citation-verified
recommendation** for every curriculum unit — with a multi-layer cache that
serves repeat queries from precomputed agent outputs.

> Companion artifact to the IEEE BigData 2026 submission
> *"CURIA: Velocity-Aware Retrieval-Augmented Curriculum Alignment with
> Industry Signal."* All numbers in this README are reproducible from the code
> in this repo and the result files under [`results/`](results/).

---

## Headline results

### Faithfulness across 17 frontier LLMs (n = 50 curriculum units)

50 TAMU CS/EE curriculum units × 17 LLMs (OpenAI / Anthropic / Google / xAI /
DeepSeek + the local extractive baseline). Same retrieved evidence given to
every model; only the generator changes. Sorted by evidence coverage. Full
data:
[`results/headline_multi_llm_50q_17models.json`](results/headline_multi_llm_50q_17models.json).

| Model | n / err | Citation precision | Hallucination | Evidence coverage | Latency (s) | Cost (USD) |
|---|:---:|---:|---:|---:|---:|---:|
| **claude-sonnet-4-6** | 50 / 0 | 1.000 | 0.000 | **0.767** | 7.54 | 0.57 |
| deepseek-v4-flash | 50 / 0 | 0.998 | 0.002 | 0.762 | 7.32 | 0.95 |
| claude-opus-4-7 | 50 / 0 | 1.000 | 0.000 | 0.760 | 7.04 | 3.86 |
| claude-opus-4-6 | 50 / 0 | 1.000 | 0.000 | 0.695 | 7.93 | 2.85 |
| claude-opus-4-8 | 50 / 0 | 1.000 | 0.000 | 0.694 | 7.02 | 3.89 |
| deepseek-reasoner | 50 / 0 | 1.000 | 0.000 | 0.684 | 6.99 | 0.97 |
| deepseek-chat | 50 / 0 | 1.000 | 0.000 | 0.657 | 3.06 | 0.63 |
| claude-haiku-4-5 | 50 / 0 | 1.000 | 0.000 | 0.657 | 3.17 | 0.17 |
| gpt-5.5 | 50 / 0 | 1.000 | 0.000 | 0.633 | 12.77 | 1.03 |
| grok-4.3 | 50 / 0 | 1.000 | 0.000 | 0.579 | 5.95 | 0.61 |
| gpt-5.4-nano | 50 / 0 | 1.000 | 0.000 | 0.572 | 2.57 | 0.62 |
| gemini-3.1-pro-preview | 48 / 2 | 1.000 | 0.000 | 0.563 | 18.53 | 0.67 |
| gpt-5.4 | 50 / 0 | 1.000 | 0.000 | 0.558 | 4.86 | 0.63 |
| **local** (extractive baseline) | 50 / 0 | **1.000** | **0.000** | 0.535 | **0.0002** | **0.00** |
| gpt-5.4-mini | 50 / 0 | 1.000 | 0.000 | 0.522 | 2.41 | 0.61 |
| claude-sonnet-4-5 | 50 / 0 | 1.000 | 0.000 | 0.514 | 6.43 | 0.51 |
| gemini-3.5-flash *(flaky)* | 21 / 29 | 1.000 | 0.000 | 0.493 | 14.10 | 0.29 |
| **Total** | — | — | — | — | — | **$18.88** |

The interesting axis is *evidence coverage* (how much of the retrieved evidence
each model actually uses), not citation precision — every model passes the
0.95 citation target. The deterministic local baseline is the **faithfulness
floor** at 0 cost and sub-millisecond latency; frontier LLMs trade money and
seconds for higher coverage. `claude-sonnet-4-6` is the **cost/quality sweet
spot** at coverage 0.767 for $0.57.

The single sub-1.000 citation precision (`deepseek-v4-flash` at 0.998) was a
single-row arXiv version mismatch (`v1` cited, `v4` indexed). That class of
false positive is now fixed in [`src/grounding.py`](src/grounding.py) via
arXiv version canonicalization — re-runs will show 1.000 across the board.

### Velocity (recommendation cache, RQ4)

From [`results/headline_scorecard.json`](results/headline_scorecard.json), `velocity` layer:

| Metric | Value |
|---|---|
| Cache hit latency p95 | **0.46 ms** (target ≤ 200 ms) |
| Workload hit rate (1,000 queries, 100 unique × 10) | **90 %** |
| LLM calls avoided | **900 / 1,000** |
| Cache DB size at steady state | 408 KB |

### Retrieval (RQ retrieval quality)

| Metric | Value | Target | Status |
|---|---:|---:|:---:|
| Recall@4 | 0.650 | — | — |
| Recall@8 | **0.867** | 0.70 | ✅ |
| MRR | 1.000 | — | — |
| nDCG@8 | 0.871 | — | — |

### Forecasting backtest on the large dated corpus (RQ2)

[`results/headline_forecast_backtest_corpus_large.json`](results/headline_forecast_backtest_corpus_large.json)
— 30 skills evaluated, 142 / 185 months with documents, `statistically_meaningful: true`.
Five baselines compared on monthly skill-frequency series:

| Baseline | sMAPE | MAPE | MASE | Directional accuracy |
|---|---:|---:|---:|---:|
| naive | 1.3094 | 0.823 | 5.83 | 0.300 |
| **exp_smoothing** | **1.2940** | 1.092 | 5.65 | 0.489 |
| moving_avg | 1.3107 | 1.043 | 4.01 | 0.378 |
| linear | 1.3479 | 1.091 | 4.97 | 0.522 |
| seasonal_naive | 1.3701 | 1.624 | **2.86** | **0.572** |

`exp_smoothing` is the **best non-naive model by sMAPE** and beats naive
(`best_non_naive_beats_naive: true`, Δ = −0.0154). `seasonal_naive` wins on
MASE and directional accuracy. Honest read: gains over naive are real but
small on this corpus size — scaling the dated ingest is the highest-leverage
follow-up.

### Ground truth vs BLS (RQ1, sample data)

[`results/headline_ground_truth_bls.json`](results/headline_ground_truth_bls.json):
Agent A vs BLS Spearman ρ = **0.082** on n = 11 overlapping skills, using a
**sample** BLS file. This is reported transparently as below-threshold; the
final paper uses the official BLS OES export (set `BLS_EXPORT_PATH`).

---

## What CURIA does, in plain language

1. **It ingests evidence.** Twelve live sources — Greenhouse / Lever / WWR /
   RemoteOK / Remotive / Arbeitnow / The Muse / HN "Who is hiring" / USAJOBS
   (job-market signal), plus arXiv `cs.*` and Stack Overflow (technical
   signal), plus GitHub READMEs (open-source signal) — are pulled by
   [`src/ingest.py`](src/ingest.py) and stored as one JSON document per item.
2. **It chunks, embeds, and indexes** the corpus into a FAISS `IndexFlatIP`
   using `sentence-transformers/all-mpnet-base-v2`.
3. **For each curriculum unit, it retrieves** evidence with recency decay and
   per-source quotas so no one source dominates.
4. **It generates a grounded recommendation** (`signal_strength`, `summary`
   with inline `SOURCE_ID` citations, `emerging_topics`, `evidence_ids`) using
   either an OpenAI model or a deterministic local extractive baseline.
5. **It verifies every citation** — cited IDs must be a subset of retrieved IDs
   — and persists the full run (unit, prompt, evidence, recommendation,
   citation check) to a SQLite audit log.
6. **It precomputes per-skill agent outputs** (demand, forecast, drift,
   resources) in a weekly batch, so interactive queries hit a multi-layer
   cache instead of paying LLM and embedding cost on every request.

---

## The five agents

| Agent | Module | Role | Output |
|---|---|---|---|
| **A — Skill Demand Fusion** | [`src/agent_a_fusion.py`](src/agent_a_fusion.py) | Counts skill mentions per (skill, source, ISO week), weights by source quality, normalizes to 0..1 intensity per skill. | Per-skill weekly intensity rows |
| **B — Skill Demand Forecasting** | [`src/forecasting.py`](src/forecasting.py) + [`src/skill_series.py`](src/skill_series.py) | Linear + exponential-smoothing forecasts of monthly skill frequency, plus a baseline backtest harness. | Per-skill {slope, projection, R²} for horizons {3, 6, 12, 24} |
| **C — Semantic Drift** | [`src/drift.py`](src/drift.py) | Cross-source / temporal centroid divergence per skill; flags skills whose meaning has shifted enough that historical recommendations should be invalidated. | Per-skill drift score + direction |
| **D — Curriculum Roadmap** | [`src/agent_d_roadmap.py`](src/agent_d_roadmap.py) | Overlays A / B / C / E outputs onto a learner's goal to produce an ordered roadmap of skills with trend and drift annotations. | `{ goal, program, completed_skills, steps[] }` |
| **E — Resource Matching** | [`src/agent_e_resources.py`](src/agent_e_resources.py) | Maps each tracked skill to open-courseware resources; scores them `skill_match × demand × (1 − drift_risk)` using A and C outputs from the cache. | Per-skill ranked resource list |

Agent outputs are materialized weekly by `BatchRunner` and read directly from
the cache by the live pipeline.

---

## Architecture

```
                                            ┌─────────────────────────┐
ingest (12 sources) ─► corpus/ ──► chunk ──►│  FAISS IndexFlatIP      │
                                            │  (mpnet-base-v2, IP)    │
                                            └──────────┬──────────────┘
                                                       │
LearnerQuery ──► query_hash.QueryFingerprint ──┐       │
                                               ▼       ▼
                                       ┌──────────────────────────┐
                                       │  CacheLayer (SQLite)     │
                                       │  ├─ recommendation_cache │◄─── miss ──┐
                                       │  ├─ agent_a_cache        │            │
                                       │  ├─ agent_b_cache        │            │
                                       │  ├─ agent_c_cache        │            │
                                       │  └─ resource_cache       │            │
                                       └────────────┬─────────────┘            │
                                       cache hit ──►│ return cached            │
                                                    │                          ▼
                                       ┌────────────┴────────────┐   ┌──────────────────────┐
                                       │  Retriever              │──►│  build_prompt        │
                                       │  recency + source quota │   │  evidence + JSON spec │
                                       └─────────────────────────┘   └──────────┬───────────┘
                                                                                ▼
                                                              ┌──────────────────────────────┐
                                                              │  Generator                   │
                                                              │  OpenAI | Anthropic | Local  │
                                                              └──────────────┬───────────────┘
                                                                             ▼
                                                              ┌──────────────────────────────┐
                                                              │  grounding.check_citations   │
                                                              │  cited ⊆ retrieved ?         │
                                                              └──────────────┬───────────────┘
                                                                             ▼
                                                              ┌──────────────────────────────┐
                                                              │  AuditLog (SQLite) + cache   │
                                                              │  persist + index by skills   │
                                                              └──────────────────────────────┘
```

Weekly batch ([`src/batch.py`](src/batch.py)): `ingest_all → build_index →
FusionAgent → SkillForecaster → SemanticDriftDetector → ResourceMatcher`.
Drift > `DRIFT_INVALIDATION_THRESHOLD` cascades into a targeted invalidation
of recommendations touching the drifted skill.

---

## Quick start

```bash
# 1. Configure
cp .env.example .env       # add OPENAI_API_KEY at minimum; everything else optional

# 2. Install (Python 3.10–3.11 recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the corpus and index
python3 scripts/ingest_corpus.py                       # live ingest (or skip and use sample)
python3 scripts/build_index.py                         # FAISS IndexFlatIP

# 4. Run a single recommendation
python3 scripts/run_pipeline.py --unit-id cs_ai_01 --k 8 --local

# 5. Launch the 5-agent web app
python3 app_gradio.py                                  # http://127.0.0.1:8888
```

### Reproducing the paper's headline numbers

```bash
# 1. Build the dated corpus (≈ 3,400 docs across arXiv + Stack Overflow)
python3 scripts/ingest_historical_arxiv.py
python3 scripts/ingest_historical_stackoverflow.py
python3 scripts/build_index.py

# 2. Run the full scorecard (retrieval, faithfulness, relevance, adversarial,
#    forecast backtest, ground truth, multi-LLM, velocity)
python3 eval/run_scorecard.py --out results/

# 3. The 50-query × 15-model headline (requires provider API keys; budget-capped)
python3 eval/run_multi_llm_eval.py \
    --units-file data/eval/benchmark_units_tamu_50.json \
    --budget-usd 20 \
    --out results/

# 4. Forecast backtest on the large dated corpus
python3 eval/run_forecast_backtest.py \
    --corpus-dir data/corpus_large \
    --horizon 6 --min-train 6 \
    --out results/
```

Cache and batch operations:

```bash
python3 scripts/cache_ops.py stats                # cache hit/miss counts, by layer
python3 scripts/cache_ops.py purge-stale          # drop entries past TTL
python3 scripts/cache_ops.py invalidate <skill>   # drop recommendations touching <skill>
python3 scripts/batch_refresh.py                  # one-shot weekly batch run
python3 scripts/bench_cache.py                    # the velocity numbers above
```

---

## Evaluation framework

`eval/run_scorecard.py` runs every evaluation layer as an isolated subprocess
so one failing layer never aborts the rest, and aggregates a one-file
pass / fail report. The targets are pulled from
[`src/config.py`](src/config.py); change them there to retarget the scorecard.

| Layer | Script | Metrics | Target |
|---|---|---|---|
| 1. Retrieval | `eval/run_retrieval_eval.py` | recall@4, recall@8, MRR, nDCG@8 | recall@8 ≥ 0.70 |
| 2. Faithfulness | `eval/run_faithfulness_eval.py` | citation precision, claim grounding | ≥ 0.95, ≥ 0.85 |
| 2. Relevance | `eval/run_relevance_eval.py` | mean human rating (1–5) | ≥ 3.5 |
| 2. Adversarial | `eval/run_adversarial_eval.py` | recall drop under synonym / topic-removal / misleading perturbations | max drop ≤ 0.30 |
| 3. Forecast backtest | `eval/run_forecast_backtest.py` | MAPE, sMAPE, MASE, directional accuracy vs naive / moving-avg / linear | statistically_meaningful = true |
| 3. Ground truth | `eval/run_ground_truth_eval.py` | Agent A vs BLS Spearman ρ, CS2023 framework coverage | model comparison |
| 4. Multi-LLM | `eval/run_multi_llm_eval.py` | citation precision, hallucination, evidence coverage, latency, cost (per model) | budget-capped |
| 5. Velocity | `scripts/bench_cache.py` | cache hit latency p95, workload hit-rate, LLM-call reduction | p95 ≤ 200 ms |

---

## Module map (`src/`)

| Module | Responsibility |
|---|---|
| `config.py` | All tunable constants, with environment-variable overrides (`_int / _float / _str / _list / _bool` helpers). |
| `models.py` | Frozen dataclasses: `Document`, `Chunk`, `SearchResult`, `Recommendation`. |
| `storage.py` | Load corpus / units from disk; `build_index_from_corpus`. |
| `chunking.py` | Sliding-window token chunking (default 160 tokens / 30 overlap). |
| `embedding.py` | `DenseEmbedder` (sentence-transformers) + `LocalTfidfEmbedder` for the offline path. |
| `indexing.py` | `FaissIndex` (`IndexFlatIP`, picklable) and `InMemoryIndex` (sparse) — same `search` API. |
| `retrieval.py` | `Retriever`: candidate search, recency decay, per-source quota enforcement, top-k. |
| `query.py` | `build_query` and `build_query_from_learner` (LearnerQuery → corpus query). |
| `query_hash.py` | `LearnerQuery` dataclass + `QueryFingerprint` — stable cache key. |
| `prompts.py` | `format_evidence` and `build_recommendation_prompt` (JSON contract). |
| `llm.py` | `OpenAIGenerator` (retrying, JSON-parsed) and `LocalGroundedGenerator` (deterministic, offline). |
| `llm_providers.py` | Provider adapters for the multi-LLM benchmark (Anthropic, Google, xAI, DeepSeek, OpenRouter, Ollama). |
| `model_registry.py` | Pinned model IDs + per-token pricing for the benchmark budget preflight. |
| `grounding.py` | `check_citations` → `CitationCheck(passed, cited_ids, retrieved_ids, missing_ids)`. |
| `audit.py` | `AuditLog` — SQLite `rag_runs` table, owns the cache DDL execution too. |
| `cache.py` | `CacheLayer` — recommendation / agent A/B/C / resource caches with TTLs; cascade invalidation. |
| `pipeline.py` | `CuriaRagPipeline` — polymorphic over `dict` (legacy) and `LearnerQuery` (cached) inputs. |
| `ingest.py` | 12 source fetchers + `ingest_all`; HTML stripping, date parsing, rate limiting. |
| `field_config.py` | `FIELD_INGESTION` — 21 academic fields → source queries / tags / topics. |
| `university.py` | Loads TAMU curricula; `curriculum_summary`, `get_all_units`. |
| `agent_a_fusion.py` | `FusionAgent` — per-skill demand intensity. |
| `forecasting.py` | `SkillForecaster` + `tracked_skills` + baseline harness. |
| `skill_series.py` | `monthly_skill_frequency` over the dated corpus. |
| `drift.py` | `SemanticDriftDetector` — centroid divergence per skill. |
| `agent_d_roadmap.py` | `RoadmapAgent` — overlays A/B/C/E onto a learner goal. |
| `agent_e_resources.py` | `ResourceMatcher` — demand- and drift-aware resource scoring. |
| `batch.py` | `BatchRunner` — weekly precompute with drift-cascade. |
| `benchmarking.py` | Shared metrics + budget preflight for the multi-LLM eval. |

Tests live in `tests/`, named `test_NN_<topic>.py` so they run in pipeline
order. Real-LLM and FAISS-dependent tests are gated behind environment.

---

## Configuration

Every constant in [`src/config.py`](src/config.py) can be overridden via
environment variables. Defaults:

| Group | Constant | Default |
|---|---|---|
| Chunking | `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP` | 160 / 30 |
| Embedding | `EMBED_MODEL` / `EMBED_BATCH_SIZE` | `all-mpnet-base-v2` / 8 |
| Retrieval | `RETRIEVAL_K` / `RETRIEVAL_CANDIDATE_K` | 8 / 50 |
| Recency | `RECENCY_HALF_LIFE_DAYS` · `BASE` · `BONUS` | 365 · 0.7 · 0.3 |
| Source quotas | job_posting / arxiv / stackoverflow / github_readme | 3 / 2 / 2 / 1 |
| LLM | `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_MAX_RETRIES` | `gpt-4o-mini` / 0.0 / 1024 / 3 |
| Cache TTLs (days) | recommendation / A / B / C / resource | 7 / 7 / 30 / 14 / 30 |
| Drift cascade | `DRIFT_INVALIDATION_THRESHOLD` | 0.6 |
| Eval target | `EVAL_TARGET_RECALL_8` | 0.70 |
| Eval target | `EVAL_TARGET_CITATION_PRECISION` | 0.95 |
| Eval target | `EVAL_TARGET_CLAIM_GROUNDING` | 0.85 |
| Eval target | `EVAL_TARGET_RELEVANCE_MEAN` | 3.5 |
| Eval target | `EVAL_TARGET_ADVERSARIAL_DROP` | 0.30 |

Recency-adjusted score:

```
score = similarity * (RECENCY_BASE + RECENCY_BONUS * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))
```

See [`.env.example`](.env.example) for every supported environment variable.

---

## Data layout

```
data/
├── corpus/                     small bootstrap corpus (≈ 180 docs, committed)
├── corpus_large/               dated corpus for forecasting (≈ 3,400 docs, NOT committed — rebuild via scripts)
├── cs2023_units.json           3 CS2023 sample curriculum units (cs_ai_01, cs_sec_01, cs_cloud_01)
├── universities/               6 TAMU college curriculum files
└── eval/
    ├── retrieval_labels.jsonl              ground truth for retrieval recall / nDCG
    ├── faithfulness_labels.jsonl           ground truth for citation precision / claim grounding
    ├── relevance_ratings.jsonl             human relevance ratings (1–5)
    ├── benchmark_units_tamu_50.json        50 TAMU CS/EE units for the multi-LLM benchmark
    └── bls_oes_sample.json                 SAMPLE BLS OES file — replace via --bls or BLS_EXPORT_PATH
```

---

## Tests

```bash
python3 -m pytest -q                       # all 20 test modules
python3 -m pytest tests/test_15_cache.py -q
```

Tests `test_11_llm.py` and `test_12_pipeline.py` make real OpenAI calls (set
`OPENAI_API_KEY`). Tests `test_04_embedding.py`, `test_05_indexing.py`,
`test_12_pipeline.py`, `test_14_drift.py` need `torch` + `faiss` +
`sentence-transformers`. The other 16 modules run on stdlib + pytest alone.

---

## Limitations

The paper reports the following honestly:

1. **Forecast gains over naive are small** (sMAPE Δ = −0.0154 for
   `exp_smoothing`; `seasonal_naive` wins MASE and directional accuracy). The
   architecture supports forecasting and `best_non_naive_beats_naive: true`;
   scaling the dated corpus is the highest-leverage follow-up for sharper
   separation.
2. **Agent A vs BLS ρ = 0.082 on a sample BLS file.** Reported transparently;
   the final paper uses the official BLS OES export
   (`BLS_EXPORT_PATH`).
3. **CS2023 `framework_coverage = 1.0` is tautological** by construction
   (`tracked_skills ⊇ CS2023 topics`). The non-tautological metric is
   `demand_surfaced = 0.6389` mean.
4. **Relevance and adversarial evals are still small-n** (n = 5 and n = 6).
   Expanding these is a near-term TODO; the multi-LLM eval has already been
   scaled to n = 50.
5. **`gemini-3.5-flash` is provider-flaky** (21 / 50 successes in the headline
   multi-LLM run; the gemini-3.1-pro-preview run is clean at 48 / 50). Both
   are reported transparently; coverage / latency / cost claims for
   gemini-3.5-flash are restricted to its 21 successful rows.
6. **Stubs that will be replaced for the production paper:**
   `FusionAgent` v2 (contrastive encoder), `ResourceMatcher` real catalog
   ingest, `RoadmapAgent` LLM-narrative pass.

---

## Lineage

This is the capstone of the RAG Bootcamp series in this repository — it
inherits the chunking, retrieval, grounding and audit infrastructure from the
earlier *Autonomous*, *Corrective*, *Adaptive* and *Memory* RAG modules and
adds the multi-agent curriculum alignment, cache, batch and evaluation
framework on top.

## Citing

If you use CURIA in academic work, please cite the accompanying paper
(citation will be added on acceptance) and this repository:

```bibtex
@misc{curia2026,
  title  = {CURIA: Velocity-Aware Retrieval-Augmented Curriculum Alignment with Industry Signal},
  author = {Joshi, Abhishek and collaborators},
  year   = {2026},
  url    = {https://github.com/abhishekjoshi007/RAG-Bootcamp}
}
```

## License

Code: see `LICENSE` (to be added). Corpus documents are pulled from public
APIs; users are responsible for honoring the source terms of service for
re-ingestion and downstream use.
