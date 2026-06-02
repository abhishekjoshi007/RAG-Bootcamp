# CURIA Paper 1 — Session Decisions & Findings Log

**Project:** CURIA — Retrieval-Augmented Curriculum Recommendation with a
Drift-Cascaded Recommendation Cache
**Target venues:** IEEE BigData 2026 (applied research track) or CIKM 2026 (short paper)
**Status as of last entry:** evidence + figures + tables paper-ready; LaTeX manuscript still to draft

Raw verbatim chat transcript (auto-saved by Claude Code):
`~/.claude/projects/-Users-abhishekjoshi-Documents-GitHub-RAG-Bootcamp-Curia-Project/0dcf49db-0ef8-4116-b597-76822bc538e4.jsonl`

This file is a curated index of the strategic decisions, headline findings, and
artifacts produced across all working sessions, ordered by date.

---

## 1. Strategic decisions

### 1.1 Submission strategy — Path C (2026-06-01)

After honest assessment of submission readiness, three paths were considered:

| Path | Description | Verdict |
|---|---|---|
| A | Submit current artifact to CIKM full paper | ~10–15% accept; multiple unresolved RQs |
| B | One more sprint, then main conference | 4–6 weeks, ~40–50% accept odds |
| **C** | **Split: velocity short paper now + full paper later** | **Chosen** |

Path C rationale: the cache + drift-cascade contribution is a defensible
standalone systems contribution; the curriculum/forecasting/BLS RQs need more
work and should not be forced into the same narrative.

### 1.2 Paper-1 framing

Lead with the **drift-cascaded recommendation cache + served-staleness metric**
as the named contribution. The 17-model LLM table and LlamaIndex baseline are
supporting evidence, not the main novelty. Forecasting, BLS correlation, and
human relevance ratings are moved to "future work" / Paper 2.

### 1.3 Paper-2 deferred items

Not blocking Paper 1; needed before any full-paper submission:

1. Official BLS OES export + ESCO crosswalk (replace sample BLS file)
2. Scale relevance ratings to n ≥ 50 with 2 raters + Cohen's κ
3. Scale adversarial perturbations to n ≥ 50
4. Add a stronger forecast baseline (Prophet / N-BEATS / light Transformer) with
   paired-t / bootstrap CI on the Δ vs naive
5. Recruit 10–15 curriculum committee members or CS instructors for a small
   user study
6. Add JSON-output LlamaIndex condition (camera-ready strengthening)
7. Re-run cache ablation with real paid LLM calls on miss path

### 1.4 Terminology fix — "TAMU units" → "courses" (2026-06-02)

The 50 benchmark units are sourced from Texas A&M, but paper claims should not
read as TAMU-specific. Generator scripts, README, table captions, and figure
labels were all changed to **"50 courses"** or **"50 benchmark courses"**.
File names and internal IDs (`benchmark_units_tamu_50.json`, `tamu_misy_erp`,
`tamu_cs.json`, ...) were intentionally left unchanged — those describe the
seed dataset, not a system restriction.

### 1.5 Faithfulness wording lock-in (2026-06-01)

Final accepted wording for the multi-LLM faithfulness claim:

> Across 17 evaluated models, successful generations met the
> citation-faithfulness target; **15 models completed all 50 units with
> citation precision 1.000 and hallucination rate 0.000**. Evidence coverage,
> not citation precision, discriminates model quality.

Rationale: keeps the Gemini-error caveat explicit (gemini-3.5-flash 21/50,
gemini-3.1-pro-preview 48/50), avoids overclaiming on the 17-model number,
and reframes the discriminator axis (coverage, not precision).

---

## 2. Headline findings (for the paper)

### 2.1 Cache-policy ablation (`results/headline_cache_ablation.json`)

Same 1,000-query workload (100 unique × 10 repeats, shuffled, rng_seed=42).
Miss-cost assumption: $0.01 / miss, 250 ms / miss.

| Policy | Hit rate | Served-staleness | Cost / 1k | LLM calls | Drift rows |
|---|---:|---:|---:|---:|---:|
| No cache | 0.000 | 0.000 | $10.00 | 1,000 | 0 |
| TTL-only | 0.900 | **0.416** | $1.00 | 100 | 0 |
| Drift-cascade | 0.783 | **0.000** | $2.17 | 217 | 145 |

**Paper claim:** drift-cascade costs **−11.7 pp hit rate** and **+$1.17 / 1k**
versus TTL-only, but **eliminates a 41.6% served-staleness rate** by
construction.

### 2.2 Multi-LLM faithfulness (`results/headline_multi_llm_50q_17models_rechecked.json`)

50 courses × 17 LLMs (OpenAI / Anthropic / Google / xAI / DeepSeek + local
extractive baseline). Same retrieved evidence per query; only the generator
changes. Total spend: **$18.87 / $20 budget**.

- 15 models complete-run at citation precision **1.000** and hallucination **0.000**
- Coverage range: 0.514 (Claude Sonnet 4.5) → **0.767 (Claude Sonnet 4.6)**
- Local extractive baseline: 0.535 coverage at 0 cost / sub-millisecond latency
- Gemini complete rates: 21/50 (3.5-flash), 48/50 (3.1-pro-preview)
- arXiv version canonicalization (`src/grounding.py:_canonical_id`) was added
  to handle v1/v4 mismatches that produced a single 0.998 row pre-fix

### 2.3 LlamaIndex baseline (`results/headline_llamaindex_baseline.json`)

50 courses × same gpt-4o-mini generator × same 3,375-doc corpus. Two embedding
configurations.

| Config | Embedder | Citation precision | Coverage | Latency | Proj. $/1k |
|---|---|---:|---:|---:|---:|
| `li_default` | text-embedding-3-small | 1.000 (vacuous) | **0.000** | 5.77 s | $0.31 |
| `li_matched` | mpnet-base-v2 | 1.000 (vacuous) | **0.000** | 4.42 s | $0.29 |

**Paper claim:** under the same system prompt, both LlamaIndex configurations
produce **zero inline source-ID citations** across all 50 queries. The 1.000
citation precision is vacuous (no citations to audit). CURIA's structured
output contract + `check_citations` post-hoc validation enforces auditable
citations by design.

### 2.4 Forecast backtest (`results/headline_forecast_backtest_corpus_large.json`)

3,375-doc dated corpus, 30 skills evaluated, 142/185 months with documents.

| Baseline | sMAPE | Best metric |
|---|---:|---|
| naive | 1.3094 | — |
| moving_avg | 1.3107 | — |
| linear | 1.3479 | directional accuracy = 0.522 |
| **exp_smoothing** | **1.2940** | **best non-naive sMAPE (Δ = −0.0154)** |
| seasonal_naive | 1.3701 | MASE = 2.86, dir-acc = 0.572 |

`statistically_meaningful: true`. `best_non_naive_beats_naive: true`.

**Paper status:** appendix / future work for Paper 1. Δ is small; useful as
"the architecture supports forecasting" claim but not as a headline result.

### 2.5 Velocity (`scripts/bench_cache.py` artifact in scorecard)

- Cache hit p95 latency: **0.46 ms** (target ≤ 200 ms)
- Workload hit rate on 1,000 shuffled queries (100 unique × 10): **90%**
- Projected LLM-call reduction: **900 / 1,000 = 90%**

### 2.6 Retrieval

`recall@8 = 0.867` (target 0.70). `MRR = 1.0`, `nDCG@8 = 0.871`.

---

## 3. Artifacts produced

### 3.1 Code

| Path | Purpose |
|---|---|
| `src/grounding.py` | `_canonical_id` for arXiv version canonicalization |
| `scripts/bench_cache_ablation.py` | Cache-policy ablation with served-staleness metric |
| `scripts/bench_llamaindex_baseline.py` | LlamaIndex baseline (two configs, real LLM calls) |
| `scripts/generate_paper_tables.py` | Auto-generates Tables 1–3 (md + tex) |
| `scripts/generate_paper_figures.py` | Auto-generates Figures 1–3 (png + pdf) |
| `scripts/generate_multi_llm_paper_table.py` | IEEE multi-LLM table generator with patch-row support |
| `eval/run_scorecard.py` | Aggregated scorecard with cache_ablation + llamaindex_baseline + multi_llm_headline read-only layers |
| `tests/test_09_grounding.py` | +3 arXiv-canonicalization tests |
| `tests/test_23_forecast_backtest.py` | +4 seasonal-naive / exp-smoothing tests |
| `tests/test_24_cache_ablation.py` | +3 ablation invariant tests |

### 3.2 Headline result artifacts

| Path | Content |
|---|---|
| `results/headline_scorecard.json` | 11-layer aggregated scorecard, current state |
| `results/headline_cache_ablation.json` | Cache-policy ablation with served-staleness |
| `results/headline_llamaindex_baseline.json` | LlamaIndex on 3,375-doc corpus, both configs |
| `results/headline_multi_llm_50q_17models_rechecked.json` | Paper-canonical 17-model table |
| `results/headline_multi_llm_50q_17models.json` | Raw pre-recheck artifact (kept for transparency) |
| `results/headline_forecast_backtest_corpus_large.json` | exp_smoothing beats naive |
| `results/headline_ground_truth_bls.json` | Spearman ρ = 0.082 (sample BLS) |

### 3.3 Paper-ready tables

| Path | Content |
|---|---|
| `results/paper_tables.md` | Full Markdown bundle, ready for outline integration |
| `results/table1_cache_ablation.tex` | IEEE LaTeX, Table 1 |
| `results/table2_llamaindex_vs_curia.tex` | IEEE LaTeX, Table 2 |
| `results/table3_multi_llm_summary.tex` | IEEE LaTeX, Table 3 (15 complete-run + Gemini subset) |
| `results/multi_llm_50q_17models_ieee_table.tex` | Full 17-model IEEE LaTeX table |
| `results/cache_velocity_algorithm_section.md` | Algorithm + invariants + complexity prose |

### 3.4 Figures

All in `results/figures/`, both PNG and PDF:

| Path | Content |
|---|---|
| `fig1_cache_ablation.{png,pdf}` | Bar chart: drift-cascade buys 0% staleness at +$1.17/1k |
| `fig2_latency_comparison.{png,pdf}` | Log-scale bars: LlamaIndex cold-path ~5 s vs CURIA cache hit ~0.5 ms |
| `fig3_multi_llm_coverage.{png,pdf}` | Horizontal bar chart: coverage discriminates 15 models |

### 3.5 Outline + architecture diagram

| Path | Content |
|---|---|
| `paper1_velocity_outline.md` | Working outline: title, abstract, RQs, method, tables, limitations, venue framing |
| (eraser.io) | Architecture diagram drafted, reviewed, and refined to paper-grade after two iterations |

---

## 4. Open items

### 4.1 Blocker resolution log

| Blocker | Resolution date | Resolution |
|---|---|---|
| LlamaIndex baseline ran on 182-doc small corpus instead of 3,375-doc `corpus_large` | 2026-06-01 | Re-ran with `--corpus-dir data/corpus_large`; n_corpus_docs verified = 3375; finding (zero inline citations) holds |
| `headline_scorecard.json` was from 2026-05-26 and stale | 2026-06-01 | Regenerated; now includes cache_ablation + llamaindex_baseline + multi_llm_headline read-only layers; 11 layers OK / 0 failed / 0 missing |
| Architecture diagram had garbled labels and tangled HIT path | 2026-06-02 | Re-prompted eraser.io; numbered cache-event legend; clean HIT bypass; A/B/C/E in one row feeding D |

### 4.2 What the author still owns

- LaTeX manuscript (IEEE conference class or ACM `sigconf`)
- Author info, affiliations, ORCID
- BibTeX references (LlamaIndex 2022, RAGAS, ESCO, CS2023, BLS OES, prior cache work)
- Venue choice (BigData applied vs CIKM short)
- Submission portal upload

---

## 5. Key methodological positions taken

These are positions the paper should explicitly defend or footnote.

1. **No model training.** Zero epochs anywhere. Forecasting uses classical
   baselines (naive, seasonal-naive, moving-avg, linear, exp_smoothing) with
   closed-form fits. Embedder (mpnet-base-v2) is pre-trained off-the-shelf.
   All 17 LLMs are API-served; no fine-tuning. *This is by design — the
   contribution is the surrounding architecture, not a new model.*

2. **Cache miss cost is simulated** ($0.01 per miss, 250 ms per miss). The
   *relative* ablation numbers (served-staleness 41.6% vs 0%) are independent
   of this assumption. A real-paid-call cache ablation is camera-ready work,
   not submission-blocking.

3. **LlamaIndex was tested under one prompt style** (CURIA's system prompt
   requesting parenthesized SOURCE_IDs). The honest finding is "under the
   same prompt, only CURIA enforces citations structurally." A stricter
   JSON-output LlamaIndex condition is camera-ready work.

4. **CS2023 `framework_coverage = 1.0` is tautological** by construction
   (tracked_skills ⊇ CS2023 topics). The non-tautological metric is
   `demand_surfaced = 0.6389` mean. Paper-2 work.

5. **Agent A vs BLS ρ = 0.082** is on sample BLS data
   (`data/eval/bls_oes_sample.json`). Reported transparently as a limitation;
   real OES export is Paper-2 blocker.

---

## 6. Where to find the raw chat transcript

Auto-saved by Claude Code at:

```
~/.claude/projects/-Users-abhishekjoshi-Documents-GitHub-RAG-Bootcamp-Curia-Project/0dcf49db-0ef8-4116-b597-76822bc538e4.jsonl
```

19 MB JSONL of every message + tool call. This curated log is the high-signal
index; the JSONL is the unabridged source-of-truth if any decision needs to be
re-traced.
