# CURIA RAG Test Project

A complete, evaluable **Retrieval-Augmented Generation** prototype for the CURIA system: it ingests live job-market and technical signals, indexes them with FAISS, retrieves recency- and source-balanced evidence for a curriculum unit, generates a **grounded, citation-checked recommendation**, and measures itself against explicit quality targets.

It also ships four analytical agents — **skill demand, trend forecasting, semantic drift, and curriculum-gap mapping** — exposed through a Gradio UI.

> This is the capstone of the [RAG Bootcamp](../README.md). It assembles ingestion, chunking, embedding, indexing, retrieval, prompting, grounded generation, auditing, and evaluation into one tested codebase.

---

## Highlights

- **12-source ingestion** — Greenhouse, Lever, We Work Remotely, RemoteOK, Remotive, Arbeitnow, The Muse, HackerNews "Who is hiring", USAJOBS, arXiv (`cs.*`), Stack Exchange, GitHub READMEs.
- **Dense retrieval** — `sentence-transformers/all-mpnet-base-v2` embeddings in a FAISS `IndexFlatIP`, with **recency-decay scoring** and **per-source quotas** so no single source dominates.
- **Grounded generation** — OpenAI `gpt-4o-mini` (temperature 0) or a deterministic offline `LocalGroundedGenerator`, both required to cite `SOURCE_ID`s.
- **Citation grounding** — every recommendation is checked: cited IDs must exist in the retrieved evidence.
- **Four evaluation harnesses** with pass/fail targets pulled directly from `src/config.py`.
- **SQLite audit log** — every pipeline run is persisted (unit, prompt, evidence, recommendation, citation check).
- **Two front-ends** — a zero-dependency `http.server` demo and a richer 4-agent Gradio app.
- **14 sequential pytest modules** covering every stage from config to drift detection.

---

## Quick start

```bash
cd curia-rag-test

# 1. Configure
cp .env.example .env          # then add your OPENAI_API_KEY

# 2. Install (Python 3.11 recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. One command: ingest → index → pipeline → evaluate → summary
./run_pipeline.sh
```

`run_pipeline.sh` writes everything (logs, per-unit pipeline output, all four eval JSONs, and a pass/fail `summary.json`) to a timestamped `results/<YYYYMMDD_HHMMSS>/` directory. Useful flags:

```bash
./run_pipeline.sh --skip-ingest                # reuse existing data/corpus/
./run_pipeline.sh --skip-ingest --skip-index   # just pipeline + eval
```

### Run pieces individually

```bash
python3 scripts/ingest_corpus.py               # fetch corpus (+ rebuild index)
python3 scripts/build_index.py                 # (re)build FAISS index only
python3 scripts/run_pipeline.py --unit-id cs_ai_01 --k 8
python3 eval/run_retrieval_eval.py             # Recall@k / MRR / nDCG
python3 app.py                                 # stdlib demo  → http://127.0.0.1:8000
python3 app_gradio.py                          # 4-agent app  → http://127.0.0.1:8888
```

---

## Architecture

### Pipeline flow (`src/pipeline.py` → `CuriaRagPipeline.run`)

```
unit-id
  │
  ▼  build_query(unit)              title + description (+ topics)
  ▼  retriever.retrieve(k, quotas)  FAISS search → recency rescoring → source quotas
  ▼  build_recommendation_prompt()  evidence blocks tagged with SOURCE_IDs + JSON spec
  ▼  generator.generate()           OpenAIGenerator | LocalGroundedGenerator
  ▼  check_citations()              cited IDs ⊆ retrieved IDs ?
  ▼  AuditLog.write()               persist run to SQLite
  ▼
returns { query, prompt, evidence, recommendation, citation_check, audit_id }
```

`Recommendation` carries `signal_strength` (`high`/`medium`/`low`), `summary` (with inline `SOURCE_ID` citations), `emerging_topics[]`, and `evidence_ids[]`.

### Module map (`src/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | All tunable constants, with environment-variable overrides (`_int/_float/_str/_list` helpers). |
| `models.py` | Frozen dataclasses: `Document`, `Chunk`, `SearchResult`, `Recommendation`. |
| `storage.py` | Load corpus/units from disk; `build_index_from_corpus()`. |
| `chunking.py` | Sliding-window token chunking (`chunk_document`, default 160 tokens / 30 overlap). |
| `embedding.py` | `DenseEmbedder` (sentence-transformers) and `LocalTfidfEmbedder` + sparse cosine helpers. |
| `indexing.py` | `FaissIndex` (`IndexFlatIP`, picklable) and `InMemoryIndex` (sparse TF-IDF) — same `search()` API. |
| `retrieval.py` | `Retriever`: candidate search, recency decay, per-source quota enforcement, top-k. |
| `query.py` | `build_query()` and `build_hyde_prompt()` from a unit. |
| `prompts.py` | `format_evidence()` and `build_recommendation_prompt()` (enforces a JSON contract). |
| `llm.py` | `OpenAIGenerator` (retrying, JSON-parsed) and `LocalGroundedGenerator` (deterministic, offline). |
| `grounding.py` | `check_citations()` → `CitationCheck(passed, cited_ids, retrieved_ids, missing_ids)`. |
| `audit.py` | `AuditLog` — SQLite `rag_runs` table, one row per pipeline run. |
| `pipeline.py` | `CuriaRagPipeline` orchestrating the flow above. |
| `ingest.py` | 12 source fetchers + `ingest_all()`; HTML stripping, date parsing, rate limiting. |
| `field_config.py` | `FIELD_INGESTION` — 21 academic fields → source queries/tags/topics. |
| `university.py` | Loads TAMU curricula; `curriculum_summary()`, `get_all_units()`, etc. |
| `drift.py` | `SemanticDriftDetector` — cross-source / temporal centroid divergence per skill. |
| `forecasting.py` | `SkillForecaster` — linear & exponential-smoothing skill-demand forecasts + backtest. |

### Front-ends

- **`app.py`** — pure Python `http.server` (`127.0.0.1:8000`). Loads the pipeline once, renders the recommendation + evidence cards as HTML. No web framework needed.
- **`app_gradio.py`** — Gradio `Blocks` app (`127.0.0.1:8888`) with four agents:
  - **Agent A — Skill demand:** scores candidate skills via the retriever, renders a ranked leaderboard.
  - **Agent B — Forecasting:** `SkillForecaster` trend arrows, 12-month projection, confidence, SVG sparklines.
  - **Agent C — Semantic drift:** `SemanticDriftDetector` bars grouped by LOW/MEDIUM/HIGH drift.
  - **Agent D — Curriculum map:** runs the pipeline per unit and overlays demand/trend/drift onto the curriculum, highlighting gaps.

---

## Configuration (`src/config.py`)

Every constant can be overridden via environment variables. Key defaults:

| Group | Constant | Default |
|-------|----------|---------|
| Chunking | `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP` | `160` / `30` |
| Embedding | `EMBED_MODEL` / `EMBED_BATCH_SIZE` | `all-mpnet-base-v2` / `8` |
| Retrieval | `RETRIEVAL_K` / `RETRIEVAL_CANDIDATE_K` | `8` / `50` |
| Recency | `RECENCY_HALF_LIFE_DAYS` · `BASE` · `BONUS` | `365` · `0.7` · `0.3` |
| Source quotas | job_posting / arxiv / stackoverflow / github_readme | `3` / `2` / `2` / `1` |
| LLM | `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_MAX_RETRIES` | `gpt-4o-mini` / `0.0` / `1024` / `3` |
| Signal | `LOCAL_SIGNAL_HIGH` / `LOCAL_SIGNAL_MEDIUM` | `0.28` / `0.14` |
| Eval target | `EVAL_TARGET_RECALL_8` | `0.70` |
| Eval target | `EVAL_TARGET_CITATION_PRECISION` | `0.95` |
| Eval target | `EVAL_TARGET_CLAIM_GROUNDING` | `0.85` |
| Eval target | `EVAL_TARGET_RELEVANCE_MEAN` | `3.5` |
| Eval target | `EVAL_TARGET_ADVERSARIAL_DROP` | `0.30` |

Recency-adjusted score:

```
score = similarity * (RECENCY_BASE_WEIGHT + RECENCY_BONUS_WEIGHT * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))
```

### Environment variables (`.env`)

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `OPENAI_API_KEY` | **Yes** (for LLM generation; `--local` avoids it) | OpenAI chat completions |
| `USAJOBS_API_KEY` / `USAJOBS_USER_AGENT` | Optional | USAJOBS federal postings ([register free](https://developer.usajobs.gov/)) |
| `GITHUB_TOKEN` | Optional | Raises GitHub rate limit 60 → 5000 req/hr |

`run_pipeline.sh` also exports `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1` to avoid HuggingFace fork deadlocks and BLAS segfaults on macOS.

---

## Data layout

```text
data/
├── corpus/                 182 ingested documents (one JSON per doc)
├── cs2023_units.json       3 CS2023 curriculum units
├── universities/           6 TAMU college curricula
└── eval/                   ground-truth label files (JSONL)
```

**Corpus documents** (182 total) — `{ id, title, source, date, text, metadata }`. ID prefixes by source:

| Prefix | Source | Count |
|--------|--------|------:|
| `gh_` | GitHub READMEs / ATS | 47 |
| `ax_` | arXiv papers | 38 |
| `so_` | Stack Overflow | 27 |
| `hn_` | HackerNews hiring | 24 |
| `wwr_` | We Work Remotely | 20 |
| `uj_` | USAJOBS | 19 |
| `an_` | Arbeitnow | 5 |
| `jp_` | other job boards | 2 |

**CS2023 units** (`data/cs2023_units.json`) — a list of 3 units, e.g.:

```json
{
  "id": "cs_ai_01",
  "title": "Generative AI and Large Language Models",
  "description": "Students should understand transformer-based language models, prompt engineering, retrieval-augmented generation, alignment, evaluation, and deployment considerations.",
  "current_topics": ["transformers", "prompt engineering", "model evaluation", "AI ethics"]
}
```
The three units are `cs_ai_01`, `cs_sec_01` (software supply-chain security), `cs_cloud_01` (cloud-native systems).

**Universities** — 6 TAMU college files (`tamu_cs`, `tamu_ee`, `tamu_engineering`, `tamu_science`, `tamu_business`, `tamu_geosciences_agriculture`) loaded by `src/university.py`.

**Eval labels** (`data/eval/`):

| File | Lines | Schema |
|------|------:|--------|
| `retrieval_labels.jsonl` | 3 | `{ query_id, relevant_doc_ids[] }` |
| `faithfulness_labels.jsonl` | 5 | `{ recommendation_id, unit_id, summary, evidence_ids[], claims[] }` |
| `relevance_ratings.jsonl` | 5 | `{ recommendation_id, unit_id, rating }` (1–5) |

---

## CLI reference

**`scripts/ingest_corpus.py`**

| Flag | Effect |
|------|--------|
| `--sources S [S ...]` | Fetch only listed sources (`greenhouse`, `lever`, `weworkremotely`, `remoteok`, `remotive`, `arbeitnow`, `themuse`, `hn_hiring`, `usajobs`, `arxiv`, `stackoverflow`, `github`) |
| `--no-rebuild-index` | Skip the automatic FAISS rebuild after ingest |
| `--force` | Re-fetch documents that already exist on disk |

**`scripts/build_index.py`** — no arguments; reads `CORPUS_DIR` / `EMBED_MODEL`, writes the FAISS index to `INDEX_PATH`.

**`scripts/run_pipeline.py`**

| Flag | Default | Effect |
|------|---------|--------|
| `--unit-id` | `cs_ai_01` | Curriculum unit to run |
| `--k` | `6` | Number of evidence chunks to retrieve |
| `--local` | off | Use the offline `LocalGroundedGenerator` instead of OpenAI |

---

## Evaluation

Run all four (or individually); `run_pipeline.sh` aggregates them into `summary.json` with `all_targets_met`.

| Script | Metric(s) | Target (from `config.py`) |
|--------|-----------|---------------------------|
| `eval/run_retrieval_eval.py` | `recall@4`, `recall@8`, `mrr`, `ndcg@8` | `recall@8 ≥ 0.70` |
| `eval/run_faithfulness_eval.py` | `citation_precision`, `claim_grounding` | `≥ 0.95`, `≥ 0.85` |
| `eval/run_relevance_eval.py` | mean human rating (1–5) + distribution | `mean ≥ 3.5` |
| `eval/run_adversarial_eval.py` | recall drop under synonym / topic-removal / misleading perturbations | `max drop ≤ 0.30` |

Each script prints one JSON object per item plus an `aggregate`/`summary` block on the last line.

---

## Tests

14 pytest modules run in pipeline order (`tests/test_01_config.py` … `tests/test_14_drift.py`): config, models, chunking, embedding, indexing, retrieval, query, prompts, grounding, audit, LLM, full pipeline, forecasting, drift.

```bash
python3 -m pytest -q                 # all
python3 -m pytest tests/test_06_retrieval.py -q
```

`test_11_llm.py` and `test_12_pipeline.py` make real OpenAI calls and need `OPENAI_API_KEY` plus the built FAISS index and corpus.

---

## Requirements

Python **3.11** recommended. Pinned in `requirements.txt`:

```
sentence-transformers==3.1.1   transformers==4.44.2   tokenizers>=0.19,<0.20
faiss-cpu>=1.8.0   openai>=1.40.0   python-dotenv>=1.0.0   numpy>=1.24.0
duckdb>=1.0.0   pydantic>=2.7.0   anthropic>=0.34.0   ragas>=0.1.15   pytest>=8.0.0
```

(`sentence-transformers`/`transformers`/`tokenizers` are pinned for `torch 2.4.x` compatibility.) There is no `pyproject.toml`/`setup.py` — run scripts directly.

---

## Roadmap

1. Hybrid dense + sparse retrieval and learned re-ranking.
2. Human-labelled faithfulness and answer-relevance sets at larger scale.
3. Ablations for `k`, recency weighting, source quotas, and HyDE.
4. Pluggable generators (Anthropic / Gemini) behind the existing generator interface.
5. Incremental index updates instead of full rebuilds on ingest.

---

## Notes

- Code follows a **minimal-comment** convention: behaviour is documented in docstrings and this README, not inline comments.
- `.env` is gitignored — never commit real keys.
- Notebook/corpus outputs and `results/` are committed so runs are reproducible to read without live network access.
