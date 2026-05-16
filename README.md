# RAG Bootcamp

A hands-on, project-based curriculum for building **Retrieval-Augmented Generation (RAG)** systems — from raw document ingestion all the way to autonomous, multi-agent retrieval pipelines.

The repository is two things in one:

1. **A learning track** — 47 Jupyter notebooks grouped by topic, each a self-contained lesson built on LangChain and LangGraph.
2. **A capstone project** — [`curia-rag-test/`](curia-rag-test/), a dependency-pinned, fully tested RAG application (the "CURIA" prototype) that puts the concepts together into a real, evaluable system with its own [README](curia-rag-test/README.md).

---

## Who this is for

You should be comfortable with Python and have basic familiarity with LLM APIs. By the end you will be able to ingest and chunk arbitrary documents, embed and index them, design retrieval and re-ranking strategies, enhance queries, add correction/reflection loops, and orchestrate agentic and multi-agent RAG with evaluation baked in.

---

## Repository layout

```text
RAG-Bootcamp/
├── README.md                                  ← you are here
│
│   ── Foundations ───────────────────────────────────────────────
├── langchain/                                 LangChain v1 primitives (6 notebooks)
│   └── updatedlangchain/                       intro, model integration, tools,
│                                               messages, structured output, middleware
├── langgraph/                                 LangGraph state machines (6 notebooks)
│                                               graphs, chatbots, state schemas,
│                                               pydantic state, chains, multi-tool bots
│
│   ── Ingestion & representation ────────────────────────────────
├── Data Ingestion & Data Parsing/             6 notebooks: text, PDF, DOCX,
│                                               CSV/Excel, JSON, databases (+ data/)
├── Advanced Chunking And Preprocessing Techniques/   semantic chunking
├── Vector Embedding and Vector Databases/     embeddings + OpenAI embeddings
├── Vector Stores/                             Chroma, FAISS, Pinecone, DataStax,
│                                               and other vector stores
│
│   ── Retrieval quality ─────────────────────────────────────────
├── Search Stratergies/                        dense vs sparse, re-ranking, MMR
├── Query Enhancement/                         query expansion, decomposition, HyDE
│
│   ── Advanced RAG patterns ─────────────────────────────────────
├── Corrective RAG/                            corrective RAG (CRAG)
├── Adaptive RAG/                              adaptive routing
├── Autonomus RAG/                             chain-of-thought, self-reflection,
│                                               query planning/decomposition,
│                                               iterative retrieval, answer synthesis
├── RAG Memory, Cache with LangGraph/          conversational memory + cache-augmented
│                                               generation
│
│   ── Agentic systems ───────────────────────────────────────────
├── Agentic Architecture/                      streaming, ReAct agents, debugging
├── Agentic Rag/                               agentic RAG + ReAct retrieval
├── Multi-Agent RAG/                           multi-agent orchestration
├── Multimodal RAG/                            multimodal (text + image) RAG
│
│   ── Capstone ──────────────────────────────────────────────────
└── curia-rag-test/                            full RAG application: ingestion →
                                               FAISS → retrieval → grounded
                                               generation → evaluation, plus a
                                               4-agent (skill / forecast / drift /
                                               curriculum) Gradio app.
```

---

## Suggested learning path

The notebooks are numbered within each folder; follow that order. Across folders, this progression works well:

| # | Module | What you learn |
|---|--------|----------------|
| 1 | `langchain/updatedlangchain/` | LangChain building blocks: models, tools, messages, structured output, middleware |
| 2 | `langgraph/` | Modelling control flow as stateful graphs |
| 3 | `Data Ingestion & Data Parsing/` | Loading text, PDF, DOCX, CSV/Excel, JSON, and SQL data |
| 4 | `Advanced Chunking And Preprocessing Techniques/` | Splitting documents so retrieval actually works |
| 5 | `Vector Embedding and Vector Databases/` | Turning chunks into vectors |
| 6 | `Vector Stores/` | Persisting and querying vectors (Chroma, FAISS, Pinecone, DataStax) |
| 7 | `Search Stratergies/` | Dense/sparse hybrid search, re-ranking, MMR diversity |
| 8 | `Query Enhancement/` | Expansion, decomposition, and HyDE to improve recall |
| 9 | `Corrective RAG/` · `Adaptive RAG/` | Grading retrievals and routing adaptively |
| 10 | `Autonomus RAG/` | CoT, self-reflection, planning, iterative retrieval, synthesis |
| 11 | `RAG Memory, Cache with LangGraph/` | Conversational memory and cache-augmented generation |
| 12 | `Agentic Architecture/` · `Agentic Rag/` | ReAct agents and agentic retrieval |
| 13 | `Multi-Agent RAG/` · `Multimodal RAG/` | Multiple cooperating agents; text + image |
| 14 | `curia-rag-test/` | Assemble everything into a tested, evaluated application |

---

## Prerequisites

- **Python 3.11+** (the capstone targets 3.11; notebooks run on 3.10+).
- **Jupyter** — `pip install jupyterlab` (or run notebooks in VS Code / Colab).
- **API keys** for the providers used by the notebooks. The most common ones:

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI embeddings & chat models (most notebooks, capstone generation) |
| `GROQ_API_KEY` | Groq-hosted models (agentic notebooks) |
| `PINECONE_API_KEY` | `Vector Stores/PineconeVectorDB.ipynb` |
| `LANGCHAIN_API_KEY` | optional LangSmith tracing |

A couple of notebooks (notably the Pinecone and DataStax ones, which were authored in Google Colab) prompt for keys inline rather than reading the environment — substitute your own where indicated.

---

## Getting started

```bash
git clone <this-repo>
cd RAG-Bootcamp

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Most notebooks rely on the modern LangChain/LangGraph stack:
pip install jupyterlab langchain langgraph langchain-openai \
            langchain-community faiss-cpu chromadb python-dotenv

# Provide your keys (create a .env at the repo root or export them):
export OPENAI_API_KEY=sk-...

jupyter lab
```

Open any notebook and run the cells top to bottom. Each topic folder is independent, so you can jump straight to what you need.

> **Note on code style:** notebook **code cells** are kept comment-free on purpose — the explanation lives in the **markdown cells** around them. Read the prose, run the code. The same minimal-comment convention applies to all `.py` files in the repo.

---

## The capstone: `curia-rag-test/`

Once the concepts click, the capstone shows them working together as a real system:

- **Ingestion** from 12 live sources (job boards, arXiv, Stack Overflow, GitHub, HackerNews, USAJOBS).
- **FAISS** dense index over `all-mpnet-base-v2` embeddings with recency-aware scoring and per-source quotas.
- **Grounded generation** (OpenAI `gpt-4o-mini` or a deterministic local generator) with **citation-grounding checks**.
- **Four evaluation harnesses**: retrieval (Recall@k/MRR/nDCG), faithfulness, answer relevance, adversarial robustness — each with pass/fail targets.
- **Four analytical agents** in a Gradio app: skill demand, trend forecasting, semantic drift, and curriculum-gap mapping.
- **15 sequential pytest suites** covering every stage.

See **[`curia-rag-test/README.md`](curia-rag-test/README.md)** for setup, the full pipeline, configuration, and evaluation details. The one-command demo:

```bash
cd curia-rag-test
cp .env.example .env          # add OPENAI_API_KEY
./run_pipeline.sh             # ingest → index → pipeline → evaluate
```

---

## Conventions

- **Folders** group lessons by theme; **numeric prefixes** give the intended order within a folder.
- `*.txt` / `*.svg` / `data/` files next to notebooks are sample inputs and diagrams used by those lessons.
- Notebook outputs are committed so you can read results without re-running (re-running requires valid API keys and live network access).

---

## License

No license file is currently included. Treat the contents as educational/reference material unless a license is added by the repository owner.
