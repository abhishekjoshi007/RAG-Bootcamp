# CURIA RAG Test Project

Standalone RAG prototype for validating the CURIA retrieval layer before building the full multi-agent system.

This first version is intentionally dependency-light: it runs with the Python standard library only. The code is structured so you can later swap the local lexical embedder for `sentence-transformers` + FAISS, and the local generator for OpenAI, Anthropic, Gemini, or another LLM provider.

## What is included

- Recursive document chunking with parent document metadata.
- Local TF-IDF-style embeddings and cosine search.
- Source quotas and recency-aware scoring.
- CS2023 query construction.
- Evidence-formatted prompt generation.
- Deterministic local recommendation generator with citations.
- Citation grounding checks.
- SQLite audit log for every run.
- Retrieval evaluation: Recall@k, MRR, nDCG.
- Small browser demo using `http.server`.

## Quick start

```bash
cd curia-rag-test
python3 scripts/build_index.py
python3 scripts/run_pipeline.py --unit-id cs_ai_01
python3 eval/run_retrieval_eval.py
python3 app.py
```

Open the app at http://127.0.0.1:8000.

## Project layout

```text
curia-rag-test/
├── app.py
├── requirements.txt
├── data/
│   ├── corpus/
│   ├── cs2023_units.json
│   └── eval/
├── scripts/
│   ├── build_index.py
│   └── run_pipeline.py
├── src/
│   ├── chunking.py
│   ├── embedding.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── query.py
│   ├── prompts.py
│   ├── llm.py
│   ├── grounding.py
│   ├── audit.py
│   ├── pipeline.py
│   └── storage.py
└── eval/
    └── run_retrieval_eval.py
```

## Next upgrades

1. Replace `LocalTfidfEmbedder` with `sentence-transformers/all-mpnet-base-v2`.
2. Replace `InMemoryIndex` with FAISS `IndexFlatIP`.
3. Add prompt-based JSON generation through your preferred LLM.
4. Add human-labeled faithfulness and answer relevance sets.
5. Run ablations for `k`, recency weighting, source quotas, and HyDE.
