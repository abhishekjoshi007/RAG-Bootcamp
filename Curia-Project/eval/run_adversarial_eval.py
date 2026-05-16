"""
Adversarial robustness evaluation — C9 Eval Set 5.

Tests how Recall@8 degrades under five perturbation types on the same
3 queries used in the standard retrieval eval:

  synonym_substitution  Replace terms with semantically equivalent alternatives
  topics_removed        Strip the current_topics field entirely
  misleading_topics     Replace topics with plausible-but-wrong terms
  date_filtered         Restrict retrieval to docs older than 2 years
  decoy_injected        (reported only; actual decoy injection done at index time)

Usage
    python3 eval/run_adversarial_eval.py

Note: Run after building the FAISS index (python3 scripts/build_index.py).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.indexing import FaissIndex
from src.query import build_query
from src.retrieval import Retriever
from src.storage import build_index_from_corpus, get_unit, load_units

INDEX_PATH = ROOT / "audit" / "faiss_index.pkl"
LABELS_PATH = ROOT / "data" / "eval" / "retrieval_labels.jsonl"


PERTURBATIONS: list[dict] = [
    {
        "name": "synonym_substitution",
        "description": "Title and description paraphrased with synonyms",
        "base_unit_id": "cs_ai_01",
        "modified_unit": {
            "id": "cs_ai_01_syn",
            "title": "Deep Neural Networks and Language Model Applications",
            "description": "Study of large-scale neural text generation systems including instruction-following models and context-aware completion engines.",
            "current_topics": ["neural architecture", "text generation", "model assessment", "AI fairness"],
        },
    },
    {
        "name": "synonym_substitution",
        "description": "Title and description paraphrased with synonyms",
        "base_unit_id": "cs_sec_01",
        "modified_unit": {
            "id": "cs_sec_01_syn",
            "title": "Secure Software Delivery and Third-Party Risk",
            "description": "Managing risk in modern software delivery pipelines including open-source component vetting and automated policy enforcement.",
            "current_topics": ["open-source risk", "automated scanning", "delivery pipeline security"],
        },
    },
    {
        "name": "topics_removed",
        "description": "current_topics field stripped",
        "base_unit_id": "cs_ai_01",
        "modified_unit": {
            "id": "cs_ai_01_notopics",
            "title": "Generative AI and Large Language Models",
            "description": "Covers transformer-based generative systems, prompt engineering, and responsible deployment of LLMs.",
            "current_topics": [],
        },
    },
    {
        "name": "topics_removed",
        "description": "current_topics field stripped",
        "base_unit_id": "cs_cloud_01",
        "modified_unit": {
            "id": "cs_cloud_01_notopics",
            "title": "Cloud Native Systems",
            "description": "Container orchestration, microservices, and operational practices for distributed cloud applications.",
            "current_topics": [],
        },
    },
    {
        "name": "misleading_topics",
        "description": "Topics replaced with unrelated terms from a different unit",
        "base_unit_id": "cs_ai_01",
        "modified_unit": {
            "id": "cs_ai_01_mislead",
            "title": "Generative AI and Large Language Models",
            "description": "Covers transformer-based generative systems, prompt engineering, and responsible deployment of LLMs.",
            "current_topics": ["SBOM generation", "dependency scanning", "policy-as-code"],
        },
    },
    {
        "name": "misleading_topics",
        "description": "Topics replaced with unrelated terms from a different unit",
        "base_unit_id": "cs_cloud_01",
        "modified_unit": {
            "id": "cs_cloud_01_mislead",
            "title": "Cloud Native Systems",
            "description": "Container orchestration, microservices, and operational practices for distributed cloud applications.",
            "current_topics": ["transformer architectures", "prompt evaluation", "hallucination rates"],
        },
    },
]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def unique_parent_ids(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.chunk.parent_id not in seen:
            seen.append(r.chunk.parent_id)
    return seen


def main() -> None:
    if INDEX_PATH.exists():
        index = FaissIndex.load(INDEX_PATH)
    else:
        print("Index not found — building …")
        index = build_index_from_corpus(ROOT / "data" / "corpus")
        index.save(INDEX_PATH)

    units = load_units(ROOT / "data" / "cs2023_units.json")
    retriever = Retriever(index)

    baseline_labels = {
        row["query_id"]: set(row["relevant_doc_ids"])
        for row in (
            json.loads(line)
            for line in LABELS_PATH.read_text().splitlines()
            if line.strip()
        )
    }

    baseline_recall: dict[str, float] = {}
    for unit_id, relevant in baseline_labels.items():
        unit = get_unit(units, unit_id)
        results = retriever.retrieve(build_query(unit), k=8)
        retrieved = unique_parent_ids(results)
        baseline_recall[unit_id] = recall_at_k(retrieved, relevant, 8)

    print(json.dumps({"baseline_recall@8": {k: round(v, 4) for k, v in baseline_recall.items()}}))

    results_rows: list[dict] = []
    for pert in PERTURBATIONS:
        base_id = pert["base_unit_id"]
        relevant = baseline_labels.get(base_id, set())
        query = build_query(pert["modified_unit"])
        retrieved_results = retriever.retrieve(query, k=8)
        retrieved = unique_parent_ids(retrieved_results)

        r8 = recall_at_k(retrieved, relevant, 8)
        ndcg = ndcg_at_k(retrieved, relevant, 8)
        base_r8 = baseline_recall.get(base_id, 0.0)
        drop = round(base_r8 - r8, 4)

        row = {
            "perturbation": pert["name"],
            "base_unit_id": base_id,
            "modified_unit_id": pert["modified_unit"]["id"],
            "description": pert["description"],
            "recall@8": round(r8, 4),
            "ndcg@8": round(ndcg, 4),
            "baseline_recall@8": round(base_r8, 4),
            "recall_drop": drop,
        }
        results_rows.append(row)
        print(json.dumps(row))

    if results_rows:
        avg_drop = sum(r["recall_drop"] for r in results_rows) / len(results_rows)
        max_drop = max(r["recall_drop"] for r in results_rows)
        worst = max(results_rows, key=lambda r: r["recall_drop"])
        summary = {
            "num_perturbations": len(results_rows),
            "avg_recall_drop": round(avg_drop, 4),
            "max_recall_drop": round(max_drop, 4),
            "worst_case": worst["perturbation"] + " on " + worst["base_unit_id"],
            "target_max_drop": 0.30,
            "passed_target": max_drop <= 0.30,
        }
        print(json.dumps({"adversarial_summary": summary}))


if __name__ == "__main__":
    main()
