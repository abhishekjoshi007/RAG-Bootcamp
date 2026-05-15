from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing import FaissIndex
from src.query import build_query
from src.retrieval import Retriever
from src.storage import build_index_from_corpus, get_unit, load_units


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def unique_parent_ids(results) -> list[str]:
    ids: list[str] = []
    for result in results:
        parent_id = result.chunk.parent_id
        if parent_id not in ids:
            ids.append(parent_id)
    return ids


def main() -> None:
    index_path = ROOT / "audit" / "faiss_index.pkl"
    if index_path.exists():
        index = FaissIndex.load(index_path)
    else:
        index = build_index_from_corpus(ROOT / "data" / "corpus")
        index.save(index_path)

    units = load_units(ROOT / "data" / "cs2023_units.json")
    retriever = Retriever(index)
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "eval" / "retrieval_labels.jsonl").read_text().splitlines()
        if line.strip()
    ]

    totals = {"recall@4": 0.0, "recall@8": 0.0, "mrr": 0.0, "ndcg@8": 0.0}
    for row in rows:
        unit = get_unit(units, row["query_id"])
        query = build_query(unit)
        results = retriever.retrieve(query, k=8)
        retrieved = unique_parent_ids(results)
        relevant = set(row["relevant_doc_ids"])
        metrics = {
            "recall@4": recall_at_k(retrieved, relevant, 4),
            "recall@8": recall_at_k(retrieved, relevant, 8),
            "mrr": mrr(retrieved, relevant),
            "ndcg@8": ndcg_at_k(retrieved, relevant, 8),
        }
        for key, value in metrics.items():
            totals[key] += value
        print(json.dumps({"query_id": row["query_id"], "retrieved": retrieved, **metrics}))

    count = max(len(rows), 1)
    aggregate = {key: round(value / count, 4) for key, value in totals.items()}
    print(json.dumps({"aggregate": aggregate}))


if __name__ == "__main__":
    main()
