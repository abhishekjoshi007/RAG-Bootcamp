#!/usr/bin/env python3
"""LlamaIndex baseline for the Path C velocity paper.

Compares CURIA's retrieval-and-grounding pipeline against a vanilla
LlamaIndex `VectorStoreIndex` over the same corpus and benchmark units, on
quality (citation precision, evidence coverage, hallucination), velocity
(end-to-end latency), and cost (OpenAI token usage). LlamaIndex ships no
recommendation cache, so the velocity comparison projects its cold-path
latency / cost over the same 1,000-query workload our cache ablation uses.

Two configurations are run to isolate the system contribution:

    li_default  OpenAI embeddings (text-embedding-3-small) + gpt-4o-mini
                (LlamaIndex's out-of-box settings with provider keys).
    li_matched  HuggingFace `all-mpnet-base-v2` embeddings + gpt-4o-mini
                (matches CURIA's embedder so the only changing variable is
                the surrounding architecture).

Outputs a single JSON artifact compatible with the headline scorecard.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.config import CORPUS_DIR  # noqa: E402
from src.grounding import _canonical_id  # noqa: E402


CITATION_RE = re.compile(r"\b[a-z]{2,}[a-z0-9-]*(?:_[a-z0-9_.-]+)+\b", re.IGNORECASE)


@dataclass
class UnitResult:
    unit_id: str
    n_retrieved: int
    retrieved_ids: list[str]
    cited_ids: list[str]
    missing_ids: list[str]
    citation_precision: float
    hallucination_rate: float
    evidence_coverage: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class ConfigResult:
    config_id: str
    embed_model: str
    llm_model: str
    units: list[UnitResult] = field(default_factory=list)
    index_build_seconds: float = 0.0
    n_docs_indexed: int = 0
    velocity_cold_latency_ms: list[float] = field(default_factory=list)
    velocity_unique_queries: int = 0

    def aggregate(self, workload_size: int) -> dict[str, Any]:
        n = len(self.units) or 1
        latencies_ms = sorted(self.velocity_cold_latency_ms) or [0.0]
        idx_p50 = latencies_ms[min(len(latencies_ms) - 1, len(latencies_ms) // 2)]
        idx_p95 = latencies_ms[min(len(latencies_ms) - 1, int(0.95 * (len(latencies_ms) - 1)))]
        mean_cold = sum(latencies_ms) / len(latencies_ms)
        cold_cost = sum(u.cost_usd for u in self.units[: self.velocity_unique_queries]) / max(
            self.velocity_unique_queries, 1
        )
        return {
            "config_id": self.config_id,
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "index_build_seconds": round(self.index_build_seconds, 3),
            "n_docs_indexed": self.n_docs_indexed,
            "quality": {
                "n_units": n,
                "citation_precision": round(sum(u.citation_precision for u in self.units) / n, 4),
                "hallucination_rate": round(sum(u.hallucination_rate for u in self.units) / n, 4),
                "evidence_coverage": round(sum(u.evidence_coverage for u in self.units) / n, 4),
                "mean_latency_s": round(sum(u.latency_s for u in self.units) / n, 3),
                "mean_cost_usd": round(sum(u.cost_usd for u in self.units) / n, 6),
                "total_cost_usd": round(sum(u.cost_usd for u in self.units), 6),
            },
            "velocity_projection": {
                "workload_size": workload_size,
                "cold_path_mean_latency_ms": round(mean_cold, 3),
                "cold_path_p50_latency_ms": round(idx_p50, 3),
                "cold_path_p95_latency_ms": round(idx_p95, 3),
                "projected_total_latency_s": round((mean_cold / 1000.0) * workload_size, 2),
                "projected_total_cost_usd": round(cold_cost * workload_size, 4),
                "cache_hit_rate_assumed": 0.0,
                "llm_calls_avoided": 0,
                "note": (
                    "LlamaIndex ships no recommendation cache. Latency and cost over "
                    "the 1,000-query workload are projected from cold-path measurements; "
                    "CURIA's measured cache hit rate avoids 783-900 of those calls."
                ),
            },
        }


def _load_corpus_docs(corpus_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    paths = sorted(corpus_dir.glob("*.json"))
    if limit:
        paths = paths[:limit]
    docs: list[dict[str, Any]] = []
    for path in paths:
        try:
            docs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return docs


def _make_llama_documents(docs: list[dict[str, Any]]):
    from llama_index.core import Document

    out = []
    for doc in docs:
        text = doc.get("text") or doc.get("title") or ""
        if not text.strip():
            continue
        out.append(
            Document(
                text=text,
                doc_id=doc["id"],
                metadata={
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "date": doc.get("date", ""),
                },
            )
        )
    return out


_SYSTEM_PROMPT = (
    "You are a curriculum-update assistant. Use ONLY the retrieved evidence "
    "below. Cite every claim with the document id in parentheses, e.g. "
    "(jp_001) or (axhist_2501.00123v1). If the evidence does not support a "
    "claim, omit it. Do not invent ids."
)


def _retrieved_doc_ids(response) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sn in getattr(response, "source_nodes", []) or []:
        node_id = (
            sn.node.metadata.get("doc_id")
            if hasattr(sn, "node") and sn.node.metadata
            else None
        ) or sn.node.ref_doc_id
        if node_id and node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered


def _score_unit(
    response_text: str,
    retrieved: list[str],
) -> tuple[float, float, float, list[str], list[str]]:
    cited = list({m.lower() for m in CITATION_RE.findall(response_text)})
    canonical_retrieved = {_canonical_id(r) for r in retrieved}
    valid_cited = [c for c in cited if _canonical_id(c) in canonical_retrieved]
    missing = [c for c in cited if _canonical_id(c) not in canonical_retrieved]
    # Convention matches our headline scoring (eval/run_multi_llm_eval.py):
    # an answer with zero cited ids is vacuously precise (no false claims to
    # be wrong about); evidence_coverage is the honest discriminator.
    if cited:
        citation_precision = len(valid_cited) / len(cited)
        hallucination_rate = len(missing) / len(cited)
    else:
        citation_precision = 1.0
        hallucination_rate = 0.0
    evidence_coverage = len(valid_cited) / len(retrieved) if retrieved else 0.0
    return (
        round(citation_precision, 4),
        round(hallucination_rate, 4),
        round(evidence_coverage, 4),
        valid_cited,
        missing,
    )


def _approx_cost_usd(input_tokens: int, output_tokens: int, llm_model: str) -> float:
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "text-embedding-3-small": (0.02, 0.0),
    }
    rate = pricing.get(llm_model, (0.15, 0.60))
    return round(input_tokens / 1e6 * rate[0] + output_tokens / 1e6 * rate[1], 6)


def _build_index(
    config_id: str,
    docs: list[dict[str, Any]],
    persist_dir: Path,
    embed_model: str,
):
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage

    if embed_model.startswith("hf:"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model[3:])
    else:
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.embed_model = OpenAIEmbedding(model=embed_model)

    persist_dir.mkdir(parents=True, exist_ok=True)
    docstore = persist_dir / "docstore.json"
    if docstore.exists():
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage), 0.0, len(docs)
    llama_docs = _make_llama_documents(docs)
    start = time.perf_counter()
    index = VectorStoreIndex.from_documents(llama_docs)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index, time.perf_counter() - start, len(llama_docs)


def _run_config(
    config_id: str,
    docs: list[dict[str, Any]],
    units: list[dict[str, Any]],
    persist_dir: Path,
    embed_model: str,
    llm_model: str,
    k: int,
    velocity_n: int,
) -> ConfigResult:
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model=llm_model, temperature=0.0)
    index, build_seconds, n_indexed = _build_index(config_id, docs, persist_dir, embed_model)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64

    query_engine = index.as_query_engine(
        similarity_top_k=k,
        system_prompt=_SYSTEM_PROMPT,
    )

    result = ConfigResult(
        config_id=config_id,
        embed_model=embed_model,
        llm_model=llm_model,
        index_build_seconds=build_seconds,
        n_docs_indexed=n_indexed,
        velocity_unique_queries=velocity_n,
    )

    for i, unit in enumerate(units):
        query_text = f"{unit['title']}. {unit['description']}"
        start = time.perf_counter()
        response = query_engine.query(query_text)
        latency = time.perf_counter() - start
        text = str(response)
        retrieved = _retrieved_doc_ids(response)
        cite_p, hall, cov, valid_cited, missing = _score_unit(text, retrieved)
        input_tokens = sum(
            len((sn.node.text or "").split()) for sn in getattr(response, "source_nodes", []) or []
        )
        output_tokens = len(text.split())
        cost = _approx_cost_usd(input_tokens, output_tokens, llm_model)
        result.units.append(
            UnitResult(
                unit_id=unit["id"],
                n_retrieved=len(retrieved),
                retrieved_ids=retrieved,
                cited_ids=valid_cited,
                missing_ids=missing,
                citation_precision=cite_p,
                hallucination_rate=hall,
                evidence_coverage=cov,
                latency_s=round(latency, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        )
        if i < velocity_n:
            result.velocity_cold_latency_ms.append(latency * 1000.0)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-file", default="data/eval/benchmark_units_tamu_50.json")
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--out", default="results/headline_llamaindex_baseline.json")
    parser.add_argument("--configs", default="li_default,li_matched")
    parser.add_argument("--max-units", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--velocity-n", type=int, default=20)
    parser.add_argument("--workload-size", type=int, default=1000)
    parser.add_argument("--budget-usd", type=float, default=10.0)
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="Limit number of corpus docs (debugging only).")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY required; aborting.")
        return 2

    units = json.loads(Path(args.units_file).read_text())[: args.max_units]
    docs = _load_corpus_docs(Path(args.corpus_dir), limit=args.corpus_limit)
    if not docs:
        print(f"No documents in {args.corpus_dir}; aborting.")
        return 2

    configs = {
        "li_default": {
            "embed": "text-embedding-3-small",
            "llm": "gpt-4o-mini",
            "persist": ROOT / "audit" / "llamaindex_default",
        },
        "li_matched": {
            "embed": "hf:sentence-transformers/all-mpnet-base-v2",
            "llm": "gpt-4o-mini",
            "persist": ROOT / "audit" / "llamaindex_matched",
        },
    }
    requested = [c.strip() for c in args.configs.split(",") if c.strip()]

    results: dict[str, ConfigResult] = {}
    budget_spent = 0.0
    for config_id in requested:
        if config_id not in configs:
            print(f"unknown config {config_id}; skipping.")
            continue
        cfg = configs[config_id]
        print(f"\n=== running {config_id} ({cfg['embed']} + {cfg['llm']}) ===")
        result = _run_config(
            config_id=config_id,
            docs=docs,
            units=units,
            persist_dir=cfg["persist"],
            embed_model=cfg["embed"],
            llm_model=cfg["llm"],
            k=args.k,
            velocity_n=args.velocity_n,
        )
        results[config_id] = result
        spent = sum(u.cost_usd for u in result.units)
        budget_spent += spent
        print(f"   spent ${spent:.4f} on quality run; total ${budget_spent:.4f}")
        if budget_spent > args.budget_usd:
            print(f"   budget ${args.budget_usd:.2f} exceeded; stopping.")
            break

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "benchmark": {
            "name": "llamaindex_vs_curia_baseline",
            "n_units": len(units),
            "n_corpus_docs": len(docs),
            "retrieval_top_k": args.k,
            "velocity_unique_queries": args.velocity_n,
            "workload_size_for_projection": args.workload_size,
            "budget_usd": args.budget_usd,
            "budget_spent_usd": round(budget_spent, 4),
        },
        "configs": {cid: r.aggregate(args.workload_size) for cid, r in results.items()},
        "rows": [
            {
                "config_id": cid,
                "unit_id": u.unit_id,
                "n_retrieved": u.n_retrieved,
                "citation_precision": u.citation_precision,
                "hallucination_rate": u.hallucination_rate,
                "evidence_coverage": u.evidence_coverage,
                "latency_s": u.latency_s,
                "cost_usd": u.cost_usd,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "missing_ids": u.missing_ids,
            }
            for cid, r in results.items()
            for u in r.units
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print(f"\n=== SUMMARY ===")
    for cid, agg in report["configs"].items():
        q = agg["quality"]
        v = agg["velocity_projection"]
        print(
            f"  {cid:12s} cite={q['citation_precision']:.3f} "
            f"cov={q['evidence_coverage']:.3f} "
            f"lat={q['mean_latency_s']:.2f}s "
            f"cost/run=${q['mean_cost_usd']:.4f} "
            f"proj_1k=${v['projected_total_cost_usd']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
