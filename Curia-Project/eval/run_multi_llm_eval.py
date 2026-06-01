"""
Multi-LLM evaluation — compares generators on the SAME retrieved evidence.

Retrieval + Agents A-E are held fixed; only the recommendation generator is
swapped. Faithfulness is scored automatically against the retrieved evidence
(hard ground truth — no human labels needed), plus latency, evidence coverage,
estimated cost, and skip reasons.

Usage
    python3 eval/run_multi_llm_eval.py
    python3 eval/run_multi_llm_eval.py --models local,gpt-5.4-mini,claude-sonnet-4-6
    python3 eval/run_multi_llm_eval.py --provider openai,anthropic --budget-usd 20
    python3 eval/run_multi_llm_eval.py --corpus-dir data/corpus_large --index-path audit/faiss_index_large.pkl
    python3 eval/run_multi_llm_eval.py --units-file data/eval/benchmark_units_tamu_50.json
    python3 eval/run_multi_llm_eval.py --models local,ollama:qwen2.5:7b --local-runtime ollama
    python3 eval/run_multi_llm_eval.py --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.benchmarking import can_afford, estimate_prompt_cost, usage_cost
from src.config import (
    CORPUS_DIR,
    EVAL_TARGET_CITATION_PRECISION,
    INDEX_PATH,
    LLM_MAX_TOKENS,
    RETRIEVAL_K,
    SOURCE_QUOTAS,
    UNITS_FILE,
)
from src.grounding import check_citations
from src.indexing import FaissIndex
from src.llm_providers import (
    LocalRuntimeUnavailableError,
    ProviderUnavailableError,
    make_generator,
)
from src.model_registry import ModelSpec, parse_csv, select_model_specs
from src.prompts import build_recommendation_prompt
from src.query import build_query
from src.retrieval import Retriever
from src.storage import build_index_from_corpus, load_units


def _load_retriever(corpus_dir: Path, index_path: Path, rebuild_index: bool) -> Retriever:
    if index_path.exists() and not rebuild_index:
        index = FaissIndex.load(index_path)
    else:
        index = build_index_from_corpus(corpus_dir)
        index.save(index_path)
    return Retriever(index)


def _score(rec, evidence) -> dict:
    cc = check_citations(rec, evidence)
    cited = cc.cited_ids
    retrieved = set(cc.retrieved_ids)
    in_set = [c for c in cited if c in retrieved]
    precision = 1.0 if not cited else len(in_set) / len(cited)
    coverage = (len(set(in_set)) / len(retrieved)) if retrieved else 0.0
    return {
        "citation_precision": round(precision, 4),
        "hallucination_rate": round(1.0 - precision, 4),
        "evidence_coverage": round(coverage, 4),
        "n_cited": len(cited),
        "n_missing": len(cc.missing_ids),
        "cited_ids": cited,
        "missing_ids": cc.missing_ids,
        "signal_strength": rec.signal_strength,
    }


def _model_meta(spec: ModelSpec) -> dict:
    return {
        "model": spec.name,
        "provider": spec.provider,
        "model_id": spec.model_id,
        "category": spec.category,
        "adapter": spec.adapter,
    }


def _skip_model(spec: ModelSpec, reason: str, detail: str) -> dict:
    return {**_model_meta(spec), "skip_reason": reason, "detail": detail}


def _row_skip(spec: ModelSpec, unit_id: str, reason: str, detail: str, projected_cost: float) -> dict:
    return {
        **_model_meta(spec),
        "unit_id": unit_id,
        "skipped": True,
        "skip_reason": reason,
        "detail": detail,
        "cost_estimate_usd": round(projected_cost, 8),
    }


def _is_hard_provider_error(detail: str) -> bool:
    text = detail.lower()
    hard_markers = (
        "credit balance is too low",
        "insufficient_quota",
        "quota exceeded",
        "billing",
        "payment required",
        "invalid api key",
        "authentication",
        "permission denied",
        "unauthorized",
        "forbidden",
    )
    return any(marker in text for marker in hard_markers)


def _usage_from_generator(generator) -> dict[str, int | None] | None:
    return getattr(generator, "last_usage", None)


def _unit_ids(units: list[dict], requested_units: str | None, max_items: int | None) -> list[str]:
    ids = parse_csv(requested_units) if requested_units else [unit["id"] for unit in units]
    if max_items is not None:
        ids = ids[:max(0, max_items)]
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="Comma-separated registry names or provider model IDs")
    parser.add_argument("--provider", help="Comma-separated provider filter, e.g. openai,anthropic,openrouter")
    parser.add_argument("--units", help="Comma-separated unit ids")
    parser.add_argument("--units-file", default=str(UNITS_FILE),
                        help="JSON file containing benchmark units")
    parser.add_argument("--max-items", type=int, help="Maximum number of unit/query items to evaluate")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                        help="Directory of corpus JSON files to retrieve from")
    parser.add_argument("--index-path", default=str(INDEX_PATH),
                        help="FAISS index path for the selected corpus")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild and overwrite the selected index before evaluation")
    parser.add_argument("--no-source-quotas", action="store_true",
                        help="Retrieve top-k evidence without fixed source quotas")
    parser.add_argument("--budget-usd", type=float, default=20.0,
                        help="Hard preflight budget cap for metered API calls (default: 20)")
    parser.add_argument("--local-runtime", choices=["none", "ollama"], default="none",
                        help="Optional runtime for local open-weight models")
    parser.add_argument("--out", help="Directory to also write a JSON report into")
    args = parser.parse_args()

    requested_models = parse_csv(args.models)
    requested_providers = parse_csv(args.provider)
    specs = select_model_specs(requested_models, requested_providers)

    units_file = Path(args.units_file)
    units = load_units(units_file)
    unit_ids = _unit_ids(units, args.units, args.max_items)
    by_id = {unit["id"]: unit for unit in units}
    missing_units = [unit_id for unit_id in unit_ids if unit_id not in by_id]
    if missing_units:
        raise SystemExit(f"Unknown unit id(s): {', '.join(missing_units)}")

    corpus_dir = Path(args.corpus_dir)
    index_path = Path(args.index_path)
    retriever = _load_retriever(corpus_dir, index_path, args.rebuild_index)
    source_quotas = None if args.no_source_quotas else SOURCE_QUOTAS
    evidence_by_unit = {
        uid: retriever.retrieve(build_query(by_id[uid]), k=args.k, source_quotas=source_quotas)
        for uid in unit_ids
    }
    prompt_by_unit = {
        uid: build_recommendation_prompt(by_id[uid], evidence_by_unit[uid])
        for uid in unit_ids
    }

    budget_usd = None if args.budget_usd < 0 else args.budget_usd
    spent_usd = 0.0
    per_model: dict[str, dict] = {}
    rows: list[dict] = []
    skipped_models: list[dict] = []

    for spec in specs:
        if spec.api_key_env and not spec.has_credentials():
            skipped_models.append(_skip_model(
                spec, "missing_key", f"{spec.api_key_env} is not set"
            ))
            continue

        output_tokens = spec.max_tokens or LLM_MAX_TOKENS
        estimates = {
            uid: estimate_prompt_cost(spec, prompt, output_tokens)
            for uid, prompt in prompt_by_unit.items()
        }
        affordable_units = [
            uid for uid, estimate in estimates.items()
            if can_afford(spent_usd, estimate, budget_usd)
        ]
        if not affordable_units and unit_ids:
            skipped_models.append(_skip_model(
                spec,
                "projected_budget_overflow",
                f"remaining budget cannot cover the next projected call; spent=${spent_usd:.4f}",
            ))
            continue

        try:
            generator = make_generator(spec, local_runtime=args.local_runtime)
        except LocalRuntimeUnavailableError as exc:
            skipped_models.append(_skip_model(spec, "local_runtime_unavailable", str(exc)))
            continue
        except (EnvironmentError, ImportError) as exc:
            skipped_models.append(_skip_model(spec, "missing_key", str(exc)))
            continue
        except ProviderUnavailableError as exc:
            skipped_models.append(_skip_model(spec, "provider_error", str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001
            skipped_models.append(_skip_model(spec, "provider_error", f"{type(exc).__name__}: {exc}"))
            continue

        agg = {
            "citation_precision": 0.0,
            "hallucination_rate": 0.0,
            "evidence_coverage": 0.0,
            "latency_s": 0.0,
        }
        cost_total = 0.0
        pricing_sources: set[str] = set()
        n_ok = 0
        n_error = 0
        n_skipped = 0

        for idx, uid in enumerate(unit_ids):
            estimate = estimates[uid]
            if not can_afford(spent_usd, estimate, budget_usd):
                row = _row_skip(
                    spec,
                    uid,
                    "projected_budget_overflow",
                    f"projected call would exceed budget; spent=${spent_usd:.4f}",
                    estimate.cost_usd,
                )
                rows.append(row)
                print(json.dumps(row))
                n_skipped += 1
                continue

            evidence = evidence_by_unit[uid]
            try:
                t0 = time.perf_counter()
                rec = generator.generate(by_id[uid], evidence)
                latency = time.perf_counter() - t0
                actual_cost = usage_cost(spec, _usage_from_generator(generator), estimate)
                metrics = _score(rec, evidence)
                metrics["latency_s"] = round(latency, 4)
            except Exception as exc:  # noqa: BLE001
                error_detail = f"{type(exc).__name__}: {exc}"
                row = {
                    **_model_meta(spec),
                    "unit_id": uid,
                    "error": error_detail,
                    "skip_reason": "provider_error",
                }
                rows.append(row)
                print(json.dumps(row))
                n_error += 1
                if _is_hard_provider_error(error_detail):
                    for remaining_uid in unit_ids[idx + 1:]:
                        remaining = _row_skip(
                            spec,
                            remaining_uid,
                            "provider_error_hard_stop",
                            f"stopped after hard provider error on {uid}: {error_detail}",
                            estimates[remaining_uid].cost_usd,
                        )
                        rows.append(remaining)
                        print(json.dumps(remaining))
                        n_skipped += 1
                    break
                continue

            spent_usd += actual_cost.cost_usd
            cost_total += actual_cost.cost_usd
            pricing_sources.add(actual_cost.pricing_source)
            row = {
                **_model_meta(spec),
                "unit_id": uid,
                **metrics,
                "input_tokens_estimate": estimate.input_tokens,
                "output_tokens_estimate": estimate.output_tokens,
                "input_tokens_billed_or_estimated": actual_cost.input_tokens,
                "output_tokens_billed_or_estimated": actual_cost.output_tokens,
                "cost_estimate_usd": round(actual_cost.cost_usd, 8),
                "pricing_source": actual_cost.pricing_source,
            }
            rows.append(row)
            print(json.dumps(row))
            for key in agg:
                agg[key] += metrics[key]
            n_ok += 1

        summary = {
            **_model_meta(spec),
            "n": n_ok,
            "n_errors": n_error,
            "n_skipped": n_skipped,
            "cost_estimate_usd": round(cost_total, 8),
            "pricing_sources": sorted(pricing_sources),
        }
        if n_ok:
            summary.update({key: round(value / n_ok, 4) for key, value in agg.items()})
            summary["passes_citation_target"] = (
                summary["citation_precision"] >= EVAL_TARGET_CITATION_PRECISION
            )
        per_model[spec.name] = summary

    ranking = sorted(
        [name for name, values in per_model.items() if "citation_precision" in values],
        key=lambda name: (-per_model[name]["citation_precision"], per_model[name]["latency_s"]),
    )
    report = {
        "models_requested": requested_models or None,
        "models_planned": [spec.name for spec in specs],
        "providers_requested": requested_providers or None,
        "units_file": str(units_file),
        "n_queries": len(unit_ids),
        "unit_ids": unit_ids,
        "corpus_dir": str(corpus_dir),
        "index_path": str(index_path),
        "source_quotas_enabled": source_quotas is not None,
        "target_citation_precision": EVAL_TARGET_CITATION_PRECISION,
        "budget_usd": budget_usd,
        "cost_estimate_usd_total": round(spent_usd, 8),
        "local_runtime": args.local_runtime,
        "per_model": per_model,
        "skipped_models": skipped_models,
        "ranking_by_faithfulness_then_latency": ranking,
    }
    print(json.dumps({"comparison": report}, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"multi_llm_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps({"rows": rows, "comparison": report}, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
