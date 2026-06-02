"""
Unified evaluation scorecard — the single reproducible artifact for the paper.

Runs every evaluation layer as a subprocess (so one failing layer never aborts
the rest) and aggregates the headline metrics + pass/fail into one JSON:

  Layer 1  Retrieval        recall@k / MRR / nDCG          run_retrieval_eval.py
  Layer 2  Faithfulness     citation precision, grounding  run_faithfulness_eval.py
           Relevance        mean human rating              run_relevance_eval.py
           Adversarial      recall drop under perturbation run_adversarial_eval.py
  Layer 3  Forecast (RQ2)   MAPE/sMAPE/MASE/dir-acc        run_forecast_backtest.py
           Ground truth     Agent A vs BLS, CS2023 cover   run_ground_truth_eval.py
  Layer 4  Multi-LLM (live) faithfulness/latency per model run_multi_llm_eval.py
  Layer 5  Velocity         cache hit-rate / latency       scripts/bench_cache.py

Path-C paper headline artifacts (read-only; produced by their own scripts):

  cache_ablation     served-staleness / hit-rate trade  results/headline_cache_ablation.json
  llamaindex         baseline vs CURIA on same corpus    results/headline_llamaindex_baseline.json
  multi_llm_headline 17-LLM n=50 rechecked run           results/headline_multi_llm_50q_17models_rechecked.json

Usage
    python3 eval/run_scorecard.py
    python3 eval/run_scorecard.py --models local,gpt-5.4-mini --budget-usd 20 --out results/
    python3 eval/run_scorecard.py --forecast-corpus data/corpus_large --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = sys.executable
ENV = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}

from src.config import (  # noqa: E402
    EVAL_TARGET_ADVERSARIAL_DROP,
    EVAL_TARGET_RECALL_8,
    EVAL_TARGET_RELEVANCE_MEAN,
)


def _run(argv: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, *argv], cwd=ROOT, env=ENV,
                          capture_output=True, text=True, timeout=timeout)


def _last_json_line(stdout: str) -> dict | None:
    for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _outfile_eval(script: str, extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run([f"eval/{script}", "--out", tmp, *extra])
        if proc.returncode != 0:
            return {"status": "error", "stderr": proc.stderr.strip()[-400:]}
        files = sorted(Path(tmp).glob("*.json"))
        if not files:
            return {"status": "error", "stderr": "no output file"}
        return {"status": "ok", "result": json.loads(files[-1].read_text())}


def _lastline_eval(script: str) -> dict:
    proc = _run([f"eval/{script}"])
    if proc.returncode != 0:
        return {"status": "error", "stderr": proc.stderr.strip()[-400:]}
    obj = _last_json_line(proc.stdout)
    return {"status": "ok", "result": obj} if obj else {
        "status": "error", "stderr": "no JSON aggregate parsed"}


def _bench(out_dir: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(["scripts/bench_cache.py", "--out", tmp])
        if proc.returncode != 0:
            return {"status": "error", "stderr": proc.stderr.strip()[-400:]}
        files = sorted(Path(tmp).glob("*.json"))
        return {"status": "ok", "result": json.loads(files[-1].read_text())} if files else {
            "status": "error", "stderr": "no output file"}


def _read_artifact(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "stderr": f"artifact {path} not found"}
    try:
        return {"status": "ok", "result": json.loads(path.read_text())}
    except json.JSONDecodeError as exc:
        return {"status": "error", "stderr": f"could not parse {path}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="local",
                        help="Models for the multi-LLM layer (default: local only)")
    parser.add_argument("--provider", help="Provider filter for the multi-LLM layer")
    parser.add_argument("--budget-usd", type=float, default=20.0,
                        help="Budget cap passed to run_multi_llm_eval.py")
    parser.add_argument("--max-items", type=int,
                        help="Maximum multi-LLM unit/query items to evaluate")
    parser.add_argument("--local-runtime", choices=["none", "ollama"], default="none",
                        help="Optional local runtime passed to run_multi_llm_eval.py")
    parser.add_argument("--forecast-corpus", default=str(ROOT / "data" / "corpus_large"),
                        help="Corpus for the forecast backtest (defaults to data/corpus_large)")
    parser.add_argument("--out", help="Directory to write the scorecard JSON into")
    args = parser.parse_args()

    layers: dict[str, dict] = {}

    layers["retrieval"] = _lastline_eval("run_retrieval_eval.py")
    layers["faithfulness"] = _lastline_eval("run_faithfulness_eval.py")
    layers["relevance"] = _lastline_eval("run_relevance_eval.py")
    layers["adversarial"] = _lastline_eval("run_adversarial_eval.py")
    forecast_args = ["--corpus-dir", args.forecast_corpus] if Path(args.forecast_corpus).is_dir() else []
    layers["forecast_backtest"] = _outfile_eval("run_forecast_backtest.py", forecast_args)
    layers["ground_truth"] = _outfile_eval("run_ground_truth_eval.py", [])
    multi_args = ["--models", args.models, "--budget-usd", str(args.budget_usd),
                  "--local-runtime", args.local_runtime]
    if args.provider:
        multi_args.extend(["--provider", args.provider])
    if args.max_items is not None:
        multi_args.extend(["--max-items", str(args.max_items)])
    layers["multi_llm"] = _outfile_eval("run_multi_llm_eval.py", multi_args)
    layers["velocity"] = _bench(ROOT)

    # Path-C paper headline artifacts (read-only references to results already
    # produced by their own scripts; do not re-run here).
    results_dir = ROOT / "results"
    layers["cache_ablation"] = _read_artifact(results_dir / "headline_cache_ablation.json")
    layers["llamaindex_baseline"] = _read_artifact(results_dir / "headline_llamaindex_baseline.json")
    layers["multi_llm_headline"] = _read_artifact(
        results_dir / "headline_multi_llm_50q_17models_rechecked.json"
    )

    # Headline pass/fail roll-up (best-effort; unknowns recorded as null)
    def _get(layer: str, *path):
        node: object = layers.get(layer, {}).get("result")
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return None
        return node

    recall8 = _get("retrieval", "aggregate", "recall@8")
    rel_mean = _get("relevance", "summary", "mean_rating")
    adv_drop = _get("adversarial", "adversarial_summary", "max_recall_drop")
    fb = layers["forecast_backtest"]
    fb_res = fb["result"] if fb.get("status") == "ok" else {}

    def _le(value, target):
        return None if value is None else value <= target

    def _ge(value, target):
        return None if value is None else value >= target

    ablation_modes = _get("cache_ablation", "modes") or {}
    ttl_mode = ablation_modes.get("ttl_only", {})
    drift_mode = ablation_modes.get("drift_cascade", {})
    li_configs = _get("llamaindex_baseline", "configs") or {}
    li_default_q = (li_configs.get("li_default") or {}).get("quality") or {}
    li_matched_q = (li_configs.get("li_matched") or {}).get("quality") or {}
    headline_models = _get("multi_llm_headline", "comparison", "per_model") or {}
    headline_complete = [
        m for m, v in headline_models.items()
        if (v.get("n") or 0) >= 50 and (v.get("citation_precision") or 0) >= 0.95
    ]

    overall = {
        "retrieval_recall@8": recall8,
        "retrieval_passed": _ge(recall8, EVAL_TARGET_RECALL_8),
        "faithfulness_passed": _get("faithfulness", "passed_target"),
        "relevance_mean": rel_mean,
        "relevance_passed": _ge(rel_mean, EVAL_TARGET_RELEVANCE_MEAN),
        "adversarial_max_drop": adv_drop,
        "adversarial_passed": _le(adv_drop, EVAL_TARGET_ADVERSARIAL_DROP),
        "velocity_hit_latency_p95_ms": _get("velocity", "deps_free",
                                            "recommendation_read_latency_ms", "hit_p95"),
        "velocity_meets_200ms": _get("velocity", "deps_free",
                                     "recommendation_read_latency_ms", "meets_200ms_target"),
        "forecast_data_sufficient": fb_res.get("data_sufficiency", {}).get("statistically_meaningful"),
        "forecast_best_non_naive_beats_naive": fb_res.get("best_non_naive_beats_naive"),
        "forecast_best_non_naive_model": fb_res.get("best_non_naive_by_smape"),
        "forecast_best_non_naive_smape_delta_vs_naive": fb_res.get("best_non_naive_smape_delta_vs_naive"),
        "agent_a_vs_bls_spearman": _get("ground_truth", "agent_a_vs_bls", "spearman_rho"),
        "cache_ablation_ttl_hit_rate": ttl_mode.get("hit_rate"),
        "cache_ablation_ttl_served_staleness": (ttl_mode.get("freshness") or {}).get("served_staleness_rate"),
        "cache_ablation_drift_hit_rate": drift_mode.get("hit_rate"),
        "cache_ablation_drift_served_staleness": (drift_mode.get("freshness") or {}).get("served_staleness_rate"),
        "cache_ablation_drift_cost_per_1k_usd": drift_mode.get("cost_estimate_usd"),
        "llamaindex_default_coverage": li_default_q.get("evidence_coverage"),
        "llamaindex_default_mean_latency_s": li_default_q.get("mean_latency_s"),
        "llamaindex_matched_coverage": li_matched_q.get("evidence_coverage"),
        "llamaindex_matched_mean_latency_s": li_matched_q.get("mean_latency_s"),
        "llamaindex_n_corpus_docs": _get("llamaindex_baseline", "benchmark", "n_corpus_docs"),
        "multi_llm_headline_complete_run_models": len(headline_complete),
        "multi_llm_headline_total_models_evaluated": len(headline_models),
        "multi_llm_headline_total_spend_usd": _get(
            "multi_llm_headline", "comparison", "cost_estimate_usd_total"
        ),
        "layers_ok": [k for k, v in layers.items() if v.get("status") == "ok"],
        "layers_failed": [k for k, v in layers.items() if v.get("status") != "ok"],
        "layers_missing": [k for k, v in layers.items() if v.get("status") == "missing"],
    }

    scorecard = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": overall,
        "layers": layers,
    }
    print(json.dumps(scorecard, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"scorecard_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(scorecard, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
