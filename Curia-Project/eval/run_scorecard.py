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
  Layer 4  Multi-LLM        faithfulness/latency per model run_multi_llm_eval.py
  Layer 5  Velocity         cache hit-rate / latency       scripts/bench_cache.py

Usage
    python3 eval/run_scorecard.py
    python3 eval/run_scorecard.py --models local,gpt-5.4-mini --budget-usd 20 --out results/
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
    parser.add_argument("--out", help="Directory to write the scorecard JSON into")
    args = parser.parse_args()

    layers: dict[str, dict] = {}

    layers["retrieval"] = _lastline_eval("run_retrieval_eval.py")
    layers["faithfulness"] = _lastline_eval("run_faithfulness_eval.py")
    layers["relevance"] = _lastline_eval("run_relevance_eval.py")
    layers["adversarial"] = _lastline_eval("run_adversarial_eval.py")
    layers["forecast_backtest"] = _outfile_eval("run_forecast_backtest.py", [])
    layers["ground_truth"] = _outfile_eval("run_ground_truth_eval.py", [])
    multi_args = ["--models", args.models, "--budget-usd", str(args.budget_usd),
                  "--local-runtime", args.local_runtime]
    if args.provider:
        multi_args.extend(["--provider", args.provider])
    if args.max_items is not None:
        multi_args.extend(["--max-items", str(args.max_items)])
    layers["multi_llm"] = _outfile_eval("run_multi_llm_eval.py", multi_args)
    layers["velocity"] = _bench(ROOT)

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
        "agent_a_vs_bls_spearman": _get("ground_truth", "agent_a_vs_bls", "spearman_rho"),
        "layers_ok": [k for k, v in layers.items() if v.get("status") == "ok"],
        "layers_failed": [k for k, v in layers.items() if v.get("status") != "ok"],
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
