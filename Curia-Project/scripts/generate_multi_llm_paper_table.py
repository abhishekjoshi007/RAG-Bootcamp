from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FRIENDLY_NAMES = {
    "local": "Local extractive",
    "gpt-5.4-nano": "GPT-5.4 Nano",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-8": "Claude Opus 4.8",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "grok-4.3": "Grok 4.3",
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-reasoner": "DeepSeek Reasoner",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _patch_rows(
    rows: list[dict[str, Any]],
    patch_reports: list[dict[str, Any]],
    patch_models: set[str] | None,
) -> list[dict[str, Any]]:
    by_key = {(row.get("model"), row.get("unit_id")): dict(row) for row in rows}
    order = [(row.get("model"), row.get("unit_id")) for row in rows]
    for report in patch_reports:
        for row in report.get("rows", []):
            if "error" in row or row.get("skipped"):
                continue
            model = row.get("model")
            if patch_models and model not in patch_models:
                continue
            key = (model, row.get("unit_id"))
            if key in by_key:
                by_key[key] = dict(row)
    return [by_key[key] for key in order]


def _recompute_comparison(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = dict(report["comparison"])
    original = comparison["per_model"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("model", "")].append(row)

    per_model: dict[str, dict[str, Any]] = {}
    target = comparison.get("target_citation_precision", 0.95)
    spent = 0.0
    for model, old_summary in original.items():
        model_rows = grouped.get(model, [])
        ok_rows = [row for row in model_rows if "error" not in row and not row.get("skipped")]
        error_rows = [row for row in model_rows if "error" in row]
        skipped_rows = [row for row in model_rows if row.get("skipped")]
        summary = {
            "model": old_summary.get("model", model),
            "provider": old_summary.get("provider"),
            "model_id": old_summary.get("model_id"),
            "category": old_summary.get("category"),
            "adapter": old_summary.get("adapter"),
            "n": len(ok_rows),
            "n_errors": len(error_rows),
            "n_skipped": len(skipped_rows),
            "cost_estimate_usd": round(sum(row.get("cost_estimate_usd", 0.0) for row in ok_rows), 8),
            "pricing_sources": sorted({row.get("pricing_source") for row in ok_rows if row.get("pricing_source")}),
        }
        spent += summary["cost_estimate_usd"]
        if ok_rows:
            for metric in ("citation_precision", "hallucination_rate", "evidence_coverage", "latency_s"):
                summary[metric] = round(sum(row[metric] for row in ok_rows) / len(ok_rows), 4)
            summary["passes_citation_target"] = summary["citation_precision"] >= target
        per_model[model] = summary

    comparison["per_model"] = per_model
    comparison["cost_estimate_usd_total"] = round(spent, 8)
    comparison["ranking_by_faithfulness_then_latency"] = sorted(
        [name for name, values in per_model.items() if "citation_precision" in values],
        key=lambda name: (-per_model[name]["citation_precision"], per_model[name]["latency_s"]),
    )
    return comparison


def _format_table(comparison: dict[str, Any], caption: str, label: str) -> str:
    rows = sorted(
        comparison["per_model"].values(),
        key=lambda row: (
            -row.get("evidence_coverage", -1.0),
            -row.get("citation_precision", -1.0),
            row.get("cost_estimate_usd", 0.0),
        ),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & $N$ & Err. & Cit. & Hall. & Cov. & Cost (\$) \\",
        r"\hline",
    ]
    for row in rows:
        model = FRIENDLY_NAMES.get(row["model"], row["model"])
        lines.append(
            f"{_latex_escape(model)} & "
            f"{row.get('n', 0)} & "
            f"{row.get('n_errors', 0)} & "
            f"{row.get('citation_precision', 0.0):.3f} & "
            f"{row.get('hallucination_rate', 0.0):.3f} & "
            f"{row.get('evidence_coverage', 0.0):.3f} & "
            f"{row.get('cost_estimate_usd', 0.0):.3f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--patch-model", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tex-out", type=Path, required=True)
    parser.add_argument(
        "--caption",
        default=(
            "Multi-LLM citation faithfulness and evidence coverage on 50 TAMU "
            "benchmark units, sorted by evidence coverage."
        ),
    )
    parser.add_argument("--label", default="tab:multi-llm-faithfulness")
    args = parser.parse_args()

    report = _load(args.input)
    patches = [_load(path) for path in args.patch]
    patch_models = set(args.patch_model) if args.patch_model else None
    rows = _patch_rows(report["rows"], patches, patch_models)
    comparison = _recompute_comparison(report, rows)
    merged = {"rows": rows, "comparison": comparison}

    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(_format_table(comparison, args.caption, args.label))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
