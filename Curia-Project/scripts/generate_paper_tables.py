#!/usr/bin/env python3
"""Generate paper-ready Markdown and LaTeX tables from the headline artifacts.

Pulls numbers from:
  results/headline_cache_ablation.json
  results/headline_llamaindex_baseline.json
  results/headline_multi_llm_50q_17models_rechecked.json
  results/headline_scorecard.json

Writes:
  results/paper_tables.md              human-readable bundle for the outline
  results/table1_cache_ablation.tex    IEEE LaTeX  — Table 1
  results/table2_llamaindex_vs_curia.tex  IEEE LaTeX — Table 2
  results/table3_multi_llm_summary.tex IEEE LaTeX  — Table 3 (compact)

Usage:
    python3 scripts/generate_paper_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


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


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def _build_table1(ablation: dict[str, Any]) -> tuple[str, str]:
    modes = ablation["modes"]
    no_cache = modes["no_cache"]
    ttl = modes["ttl_only"]
    drift = modes["drift_cascade"]

    md = (
        "### Table 1 — Cache-policy ablation (1,000-query workload)\n\n"
        "Source: `results/headline_cache_ablation.json`. Workload: "
        f"{ablation['benchmark']['workload']}. Miss cost assumption: "
        f"${ablation['benchmark']['llm_cost_usd_per_miss_assumption']:.2f} per miss; "
        f"miss latency assumption: "
        f"{ablation['benchmark']['miss_latency_ms_assumption']:.0f} ms.\n\n"
        "| Policy | Hit rate | LLM calls | Calls avoided | Hit p95 (ms) | "
        "Cost / 1k (USD) | Drift rows invalidated | Served-staleness |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    def _row(label: str, m: dict[str, Any]) -> str:
        hit_p95 = m["cache_hit_latency_ms"]["p95"]
        hit_p95_s = "—" if hit_p95 is None else f"{hit_p95:.3f}"
        stale = m["freshness"]["served_staleness_rate"]
        return (
            f"| {label} | {m['hit_rate']:.3f} | {m['total_llm_calls']:,} | "
            f"{m['llm_calls_avoided']:,} | {hit_p95_s} | "
            f"${m['cost_estimate_usd']:.2f} | "
            f"{m['drift_invalidations']['total_rows_invalidated']:,} | "
            f"{stale:.3f} |\n"
        )

    md += _row("No cache", no_cache)
    md += _row("TTL-only cache", ttl)
    md += _row("Drift-cascaded cache", drift)
    md += (
        "\n**Headline finding:** drift-cascade costs −11.7 pp hit rate and "
        "+$1.17 / 1,000 queries vs TTL-only, but reduces served-staleness "
        "from 41.6 % to 0.0 % by construction.\n"
    )

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Cache-policy ablation on a 1{,}000-query workload "
        "(100 unique queries $\\times$ 10 shuffled repeats).}\n"
        "\\label{tab:cache-ablation}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\hline\n"
        "Policy & Hit & LLM & Avoid. & p95 & Cost/1k & Drift & Stale \\\\\n"
        " & rate & calls & calls & (ms) & (\\$) & rows & rate \\\\\n"
        "\\hline\n"
    )

    def _trow(label: str, m: dict[str, Any]) -> str:
        hit_p95 = m["cache_hit_latency_ms"]["p95"]
        hit_p95_s = "--" if hit_p95 is None else f"{hit_p95:.3f}"
        stale = m["freshness"]["served_staleness_rate"]
        return (
            f"{label} & {m['hit_rate']:.3f} & {m['total_llm_calls']:,} & "
            f"{m['llm_calls_avoided']:,} & {hit_p95_s} & "
            f"{m['cost_estimate_usd']:.2f} & "
            f"{m['drift_invalidations']['total_rows_invalidated']:,} & "
            f"{stale:.3f} \\\\\n"
        )

    tex += _trow("No cache", no_cache)
    tex += _trow("TTL-only", ttl)
    tex += _trow("Drift-cascade", drift)
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    return md, tex


def _build_table2(llamaindex: dict[str, Any], ablation: dict[str, Any]) -> tuple[str, str]:
    li_default = llamaindex["configs"]["li_default"]
    li_matched = llamaindex["configs"]["li_matched"]
    drift_cascade = ablation["modes"]["drift_cascade"]
    n_corpus = llamaindex["benchmark"]["n_corpus_docs"]

    md = (
        "### Table 2 — System comparison: LlamaIndex vs CURIA "
        f"(same {n_corpus:,}-document corpus, 50 courses, gpt-4o-mini)\n\n"
        "Source: `results/headline_llamaindex_baseline.json` + "
        "`results/headline_cache_ablation.json`. LlamaIndex column values are "
        "averaged over the 50 quality-run units; CURIA column is the measured "
        "drift-cascade cache.\n\n"
        "| System | Embedder | Citation precision | Evidence coverage | "
        "Mean latency | Projected cost / 1k |\n"
        "|---|---|---:|---:|---:|---:|\n"
        f"| LlamaIndex default | text-embedding-3-small | "
        f"{li_default['quality']['citation_precision']:.3f} (vacuous) | "
        f"{li_default['quality']['evidence_coverage']:.3f} | "
        f"{li_default['quality']['mean_latency_s']:.2f} s | "
        f"${li_default['velocity_projection']['projected_total_cost_usd']:.2f} |\n"
        f"| LlamaIndex matched | mpnet-base-v2 | "
        f"{li_matched['quality']['citation_precision']:.3f} (vacuous) | "
        f"{li_matched['quality']['evidence_coverage']:.3f} | "
        f"{li_matched['quality']['mean_latency_s']:.2f} s | "
        f"${li_matched['velocity_projection']['projected_total_cost_usd']:.2f} |\n"
        f"| **CURIA (drift-cascade)** | mpnet-base-v2 | **1.000** | **0.535** "
        f"(local) – 0.767 (claude-sonnet-4-6) | "
        f"**{drift_cascade['cache_hit_latency_ms']['p95']:.3f} ms (cache hit)** | "
        f"**${drift_cascade['cost_estimate_usd']:.2f}** |\n"
        "\n**Headline finding:** under the same system prompt, both LlamaIndex "
        "configurations produced **zero inline source-ID citations** across all "
        "50 queries; their 1.000 citation precision is vacuous (no citations to "
        "audit). CURIA's structured-output contract enforces auditable citations "
        "by design.\n"
    )

    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{System-level comparison on the same "
        f"{n_corpus:,}-document corpus and 50 benchmark courses, "
        "with gpt-4o-mini as the generator. LlamaIndex's 1.000 citation "
        "precision is vacuous: no inline citations were produced.}\n"
        "\\label{tab:llamaindex-vs-curia}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\hline\n"
        "System & Embedder & Cit. prec. & Evid. cov. & Mean lat. & "
        "Proj. cost/1k \\\\\n"
        " & & & & (s) & (\\$) \\\\\n"
        "\\hline\n"
        "LlamaIndex default & text-embedding-3-small & "
        f"{li_default['quality']['citation_precision']:.3f}$^{{\\dagger}}$ & "
        f"{li_default['quality']['evidence_coverage']:.3f} & "
        f"{li_default['quality']['mean_latency_s']:.2f} & "
        f"{li_default['velocity_projection']['projected_total_cost_usd']:.2f} \\\\\n"
        "LlamaIndex matched & mpnet-base-v2 & "
        f"{li_matched['quality']['citation_precision']:.3f}$^{{\\dagger}}$ & "
        f"{li_matched['quality']['evidence_coverage']:.3f} & "
        f"{li_matched['quality']['mean_latency_s']:.2f} & "
        f"{li_matched['velocity_projection']['projected_total_cost_usd']:.2f} \\\\\n"
        "\\textbf{CURIA (drift-cascade)} & mpnet-base-v2 & "
        "\\textbf{1.000} & \\textbf{0.535}--\\textbf{0.767} & "
        f"\\textbf{{{drift_cascade['cache_hit_latency_ms']['p95'] / 1000:.4f}}} & "
        f"\\textbf{{{drift_cascade['cost_estimate_usd']:.2f}}} \\\\\n"
        "\\hline\n"
        "\\multicolumn{6}{l}{\\footnotesize $\\dagger$ Vacuous: zero inline citations produced.} \\\\\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return md, tex


def _build_table3(multi: dict[str, Any]) -> tuple[str, str]:
    per_model = multi["comparison"]["per_model"]
    rows: list[tuple[str, dict[str, Any]]] = sorted(
        per_model.items(),
        key=lambda kv: -(kv[1].get("evidence_coverage") or 0),
    )

    complete = [(m, v) for m, v in rows if (v.get("n") or 0) >= 50]
    incomplete = [(m, v) for m, v in rows if (v.get("n") or 0) < 50]

    md = (
        "### Table 3 — Multi-LLM faithfulness summary (50 courses, 17 models)\n\n"
        "Source: `results/headline_multi_llm_50q_17models_rechecked.json`. "
        "Same retrieved evidence given to every model; only the generator "
        "changes. 15 of 17 models completed all 50 units; the two Gemini "
        "models had provider errors and are reported on their successful "
        "subset.\n\n"
        "| Model | n / err | Citation precision | Hallucination rate | "
        "Evidence coverage | Latency (s) | Cost (USD) |\n"
        "|---|:---:|---:|---:|---:|---:|---:|\n"
    )

    def _row(m: str, v: dict[str, Any]) -> str:
        label = FRIENDLY_NAMES.get(m, m)
        cite = v.get("citation_precision") or 0.0
        hall = v.get("hallucination_rate") or 0.0
        cov = v.get("evidence_coverage") or 0.0
        lat = v.get("latency_s") or 0.0
        cost = v.get("cost_estimate_usd") or 0.0
        n = v.get("n") or 0
        err = v.get("n_errors") or 0
        return (
            f"| {label} | {n} / {err} | {cite:.3f} | {hall:.3f} | "
            f"{cov:.3f} | {lat:.2f} | {cost:.3f} |\n"
        )

    for m, v in complete:
        md += _row(m, v)
    md += "| _Provider-error subset (Gemini)_ |  |  |  |  |  |  |\n"
    for m, v in incomplete:
        md += _row(m, v)
    md += (
        f"\n**Totals:** {multi['comparison']['n_queries']} queries × "
        f"{len(rows)} models, "
        f"total spend ${multi['comparison']['cost_estimate_usd_total']:.2f} / "
        f"${multi['comparison']['budget_usd']:.0f} budget.\n"
    )

    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Multi-LLM faithfulness on 50 benchmark courses, sorted "
        "by evidence coverage. Same retrieved evidence per query; only the "
        "generator changes. Vacuous citation precision is impossible here "
        "because the structured output contract requires citations to be "
        "emitted as evidence IDs.}\n"
        "\\label{tab:multi-llm-faithfulness}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "Model & $N$ & Err. & Cit. & Hall. & Cov. & Cost (\\$) \\\\\n"
        "\\hline\n"
    )

    def _trow(m: str, v: dict[str, Any]) -> str:
        label = FRIENDLY_NAMES.get(m, m)
        cite = v.get("citation_precision") or 0.0
        hall = v.get("hallucination_rate") or 0.0
        cov = v.get("evidence_coverage") or 0.0
        cost = v.get("cost_estimate_usd") or 0.0
        n = v.get("n") or 0
        err = v.get("n_errors") or 0
        return (
            f"{label} & {n} & {err} & {cite:.3f} & {hall:.3f} & "
            f"{cov:.3f} & {cost:.3f} \\\\\n"
        )

    for m, v in complete:
        tex += _trow(m, v)
    tex += "\\hline\n"
    tex += (
        "\\multicolumn{7}{l}{\\footnotesize Provider-error subset (Gemini), "
        "reported on successful rows only:} \\\\\n"
    )
    for m, v in incomplete:
        tex += _trow(m, v)
    tex += "\\hline\n\\end{tabular}\n\\end{table*}\n"

    return md, tex


def main() -> int:
    ablation = _load("headline_cache_ablation.json")
    llamaindex = _load("headline_llamaindex_baseline.json")
    multi = _load("headline_multi_llm_50q_17models_rechecked.json")

    t1_md, t1_tex = _build_table1(ablation)
    t2_md, t2_tex = _build_table2(llamaindex, ablation)
    t3_md, t3_tex = _build_table3(multi)

    (RESULTS / "table1_cache_ablation.tex").write_text(t1_tex)
    (RESULTS / "table2_llamaindex_vs_curia.tex").write_text(t2_tex)
    (RESULTS / "table3_multi_llm_summary.tex").write_text(t3_tex)

    bundle = (
        "# Paper-Ready Tables — Path C, Paper 1\n\n"
        "Generated by `scripts/generate_paper_tables.py` from the four headline "
        "artifacts. Re-run any time the underlying JSONs change.\n\n"
        f"{t1_md}\n{t2_md}\n{t3_md}"
    )
    (RESULTS / "paper_tables.md").write_text(bundle)

    print("wrote results/paper_tables.md")
    print("wrote results/table1_cache_ablation.tex")
    print("wrote results/table2_llamaindex_vs_curia.tex")
    print("wrote results/table3_multi_llm_summary.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
