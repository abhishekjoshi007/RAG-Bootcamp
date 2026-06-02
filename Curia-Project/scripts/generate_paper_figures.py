#!/usr/bin/env python3
"""Generate paper figures from the headline artifacts.

Writes both PNG (preview) and PDF (paper-ready, vector) for each figure under
`results/figures/`. Designed to be camera-ready: greyscale-safe palette, sans
font, small legend, IEEE-conference column widths.

Figures:
  fig1_cache_ablation.{png,pdf}     — hit rate vs served-staleness vs cost,
                                       three policies (paper headline)
  fig2_latency_comparison.{png,pdf} — log-scale latency: LlamaIndex cold path
                                       vs CURIA cache hit/miss
  fig3_multi_llm_coverage.{png,pdf} — evidence-coverage bar chart for the
                                       15 complete-run models

Usage:
    python3 scripts/generate_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote results/figures/{name}.png")
    print(f"wrote results/figures/{name}.pdf")


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def figure_cache_ablation(ablation: dict[str, Any]) -> None:
    modes = ablation["modes"]
    labels = ["No cache", "TTL-only", "Drift-cascade"]
    keys = ["no_cache", "ttl_only", "drift_cascade"]
    hit_rates = [modes[k]["hit_rate"] for k in keys]
    staleness = [modes[k]["freshness"]["served_staleness_rate"] for k in keys]
    costs = [modes[k]["cost_estimate_usd"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.3))
    x = list(range(len(labels)))
    width = 0.32

    bars_hit = ax1.bar(
        [p - width for p in x],
        hit_rates,
        width=width,
        color="#2b7bba",
        label="Hit rate",
    )
    bars_stale = ax1.bar(
        x,
        staleness,
        width=width,
        color="#e58a2b",
        label="Served-staleness",
    )
    ax1.set_ylabel("Rate (0–1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    bars_cost = ax2.bar(
        [p + width for p in x],
        costs,
        width=width,
        color="#666666",
        label="Cost / 1k (USD)",
    )
    ax2.set_ylabel("Cost / 1,000 queries (USD)")
    ax2.set_ylim(0, max(costs) * 1.25)
    ax2.spines["top"].set_visible(False)

    for bars, values, fmt in (
        (bars_hit, hit_rates, "{:.2f}"),
        (bars_stale, staleness, "{:.2f}"),
    ):
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for bar, val in zip(bars_cost, costs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(costs) * 0.02,
            f"${val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        frameon=False,
    )
    ax1.set_title(
        "Cache-policy ablation: drift-cascade buys 0% served-staleness "
        "at +$1.17 / 1k",
        pad=8,
    )
    _save(fig, "fig1_cache_ablation")


def figure_latency_comparison(
    llamaindex: dict[str, Any],
    ablation: dict[str, Any],
) -> None:
    li_default = llamaindex["configs"]["li_default"]["quality"]
    li_matched = llamaindex["configs"]["li_matched"]["quality"]
    drift = ablation["modes"]["drift_cascade"]
    miss_assumption_ms = ablation["benchmark"]["miss_latency_ms_assumption"]

    labels = [
        "LlamaIndex\ndefault\n(cold)",
        "LlamaIndex\nmatched\n(cold)",
        "CURIA\nmiss\n(assumed)",
        "CURIA\ncache hit\n(p50)",
        "CURIA\ncache hit\n(p95)",
    ]
    values_ms = [
        li_default["mean_latency_s"] * 1000.0,
        li_matched["mean_latency_s"] * 1000.0,
        miss_assumption_ms,
        drift["cache_hit_latency_ms"]["p50"],
        drift["cache_hit_latency_ms"]["p95"],
    ]
    colors = ["#aaaaaa", "#777777", "#cccccc", "#2b7bba", "#1c4d75"]

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    bars = ax.bar(labels, values_ms, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title(
        "End-to-end query latency: LlamaIndex cold-path vs CURIA cache hit",
        pad=8,
    )
    for bar, val in zip(bars, values_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val:.3f}" if val < 10 else f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.1, max(values_ms) * 5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig2_latency_comparison")


def figure_multi_llm_coverage(multi: dict[str, Any]) -> None:
    per_model = multi["comparison"]["per_model"]
    complete = sorted(
        [
            (m, v)
            for m, v in per_model.items()
            if (v.get("n") or 0) >= 50
        ],
        key=lambda kv: -(kv[1].get("evidence_coverage") or 0),
    )
    friendly = {
        "local": "Local extractive",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
        "claude-opus-4-7": "Claude Opus 4.7",
        "claude-opus-4-6": "Claude Opus 4.6",
        "claude-opus-4-8": "Claude Opus 4.8",
        "deepseek-reasoner": "DeepSeek Reasoner",
        "deepseek-chat": "DeepSeek Chat",
        "claude-haiku-4-5": "Claude Haiku 4.5",
        "gpt-5.5": "GPT-5.5",
        "grok-4.3": "Grok 4.3",
        "gpt-5.4-nano": "GPT-5.4 Nano",
        "gpt-5.4": "GPT-5.4",
        "gpt-5.4-mini": "GPT-5.4 Mini",
        "claude-sonnet-4-5": "Claude Sonnet 4.5",
    }
    labels = [friendly.get(m, m) for m, _ in complete]
    coverages = [(v.get("evidence_coverage") or 0.0) for _, v in complete]
    colors = ["#1c4d75" if "local" in m else "#2b7bba" for m, _ in complete]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.barh(labels[::-1], coverages[::-1], color=colors[::-1])
    ax.set_xlabel("Evidence coverage (mean over 50 courses)")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Evidence coverage discriminates models;\n"
        "citation precision is 1.000 / hallucination 0.000 for all 15 "
        "complete-run models",
        pad=8,
    )
    for bar, val in zip(bars, coverages[::-1]):
        ax.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7,
        )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    _save(fig, "fig3_multi_llm_coverage")


def main() -> int:
    ablation = _load("headline_cache_ablation.json")
    llamaindex = _load("headline_llamaindex_baseline.json")
    multi = _load("headline_multi_llm_50q_17models_rechecked.json")

    figure_cache_ablation(ablation)
    figure_latency_comparison(llamaindex, ablation)
    figure_multi_llm_coverage(multi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
