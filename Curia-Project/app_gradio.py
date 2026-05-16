"""
CURIA — Curriculum Intelligence via Industry Agents
Single-page, step-by-step: Field → A (skills) → B → C → D (curriculum map)
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gradio as gr

from src.config import AUDIT_DB_PATH, CORPUS_DIR, INDEX_PATH, SOURCE_QUOTAS
from src.drift import DriftResult, SemanticDriftDetector
from src.field_config import get_ingestion_config
from src.forecasting import ForecastResult, SkillForecaster, _canonical_skill
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus
from src.university import ALL_FIELDS, get_all_units

_forecaster = SkillForecaster(history_months=40, forecast_months=12)

_drift_detector: SemanticDriftDetector | None = None


def _get_drift_detector() -> SemanticDriftDetector:
    global _drift_detector
    if _drift_detector is None:
        pipe = _get_pipeline()
        _drift_detector = SemanticDriftDetector(
            pipe.retriever.index,
            chunks_per_skill=40,
            min_chunks_per_bucket=2,
            drift_threshold=0.15,
            mode="cross_source",
        )
    return _drift_detector


_pipeline: CuriaRagPipeline | None = None


def _get_pipeline() -> CuriaRagPipeline:
    global _pipeline
    if _pipeline is None:
        if INDEX_PATH.exists():
            index = FaissIndex.load(INDEX_PATH)
        else:
            index = build_index_from_corpus(CORPUS_DIR)
            index.save(INDEX_PATH)
        _pipeline = CuriaRagPipeline(
            index, audit_path=AUDIT_DB_PATH, source_quotas=SOURCE_QUOTAS
        )
    return _pipeline


def _card(letter: str, title: str, status: str, body: str) -> str:
    colors = {
        "done":        ("#22c55e", "✓ Done"),
        "running":     ("#3b82f6", "⟳ Running…"),
        "pending":     ("#64748b", "Waiting"),
        "unavailable": ("#f59e0b", "⚠ Needs more data"),
    }
    col, label = colors.get(status, ("#64748b", status))
    return f"""
<div style="background:#1e293b;border-radius:12px;border-left:5px solid {col};
            padding:20px 22px;margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
    <div style="background:{col}22;border-radius:8px;padding:6px 12px;
                font-size:18px;font-weight:800;color:{col};">Agent {letter}</div>
    <div style="font-weight:700;color:#f1f5f9;font-size:15px;">{title}</div>
    <div style="margin-left:auto;font-size:12px;color:{col};font-weight:600;">{label}</div>
  </div>
  {body}
</div>"""


def _pending_card(letter: str, title: str) -> str:
    return _card(letter, title, "pending",
                 f'<div style="color:#64748b;font-size:13px;">Waiting for previous step…</div>')


def _leaderboard_row(skill: str, score: float, rank: int) -> str:
    pct   = min(int(score * 100), 100)
    level = "HIGH" if score > 0.55 else "MEDIUM" if score > 0.35 else "LOW"
    col   = "#22c55e" if level == "HIGH" else "#f59e0b" if level == "MEDIUM" else "#ef4444"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    bg    = "#111827" if rank % 2 else "#0f172a"
    medal_size = "20px" if rank <= 3 else "12px"
    medal_col  = "#fbbf24" if rank <= 3 else "#64748b"
    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;
                background:{bg};border-radius:6px;margin-bottom:4px;">
      <div style="font-size:{medal_size};font-weight:700;min-width:40px;text-align:center;
                  color:{medal_col};">{medal}</div>
      <div style="flex:1;font-size:13px;color:#e2e8f0;font-weight:500;
                  text-transform:capitalize;">{skill}</div>
      <div style="width:140px;background:#0f172a;border-radius:4px;height:8px;">
        <div style="width:{pct}%;background:{col};height:8px;border-radius:4px;"></div>
      </div>
      <div style="min-width:52px;font-size:11px;font-weight:700;color:{col};
                  text-align:center;">{level}</div>
      <div style="min-width:42px;font-size:11px;color:#64748b;text-align:right;
                  font-variant-numeric:tabular-nums;">{score:.2f}</div>
    </div>"""


MAX_AGENT_A_SKILLS = 20


def _run_agent_a(field: str) -> tuple[str, list[tuple[str, float]]]:
    """Returns (html_card, ranked_skill_list) so Agent B can consume the skills."""
    cfg  = get_ingestion_config(field)
    pipe = _get_pipeline()
    corpus_size = len(pipe.retriever.index.chunks)

    raw_candidates = (
        cfg.get("job_titles", [])
        + cfg.get("so_tags", [])
        + cfg.get("github_topics", [])
        + cfg.get("remotive_queries", [])
        + cfg.get("arbeitnow_queries", [])
    )

    seen: set[str] = set()
    candidates: list[str] = []
    for term in raw_candidates:
        norm = term.lower().replace("-", " ").strip()
        if norm and norm not in seen:
            seen.add(norm)
            candidates.append(norm)

    signals: list[tuple[str, float]] = []
    for term in candidates:
        results = pipe.retriever.retrieve(term, k=8)
        if results:
            top_score = max(r.score for r in results)
            signals.append((term, top_score))

    signals.sort(key=lambda x: x[1], reverse=True)
    top = signals[:MAX_AGENT_A_SKILLS]

    rows = "".join(
        _leaderboard_row(skill, score, i + 1)
        for i, (skill, score) in enumerate(top)
    )

    arxiv = " · ".join(cfg.get("arxiv_cats", []))
    body = f"""
    <div style="font-size:12px;color:#94a3b8;margin-bottom:12px;">
      Corpus: <b style="color:#38bdf8">{corpus_size} chunks</b> ·
      Candidates evaluated: <b style="color:#38bdf8">{len(candidates)}</b> ·
      Ranked: <b style="color:#38bdf8">{len(top)}</b> ·
      arXiv: <span style="color:#94a3b8">{arxiv}</span>
    </div>
    <div style="font-size:12px;font-weight:600;color:#fbbf24;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
      🏆 Industry Demand Leaderboard
    </div>
    {rows}"""

    return _card("A", "Signal Fusion", "done", body), top


def _sparkline_svg(values: list[float], forecast_values: list[float],
                   w: int = 160, h: int = 36) -> str:
    """Render historical (blue) + forecast (dashed orange) as an inline SVG."""
    all_vals = values + forecast_values
    if not all_vals or max(all_vals) == min(all_vals):
        return ""
    mn, mx = min(all_vals), max(all_vals)

    def sx(i: int, total: int) -> float:
        return i / max(total - 1, 1) * w

    def sy(v: float) -> float:
        return h - (v - mn) / (mx - mn) * h * 0.85 - h * 0.075

    n_hist = len(values)
    n_fc   = len(forecast_values)
    total  = n_hist + n_fc

    hist_pts  = " ".join(f"{sx(i, total):.1f},{sy(v):.1f}"
                         for i, v in enumerate(values))
    fc_pts    = " ".join(f"{sx(n_hist + i, total):.1f},{sy(v):.1f}"
                         for i, v in enumerate(forecast_values))

    div_x = sx(n_hist, total)

    return (
        f'<svg width="{w}" height="{h}" style="vertical-align:middle;margin-left:8px;">'
        f'<polyline points="{hist_pts}" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-linejoin="round"/>'
        f'<polyline points="{fc_pts}"   fill="none" stroke="#f59e0b" stroke-width="1.5" '
        f'stroke-dasharray="3,2" stroke-linejoin="round"/>'
        f'<line x1="{div_x:.1f}" y1="0" x2="{div_x:.1f}" y2="{h}" '
        f'stroke="#334155" stroke-width="1" stroke-dasharray="2,2"/>'
        f'</svg>'
    )


def _trend_arrow(trend: str) -> str:
    return {"rising": "↑", "declining": "↓", "stable": "→"}.get(trend, "→")


def _trend_color(trend: str) -> str:
    return {"rising": "#22c55e", "declining": "#ef4444", "stable": "#94a3b8"}.get(trend, "#94a3b8")


def _run_agent_b(
    field: str,
    agent_a_skills: list[tuple[str, float]] | None = None,
) -> tuple[str, list[ForecastResult]]:
    if agent_a_skills:
        skill_names = [s for s, _ in agent_a_skills]
        results = _forecaster.forecast_skills(skill_names, top_n=10)
    else:
        results = _forecaster.forecast_field(field, top_n=8)
    if not results:
        return (
            _card("B", "Skill Demand Forecasting", "done",
                  '<div style="color:#94a3b8">No skill data available for this field.</div>'),
            [],
        )

    bt_skill = _canonical_skill(results[0].skill)
    bt = _forecaster.backtest(bt_skill, cutoff_months_ago=6)
    mape_str    = f"{bt['mape']*100:.1f}%" if bt["mape"] == bt["mape"] else "N/A"
    dir_str     = f"{bt['direction_accuracy']*100:.0f}%" if bt["direction_accuracy"] == bt["direction_accuracy"] else "N/A"

    rows = ""
    for r in results:
        hist_vals = [dp.frequency for dp in r.historical[-20:]]
        fc_vals   = [dp.frequency for dp in r.forecast[:6]]
        svg       = _sparkline_svg(hist_vals, fc_vals)
        arrow     = _trend_arrow(r.trend)
        col       = _trend_color(r.trend)
        fc_12m    = r.forecast[-1].frequency
        conf_pct  = int(r.confidence * 100)

        rows += f"""
        <tr>
          <td style="padding:8px 10px;border-bottom:1px solid #334155;color:#e2e8f0;font-size:13px;">
            {r.skill}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #334155;text-align:center;">
            <span style="color:{col};font-weight:700;font-size:15px;">{arrow}</span>
            <span style="color:{col};font-size:11px;font-weight:600;margin-left:4px;">
              {r.trend.upper()}</span></td>
          <td style="padding:8px 10px;border-bottom:1px solid #334155;color:#38bdf8;font-size:12px;">
            {fc_12m:.2f}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #334155;color:#64748b;font-size:11px;">
            {conf_pct}%</td>
          <td style="padding:8px 10px;border-bottom:1px solid #334155;">{svg}</td>
        </tr>"""

    body = f"""
    <div style="font-size:11px;color:#64748b;margin-bottom:10px;display:flex;gap:20px;">
      <span>Method: <b style="color:#f1f5f9">Linear regression baseline</b></span>
      <span>Horizon: <b style="color:#f1f5f9">12 months</b></span>
      <span>Backtest MAPE: <b style="color:#22c55e">{mape_str}</b></span>
      <span>Direction accuracy: <b style="color:#22c55e">{dir_str}</b></span>
    </div>
    <div style="font-size:10px;color:#475569;margin-bottom:12px;">
      📘 Historical data: Jan 2022 – present (synthetic trend model) ·
      <span style="color:#38bdf8">━━</span> History &nbsp;
      <span style="color:#f59e0b">╌╌</span> 6-month forecast
    </div>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#0f172a;">
            <th style="padding:8px 10px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Skill</th>
            <th style="padding:8px 10px;text-align:center;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Trend</th>
            <th style="padding:8px 10px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">12m Forecast</th>
            <th style="padding:8px 10px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Confidence</th>
            <th style="padding:8px 10px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Historical → Forecast</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return _card("B", f"Skill Demand Forecasting — {field}", "done", body), results


_DRIFT_BAND_COLOURS = [
    (0.30, "#ef4444", "HIGH"),
    (0.15, "#f59e0b", "MEDIUM"),
    (0.00, "#22c55e", "LOW"),
]


def _drift_band(score: float) -> tuple[str, str]:
    for cutoff, col, label in _DRIFT_BAND_COLOURS:
        if score >= cutoff:
            return col, label
    return "#22c55e", "LOW"


def _drift_bar(score: float, max_scale: float = 0.6) -> str:
    pct = min(int(score / max_scale * 100), 100)
    col, _ = _drift_band(score)
    return (
        f'<div style="width:120px;background:#0f172a;border-radius:4px;height:8px;">'
        f'<div style="width:{pct}%;background:{col};height:8px;border-radius:4px;"></div>'
        f'</div>'
    )


def _drift_row(r: DriftResult) -> str:
    col, label = _drift_band(r.max_drift)
    top_pair = r.pairs[0] if r.pairs else None
    pair_html = (
        f'{top_pair.from_label} ↔ {top_pair.to_label}'
        if top_pair else "—"
    )
    buckets_html = " · ".join(
        f'<span style="color:#94a3b8;">{b.label} ({b.chunk_count})</span>'
        for b in r.buckets
    )
    return f"""
    <tr>
      <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#e2e8f0;font-size:13px;
                 text-transform:capitalize;font-weight:500;">{r.skill}</td>
      <td style="padding:10px 12px;border-bottom:1px solid #334155;">
        <div style="display:flex;align-items:center;gap:8px;">
          {_drift_bar(r.max_drift)}
          <span style="color:{col};font-weight:700;font-size:12px;min-width:42px;">{label}</span>
          <span style="color:#64748b;font-size:11px;font-variant-numeric:tabular-nums;">{r.max_drift:.3f}</span>
        </div>
      </td>
      <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#cbd5e1;font-size:12px;">
        {pair_html}
      </td>
      <td style="padding:10px 12px;border-bottom:1px solid #334155;font-size:11px;">
        {buckets_html}
      </td>
    </tr>"""


def _run_agent_c(
    field: str,
    agent_a_skills: list[tuple[str, float]] | None = None,
) -> tuple[str, list[DriftResult]]:
    """Compute cross-community drift for each Agent-A skill.

    Returns (html, drift_results) so Agent D can highlight drifted skills.
    """
    skills = [s for s, _ in (agent_a_skills or [])]
    if not skills:
        body = (
            '<div style="color:#94a3b8;font-size:12px;">'
            'No skills to analyse — Agent A produced no leaderboard.</div>'
        )
        return _card("C", "Semantic Drift Detection", "done", body), []

    detector = _get_drift_detector()
    results  = detector.analyze_skills(skills)
    analysed = len(results)
    skipped  = len(skills) - analysed
    drifted  = [r for r in results if r.drifted]

    if not results:
        body = (
            '<div style="color:#f59e0b;font-size:12px;">'
            f'⚠ None of the {len(skills)} Agent-A skills had enough chunks across '
            '≥2 communities to compute drift. Add more corpus coverage.</div>'
        )
        return _card("C", "Semantic Drift Detection", "done", body), []

    sorted_results = sorted(
        results,
        key=lambda r: (not r.drifted, -r.max_drift),
    )

    max_drift_overall = max((r.max_drift for r in results), default=0.0)
    top_drifter = sorted_results[0] if sorted_results else None

    rows = "".join(_drift_row(r) for r in sorted_results)

    top_pair_html = "—"
    if top_drifter and top_drifter.pairs:
        p = top_drifter.pairs[0]
        top_pair_html = (
            f'{p.from_label} ↔ {p.to_label} '
            f'<span style="color:#64748b;">on</span> '
            f'<b style="color:#e2e8f0;text-transform:capitalize;">{top_drifter.skill}</b>'
        )

    body = f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#38bdf8;">{analysed}</div>
        <div style="font-size:11px;color:#64748b;">Skills analysed</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#ef4444;">{len(drifted)}</div>
        <div style="font-size:11px;color:#64748b;">Drifted (≥ 0.15)</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#f59e0b;">{max_drift_overall:.3f}</div>
        <div style="font-size:11px;color:#64748b;">Max drift score</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#64748b;">{skipped}</div>
        <div style="font-size:11px;color:#64748b;">Insufficient data</div>
      </div>
    </div>
    <div style="font-size:11px;color:#64748b;margin-bottom:10px;display:flex;gap:20px;flex-wrap:wrap;">
      <span>Method: <b style="color:#f1f5f9">cosine drift</b> between embedding centroids</span>
      <span>Communities: <b style="color:#f1f5f9">Industry · Research · Practitioners · OSS</b></span>
      <span>Top divergence: {top_pair_html}</span>
    </div>
    <div style="font-size:10px;color:#475569;margin-bottom:10px;">
      Drift = cosine distance between L2-normalised centroids of chunks from each source class.
      <b style="color:#22c55e;">LOW &lt; 0.15</b> ·
      <b style="color:#f59e0b;">MEDIUM 0.15–0.30</b> ·
      <b style="color:#ef4444;">HIGH ≥ 0.30</b>
    </div>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#0f172a;">
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Skill</th>
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Max Drift</th>
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Top Divergent Pair</th>
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:10px;
                       text-transform:uppercase;letter-spacing:1px;">Communities (chunks)</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return _card("C", f"Semantic Drift — {field}", "done", body), results


SIGNAL_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}
_TREND_PALETTE = {
    "rising":    ("#22c55e", "↑"),
    "stable":    ("#94a3b8", "→"),
    "declining": ("#ef4444", "↓"),
    "unknown":   ("#64748b", "•"),
}


_JOB_TITLE_SUFFIXES = (
    " engineer", " scientist", " analyst", " developer",
    " researcher", " specialist", " architect", " manager",
)


def _skill_search_terms(skill: str) -> list[str]:
    """Substrings to look for in unit text. Prefers canonical form for short tokens.

    Also handles the job-title / topic mismatch: when Agent A produces
    'process engineer' but the unit text says 'process control', we
    strip the role suffix and match on the topical stem ('process').
    The stem must be >=5 chars to avoid over-generic matches like 'data'.
    """
    raw       = skill.lower().replace("-", " ").strip()
    canonical = _canonical_skill(skill)
    terms: list[str] = []
    if canonical and canonical != raw:
        terms.append(canonical)
    if len(raw) >= 5 or " " in raw:
        terms.append(raw)
    for suf in _JOB_TITLE_SUFFIXES:
        if raw.endswith(suf):
            stem = raw[: -len(suf)].strip()
            if len(stem) >= 5 and stem not in terms:
                terms.append(stem)
            break
    return terms or [raw]


def _match_skills_to_unit(
    unit: dict,
    emerging_topics: list[str],
    agent_a_skills: list[tuple[str, float]],
    b_trend_by_skill: dict[str, str],
) -> list[tuple[str, float, str]]:
    """Find which Agent-A skills this unit covers, each tagged with B's trend."""
    haystack = " ".join([
        unit.get("title", ""),
        unit.get("description", ""),
        " ".join(unit.get("current_topics", [])),
        " ".join(emerging_topics),
    ]).lower().replace("-", " ")

    matched: list[tuple[str, float, str]] = []
    for skill, score in agent_a_skills:
        for term in _skill_search_terms(skill):
            if term in haystack:
                trend = b_trend_by_skill.get(skill, "unknown")
                matched.append((skill, score, trend))
                break
    return matched


def _skill_chip(skill: str, trend: str, drifted: bool = False) -> str:
    col, arrow = _TREND_PALETTE.get(trend, _TREND_PALETTE["unknown"])
    drift_mark = ' <span title="semantic drift across communities">⚠</span>' if drifted else ""
    border = "2px solid #ef4444" if drifted else f"1px solid {col}66"
    return (
        f'<span style="display:inline-block;background:{col}22;color:{col};'
        f'padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;'
        f'margin:2px 3px 2px 0;border:{border};text-transform:capitalize;">'
        f'{skill} {arrow}{drift_mark}</span>'
    )


def _run_agent_d(
    field: str,
    agent_a_skills: list[tuple[str, float]] | None = None,
    agent_b_results: list[ForecastResult] | None = None,
    agent_c_results: list[DriftResult] | None = None,
    progress_cb=None,
) -> str:
    units = get_all_units(field)
    pipe  = _get_pipeline()
    rows  = []

    a_skills         = agent_a_skills or []
    b_trend_by_skill = {r.skill: r.trend for r in (agent_b_results or [])}
    drifted_skills:  set[str] = {r.skill for r in (agent_c_results or []) if r.drifted}
    covered_skills:  set[str] = set()

    for i, unit in enumerate(units):
        if progress_cb:
            progress_cb(0.65 + 0.33 * (i / max(len(units), 1)),
                        desc=f"Agent D — {unit['title'][:40]} …")
        result   = pipe.run(unit)
        rec      = result["recommendation"]
        emerging = rec.get("emerging_topics", [])
        matched  = _match_skills_to_unit(unit, emerging, a_skills, b_trend_by_skill)
        for s, _, _ in matched:
            covered_skills.add(s)
        rows.append({
            "unit":    unit["title"],
            "courses": ", ".join(unit.get("courses", [])),
            "area":    unit.get("cs2023_area", "—"),
            "signal":  rec["signal_strength"],
            "matched": matched,
            "topics":  emerging,
            "audit":   result["audit_id"],
        })

    high   = sum(1 for r in rows if r["signal"] == "high")
    medium = sum(1 for r in rows if r["signal"] == "medium")
    low    = sum(1 for r in rows if r["signal"] == "low")

    n_a = len(a_skills)
    n_covered = len(covered_skills)
    coverage_pct = (n_covered / n_a * 100) if n_a else 0

    trend_rank = {"rising": 0, "unknown": 1, "stable": 2, "declining": 3}
    gap_skills = sorted(
        [
            (s, score, b_trend_by_skill.get(s, "unknown"))
            for s, score in a_skills if s not in covered_skills
        ],
        key=lambda x: (trend_rank.get(x[2], 4), -x[1]),
    )

    table_rows = ""
    for r in rows:
        icon  = SIGNAL_ICON.get(r["signal"], "⚪")
        if r["matched"]:
            chips = "".join(
                _skill_chip(s, t, drifted=(s in drifted_skills))
                for s, _, t in r["matched"]
            )
            n_drifted_here = sum(1 for s, _, _ in r["matched"] if s in drifted_skills)
            drift_note = (
                f' · <span style="color:#ef4444;">⚠ {n_drifted_here} drifted</span>'
                if n_drifted_here else ""
            )
            match_count = (
                f'<div style="font-size:10px;color:#22c55e;font-weight:600;'
                f'margin-bottom:6px;">✓ {len(r["matched"])} industry skill'
                f'{"s" if len(r["matched"]) != 1 else ""} covered{drift_note}</div>{chips}'
            )
        else:
            extra = ", ".join(r["topics"][:3]) if r["topics"] else "—"
            match_count = (
                f'<div style="font-size:10px;color:#f59e0b;font-weight:600;'
                f'margin-bottom:4px;">⚠ no Agent-A skills matched</div>'
                f'<div style="font-size:11px;color:#64748b;">{extra}</div>'
            )
        table_rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#e2e8f0;font-size:13px;
                     vertical-align:top;">{r['unit']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#64748b;font-size:12px;
                     vertical-align:top;">{r['courses']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;text-align:center;
                     font-size:14px;vertical-align:top;">{icon} {r['signal'].upper()}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;
                     vertical-align:top;">{match_count}</td>
        </tr>"""

    if gap_skills:
        gap_chips = "".join(
            _skill_chip(s, t, drifted=(s in drifted_skills))
            for s, _, t in gap_skills
        )
        gap_panel = f"""
        <div style="margin-top:16px;padding:12px 14px;background:#0f172a;
                    border-left:3px solid #ef4444;border-radius:6px;">
          <div style="font-size:11px;color:#ef4444;font-weight:700;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
            🚨 Curriculum Gap — Industry skills not covered by any unit ({len(gap_skills)})
          </div>
          <div>{gap_chips}</div>
        </div>"""
    else:
        gap_panel = ""

    coverage_col = "#22c55e" if coverage_pct >= 60 else "#f59e0b" if coverage_pct >= 30 else "#ef4444"

    body = f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:{coverage_col};">
          {n_covered}/{n_a}
        </div>
        <div style="font-size:11px;color:#64748b;">A-skills covered ({coverage_pct:.0f}%)</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#22c55e;">{high}</div>
        <div style="font-size:11px;color:#64748b;">High signal</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#f59e0b;">{medium}</div>
        <div style="font-size:11px;color:#64748b;">Medium signal</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#ef4444;">{low}</div>
        <div style="font-size:11px;color:#64748b;">Low signal</div>
      </div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 18px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:#38bdf8;">{len(rows)}</div>
        <div style="font-size:11px;color:#64748b;">Units analysed</div>
      </div>
    </div>
    <div style="font-size:10px;color:#64748b;margin-bottom:10px;">
      Chips show <b style="color:#e2e8f0">Agent A skills</b> covered by each unit, coloured by
      <span style="color:#22c55e">↑ rising</span> /
      <span style="color:#94a3b8">→ stable</span> /
      <span style="color:#ef4444">↓ declining</span> trend from Agent B.
    </div>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#0f172a;">
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:11px;
                       text-transform:uppercase;letter-spacing:1px;">Knowledge Unit</th>
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:11px;
                       text-transform:uppercase;letter-spacing:1px;">Courses</th>
            <th style="padding:10px 12px;text-align:center;color:#64748b;font-size:11px;
                       text-transform:uppercase;letter-spacing:1px;">Industry Signal</th>
            <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:11px;
                       text-transform:uppercase;letter-spacing:1px;">Industry Skills Covered</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    {gap_panel}"""

    return _card("D", f"Curriculum Mapping — TAMU {field}", "done", body)


EMPTY_A = _pending_card("A", "Signal Fusion")
EMPTY_B = _pending_card("B", "Skill Demand Forecasting")
EMPTY_C = _pending_card("C", "Semantic Drift Detection")
EMPTY_D = _pending_card("D", "Curriculum Mapping")


def run_full_analysis(field: str, progress=gr.Progress()):
    if not field:
        yield EMPTY_A, EMPTY_B, EMPTY_C, EMPTY_D
        return

    yield EMPTY_A, EMPTY_B, EMPTY_C, EMPTY_D

    progress(0.05, desc="Agent A: computing industry skill signals …")
    a_html, a_skills = _run_agent_a(field)
    yield a_html, _card("B", "Skill Demand Forecasting", "running",
                         f'<div style="color:#94a3b8;">Forecasting {len(a_skills)} skills from Agent A…</div>'), EMPTY_C, EMPTY_D

    progress(0.35, desc=f"Agent B: forecasting {len(a_skills)} skills from Agent A …")
    b_html, b_results = _run_agent_b(field, agent_a_skills=a_skills)
    yield a_html, b_html, _card("C", "Semantic Drift Detection", "running",
                                  '<div style="color:#94a3b8;">Checking drift data…</div>'), EMPTY_D

    progress(0.50, desc=f"Agent C: measuring cross-community drift for {len(a_skills)} skills …")
    c_html, c_results = _run_agent_c(field, agent_a_skills=a_skills)
    yield a_html, b_html, c_html, _card("D", "Curriculum Mapping", "running",
                                          f'<div style="color:#94a3b8;">Mapping {len(a_skills)} Agent-A skills onto TAMU curriculum…</div>')

    progress(0.65, desc="Agent D: matching A skills + B trends + C drift to TAMU units …")
    d_html = _run_agent_d(
        field,
        agent_a_skills=a_skills,
        agent_b_results=b_results,
        agent_c_results=c_results,
        progress_cb=progress,
    )
    yield a_html, b_html, c_html, d_html


CSS = """
body, .gradio-container { background:#0f172a !important; color:#f1f5f9 !important; }
.gr-panel, .gr-box, .gr-form { background:#0f172a !important; border:none !important; }
h1,h2,h3 { color:#38bdf8 !important; }
footer { display:none !important; }
label { color:#94a3b8 !important; }
"""

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
            padding:20px 24px;border-radius:12px;border:1px solid #1e40af;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:32px;">🎓</span>
    <div>
      <div style="font-size:22px;font-weight:800;color:#38bdf8;letter-spacing:-0.5px;">CURIA</div>
      <div style="font-size:11px;color:#64748b;margin-top:1px;">
        Curriculum Intelligence via Industry Agents 
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-size:10px;color:#475569;">LLM · Embeddings</div>
      <div style="font-size:12px;font-weight:600;color:#34d399;">GPT-4o · all-mpnet-base-v2</div>
    </div>
  </div>
</div>"""


with gr.Blocks(title="CURIA") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        field_dd = gr.Dropdown(
            choices=ALL_FIELDS,
            value="Computer Science",
            label="Field of Study (Texas A&M University)",
            interactive=True,
            scale=4,
        )
        run_btn = gr.Button("▶  Run All 4 Agents", variant="primary", scale=1, size="lg")

    agent_a = gr.HTML(value=EMPTY_A)
    agent_b = gr.HTML(value=EMPTY_B)
    agent_c = gr.HTML(value=EMPTY_C)
    agent_d = gr.HTML(value=EMPTY_D)

    run_btn.click(
        run_full_analysis,
        inputs=[field_dd],
        outputs=[agent_a, agent_b, agent_c, agent_d],
    )


PORT = 7888


def _free_port(port: int) -> None:
    """Kill any process holding the port so re-runs work cleanly."""
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=3,
        )
        for pid in result.stdout.strip().split():
            if pid and pid.isdigit():
                subprocess.run(["kill", "-9", pid], timeout=2)
    except Exception:
        pass


def _shutdown(*_: object) -> None:
    _free_port(PORT)
    sys.exit(0)


atexit.register(_free_port, PORT)
signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8888,
        share=False,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
