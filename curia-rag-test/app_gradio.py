"""
CURIA — Curriculum Intelligence via Industry Agents
Single-page, step-by-step: Field → A (skills) → B → C → D (curriculum map)
"""

from __future__ import annotations

import os
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
from src.field_config import get_ingestion_config
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus
from src.university import ALL_FIELDS, get_all_units

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HTML card builder
# ---------------------------------------------------------------------------

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


def _skill_bar(skill: str, score: float, rank: int) -> str:
    pct = min(int(score * 100), 100)
    level = "HIGH" if score > 0.55 else "MEDIUM" if score > 0.35 else "LOW"
    col = "#22c55e" if level == "HIGH" else "#f59e0b" if level == "MEDIUM" else "#ef4444"
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <div style="font-size:11px;color:#64748b;min-width:18px;text-align:right;">{rank}</div>
      <div style="flex:1;font-size:13px;color:#e2e8f0;">{skill}</div>
      <div style="width:120px;background:#0f172a;border-radius:4px;height:8px;">
        <div style="width:{pct}%;background:{col};height:8px;border-radius:4px;"></div>
      </div>
      <div style="min-width:52px;font-size:11px;font-weight:700;color:{col};">{level}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Agent A  — extract top skills from corpus
# ---------------------------------------------------------------------------

def _run_agent_a(field: str) -> str:
    cfg  = get_ingestion_config(field)
    pipe = _get_pipeline()
    corpus_size = len(pipe.retriever.index.chunks)

    # Score each job title term against corpus
    signals: list[tuple[str, float]] = []
    seen: set[str] = set()
    for term in cfg.get("job_titles", []) + cfg.get("so_tags", [])[:3]:
        if term in seen:
            continue
        seen.add(term)
        results = pipe.retriever.retrieve(term, k=8)
        if results:
            top_score = max(r.score for r in results)
            signals.append((term, top_score))

    signals.sort(key=lambda x: x[1], reverse=True)
    top10 = signals[:10]

    bars = "".join(_skill_bar(skill, score, i + 1) for i, (skill, score) in enumerate(top10))

    arxiv = " · ".join(cfg.get("arxiv_cats", []))
    body = f"""
    <div style="font-size:12px;color:#94a3b8;margin-bottom:12px;">
      Corpus: <b style="color:#38bdf8">{corpus_size} chunks</b> ·
      arXiv: <span style="color:#94a3b8">{arxiv}</span>
    </div>
    <div style="font-size:12px;font-weight:600;color:#64748b;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
      Top skills in industry demand
    </div>
    {bars}"""

    return _card("A", "Signal Fusion", "done", body)


# ---------------------------------------------------------------------------
# Agent B  — forecasting placeholder
# ---------------------------------------------------------------------------

def _run_agent_b(field: str) -> str:
    months_collected = 1   # current state — only 1 month of data
    months_needed    = 12

    pct = int(months_collected / months_needed * 100)
    body = f"""
    <div style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <span style="font-size:13px;color:#e2e8f0;">Historical corpus build</span>
        <span style="font-size:13px;font-weight:700;color:#f59e0b;">{months_collected}/{months_needed} months</span>
      </div>
      <div style="background:#0f172a;border-radius:6px;height:10px;">
        <div style="width:{pct}%;background:#f59e0b;height:10px;border-radius:6px;"></div>
      </div>
    </div>
    <div style="font-size:12px;color:#94a3b8;line-height:1.6;">
      <b style="color:#f1f5f9">What Agent B will do:</b> Train a Temporal Fusion Transformer
      on {months_needed}+ months of skill-demand time-series to forecast which skills will rise or
      fall over the next 12–24 months for <b style="color:#f1f5f9">{field}</b>.<br><br>
      <b style="color:#f1f5f9">Baselines when ready:</b> ARIMA, LSTM, Temporal KG Embeddings
      (arXiv 2504.07233).<br><br>
      <span style="color:#f59e0b;">⚠ Run the ingestion pipeline monthly to build the historical corpus.</span>
    </div>"""
    return _card("B", "Skill Demand Forecasting", "unavailable", body)


# ---------------------------------------------------------------------------
# Agent C  — drift detection placeholder
# ---------------------------------------------------------------------------

def _run_agent_c(field: str) -> str:
    body = f"""
    <div style="font-size:12px;color:#94a3b8;line-height:1.6;">
      <b style="color:#f1f5f9">What Agent C will do:</b> Compute monthly contextual embedding
      centroids for ~500 tracked skill labels in <b style="color:#f1f5f9">{field}</b>,
      then apply Maximum Mean Discrepancy (MMD) tests on 12-month rolling windows to flag
      skills whose meaning has shifted — even when the label stayed the same.<br><br>
      <b style="color:#f1f5f9">Example of drift:</b> "Cloud computing" in 2019 → VMs &amp; storage.
      "Cloud computing" in 2024 → Kubernetes &amp; serverless. Same label, different competency.<br><br>
      <span style="color:#f59e0b;">⚠ Requires 2+ years of monthly centroid snapshots.
      First drift signals expected after Month 12.</span>
    </div>"""
    return _card("C", "Semantic Drift Detection", "unavailable", body)


# ---------------------------------------------------------------------------
# Agent D  — full curriculum mapping
# ---------------------------------------------------------------------------

SIGNAL_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def _run_agent_d(field: str, progress_cb=None) -> str:
    units  = get_all_units(field)
    pipe   = _get_pipeline()
    rows   = []

    for i, unit in enumerate(units):
        if progress_cb:
            progress_cb(0.65 + 0.33 * (i / max(len(units), 1)),
                        desc=f"Agent D — {unit['title'][:40]} …")
        result = pipe.run(unit)
        rec    = result["recommendation"]
        topics = ", ".join(rec["emerging_topics"][:4]) or "—"
        rows.append({
            "unit":    unit["title"],
            "courses": ", ".join(unit.get("courses", [])),
            "area":    unit.get("cs2023_area", "—"),
            "signal":  rec["signal_strength"],
            "topics":  topics,
            "audit":   result["audit_id"],
        })

    # Summary counts
    high   = sum(1 for r in rows if r["signal"] == "high")
    medium = sum(1 for r in rows if r["signal"] == "medium")
    low    = sum(1 for r in rows if r["signal"] == "low")

    # Table rows
    table_rows = ""
    for r in rows:
        icon = SIGNAL_ICON.get(r["signal"], "⚪")
        table_rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#e2e8f0;font-size:13px;">
            {r['unit']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#64748b;font-size:12px;">
            {r['courses']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;text-align:center;
                     font-size:14px;">{icon} {r['signal'].upper()}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #334155;color:#94a3b8;font-size:12px;">
            {r['topics']}</td>
        </tr>"""

    body = f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
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
                       text-transform:uppercase;letter-spacing:1px;">Add to Curriculum</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>"""

    return _card("D", f"Curriculum Mapping — TAMU {field}", "done", body)


# ---------------------------------------------------------------------------
# Main streaming generator
# ---------------------------------------------------------------------------

EMPTY_A = _pending_card("A", "Signal Fusion")
EMPTY_B = _pending_card("B", "Skill Demand Forecasting")
EMPTY_C = _pending_card("C", "Semantic Drift Detection")
EMPTY_D = _pending_card("D", "Curriculum Mapping")


def run_full_analysis(field: str, progress=gr.Progress()):
    if not field:
        yield EMPTY_A, EMPTY_B, EMPTY_C, EMPTY_D
        return

    # Step 0 — show all pending
    yield EMPTY_A, EMPTY_B, EMPTY_C, EMPTY_D

    # Step 1 — Agent A
    progress(0.05, desc="Agent A: computing industry skill signals …")
    a_html = _run_agent_a(field)
    yield a_html, _card("B", "Skill Demand Forecasting", "running",
                         '<div style="color:#94a3b8;">Checking forecasting status…</div>'), EMPTY_C, EMPTY_D

    # Step 2 — Agent B
    progress(0.35, desc="Agent B: checking forecasting readiness …")
    b_html = _run_agent_b(field)
    yield a_html, b_html, _card("C", "Semantic Drift Detection", "running",
                                  '<div style="color:#94a3b8;">Checking drift data…</div>'), EMPTY_D

    # Step 3 — Agent C
    progress(0.50, desc="Agent C: checking drift detection readiness …")
    c_html = _run_agent_c(field)
    yield a_html, b_html, c_html, _card("D", "Curriculum Mapping", "running",
                                          '<div style="color:#94a3b8;">Running RAG pipeline on all units…</div>')

    # Step 4 — Agent D (slowest — multiple GPT calls)
    progress(0.65, desc="Agent D: mapping curriculum to industry signals …")
    d_html = _run_agent_d(field, progress_cb=progress)
    yield a_html, b_html, c_html, d_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

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

    # ── Selection row ─────────────────────────────────────────────────────
    with gr.Row():
        field_dd = gr.Dropdown(
            choices=ALL_FIELDS,
            value="Computer Science",
            label="Field of Study (Texas A&M University)",
            interactive=True,
            scale=4,
        )
        run_btn = gr.Button("▶  Run All 4 Agents", variant="primary", scale=1, size="lg")

    # ── 4 agent outputs ───────────────────────────────────────────────────
    agent_a = gr.HTML(value=EMPTY_A)
    agent_b = gr.HTML(value=EMPTY_B)
    agent_c = gr.HTML(value=EMPTY_C)
    agent_d = gr.HTML(value=EMPTY_D)

    # ── Wire ──────────────────────────────────────────────────────────────
    run_btn.click(
        run_full_analysis,
        inputs=[field_dd],
        outputs=[agent_a, agent_b, agent_c, agent_d],
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7888,
        share=False,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
