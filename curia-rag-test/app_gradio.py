"""
CURIA — Curriculum Intelligence via Industry Agents
Gradio frontend: select university + field of study, run the 4-agent pipeline.
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
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

from src.config import CORPUS_DIR, SOURCE_QUOTAS, INDEX_PATH, AUDIT_DB_PATH
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus
from src.university import (
    UNIVERSITY_OPTIONS,
    FIELD_OPTIONS,
    curriculum_summary,
    get_unit_titles,
    get_unit_by_title,
    get_all_units,
)

# ---------------------------------------------------------------------------
# Pipeline singleton (loaded once, reused across requests)
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
            index,
            audit_path=AUDIT_DB_PATH,
            source_quotas=SOURCE_QUOTAS,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

SIGNAL_EMOJI = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}

AGENT_STATUS = {
    "A": ("Signal Fusion Agent", "Aggregates job postings, GitHub, arXiv, Stack Overflow"),
    "B": ("Forecasting Agent",   "Predicts 12–24 month skill demand trajectories"),
    "C": ("Drift Detection Agent","Detects semantic shift within stable skill labels"),
    "D": ("Curriculum Mapper",   "Maps industry signals onto CS2023 competency framework"),
}


def _agent_html(letter: str, status: str) -> str:
    colours = {"done": "#22c55e", "running": "#3b82f6", "pending": "#94a3b8", "unavailable": "#f59e0b"}
    labels  = {"done": "✓ Done", "running": "⟳ Running", "pending": "Queued", "unavailable": "⚠ Not yet built"}
    colour = colours[status]
    label  = labels[status]
    name, desc = AGENT_STATUS[letter]
    return f"""
    <div style="
        display:flex; align-items:center; gap:14px;
        background:#1e293b; border-radius:10px; padding:14px 18px;
        border-left:4px solid {colour}; margin-bottom:8px;">
      <div style="font-size:22px; font-weight:800; color:{colour}; min-width:28px;">
        {letter}
      </div>
      <div style="flex:1">
        <div style="font-weight:600; color:#f1f5f9; font-size:14px;">{name}</div>
        <div style="color:#94a3b8; font-size:12px; margin-top:2px;">{desc}</div>
      </div>
      <div style="font-size:12px; color:{colour}; font-weight:600; white-space:nowrap;">{label}</div>
    </div>"""


def _evidence_html(evidence: list[dict]) -> str:
    cards = []
    for i, ev in enumerate(evidence, 1):
        source_icon = {"job_posting": "💼", "arxiv": "📄", "stackoverflow": "💬",
                       "github_readme": "🐙"}.get(ev["source"], "📎")
        score_pct = min(int(ev["score"] * 100), 100)
        bar_col = "#22c55e" if score_pct > 60 else "#f59e0b" if score_pct > 35 else "#ef4444"
        cards.append(f"""
        <div style="background:#1e293b; border-radius:10px; padding:14px 16px;
                    margin-bottom:10px; border:1px solid #334155;">
          <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-weight:600; color:#f1f5f9; font-size:13px;">
              {source_icon} [{i}] {ev['parent_id']}
            </span>
            <span style="font-size:11px; color:#64748b;">{ev['source']} · {ev['date']}</span>
          </div>
          <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">{ev['title']}</div>
          <div style="font-size:12px; color:#cbd5e1; line-height:1.5;">
            {ev['text'][:300]}{'…' if len(ev['text'])>300 else ''}
          </div>
          <div style="margin-top:8px; display:flex; align-items:center; gap:8px;">
            <div style="flex:1; background:#0f172a; border-radius:4px; height:5px;">
              <div style="width:{score_pct}%; background:{bar_col};
                          height:5px; border-radius:4px;"></div>
            </div>
            <span style="font-size:11px; color:#64748b;">score {ev['score']:.3f}</span>
          </div>
        </div>""")
    return "\n".join(cards)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(university: str, field: str, unit_title: str, progress=gr.Progress()):
    if not unit_title:
        return ("", "", "", "", _agent_html("A","pending")+_agent_html("B","unavailable")+
                _agent_html("C","unavailable")+_agent_html("D","pending"), "")

    progress(0.05, desc="Loading pipeline …")
    pipeline = _get_pipeline()

    # Agent A — corpus already ingested, show as done
    agent_html = (
        _agent_html("A", "done") +
        _agent_html("B", "unavailable") +
        _agent_html("C", "unavailable") +
        _agent_html("D", "running")
    )

    progress(0.30, desc="Retrieving evidence …")
    unit = get_unit_by_title(university, field, unit_title)
    result = pipeline.run(unit)

    progress(0.90, desc="Processing results …")
    rec   = result["recommendation"]
    cc    = result["citation_check"]
    evs   = result["evidence"]

    # Signal badge
    sig_label = SIGNAL_EMOJI.get(rec["signal_strength"], rec["signal_strength"])
    citation_ok = "✅ All citations verified" if cc["passed"] else f"⚠️ Unverified: {cc['missing_ids']}"

    # Recommendation markdown
    rec_md = f"""### {sig_label}  ·  Audit #{result['audit_id']}

{rec['summary']}

**Emerging topics to add:**
{chr(10).join(f'- `{t}`' for t in rec['emerging_topics'])}

**Citation check:** {citation_ok}

---
*Query used:* `{result['query']}`
"""

    # Topics plain text for the chips display
    topics_text = "  ·  ".join(rec["emerging_topics"]) if rec["emerging_topics"] else "None identified"

    # Agent pipeline — all done
    agent_done_html = (
        _agent_html("A", "done") +
        _agent_html("B", "unavailable") +
        _agent_html("C", "unavailable") +
        _agent_html("D", "done")
    )

    # Evidence HTML
    ev_html = _evidence_html(evs)

    # Metrics table
    metrics_md = f"""| Metric | Value |
|---|---|
| Signal strength | **{rec['signal_strength'].upper()}** |
| Documents retrieved | {len(evs)} |
| Documents cited | {len(cc['cited_ids'])} |
| Citation check | {'✅ Passed' if cc['passed'] else '❌ Failed'} |
| Emerging topics | {len(rec['emerging_topics'])} |
| Audit ID | #{result['audit_id']} |
"""

    progress(1.0, desc="Done")
    return rec_md, topics_text, ev_html, metrics_md, agent_done_html, json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Run all units
# ---------------------------------------------------------------------------

def run_all_units(university: str, field: str, progress=gr.Progress()):
    units = get_all_units(university, field)
    if not units:
        return "No units found.", "", ""

    pipeline = _get_pipeline()
    rows = []
    for i, unit in enumerate(units):
        progress((i + 1) / len(units), desc=f"Analysing: {unit['title'][:40]} …")
        result = pipeline.run(unit)
        rec = result["recommendation"]
        cc  = result["citation_check"]
        rows.append({
            "Unit": unit["title"],
            "CS2023 Area": unit.get("cs2023_area", "—"),
            "Signal": rec["signal_strength"].upper(),
            "Emerging Topics": ", ".join(rec["emerging_topics"][:3]),
            "Citations OK": "✅" if cc["passed"] else "⚠️",
            "Audit #": result["audit_id"],
        })

    # Summary markdown table
    table_md = "| Unit | Area | Signal | Top Topics | Citations |\n|---|---|---|---|---|\n"
    for r in rows:
        table_md += f"| {r['Unit']} | {r['CS2023 Area']} | **{r['Signal']}** | {r['Emerging Topics']} | {r['Citations OK']} |\n"

    high   = sum(1 for r in rows if r["Signal"] == "HIGH")
    medium = sum(1 for r in rows if r["Signal"] == "MEDIUM")
    low    = sum(1 for r in rows if r["Signal"] == "LOW")
    summary = f"**{len(rows)} units analysed** — 🟢 High: {high}  🟡 Medium: {medium}  🔴 Low: {low}"

    return summary, table_md, json.dumps(rows, indent=2)


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container { background: #0f172a !important; color: #f1f5f9 !important; }
.gr-box, .gr-form { background: #1e293b !important; border-color: #334155 !important; }
h1, h2, h3 { color: #38bdf8 !important; }
.gr-button-primary { background: #3b82f6 !important; border-color: #3b82f6 !important; }
.gr-button-secondary { background: #1e293b !important; border-color: #475569 !important; color: #94a3b8 !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
            padding:28px 32px; border-radius:14px; margin-bottom:4px;
            border:1px solid #1e40af;">
  <div style="display:flex; align-items:center; gap:16px;">
    <div style="font-size:40px;">🎓</div>
    <div>
      <div style="font-size:26px; font-weight:800; color:#38bdf8; letter-spacing:-0.5px;">
        CURIA
      </div>
      <div style="font-size:13px; color:#94a3b8; margin-top:2px;">
        Curriculum Intelligence via Industry Agents — IEEE BigData 2026
      </div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div style="font-size:11px; color:#475569;">Model</div>
      <div style="font-size:13px; font-weight:600; color:#34d399;">GPT-4o + all-mpnet-base-v2</div>
    </div>
  </div>
</div>
"""


def update_fields(university: str) -> gr.update:
    fields = FIELD_OPTIONS.get(university, [])
    return gr.update(choices=fields, value=fields[0] if fields else None)


def update_units(university: str, field: str) -> gr.update:
    if not university or not field:
        return gr.update(choices=[], value=None)
    titles = get_unit_titles(university, field)
    return gr.update(choices=titles, value=titles[0] if titles else None)


def show_curriculum(university: str, field: str) -> str:
    if not university or not field:
        return ""
    try:
        return curriculum_summary(university, field)
    except KeyError:
        return "Curriculum not found."


with gr.Blocks(title="CURIA — Curriculum Intelligence") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Single Unit Analysis ───────────────────────────────────
        with gr.Tab("🔬 Analyse a Unit"):
            with gr.Row():
                # Left panel — controls
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### Select Curriculum")
                    university_dd = gr.Dropdown(
                        choices=UNIVERSITY_OPTIONS,
                        value=UNIVERSITY_OPTIONS[0],
                        label="University",
                        interactive=True,
                    )
                    field_dd = gr.Dropdown(
                        choices=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]],
                        value=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]][0],
                        label="Field of Study",
                        interactive=True,
                    )
                    unit_dd = gr.Dropdown(
                        choices=get_unit_titles(UNIVERSITY_OPTIONS[0], FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]][0]),
                        value=get_unit_titles(UNIVERSITY_OPTIONS[0], FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]][0])[0],
                        label="Knowledge Unit",
                        interactive=True,
                    )
                    run_btn = gr.Button("▶  Run Analysis", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### Agent Pipeline")
                    agent_status = gr.HTML(
                        _agent_html("A", "pending") +
                        _agent_html("B", "unavailable") +
                        _agent_html("C", "unavailable") +
                        _agent_html("D", "pending")
                    )

                # Right panel — results
                with gr.Column(scale=2):
                    gr.Markdown("### Recommendation")
                    rec_md = gr.Markdown(
                        value="*Select a unit and click Run Analysis.*",
                        label="",
                    )

                    gr.Markdown("### Emerging Topics")
                    topics_out = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=2,
                    )

                    gr.Markdown("### Metrics")
                    metrics_md = gr.Markdown()

            gr.Markdown("### Retrieved Evidence")
            evidence_html = gr.HTML()

            with gr.Accordion("📋 Raw JSON output", open=False):
                raw_json = gr.Code(language="json", label="Full pipeline output")

        # ── Tab 2: Full Curriculum Scan ───────────────────────────────────
        with gr.Tab("📊 Scan Full Curriculum"):
            with gr.Row():
                uni2 = gr.Dropdown(
                    choices=UNIVERSITY_OPTIONS,
                    value=UNIVERSITY_OPTIONS[0],
                    label="University",
                    interactive=True,
                    scale=2,
                )
                field2 = gr.Dropdown(
                    choices=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]],
                    value=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]][0],
                    label="Field of Study",
                    interactive=True,
                    scale=2,
                )
                scan_btn = gr.Button("▶  Scan All Units", variant="primary", scale=1)

            scan_summary = gr.Markdown()
            scan_table   = gr.Markdown()
            with gr.Accordion("📋 Raw JSON", open=False):
                scan_json = gr.Code(language="json")

        # ── Tab 3: Curriculum Overview ────────────────────────────────────
        with gr.Tab("🏛️ Curriculum Overview"):
            with gr.Row():
                uni3   = gr.Dropdown(choices=UNIVERSITY_OPTIONS, value=UNIVERSITY_OPTIONS[0],
                                     label="University", interactive=True, scale=2)
                field3 = gr.Dropdown(choices=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]],
                                     value=FIELD_OPTIONS[UNIVERSITY_OPTIONS[0]][0],
                                     label="Field", interactive=True, scale=2)
                view_btn = gr.Button("View", scale=1)
            curr_md = gr.Markdown()

    # ── Wire events ──────────────────────────────────────────────────────

    # Cascading dropdowns — Tab 1
    university_dd.change(update_fields, university_dd, field_dd)
    university_dd.change(update_units,  [university_dd, field_dd], unit_dd)
    field_dd.change(update_units, [university_dd, field_dd], unit_dd)

    # Run single unit
    run_btn.click(
        run_analysis,
        inputs=[university_dd, field_dd, unit_dd],
        outputs=[rec_md, topics_out, evidence_html, metrics_md, agent_status, raw_json],
    )

    # Tab 2 cascading
    uni2.change(update_fields, uni2, field2)
    scan_btn.click(
        run_all_units,
        inputs=[uni2, field2],
        outputs=[scan_summary, scan_table, scan_json],
    )

    # Tab 3
    uni3.change(update_fields, uni3, field3)
    view_btn.click(show_curriculum, [uni3, field3], curr_md)
    # Auto-load on startup
    demo.load(show_curriculum, [uni3, field3], curr_md)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7880,
        share=False,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
