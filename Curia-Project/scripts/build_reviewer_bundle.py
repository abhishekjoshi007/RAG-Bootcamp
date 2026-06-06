#!/usr/bin/env python3
"""Build a human-reviewer bundle from existing CURIA results.

Produces one per-course Markdown packet, a ratings CSV template, a rubric
README, and a copy of the existing ground-truth labels. The packet contains
everything a human rater (CS instructor, curriculum committee member, RA)
needs to assess CURIA's output for one course without re-running the system.

The bundle reuses our PRIOR multi-LLM results: per-(model, course) metrics
and cited IDs from `results/headline_multi_llm_50q_17models_rechecked.json`.
The prior run did not persist the actual recommendation prose; this script
regenerates the LOCAL extractive baseline (deterministic, free) for each
course so raters have one reviewable model end-to-end. Other 16 models are
shown as a cited-IDs + metrics comparison table per course.

Output: writes to ./reviewer_bundle/ (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing import FaissIndex  # noqa: E402
from src.llm import LocalGroundedGenerator  # noqa: E402
from src.pipeline import CuriaRagPipeline  # noqa: E402


FRIENDLY: dict[str, str] = {
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


@dataclass(frozen=True)
class CourseBundle:
    course: dict[str, Any]
    evidence: list[dict[str, Any]]
    local_recommendation: dict[str, Any]
    model_rows: list[dict[str, Any]]
    ground_truth: dict[str, Any]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _evidence_payload(search_results: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sr in search_results:
        if isinstance(sr, dict):
            out.append({
                "doc_id": sr.get("parent_id"),
                "source": sr.get("source"),
                "date": str(sr.get("date")),
                "score": round(float(sr.get("score") or 0.0), 4),
                "similarity": round(float(sr.get("similarity") or 0.0), 4),
                "title": sr.get("title"),
                "text": sr.get("text") or "",
            })
            continue
        chunk = sr.chunk
        out.append({
            "doc_id": chunk.parent_id,
            "source": chunk.source,
            "date": str(chunk.date),
            "score": round(sr.score, 4),
            "similarity": round(sr.similarity, 4),
            "title": chunk.title,
            "text": chunk.text,
        })
    return out


def _ground_truth_for_unit(
    unit_id: str,
    retrieval_labels: list[dict[str, Any]],
    faithfulness_labels: list[dict[str, Any]],
    relevance_ratings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "retrieval": next(
            (r for r in retrieval_labels if r.get("query_id") == unit_id),
            None,
        ),
        "faithfulness": [r for r in faithfulness_labels if r.get("unit_id") == unit_id],
        "relevance": [r for r in relevance_ratings if r.get("unit_id") == unit_id],
    }


def _build_one_course(
    pipeline: CuriaRagPipeline,
    course: dict[str, Any],
    k: int,
    multi_rows: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> CourseBundle:
    out = pipeline.run(course, k=k)
    evidence = _evidence_payload(out["evidence"])
    rec = out["recommendation"]
    cc = out["citation_check"]
    cc_passed = cc.get("passed") if isinstance(cc, dict) else cc.passed
    cc_cited = cc.get("cited_ids") if isinstance(cc, dict) else cc.cited_ids
    cc_missing = cc.get("missing_ids") if isinstance(cc, dict) else cc.missing_ids
    local_rec = {
        "signal_strength": rec.get("signal_strength"),
        "summary": rec.get("summary"),
        "emerging_topics": list(rec.get("emerging_topics") or []),
        "evidence_ids": list(rec.get("evidence_ids") or []),
        "citation_check_passed": bool(cc_passed),
        "cited_ids": list(cc_cited or []),
        "missing_ids": list(cc_missing or []),
    }
    rows = [r for r in multi_rows if r.get("unit_id") == course["id"]]
    return CourseBundle(
        course=course,
        evidence=evidence,
        local_recommendation=local_rec,
        model_rows=rows,
        ground_truth=ground_truth,
    )


def _format_packet(bundle: CourseBundle) -> str:
    c = bundle.course
    lines: list[str] = []
    lines.append(f"# Review packet — `{c['id']}`")
    lines.append("")
    lines.append(f"**Title:** {c.get('title', '')}")
    lines.append("")
    desc = c.get("description") or ""
    lines.append(f"**Description:** {desc}")
    lines.append("")
    topics = c.get("current_topics") or []
    if topics:
        lines.append(f"**Current topics taught:** {', '.join(topics)}")
        lines.append("")
    courses = c.get("courses") or []
    if courses:
        lines.append(f"**Course codes:** {', '.join(courses)}")
        lines.append("")
    if c.get("cs2023_area"):
        lines.append(f"**CS2023 area:** {c['cs2023_area']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 1. Retrieved evidence (top-k chunks given to every model)")
    lines.append("")
    for i, ev in enumerate(bundle.evidence, 1):
        lines.append(f"### Evidence {i} — `{ev['doc_id']}`")
        lines.append(
            f"- Source: **{ev['source']}** · Date: **{ev['date']}** · "
            f"Score: {ev['score']:.3f} (sim {ev['similarity']:.3f})"
        )
        title = ev.get("title") or "(no title)"
        lines.append(f"- Title: {title}")
        lines.append("")
        text = ev.get("text") or ""
        if len(text) > 800:
            text = text[:800].rstrip() + " …"
        lines.append("> " + text.replace("\n", "\n> "))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. System output — local extractive baseline (reviewable prose)")
    lines.append("")
    local = bundle.local_recommendation
    lines.append(f"**Signal strength:** `{local['signal_strength']}`")
    lines.append("")
    lines.append("**Recommendation summary:**")
    lines.append("")
    lines.append("> " + (local.get("summary") or "").replace("\n", "\n> "))
    lines.append("")
    emerging = local.get("emerging_topics") or []
    if emerging:
        lines.append(f"**Emerging topics suggested:** {', '.join(emerging)}")
        lines.append("")
    cited = local.get("cited_ids") or []
    lines.append(f"**Cited evidence IDs ({len(cited)}):** "
                 f"{', '.join(f'`{c}`' for c in cited) if cited else '_none_'}")
    if local.get("missing_ids"):
        lines.append(f"**MISSING IDs (citation check failed):** "
                     f"{', '.join(f'`{c}`' for c in local['missing_ids'])}")
    lines.append("")
    lines.append(f"**Citation check passed:** "
                 f"{'YES' if local.get('citation_check_passed') else 'NO'}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Cross-model comparison (auto-scored, from headline run)")
    lines.append("")
    lines.append("Same retrieved evidence shown to every model; only the generator "
                 "changes. Sorted by evidence coverage. Prose for non-local models was "
                 "not persisted in the prior run (rerun the eval to capture).")
    lines.append("")
    lines.append("| Model | n / err | Citation prec. | Hallucination | Coverage | "
                 "Latency (s) | Cost ($) | Cited IDs |")
    lines.append("|---|:---:|---:|---:|---:|---:|---:|---|")
    rows = sorted(
        bundle.model_rows,
        key=lambda r: -(r.get("evidence_coverage") or 0.0),
    )
    for row in rows:
        if "error" in row:
            lines.append(
                f"| {FRIENDLY.get(row['model'], row['model'])} | 0 / err | "
                f"— | — | — | — | — | _{row.get('error', 'provider error')[:60]}_ |"
            )
            continue
        cited = row.get("cited_ids") or []
        cited_str = ", ".join(f"`{c}`" for c in cited[:5])
        if len(cited) > 5:
            cited_str += f", … (+{len(cited) - 5})"
        cite = row.get("citation_precision") or 0.0
        hall = row.get("hallucination_rate") or 0.0
        cov = row.get("evidence_coverage") or 0.0
        lat = row.get("latency_s") or 0.0
        cost = row.get("cost_estimate_usd") or 0.0
        lines.append(
            f"| {FRIENDLY.get(row['model'], row['model'])} | 1 / 0 | "
            f"{cite:.3f} | {hall:.3f} | {cov:.3f} | {lat:.2f} | {cost:.4f} | "
            f"{cited_str or '_none_'} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Ground truth (if available)")
    lines.append("")
    gt = bundle.ground_truth
    if gt["retrieval"]:
        rel_docs = gt["retrieval"].get("relevant_doc_ids") or []
        lines.append("**Retrieval gold (doc IDs that SHOULD have been retrieved):**")
        lines.append("")
        for did in rel_docs:
            in_evidence = any(ev["doc_id"] == did for ev in bundle.evidence)
            mark = "✓ retrieved" if in_evidence else "✗ MISSED"
            lines.append(f"- `{did}` ({mark})")
        lines.append("")
    if gt["faithfulness"]:
        lines.append("**Faithfulness gold (prior claim-level labels):**")
        lines.append("")
        for fr in gt["faithfulness"]:
            lines.append(f"- Recommendation `{fr['recommendation_id']}`:")
            for claim in fr.get("claims", []):
                supported = "✓" if claim.get("supported") else "✗"
                cited_ok = "✓" if claim.get("cited_correctly") else "✗"
                lines.append(
                    f"  - {supported} supported, {cited_ok} cited — "
                    f"\"{claim.get('text', '')}\""
                )
        lines.append("")
    if gt["relevance"]:
        lines.append("**Prior relevance ratings (1–5):**")
        lines.append("")
        for rr in gt["relevance"]:
            lines.append(
                f"- `{rr['recommendation_id']}`: **{rr.get('rating')}/5** — "
                f"_{rr.get('notes', '')}_"
            )
        lines.append("")
    if not any([gt["retrieval"], gt["faithfulness"], gt["relevance"]]):
        lines.append("_No prior ground truth labels exist for this course. "
                     "This packet establishes the first reviewer baseline._")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Reviewer ratings")
    lines.append("")
    lines.append("Enter your ratings in `ratings_template.csv` (one row per course). "
                 "Fields:")
    lines.append("")
    lines.append("- **relevance_1to5** — does the recommendation address what the "
                 "course actually teaches?")
    lines.append("- **faithfulness_1to5** — are all claims supported by the cited "
                 "evidence?")
    lines.append("- **actionability_1to5** — could a curriculum committee act on "
                 "this without further research?")
    lines.append("- **freetext_notes** — anything notable, concerns, or "
                 "suggestions for the prose.")
    lines.append("")
    return "\n".join(lines)


def _ratings_template_rows(bundles: list[CourseBundle]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in bundles:
        c = b.course
        out.append({
            "course_id": c["id"],
            "title": c.get("title", ""),
            "cs2023_area": c.get("cs2023_area", ""),
            "rater_id": "",
            "relevance_1to5": "",
            "faithfulness_1to5": "",
            "actionability_1to5": "",
            "freetext_notes": "",
        })
    return out


def _rubric_readme(n_courses: int, n_with_truth: int) -> str:
    return f"""# CURIA Reviewer Bundle

Welcome — and thank you for reviewing CURIA's recommendations.

This bundle contains **{n_courses} per-course review packets** for the
CURIA system, plus a ratings template for you to fill in. The system
ingests live industry signal (job postings, arXiv, Stack Overflow,
GitHub) and recommends what a curriculum unit might want to add or
update. Your job is to judge whether those recommendations are
reasonable, grounded in the cited evidence, and actionable.

## Workflow

1. Open `packets/<course_id>.md` for one course at a time.
2. Read sections 1–4: the input, the retrieved evidence, the system's
   recommendation, and any existing ground truth.
3. Open `ratings_template.csv` in a spreadsheet, find that course's
   row, and fill in:
   - **relevance_1to5** — does the recommendation address what the
     course actually teaches? (5 = directly applicable; 1 = off-topic)
   - **faithfulness_1to5** — are all claims supported by the cited
     evidence? (5 = every claim grounded; 1 = unsupported or
     hallucinated)
   - **actionability_1to5** — could a curriculum committee act on
     this without further research? (5 = ready to discuss; 1 = needs
     significant rewriting)
   - **freetext_notes** — anything notable; specifically flag any
     hallucinated citations or claims you can't verify against the
     evidence.
4. Move to the next packet.

## Time estimate

Roughly 15–30 minutes per packet × {n_courses} packets =
**~{n_courses // 4}–{n_courses // 2} hours total**. You do not need to
complete in one sitting.

## Inter-rater agreement target

If we have ≥ 2 raters, we target Cohen's κ ≥ 0.6 on faithfulness and
≥ 0.4 on relevance. You don't need to coordinate with other raters
mid-review; we'll compute κ on completed ratings afterward.

## What the recommendation looks like

The packet shows ONE model end-to-end: the **local extractive baseline**.
This is a deterministic, offline model that only emits sentences that
exist in the retrieved evidence — so it can't hallucinate by
construction. It's the faithfulness floor.

Section 3 of each packet shows how the other 16 frontier LLMs scored on
the same course (citation precision, coverage, latency, cost, cited
IDs). The other models' prose was not persisted in the prior run, so
you only see their citation behavior, not their full text.

## What "ground truth" means here

Of the {n_courses} courses in this bundle, **{n_with_truth}** have prior
labels (retrieval gold, faithfulness gold, or earlier relevance
ratings). The other {n_courses - n_with_truth} are unlabeled — your
ratings will be the first reviewer baseline. The packet notes which
ones have prior labels and which are first-baseline.

## Bundle contents

```
reviewer_bundle/
├── README.md                    (this file)
├── ratings_template.csv         (fill in one row per course)
├── packets/                     (one .md per course)
├── ground_truth/                (existing labels, jsonl)
└── reference/
    ├── paper_tables.md          (how your ratings feed the paper)
    └── figures/                 (the three paper figures)
```

## Questions / errata

If a packet is missing data, the recommendation prose is empty, or the
evidence text looks wrong, leave a note in the corresponding
`freetext_notes` cell and move on.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-file", default="data/eval/benchmark_units_tamu_50.json")
    parser.add_argument("--index-path", default="audit/faiss_index_large.pkl")
    parser.add_argument("--multi-artifact",
                        default="results/headline_multi_llm_50q_17models_rechecked.json")
    parser.add_argument("--retrieval-labels", default="data/eval/retrieval_labels.jsonl")
    parser.add_argument("--faithfulness-labels", default="data/eval/faithfulness_labels.jsonl")
    parser.add_argument("--relevance-ratings", default="data/eval/relevance_ratings.jsonl")
    parser.add_argument("--paper-tables", default="results/paper_tables.md")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--out-dir", default="reviewer_bundle")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-courses", type=int, default=None)
    args = parser.parse_args()

    units = json.loads(Path(args.units_file).read_text())
    if args.max_courses:
        units = units[: args.max_courses]
    print(f"loaded {len(units)} courses")

    index = FaissIndex.load(Path(args.index_path))
    pipeline = CuriaRagPipeline(index=index, generator=LocalGroundedGenerator())

    multi = json.loads(Path(args.multi_artifact).read_text())
    multi_rows = multi.get("rows", [])
    retrieval_labels = _load_jsonl(Path(args.retrieval_labels))
    faithfulness_labels = _load_jsonl(Path(args.faithfulness_labels))
    relevance_ratings = _load_jsonl(Path(args.relevance_ratings))

    out_dir = Path(args.out_dir)
    packets_dir = out_dir / "packets"
    truth_dir = out_dir / "ground_truth"
    ref_dir = out_dir / "reference"
    for d in (packets_dir, truth_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    bundles: list[CourseBundle] = []
    n_with_truth = 0
    for course in units:
        gt = _ground_truth_for_unit(
            course["id"], retrieval_labels, faithfulness_labels, relevance_ratings,
        )
        if any([gt["retrieval"], gt["faithfulness"], gt["relevance"]]):
            n_with_truth += 1
        bundle = _build_one_course(pipeline, course, args.k, multi_rows, gt)
        bundles.append(bundle)
        packet_text = _format_packet(bundle)
        (packets_dir / f"{course['id']}.md").write_text(packet_text)
        print(f"  packet: {course['id']}")

    csv_path = out_dir / "ratings_template.csv"
    rows = _ratings_template_rows(bundles)
    if rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {csv_path}")

    (out_dir / "README.md").write_text(_rubric_readme(len(units), n_with_truth))
    print(f"wrote {out_dir / 'README.md'}")

    for src in (
        Path(args.retrieval_labels),
        Path(args.faithfulness_labels),
        Path(args.relevance_ratings),
    ):
        if src.exists():
            shutil.copy2(src, truth_dir / src.name)
    print(f"copied ground truth -> {truth_dir}")

    paper_tables_path = Path(args.paper_tables)
    if paper_tables_path.exists():
        shutil.copy2(paper_tables_path, ref_dir / paper_tables_path.name)
    figures_dir = Path(args.figures_dir)
    if figures_dir.is_dir():
        ref_figs = ref_dir / "figures"
        ref_figs.mkdir(exist_ok=True)
        for fig in figures_dir.glob("*.png"):
            shutil.copy2(fig, ref_figs / fig.name)
    print(f"copied reference materials -> {ref_dir}")

    print()
    print(f"DONE. Bundle for {len(bundles)} courses at {out_dir.resolve()}")
    print(f"  ground-truth available for {n_with_truth}/{len(bundles)} courses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
