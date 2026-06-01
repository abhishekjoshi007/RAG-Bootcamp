"""Real monthly skill-frequency series derived from the dated corpus.

For each tracked skill, computes the share of documents dated in each calendar
month whose title+text mention the skill. This is the REAL historical signal
that replaces the synthetic _TREND_DEFS history for forecasting/backtest
validation — every value traces back to dated source documents.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import CORPUS_DIR
from .forecasting import tracked_skills


def _month(iso_date: str) -> str:
    return iso_date[:7]


def _month_range(months: list[str]) -> list[str]:
    if not months:
        return []
    lo, hi = min(months), max(months)
    y, m = int(lo[:4]), int(lo[5:7])
    ey, em = int(hi[:4]), int(hi[5:7])
    out: list[str] = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(Path(corpus_dir).glob("*.json")):
        try:
            docs.append(json.loads(path.read_text()))
        except Exception:
            continue
    return docs


def _normalize_text(text: str) -> str:
    return text.lower().replace("-", " ")


def _skill_variants(skill: str) -> set[str]:
    normalized = _normalize_text(skill)
    variants = {normalized}
    if normalized.endswith("s"):
        variants.add(normalized[:-1])
    variants.update({
        normalized.replace(" and ", " "),
        normalized.replace(" augmented ", " "),
    })
    aliases: dict[str, set[str]] = {
        "machine learning": {"ml", "machine learning engineer"},
        "large language models": {"large language model", "llm", "llms"},
        "retrieval augmented generation": {"retrieval augmented", "rag"},
        "prompt engineering": {"prompt engineer"},
        "vector databases": {"vector database"},
        "cloud native": {"cloud", "cloud engineer"},
        "supply chain security": {"devsecops", "supply chain"},
        "cybersecurity": {"security", "security engineer"},
        "data analyst": {"data analysis", "analytics engineer"},
        "data scientist": {"data science"},
        "embedded systems": {"embedded"},
        "guidance navigation control": {"gnc"},
        "gis analyst": {"gis", "geospatial", "remote sensing"},
        "medical device engineer": {"biomedical engineer"},
        "quantitative analyst": {"quantitative finance", "quantitative researcher"},
        "fintech engineer": {"fintech"},
        "precision agriculture": {"agtech"},
        "atmospheric scientist": {"climate scientist", "weather"},
    }
    variants.update(aliases.get(normalized, set()))
    return {v for v in variants if v}


def monthly_skill_frequency(
    corpus_dir: Path | str = CORPUS_DIR,
    skills=None,
    normalize: bool = True,
) -> tuple[dict[str, list[tuple[str, float]]], list[str], dict[str, int]]:
    """Return (series_by_skill, months, month_doc_counts).

    series_by_skill[skill] = [(month, value), ...] over the full month range.
    value = share of that month's docs mentioning the skill (normalize=True)
    or raw mention count (normalize=False).
    """
    docs = load_corpus(corpus_dir)
    skills = list(skills) if skills else tracked_skills()
    variants_by_skill = {skill: _skill_variants(skill) for skill in skills}
    month_total: dict[str, int] = defaultdict(int)
    month_skill: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    present: list[str] = []

    for doc in docs:
        mo = _month(doc.get("date", ""))
        if len(mo) != 7:
            continue
        present.append(mo)
        month_total[mo] += 1
        text = _normalize_text(f"{doc.get('title', '')} {doc.get('text', '')}")
        for skill in skills:
            if any(variant in text for variant in variants_by_skill[skill]):
                month_skill[skill][mo] += 1

    months = _month_range(present)
    series: dict[str, list[tuple[str, float]]] = {}
    for skill in skills:
        seq: list[tuple[str, float]] = []
        for mo in months:
            cnt = month_skill[skill].get(mo, 0)
            if normalize:
                tot = month_total.get(mo, 0)
                val = (cnt / tot) if tot else 0.0
            else:
                val = float(cnt)
            seq.append((mo, round(val, 6)))
        series[skill] = seq
    return series, months, dict(month_total)
