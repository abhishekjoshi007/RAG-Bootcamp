"""
University curriculum loader for CURIA.

Loads structured curriculum JSON files from data/universities/ and
converts them into the unit dicts the RAG pipeline expects.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT

UNIVERSITIES_DIR = ROOT / "data" / "universities"

# Registry: (university_short_name, field) -> filename
_REGISTRY: dict[tuple[str, str], str] = {
    ("TAMU", "Computer Science"):       "tamu_cs.json",
    ("TAMU", "Electrical Engineering"): "tamu_ee.json",
}

UNIVERSITY_OPTIONS = ["Texas A&M University (TAMU)"]
FIELD_OPTIONS: dict[str, list[str]] = {
    "Texas A&M University (TAMU)": ["Computer Science", "Electrical Engineering"],
}


def _short_name(university_label: str) -> str:
    """Extract short name from label like 'Texas A&M University (TAMU)' → 'TAMU'."""
    if "(" in university_label:
        return university_label.split("(")[1].rstrip(")")
    return university_label


def load_curriculum(university_label: str, field: str) -> dict:
    """
    Load the full curriculum dict for a given university + field combination.
    Returns the raw JSON dict with all units and metadata.
    """
    short = _short_name(university_label)
    key = (short, field)
    if key not in _REGISTRY:
        raise KeyError(
            f"No curriculum found for ({university_label}, {field}). "
            f"Available: {list(_REGISTRY.keys())}"
        )
    path = UNIVERSITIES_DIR / _REGISTRY[key]
    return json.loads(path.read_text())


def get_unit_titles(university_label: str, field: str) -> list[str]:
    """Return list of unit titles for the dropdown."""
    curriculum = load_curriculum(university_label, field)
    return [u["title"] for u in curriculum["units"]]


def get_unit_by_title(university_label: str, field: str, title: str) -> dict:
    """Return the pipeline-compatible unit dict for a given title."""
    curriculum = load_curriculum(university_label, field)
    for unit in curriculum["units"]:
        if unit["title"] == title:
            return {
                "id": unit["id"],
                "title": unit["title"],
                "description": unit["description"],
                "current_topics": unit["current_topics"],
                "courses": unit.get("courses", []),
                "cs2023_area": unit.get("cs2023_area", ""),
            }
    raise KeyError(f"Unit '{title}' not found in {university_label} {field} curriculum")


def get_all_units(university_label: str, field: str) -> list[dict]:
    """Return all pipeline-compatible unit dicts for a curriculum."""
    curriculum = load_curriculum(university_label, field)
    return [
        {
            "id": u["id"],
            "title": u["title"],
            "description": u["description"],
            "current_topics": u["current_topics"],
            "courses": u.get("courses", []),
            "cs2023_area": u.get("cs2023_area", ""),
        }
        for u in curriculum["units"]
    ]


def curriculum_summary(university_label: str, field: str) -> str:
    """Return a markdown summary of the curriculum for display."""
    curriculum = load_curriculum(university_label, field)
    lines = [
        f"**{curriculum['university']}** — {curriculum['degree']}",
        f"Department: {curriculum['department']}",
        f"Curriculum year: {curriculum['curriculum_year']}",
        f"Total units: {len(curriculum['units'])}",
        "",
        "| Unit | CS2023 Area | Courses |",
        "|---|---|---|",
    ]
    for u in curriculum["units"]:
        courses = ", ".join(u.get("courses", []))
        lines.append(f"| {u['title']} | {u.get('cs2023_area', '—')} | {courses} |")
    return "\n".join(lines)
