"""
University curriculum loader for CURIA.

Loads all TAMU college JSON files and exposes a College → Field → Units
hierarchy for the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT

UNIVERSITIES_DIR = ROOT / "data" / "universities"

# ---------------------------------------------------------------------------
# Raw college files to load
# ---------------------------------------------------------------------------
_COLLEGE_FILES = [
    "tamu_engineering.json",
    "tamu_science.json",
    "tamu_business.json",
    "tamu_geosciences_agriculture.json",
]

# ---------------------------------------------------------------------------
# Build the registry from loaded JSON
# ---------------------------------------------------------------------------

# college_name -> list of program dicts
_COLLEGE_PROGRAMS: dict[str, list[dict]] = {}

# (field) -> list of unit dicts (pipeline-compatible)
_FIELD_UNITS: dict[str, list[dict]] = {}

# field -> degree title
_FIELD_DEGREES: dict[str, str] = {}


def _load_all() -> None:
    for fname in _COLLEGE_FILES:
        path = UNIVERSITIES_DIR / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        college = data.get("college", "Unknown College")
        programs = data.get("programs", [])
        _COLLEGE_PROGRAMS.setdefault(college, []).extend(programs)
        for prog in programs:
            field = prog["field"]
            _FIELD_DEGREES[field] = prog.get("degree", field)
            units = [
                {
                    "id": u["id"],
                    "title": u["title"],
                    "description": u["description"],
                    "current_topics": u["current_topics"],
                    "courses": u.get("courses", []),
                    "cs2023_area": u.get("cs2023_area", ""),
                    "field": field,
                }
                for u in prog.get("units", [])
            ]
            _FIELD_UNITS[field] = units


_load_all()

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

UNIVERSITY_OPTIONS = ["Texas A&M University (TAMU)"]

COLLEGE_OPTIONS: dict[str, list[str]] = {
    "Texas A&M University (TAMU)": sorted(_COLLEGE_PROGRAMS.keys())
}

FIELD_OPTIONS_BY_COLLEGE: dict[str, list[str]] = {
    college: sorted(p["field"] for p in programs)
    for college, programs in _COLLEGE_PROGRAMS.items()
}

# Flat field list for the university (all colleges combined)
ALL_FIELDS: list[str] = sorted(_FIELD_UNITS.keys())


def get_college_for_field(field: str) -> str:
    for college, programs in _COLLEGE_PROGRAMS.items():
        for p in programs:
            if p["field"] == field:
                return college
    return "Unknown"


def get_unit_titles(field: str) -> list[str]:
    return [u["title"] for u in _FIELD_UNITS.get(field, [])]


def get_unit_by_title(field: str, title: str) -> dict:
    for unit in _FIELD_UNITS.get(field, []):
        if unit["title"] == title:
            return unit
    raise KeyError(f"Unit '{title}' not found in field '{field}'")


def get_all_units(field: str) -> list[dict]:
    return list(_FIELD_UNITS.get(field, []))


def curriculum_summary(field: str) -> str:
    units = _FIELD_UNITS.get(field, [])
    degree = _FIELD_DEGREES.get(field, field)
    college = get_college_for_field(field)
    lines = [
        f"**Texas A&M University** — {degree}",
        f"College: {college}",
        f"Total units: {len(units)}",
        "",
        "| # | Unit | CS2023 Area | Courses |",
        "|---|---|---|---|",
    ]
    for i, u in enumerate(units, 1):
        courses = ", ".join(u.get("courses", []))
        lines.append(f"| {i} | {u['title']} | {u.get('cs2023_area','—')} | {courses} |")
    return "\n".join(lines)


def all_programs_summary() -> str:
    lines = ["# Texas A&M University — All Programs\n"]
    for college, programs in sorted(_COLLEGE_PROGRAMS.items()):
        lines.append(f"## {college}\n")
        for p in sorted(programs, key=lambda x: x["field"]):
            n = len(p.get("units", []))
            lines.append(f"- **{p['field']}** ({p['degree']}) — {n} knowledge units")
        lines.append("")
    return "\n".join(lines)
