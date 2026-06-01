from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def extract_curriculum_units(payload: dict, source_path: Path | None = None) -> list[dict]:
    """Flatten one university curriculum JSON payload into benchmark units."""
    shared_context = {
        "university": payload.get("university"),
        "short_name": payload.get("short_name"),
        "college": payload.get("college"),
        "website": payload.get("website"),
        "curriculum_year": payload.get("curriculum_year"),
    }
    units: list[dict] = []

    if payload.get("units"):
        context = {
            **shared_context,
            "field": payload.get("field"),
            "degree": payload.get("degree"),
            "department": payload.get("department"),
        }
        units.extend(_normalize_unit(unit, context, source_path) for unit in payload["units"])

    for program in payload.get("programs", []):
        context = {
            **shared_context,
            "field": program.get("field"),
            "degree": program.get("degree"),
            "department": program.get("department"),
        }
        units.extend(_normalize_unit(unit, context, source_path) for unit in program.get("units", []))

    return units


def collect_curriculum_units(paths: Iterable[Path], limit: int | None = None) -> tuple[list[dict], list[str]]:
    units: list[dict] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        for unit in extract_curriculum_units(payload, source_path=path):
            unit_id = unit["id"]
            if unit_id in seen:
                duplicates.append(unit_id)
                continue
            seen.add(unit_id)
            units.append(unit)
            if limit is not None and len(units) >= limit:
                return units, duplicates
    return units, duplicates


def _normalize_unit(unit: dict, context: dict, source_path: Path | None) -> dict:
    if not unit.get("id") or not unit.get("title"):
        raise ValueError("Curriculum units must include id and title")

    normalized = {
        "id": unit["id"],
        "title": unit["title"],
        "description": unit.get("description", ""),
        "current_topics": list(unit.get("current_topics", [])),
    }
    for key in ("courses", "cs2023_area"):
        if key in unit:
            normalized[key] = unit[key]

    metadata = {key: value for key, value in context.items() if value is not None}
    if source_path is not None:
        metadata["source_file"] = str(source_path)
    if metadata:
        normalized["metadata"] = metadata
    return normalized
