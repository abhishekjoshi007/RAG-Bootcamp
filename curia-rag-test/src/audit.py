from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .grounding import CitationCheck
from .models import Recommendation, SearchResult


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    unit_json TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    recommendation_json TEXT NOT NULL,
                    citation_check_json TEXT NOT NULL
                )
                """
            )

    def write(
        self,
        unit: dict,
        prompt: str,
        evidence: list[SearchResult],
        recommendation: Recommendation,
        citation_check: CitationCheck,
    ) -> int:
        evidence_rows = [
            {
                "chunk_id": result.chunk.chunk_id,
                "parent_id": result.chunk.parent_id,
                "title": result.chunk.title,
                "source": result.chunk.source,
                "date": result.chunk.date.isoformat(),
                "similarity": result.similarity,
                "score": result.score,
                "text": result.chunk.text,
            }
            for result in evidence
        ]
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO rag_runs (
                    created_at,
                    unit_json,
                    prompt,
                    evidence_json,
                    recommendation_json,
                    citation_check_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(unit, sort_keys=True),
                    prompt,
                    json.dumps(evidence_rows, sort_keys=True),
                    json.dumps(asdict(recommendation), sort_keys=True),
                    json.dumps(asdict(citation_check), sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)
