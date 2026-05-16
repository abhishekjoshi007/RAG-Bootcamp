"""Step 10 — Audit: writes to SQLite, returns integer ID, row is readable."""
import json
import sqlite3
import tempfile
from datetime import date
from pathlib import Path
from src.models import Chunk, Recommendation, SearchResult
from src.grounding import check_citations, CitationCheck
from src.audit import AuditLog


def _result(doc_id: str) -> SearchResult:
    chunk = Chunk(chunk_id=f"{doc_id}_c0", parent_id=doc_id, title="T",
                  source="job_posting", date=date(2025, 1, 1),
                  text="evidence text", chunk_index=0)
    return SearchResult(chunk=chunk, similarity=0.8, score=0.75)


UNIT = {"id": "cs_ai_01", "title": "Generative AI", "description": "LLMs."}
EVIDENCE = [_result("jp_001"), _result("ax_002")]
REC = Recommendation(signal_strength="high", summary="Good (jp_001).",
                     emerging_topics=["llm"], evidence_ids=["jp_001"])
CITATION = check_citations(REC, EVIDENCE)
PROMPT = "Test prompt"


def test_write_returns_integer_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = AuditLog(Path(tmpdir) / "test.db")
        audit_id = log.write(UNIT, PROMPT, EVIDENCE, REC, CITATION)
        assert isinstance(audit_id, int) and audit_id > 0


def test_write_creates_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        log = AuditLog(db_path)
        log.write(UNIT, PROMPT, EVIDENCE, REC, CITATION)
        with sqlite3.connect(db_path) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "rag_runs" in tables


def test_written_row_is_readable():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        log = AuditLog(db_path)
        audit_id = log.write(UNIT, PROMPT, EVIDENCE, REC, CITATION)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT unit_json, prompt, recommendation_json FROM rag_runs WHERE id=?",
                (audit_id,)
            ).fetchone()
        assert row is not None
        unit_data = json.loads(row[0])
        assert unit_data["id"] == "cs_ai_01"
        assert row[1] == PROMPT
        rec_data = json.loads(row[2])
        assert rec_data["signal_strength"] == "high"


def test_multiple_writes_increment_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = AuditLog(Path(tmpdir) / "test.db")
        id1 = log.write(UNIT, PROMPT, EVIDENCE, REC, CITATION)
        id2 = log.write(UNIT, PROMPT, EVIDENCE, REC, CITATION)
        assert id2 > id1


def test_audit_log_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "nested" / "dir" / "audit.db"
        log = AuditLog(nested)
        assert nested.parent.exists()
