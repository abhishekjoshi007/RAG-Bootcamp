"""Weekly batch precomputation.

Idempotent — safe to re-run. Each step writes to the cache and logs to
batch_runs. Invoked from scripts/batch_refresh.py via cron, or directly in tests.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .agent_a_fusion import FusionAgent
from .agent_e_resources import ResourceMatcher
from .audit import AuditLog
from .cache import CacheLayer
from .config import AUDIT_DB_PATH, CORPUS_DIR, DRIFT_INVALIDATION_THRESHOLD
from .drift import SemanticDriftDetector
from .forecasting import SkillForecaster
from .ingest import ingest_all
from .storage import build_index_from_corpus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BatchResult:
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    n_docs_added: int = 0
    n_agent_a: int = 0
    n_agent_b: int = 0
    n_agent_c: int = 0
    n_resources: int = 0
    errors: list[str] = field(default_factory=list)
    run_id: Optional[int] = None


class BatchRunner:
    def __init__(
        self,
        cache: CacheLayer,
        audit: AuditLog,
        run_drift: bool = False,
        run_forecast: bool = True,
        skip_ingest: bool = False,
    ) -> None:
        self.cache = cache
        self.audit = audit
        self.run_drift = run_drift
        self.run_forecast = run_forecast
        self.skip_ingest = skip_ingest

    def run_full_refresh(self) -> BatchResult:
        result = BatchResult(started_at=_now())
        result.run_id = self._log_batch_start(result.started_at)
        try:
            if not self.skip_ingest:
                result.n_docs_added = self._ingest_new()
                self._reindex_if_needed(result.n_docs_added)
            result.n_agent_a = self._refresh_agent_a()
            if self.run_forecast:
                result.n_agent_b = self._refresh_agent_b()
            if self.run_drift:
                result.n_agent_c = self._refresh_agent_c()
            result.n_resources = self._refresh_resources()
            result.status = "success"
        except Exception as exc:  # noqa: BLE001 - surface any step failure
            result.status = "failed"
            result.errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Batch refresh failed")
        finally:
            result.completed_at = _now()
            self._log_batch_complete(result)
        return result

    # ---------- Step implementations ----------
    def _ingest_new(self) -> int:
        since = self._last_successful_batch_timestamp()
        return ingest_all(CORPUS_DIR, since=since)

    def _reindex_if_needed(self, n_new: int) -> None:
        if n_new == 0:
            return
        build_index_from_corpus(CORPUS_DIR)

    def _refresh_agent_a(self) -> int:
        rows = FusionAgent().compute_for_window(weeks=13)
        return self.cache.set_agent_a(rows)

    def _refresh_agent_b(self) -> int:
        forecaster = SkillForecaster(cache=self.cache)
        rows = forecaster.forecast_all_skills(horizons=(3, 6, 12, 24))
        return self.cache.set_agent_b(rows)

    def _refresh_agent_c(self) -> int:
        detector = SemanticDriftDetector(cache=self.cache)
        rows = detector.detect_all_skills()
        n_written = self.cache.set_agent_c(rows)
        for row in rows:
            if row["drift_score"] > DRIFT_INVALIDATION_THRESHOLD:
                self.cache.invalidate_skill(
                    skill_id=row["skill_id"], layers=("b",), reason="drift_detected"
                )
                self.cache.invalidate_recommendations_touching_skill(
                    row["skill_id"], reason="drift_detected"
                )
        return n_written

    def _refresh_resources(self) -> int:
        rows = ResourceMatcher(cache=self.cache).match_all_skills()
        return self.cache.set_resources(rows)

    # ---------- Helpers ----------
    def _last_successful_batch_timestamp(self) -> Optional[datetime]:
        with sqlite3.connect(self.audit.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(started_at) AS s FROM batch_runs WHERE status = 'success'"
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def _log_batch_start(self, started_at: datetime) -> int:
        with sqlite3.connect(self.audit.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO batch_runs (started_at, status) VALUES (?, 'running')",
                (started_at.isoformat(),),
            )
            return int(cur.lastrowid)

    def _log_batch_complete(self, result: BatchResult) -> None:
        with sqlite3.connect(self.audit.db_path) as conn:
            conn.execute(
                "UPDATE batch_runs SET completed_at = ?, status = ?, "
                "n_docs_added = ?, n_agent_a = ?, n_agent_b = ?, n_agent_c = ?, "
                "n_resources = ?, error_message = ? WHERE id = ?",
                (
                    result.completed_at.isoformat() if result.completed_at else None,
                    result.status,
                    result.n_docs_added, result.n_agent_a, result.n_agent_b,
                    result.n_agent_c, result.n_resources,
                    "; ".join(result.errors) if result.errors else None,
                    result.run_id,
                ),
            )
