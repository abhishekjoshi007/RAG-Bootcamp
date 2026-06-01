"""Multi-layer cache for CURIA agent outputs and recommendations.

All cache I/O goes through this module, backed by the same SQLite database as
the audit log (AUDIT_DB_PATH). Stale entries are never silently returned: every
get_* returns None on miss or expiry, leaving the compute decision to the caller.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .config import (
    AUDIT_DB_PATH,
    CACHE_TTL_AGENT_A_DAYS,
    CACHE_TTL_AGENT_B_DAYS,
    CACHE_TTL_AGENT_C_DAYS,
    CACHE_TTL_RECOMMENDATION_DAYS,
    CACHE_TTL_RESOURCE_DAYS,
)

_CACHE_DDL = (Path(__file__).parent / "_cache_ddl.sql").read_text(encoding="utf-8")

_LAYER_TO_TABLE = {
    "a": "agent_a_cache",
    "b": "agent_b_cache",
    "c": "agent_c_cache",
    "resources": "resource_cache",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    value: Mapping[str, Any]
    computed_at: datetime
    expires_at: datetime

    def is_fresh(self, now: Optional[datetime] = None) -> bool:
        return (now or _now()) < self.expires_at


class CacheLayer:
    def __init__(self, db_path: Path | str = AUDIT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CACHE_DDL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------- Recommendation cache ----------
    def get_recommendation(self, query_hash: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT recommendation_json, evidence_ids, expires_at "
                "FROM recommendation_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) < _now():
                return None
            conn.execute(
                "UPDATE recommendation_cache SET hit_count = hit_count + 1, "
                "last_hit_at = ? WHERE query_hash = ?",
                (_iso(_now()), query_hash),
            )
            return {
                "recommendation": json.loads(row["recommendation_json"]),
                "evidence_ids": json.loads(row["evidence_ids"]),
            }

    def set_recommendation(
        self,
        query_hash: str,
        normalized_query: dict,
        recommendation: dict,
        evidence_ids: Sequence[str],
        llm_model: str,
        citation_check_ok: bool,
        ttl_days: Optional[int] = None,
    ) -> None:
        ttl = ttl_days if ttl_days is not None else CACHE_TTL_RECOMMENDATION_DAYS
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO recommendation_cache "
                "(query_hash, normalized_query, recommendation_json, evidence_ids, "
                "citation_check_ok, llm_model, created_at, expires_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT hit_count FROM recommendation_cache WHERE query_hash = ?), 0))",
                (
                    query_hash,
                    json.dumps(normalized_query, sort_keys=True),
                    json.dumps(recommendation),
                    json.dumps(list(evidence_ids)),
                    1 if citation_check_ok else 0,
                    llm_model,
                    _iso(now),
                    _iso(now + timedelta(days=ttl)),
                    query_hash,
                ),
            )

    # ---------- Agent A cache ----------
    def get_agent_a(
        self, skill_id: str, sources: Optional[Iterable[str]] = None
    ) -> list[CacheEntry]:
        query = (
            "SELECT skill_id, source, week_iso, intensity, attribution, n_mentions, "
            "computed_at, expires_at FROM agent_a_cache "
            "WHERE skill_id = ? AND expires_at > ?"
        )
        params: list[Any] = [skill_id, _iso(_now())]
        if sources:
            sources = list(sources)
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            CacheEntry(
                key=f"{r['skill_id']}::{r['source']}::{r['week_iso']}",
                value={
                    "intensity": r["intensity"],
                    "attribution": json.loads(r["attribution"]),
                    "n_mentions": r["n_mentions"],
                    "source": r["source"],
                    "week_iso": r["week_iso"],
                },
                computed_at=datetime.fromisoformat(r["computed_at"]),
                expires_at=datetime.fromisoformat(r["expires_at"]),
            )
            for r in rows
        ]

    def set_agent_a(self, rows: Sequence[dict]) -> int:
        now = _now()
        expires = _iso(now + timedelta(days=CACHE_TTL_AGENT_A_DAYS))
        records = [
            (
                r["skill_id"], r["source"], r["week_iso"], r["intensity"],
                json.dumps(r["attribution"]), r["n_mentions"],
                _iso(now), expires,
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO agent_a_cache "
                "(skill_id, source, week_iso, intensity, attribution, n_mentions, "
                "computed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        return len(records)

    # ---------- Agent B cache ----------
    def get_agent_b(self, skill_id: str, horizon_months: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT forecast_value, ci_lower, ci_upper, slope, model_name, "
                "backtest_mape, expires_at FROM agent_b_cache "
                "WHERE skill_id = ? AND horizon_months = ?",
                (skill_id, horizon_months),
            ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < _now():
            return None
        return dict(row)

    def set_agent_b(self, rows: Sequence[dict]) -> int:
        now = _now()
        expires = _iso(now + timedelta(days=CACHE_TTL_AGENT_B_DAYS))
        records = [
            (
                r["skill_id"], r["horizon_months"], r["forecast_value"],
                r["ci_lower"], r["ci_upper"], r["slope"],
                r["model_name"], r.get("backtest_mape"),
                _iso(now), expires,
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO agent_b_cache "
                "(skill_id, horizon_months, forecast_value, ci_lower, ci_upper, "
                "slope, model_name, backtest_mape, computed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        return len(records)

    # ---------- Agent C cache ----------
    def get_agent_c(self, skill_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_c_cache WHERE skill_id = ?", (skill_id,)
            ).fetchone()
        if row is None or datetime.fromisoformat(row["expires_at"]) < _now():
            return None
        result = dict(row)
        result["evidence_blob"] = json.loads(result["evidence_blob"])
        return result

    def set_agent_c(self, rows: Sequence[dict]) -> int:
        now = _now()
        expires = _iso(now + timedelta(days=CACHE_TTL_AGENT_C_DAYS))
        records = [
            (
                r["skill_id"], r["drift_score"], r.get("drift_p_value"),
                r["direction"], json.dumps(r["evidence_blob"]),
                r["window_start"], r["window_end"],
                _iso(now), expires,
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO agent_c_cache "
                "(skill_id, drift_score, drift_p_value, direction, evidence_blob, "
                "window_start, window_end, computed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        return len(records)

    # ---------- Resource cache ----------
    def get_resources_for_skill(self, skill_id: str, top_k: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT resource_id, match_score, prerequisite_depth, "
                "estimated_hours, resource_meta FROM resource_cache "
                "WHERE skill_id = ? AND expires_at > ? "
                "ORDER BY match_score DESC LIMIT ?",
                (skill_id, _iso(_now()), top_k),
            ).fetchall()
        return [
            {
                "resource_id": r["resource_id"],
                "match_score": r["match_score"],
                "prerequisite_depth": r["prerequisite_depth"],
                "estimated_hours": r["estimated_hours"],
                "meta": json.loads(r["resource_meta"]),
            }
            for r in rows
        ]

    def set_resources(self, rows: Sequence[dict]) -> int:
        now = _now()
        expires = _iso(now + timedelta(days=CACHE_TTL_RESOURCE_DAYS))
        records = [
            (
                r["skill_id"], r["resource_id"], r["match_score"],
                r["prerequisite_depth"], r.get("estimated_hours"),
                json.dumps(r["meta"]),
                _iso(now), expires,
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO resource_cache "
                "(skill_id, resource_id, match_score, prerequisite_depth, "
                "estimated_hours, resource_meta, computed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )
        return len(records)

    # ---------- Invalidation ----------
    def invalidate_recommendation(self, query_hash: str, reason: str) -> int:
        return self._delete_and_log(
            "recommendation_cache", "query_hash = ?", (query_hash,), reason
        )

    def invalidate_skill(
        self, skill_id: str, layers: Iterable[str], reason: str
    ) -> dict[str, int]:
        affected: dict[str, int] = {}
        for layer in layers:
            table = _LAYER_TO_TABLE[layer]
            affected[layer] = self._delete_and_log(
                table, "skill_id = ?", (skill_id,), reason
            )
        return affected

    def link_recommendation_skills(
        self, query_hash: str, skill_ids: Iterable[str]
    ) -> int:
        records = [(query_hash, s) for s in dict.fromkeys(skill_ids)]
        if not records:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO recommendation_skill_links "
                "(query_hash, skill_id) VALUES (?, ?)",
                records,
            )
        return len(records)

    def recommendations_touching_skill(self, skill_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT query_hash FROM recommendation_skill_links WHERE skill_id = ?",
                (skill_id,),
            ).fetchall()
        return [r["query_hash"] for r in rows]

    def invalidate_recommendations_touching_skill(
        self, skill_id: str, reason: str
    ) -> int:
        total = 0
        for query_hash in self.recommendations_touching_skill(skill_id):
            total += self._delete_and_log(
                "recommendation_cache", "query_hash = ?", (query_hash,), reason
            )
        return total

    def _delete_and_log(
        self, table: str, where: str, params: tuple, reason: str
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE {where}", params)
            rows_affected = cur.rowcount
            conn.execute(
                "INSERT INTO cache_invalidations "
                "(cache_table, entity_id, invalidated_at, reason, rows_affected) "
                "VALUES (?, ?, ?, ?, ?)",
                (table, params[0] if params else None, _iso(_now()), reason, rows_affected),
            )
        return rows_affected

    def purge_stale(self) -> dict[str, int]:
        now = _iso(_now())
        purged: dict[str, int] = {}
        with self._conn() as conn:
            for layer, table in _LAYER_TO_TABLE.items():
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE expires_at < ?", (now,)
                )
                purged[layer] = cur.rowcount
            cur = conn.execute(
                "DELETE FROM recommendation_cache WHERE expires_at < ?", (now,)
            )
            purged["recommendations"] = cur.rowcount
        return purged

    # ---------- Monitoring ----------
    def stats(self) -> dict:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT 'agent_a' AS t, COUNT(*) AS n FROM agent_a_cache "
                "UNION ALL SELECT 'agent_b', COUNT(*) FROM agent_b_cache "
                "UNION ALL SELECT 'agent_c', COUNT(*) FROM agent_c_cache "
                "UNION ALL SELECT 'resources', COUNT(*) FROM resource_cache "
                "UNION ALL SELECT 'recommendations', COUNT(*) FROM recommendation_cache"
            )
            counts = {row["t"]: row["n"] for row in cur.fetchall()}
            hit_row = conn.execute(
                "SELECT SUM(hit_count) AS h, COUNT(*) AS n FROM recommendation_cache"
            ).fetchone()
        return {
            "counts": counts,
            "recommendation_total_hits": hit_row["h"] or 0,
            "recommendation_entries": hit_row["n"] or 0,
        }
