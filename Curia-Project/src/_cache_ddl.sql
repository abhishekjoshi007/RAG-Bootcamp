-- CURIA multi-layer cache schema. Lives in the same SQLite DB as the audit log.
-- All statements are idempotent (CREATE ... IF NOT EXISTS), safe on existing installs.

CREATE TABLE IF NOT EXISTS agent_a_cache (
    skill_id    TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    week_iso    TEXT    NOT NULL,
    intensity   REAL    NOT NULL,
    attribution TEXT    NOT NULL,
    n_mentions  INTEGER NOT NULL,
    computed_at TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    PRIMARY KEY (skill_id, source, week_iso)
);
CREATE INDEX IF NOT EXISTS idx_agent_a_skill_week ON agent_a_cache (skill_id, week_iso);
CREATE INDEX IF NOT EXISTS idx_agent_a_expires    ON agent_a_cache (expires_at);

CREATE TABLE IF NOT EXISTS agent_b_cache (
    skill_id       TEXT    NOT NULL,
    horizon_months INTEGER NOT NULL,
    forecast_value REAL    NOT NULL,
    ci_lower       REAL    NOT NULL,
    ci_upper       REAL    NOT NULL,
    slope          REAL    NOT NULL,
    model_name     TEXT    NOT NULL,
    backtest_mape  REAL,
    computed_at    TEXT    NOT NULL,
    expires_at     TEXT    NOT NULL,
    PRIMARY KEY (skill_id, horizon_months)
);
CREATE INDEX IF NOT EXISTS idx_agent_b_expires ON agent_b_cache (expires_at);

CREATE TABLE IF NOT EXISTS agent_c_cache (
    skill_id      TEXT PRIMARY KEY,
    drift_score   REAL NOT NULL,
    drift_p_value REAL,
    direction     TEXT,
    evidence_blob TEXT NOT NULL,
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    computed_at   TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_c_expires     ON agent_c_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_agent_c_drift_score ON agent_c_cache (drift_score DESC);

CREATE TABLE IF NOT EXISTS resource_cache (
    skill_id           TEXT    NOT NULL,
    resource_id        TEXT    NOT NULL,
    match_score        REAL    NOT NULL,
    prerequisite_depth INTEGER NOT NULL,
    estimated_hours    INTEGER,
    resource_meta      TEXT    NOT NULL,
    computed_at        TEXT    NOT NULL,
    expires_at         TEXT    NOT NULL,
    PRIMARY KEY (skill_id, resource_id)
);
CREATE INDEX IF NOT EXISTS idx_resource_skill_score ON resource_cache (skill_id, match_score DESC);
CREATE INDEX IF NOT EXISTS idx_resource_expires     ON resource_cache (expires_at);

CREATE TABLE IF NOT EXISTS recommendation_cache (
    query_hash          TEXT PRIMARY KEY,
    normalized_query    TEXT    NOT NULL,
    recommendation_json TEXT    NOT NULL,
    evidence_ids        TEXT    NOT NULL,
    citation_check_ok   INTEGER NOT NULL,
    llm_model           TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    expires_at          TEXT    NOT NULL,
    hit_count           INTEGER NOT NULL DEFAULT 0,
    last_hit_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_expires ON recommendation_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_rec_created ON recommendation_cache (created_at DESC);

CREATE TABLE IF NOT EXISTS recommendation_skill_links (
    query_hash TEXT NOT NULL,
    skill_id   TEXT NOT NULL,
    PRIMARY KEY (query_hash, skill_id)
);
CREATE INDEX IF NOT EXISTS idx_rsl_skill ON recommendation_skill_links (skill_id);

CREATE TABLE IF NOT EXISTS cache_invalidations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_table    TEXT    NOT NULL,
    entity_id      TEXT,
    invalidated_at TEXT    NOT NULL,
    reason         TEXT    NOT NULL,
    rows_affected  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_invalidations_at ON cache_invalidations (invalidated_at DESC);

CREATE TABLE IF NOT EXISTS batch_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT    NOT NULL,
    completed_at TEXT,
    status       TEXT    NOT NULL,
    n_docs_added INTEGER DEFAULT 0,
    n_agent_a    INTEGER DEFAULT 0,
    n_agent_b    INTEGER DEFAULT 0,
    n_agent_c    INTEGER DEFAULT 0,
    n_resources  INTEGER DEFAULT 0,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_batch_runs_started ON batch_runs (started_at DESC);
