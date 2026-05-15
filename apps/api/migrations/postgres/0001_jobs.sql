-- Jobs table — Postgres port of migrations/sqlite/0001_jobs.sql.
--
-- The two ports share string semantics for UUIDs (TEXT) and timestamps
-- (RFC3339 TEXT) on purpose: it keeps `Job` rows interchangeable with a
-- `pg_dump --inserts` -> `sqlite3 .read` round-trip during an emergency
-- failback. We deliberately do NOT use TIMESTAMPTZ / UUID native types
-- on the Postgres side — that would force the application layer to
-- carry two SELECT/INSERT code paths and undermine the trait abstraction.
--
-- The only structural rewrites are integer width (Postgres BIGINT is
-- the canonical match for sqlx's `i64`; SQLite's INTEGER widens to
-- i64 implicitly) and float type (DOUBLE PRECISION is the analogue of
-- SQLite's REAL when bound as f32).
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY NOT NULL,
    preset              TEXT NOT NULL,
    filename            TEXT NOT NULL,
    size_bytes          BIGINT NOT NULL,
    label               TEXT,
    status              TEXT NOT NULL,
    blob_key            TEXT NOT NULL,
    blob_url            TEXT,
    source_url          TEXT,
    upload_size_bytes   BIGINT,
    output_url          TEXT,
    preview_url         TEXT,
    phase               TEXT,
    percent             DOUBLE PRECISION,
    webhook_url         TEXT,
    batch_id            TEXT,
    tier                TEXT NOT NULL DEFAULT 'free',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_batch_id   ON jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status);
