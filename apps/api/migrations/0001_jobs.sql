-- Jobs table — one row per optimize/repack call.
--
-- All UUIDs and timestamps are stored as TEXT (hyphenated UUID, RFC3339)
-- because sqlx maps them to the Rust types via the `uuid` + `chrono`
-- features. Storing them as BLOB / INTEGER would be marginally smaller
-- but breaks `sqlite3 jobs.db ".dump"` for human inspection.
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY NOT NULL,
    preset              TEXT NOT NULL,
    filename            TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL,
    label               TEXT,
    status              TEXT NOT NULL,
    blob_key            TEXT NOT NULL,
    blob_url            TEXT,
    source_url          TEXT,
    upload_size_bytes   INTEGER,
    output_url          TEXT,
    preview_url         TEXT,
    phase               TEXT,
    percent             REAL,
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
