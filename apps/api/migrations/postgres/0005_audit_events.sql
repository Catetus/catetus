-- Audit log for mutating routes (Postgres port of sqlite/0005).
--
-- Same shape as the SQLite variant; `INTEGER` → `BIGINT` / `INTEGER`
-- to match Postgres native types. `created_at` is kept as TEXT so the
-- shared row decoder in `store/postgres.rs` can parse it identically
-- to the SQLite path (RFC3339 strings throughout).
CREATE TABLE IF NOT EXISTS audit_events (
    id              TEXT PRIMARY KEY NOT NULL,
    key_prefix      TEXT NOT NULL,
    route           TEXT NOT NULL,
    method          TEXT NOT NULL,
    status          INTEGER NOT NULL,
    body_size       BIGINT NOT NULL DEFAULT 0,
    duration_ms     BIGINT NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_key_prefix ON audit_events(key_prefix);
CREATE INDEX IF NOT EXISTS idx_audit_events_route      ON audit_events(route);
