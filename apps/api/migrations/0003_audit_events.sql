-- Audit log for mutating routes.
--
-- One row per request to a mutating /v1 endpoint, written at request
-- completion time (best-effort — a failing INSERT must NOT fail the
-- underlying request). Used for forensics, billing reconciliation, and
-- to satisfy the SOC2-Type-I "tamper-evident operations log" control
-- once we file. Until then, this is just the operator's audit trail.
--
-- We deliberately log only the **key prefix** (first 8 chars), never the
-- full bearer token, so a dump of this table can't be replayed against
-- the API. The route is the templated path (e.g. `/v1/jobs/:id/repack`)
-- not the concrete one — so we never accidentally persist a user-supplied
-- uuid in a column that's queried by route class.
--
-- `error` carries a *short* operator-facing reason (rate-limited / bad
-- request / etc.) — never the raw error envelope.
CREATE TABLE IF NOT EXISTS audit_events (
    id              TEXT PRIMARY KEY NOT NULL,
    key_prefix      TEXT NOT NULL,
    route           TEXT NOT NULL,
    method          TEXT NOT NULL,
    status          INTEGER NOT NULL,
    body_size       INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_key_prefix ON audit_events(key_prefix);
CREATE INDEX IF NOT EXISTS idx_audit_events_route      ON audit_events(route);
