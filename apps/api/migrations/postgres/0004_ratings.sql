-- fidelity-ml v0.4 pairwise ratings (Postgres port of sqlite/0004).
--
-- `AUTOINCREMENT` -> `BIGSERIAL`. Postgres BIGSERIAL is internally
-- backed by a sequence; the next-id semantic is "monotonic per session,
-- not necessarily contiguous on rollback". SQLite's AUTOINCREMENT is
-- contiguous-on-success but the public surface only echoes the id
-- back to the page (it isn't used as a join key from another table),
-- so the gap behavior is invisible to callers.
--
-- One application-layer adapter: SQLite returns `last_insert_rowid`
-- via `ExecuteResult::last_insert_rowid()`. Postgres returns it via
-- `INSERT … RETURNING id` because Postgres doesn't have a session-
-- level "last id" concept. The store impl in `store/postgres.rs`
-- uses RETURNING; the SQLite impl uses last_insert_rowid().
CREATE TABLE IF NOT EXISTS ratings (
    id              BIGSERIAL PRIMARY KEY,
    scene_id        TEXT NOT NULL,
    left_preset     TEXT NOT NULL,
    right_preset    TEXT NOT NULL,
    winner          TEXT NOT NULL,
    respondent_hash TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_scene       ON ratings(scene_id);
CREATE INDEX IF NOT EXISTS idx_ratings_respondent  ON ratings(respondent_hash, created_at);
CREATE INDEX IF NOT EXISTS idx_ratings_summary     ON ratings(scene_id, left_preset, right_preset);
