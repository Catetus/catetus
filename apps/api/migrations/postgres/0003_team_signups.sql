-- Self-serve Team-tier signup ledger (Postgres port of sqlite/0003).
--
-- The UNIQUE(stripe_session_id) constraint plus the application-layer
-- `ON CONFLICT(stripe_session_id) DO NOTHING` is the no-double-issuance
-- gate. `ON CONFLICT … DO NOTHING` is identical syntax on Postgres ≥9.5
-- and SQLite ≥3.24, which is the only reason the trait abstraction can
-- share a single SQL string between the two backends for this INSERT.
CREATE TABLE IF NOT EXISTS team_signups (
    id                     TEXT PRIMARY KEY NOT NULL,
    stripe_session_id      TEXT NOT NULL UNIQUE,
    stripe_customer_id     TEXT NOT NULL,
    stripe_subscription_id TEXT,
    email                  TEXT NOT NULL,
    claim_token            TEXT NOT NULL,
    key_prefix             TEXT NOT NULL,
    key_hash               TEXT NOT NULL,
    seats                  BIGINT NOT NULL DEFAULT 1,
    created_at             TEXT NOT NULL,
    key_revealed_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_team_signups_customer ON team_signups(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_team_signups_email    ON team_signups(email);
