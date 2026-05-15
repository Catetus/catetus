-- Stripe customer linkage (Postgres port of migrations/sqlite/0002).
--
-- Same semantics as the SQLite version: customer_id is NULLable, NULL
-- means "free / untracked / no charge". The most load-bearing portability
-- note lives here: Postgres and SQLite both allow multiple NULLs in a
-- non-UNIQUE column, but they DIFFER on multiple NULLs under a UNIQUE
-- constraint — SQLite allows them, Postgres rejects (treats NULLs as
-- distinct only when the constraint is `UNIQUE NULLS DISTINCT`, which
-- is the default in Postgres ≥15). Nothing in this schema relies on
-- that distinction, but anyone adding a UNIQUE constraint on a NULLable
-- column in a future migration must check the Postgres-side behavior
-- and (probably) make the column NOT NULL with a sentinel value.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS customer_id TEXT;

CREATE INDEX IF NOT EXISTS idx_jobs_customer_id ON jobs(customer_id);

-- Billing-event ledger. UNIQUE(job_id, sku) is the no-double-charge
-- invariant. Both job_id and sku are NOT NULL, so the SQLite/Postgres
-- multi-NULL semantic divergence (above) is sidestepped by construction.
CREATE TABLE IF NOT EXISTS billing_events (
    id              TEXT PRIMARY KEY NOT NULL,
    job_id          TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    sku             TEXT NOT NULL,
    units           BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    stripe_event_id TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (job_id, sku)
);

CREATE INDEX IF NOT EXISTS idx_billing_events_customer ON billing_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_job      ON billing_events(job_id);
