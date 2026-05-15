-- Stripe customer linkage for usage-based billing.
--
-- One bearer API key maps to one Stripe customer (cus_xxx). We stamp the
-- customer_id on every Job at creation time so the billing path doesn't
-- have to round-trip back to the key→customer map when it fires meter
-- events from the (asynchronous) Modal callback. Free-tier jobs leave
-- this NULL and never emit billing events.
--
-- Additive change: existing rows keep customer_id = NULL, which the
-- billing module treats as "untracked / no charge".
ALTER TABLE jobs ADD COLUMN customer_id TEXT;

CREATE INDEX IF NOT EXISTS idx_jobs_customer_id ON jobs(customer_id);

-- Billing-event ledger. One row per (job_id, sku) we've sent to Stripe,
-- with the idempotency key we used. The UNIQUE constraint is the
-- no-double-charge invariant: a Modal callback that fires twice can't
-- create two rows, so it can't post two meter events.
--
--   sku          — e.g. "splatforge_repack_runs" / "splatforge_repack_seconds"
--   units        — integer count we billed for (runs=1, seconds=elapsed_s)
--   idempotency  — the Stripe-Idempotency-Key we sent (sha256-derived)
--   stripe_event — Stripe's returned meter-event identifier, when present
--   created_at   — RFC3339, server clock
CREATE TABLE IF NOT EXISTS billing_events (
    id              TEXT PRIMARY KEY NOT NULL,
    job_id          TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    sku             TEXT NOT NULL,
    units           INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    stripe_event_id TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (job_id, sku)
);

CREATE INDEX IF NOT EXISTS idx_billing_events_customer ON billing_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_job      ON billing_events(job_id);
