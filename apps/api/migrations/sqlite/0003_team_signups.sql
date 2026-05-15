-- Self-serve Team-tier signup ledger.
--
-- One row per Stripe Checkout Session that resulted in a paid Team
-- subscription. The webhook handler INSERTs into this table with
-- ON CONFLICT(stripe_session_id) DO NOTHING — that conflict clause is
-- the load-bearing "no double key issuance" invariant. A Stripe webhook
-- retry that lands a second time finds the row already present, skips
-- key minting, and returns 200 so Stripe stops retrying.
--
-- The plaintext key is *never* stored. Only:
--   * `key_hash`   — SHA-256 hex of the plaintext (matches the contract
--                    on the WorkOS branch's `api_keys.key_hash` column,
--                    so post-merge this row can be promoted into the
--                    full api_keys schema without re-hashing).
--   * `key_prefix` — first 12 chars of the plaintext (`sf_live_XXXX`),
--                    safe to display in the dashboard / logs.
--
-- The plaintext itself is held in a side-channel (the in-memory
-- `PendingKeyCache` in `checkout.rs`) for at most 10 minutes between
-- the webhook completing and the user landing on /welcome. Once the
-- user fetches /v1/checkout/reveal, `key_revealed_at` is stamped and
-- the plaintext is evicted from memory. A second reveal attempt finds
-- `key_revealed_at IS NOT NULL` and returns 410 Gone — see
-- `tests/checkout.rs::reveal_is_strictly_one_shot`.
CREATE TABLE IF NOT EXISTS team_signups (
    id                  TEXT PRIMARY KEY NOT NULL,
    stripe_session_id   TEXT NOT NULL UNIQUE,
    stripe_customer_id  TEXT NOT NULL,
    stripe_subscription_id TEXT,
    email               TEXT NOT NULL,
    -- Random unguessable token returned to the buyer via success_url so
    -- only the actual checkout return URL can hit /reveal. Without this,
    -- anyone who guessed a session id could steal a fresh customer's
    -- plaintext key.
    claim_token         TEXT NOT NULL,
    key_prefix          TEXT NOT NULL,
    key_hash            TEXT NOT NULL,
    seats               INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    key_revealed_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_team_signups_customer ON team_signups(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_team_signups_email    ON team_signups(email);
