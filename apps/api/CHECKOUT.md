# SplatForge Cloud — Self-serve Team-tier signup

Operator runbook for the Stripe Checkout flow that lands at
`/pricing` and emits a fresh `sf_live_<24chars>` API key.

The surface lives in `apps/api/src/checkout.rs`. It is intentionally
distinct from `billing.rs` (which posts usage meter events for the
A100 repack worker): checkout is a one-shot signup transaction;
billing is an ongoing per-job ledger.

---

## End-to-end flow

```
                 ┌────────────────────┐
  Buyer ─POST─► /v1/checkout/         (1)  axum/main.rs
                │ create-session     │       → checkout::create_session_and_register
                └──────────┬─────────┘       → POST https://api.stripe.com/v1/checkout/sessions
                           │                 → caches (session_id → claim_token) in memory
                           ▼
                ┌────────────────────┐
                │  Stripe Checkout   │   (2) Hosted by Stripe.
                │  (hosted payment)  │       success_url has {CHECKOUT_SESSION_ID}+token.
                └──────────┬─────────┘
                           │ paid
                           ▼
                ┌────────────────────┐
                │ checkout.session.  │   (3) Stripe → POST /v1/checkout/webhook
                │  completed (whk)   │       → checkout::provision_from_session
                └──────────┬─────────┘       → mint sf_live_… + insert team_signups
                           │                   (ON CONFLICT DO NOTHING idempotent)
                           ▼                 → cache plaintext in PendingKeyCache (10min TTL)
                ┌────────────────────┐
  Buyer  ◄──redirect──     │      ─►/welcome?session_id=cs_…&token=<claim>
                           │                   (4) JS POSTs /v1/checkout/reveal
                ┌────────────────────┐
                │ /v1/checkout/reveal│   (5) Three gates:
                │                    │        a) constant-time token compare
                └──────────┬─────────┘        b) atomic UPDATE … WHERE key_revealed_at IS NULL
                           │                  c) take_if_fresh from PendingKeyCache
                           ▼
                Plaintext shown ONCE.
                Server keeps SHA-256 hash + 12-char display prefix only.
```

---

## Env vars

| Var                              | Purpose                                                                          |
|----------------------------------|----------------------------------------------------------------------------------|
| `STRIPE_SECRET_KEY`              | `sk_test_…` or `sk_live_…`. Unset → `/create-session` returns 503.               |
| `STRIPE_TEAM_PRICE_ID`           | The `price_…` id of the $99/seat/mo recurring price (created in dashboard).      |
| `STRIPE_LIVE_MODE`               | Must be `"true"` *and* the secret must be `sk_live_…` to bill real cards.        |
| `STRIPE_WEBHOOK_SECRET`          | HMAC for both `/v1/stripe/webhook` (meter events) and `/v1/checkout/webhook`.    |
| `STRIPE_CHECKOUT_WEBHOOK_SECRET` | Optional override if you configured a *separate* Stripe webhook endpoint.        |
| `SPLATFORGE_PUBLIC_SITE_URL`     | Base URL the success_url is built against (default `https://splatforge.dev`).    |

The two-gate safety pattern from `billing.rs` carries over: a
`sk_live_` secret without `STRIPE_LIVE_MODE=true` falls back to a
state where `/create-session` returns 503, so a misconfigured deploy
can't accidentally charge real cards.

---

## Bootstrap

1. **Create the Team tier price in the Stripe dashboard.** Recurring
   monthly product, $99/seat, USD. Capture the resulting `price_…`
   id. We deliberately do NOT mint this from code — pricing is a
   commercial decision that lives in the dashboard.

2. **Add the webhook endpoint** in the Stripe dashboard:
   `https://api.splatforge.dev/v1/checkout/webhook`, subscribed to
   `checkout.session.completed`. Copy the `whsec_…` value.

3. **Set the env vars on Fly**:
   ```bash
   fly secrets set \
     STRIPE_SECRET_KEY="sk_test_…" \
     STRIPE_TEAM_PRICE_ID="price_…" \
     STRIPE_WEBHOOK_SECRET="whsec_…" \
     STRIPE_LIVE_MODE=false \
     SPLATFORGE_PUBLIC_SITE_URL="https://splatforge.dev"
   ```

4. **Local dev**: `stripe listen --forward-to http://localhost:8080/v1/checkout/webhook`
   and pipe the printed `whsec_…` into `STRIPE_WEBHOOK_SECRET`.

5. **Flip live** when you're ready to bill real cards:
   ```bash
   fly secrets set \
     STRIPE_SECRET_KEY="sk_live_…" \
     STRIPE_WEBHOOK_SECRET="whsec_…" \
     STRIPE_LIVE_MODE=true
   ```

---

## Plaintext-exactly-once invariant

The plaintext API key crosses the wire once: on the response of
`POST /v1/checkout/reveal`. Three gates guarantee no second reveal:

1. **DB.** `mark_team_signup_revealed` is
   `UPDATE … SET key_revealed_at = now() WHERE key_revealed_at IS NULL`.
   First call returns 1 row affected; subsequent calls return 0
   and the handler responds 410 Gone.

2. **Process memory.** `PendingKeyCache::take_if_fresh` removes the
   entry on read. Even if the DB column flag is missing, the cache
   is empty after the first reveal and the second call returns 410.

3. **TTL.** Both `PendingKeyCache` and `PendingClaimTokens` are
   swept every 60s and drop entries older than 10 minutes.

The plaintext NEVER:
* hits the SQLite database (we store SHA-256 hex + 12-char prefix),
* hits the log pipeline (`info!` calls log only `key_prefix`),
* gets returned by any endpoint other than `/v1/checkout/reveal`
  on its first call.

Tested in `apps/api/tests/checkout.rs`:
* `reveal_is_strictly_one_shot` — the "key shown twice" anti-test.
  Second reveal returns Gone. Post simulated process restart still Gone.
* `key_plaintext_never_persisted` — scans every column of every row in
  `team_signups` and asserts no value starts with `sf_live_`.
* `webhook_idempotent_under_double_delivery` — three identical webhook
  deliveries produce one row, one key, one customer record.

---

## Webhook idempotency

Stripe retries `checkout.session.completed` on any non-2xx response.
Protection is two layers:

1. **Signature freshness.** `billing::verify_webhook` rejects events
   where `|t - now| > 300s`. A replay 6 minutes later fails the
   signature gate before we touch SQL.

2. **DB constraint.** `team_signups.stripe_session_id UNIQUE`.
   `claim_team_signup` does `INSERT … ON CONFLICT(stripe_session_id)
   DO NOTHING`. Second delivery returns `Ok(false)` and we 200 back
   without minting a key.

SQLite serializes the conflict on the UNIQUE index, so two
simultaneous webhook deliveries can't both win.

---

## Customer-loss risk

The single biggest churn point in this funnel is: **buyer pays,
sees the Stripe success page, closes the tab before `/welcome`
loads.** Their plaintext is in memory only, gets swept after 10
minutes, and there is no recovery path.

Mitigations in this PR:
* Stripe redirects directly to `/welcome` with the session id and
  claim_token already in the URL — no extra click.
* `/welcome` displays the plaintext as soon as `/reveal` returns,
  with a prominent "save this now, we never show it again"
  warning. 60-second auto-redirect to the docs.
* The copy button is the highest-contrast element on the page.
* On `/reveal` returning 410, the error region copy is
  intentionally support-friendly: "email support, we'll rotate
  you a fresh key" — failure mode resolved with one
  `POST /v1/admin/api-keys` call once WorkOS branch merges.

Recommended post-launch mitigations:
* Email the buyer a one-time magic-link to `/welcome` when the
  webhook fires, so closing the Stripe tab is recoverable.
* Server-side "your dashboard" link that lets them mint a *new*
  key (revoking the unrevealed one). The DB row is already shaped
  for this — `key_revealed_at IS NULL` is the rotation predicate.

---

## Mapping the new key to billing

The Team-tier signup row carries `stripe_customer_id`. Until the
WorkOS SSO branch merges (it adds a DB-backed equivalent of
`SPLATFORGE_KEY_CUSTOMERS`), the operator manually appends
`<plaintext_prefix>:<stripe_customer_id>` to that env var from the
row's persisted fields:

```bash
sqlite3 data/jobs.db <<EOF
SELECT key_prefix, stripe_customer_id, email
FROM team_signups
WHERE email = 'alice@acme.com';
EOF
```

Then `fly secrets set SPLATFORGE_KEY_CUSTOMERS="<existing>,<prefix>:<cus_id>"`.
Temporary bridge — the WorkOS branch's `api_keys` table is the
canonical mapping post-merge.

---

## Launch day runbook

This section is the operator's flight checklist for promoting the
Stripe surface from "merged but inert" to "live and billing real
cards". Do these in order. Each step links back to the script or
endpoint that owns the contract.

### Prereqs

* `stripe` CLI authenticated with both test + live keys
  (`stripe login` once per mode).
* `jq`, `openssl`, `curl` on the operator's PATH.
* The merge train carrying `feat/api-production-hardening` and
  the Stripe Checkout agent's branch is already on `main`.
* Fly app is deployed and reachable at `$SF_API_BASE`
  (default `https://api.splatforge.dev`).

### 1. Test mode bootstrap

```bash
# Provisions meters + product placeholders in test mode.
tasks/scripts/stripe-bootstrap.sh --mode test

# Provisions the $99/seat/mo recurring price in test mode.
# Capture the printed STRIPE_TEAM_PRICE_ID.
tasks/scripts/stripe-bootstrap-team-tier.sh --mode test
```

Both scripts are idempotent — safe to re-run.

### 2. Test mode smoke (against a staging Fly app)

Set the resulting test-mode secrets on the staging Fly app:

```bash
fly secrets set --app splatforge-api-staging \
  STRIPE_SECRET_KEY="sk_test_…" \
  STRIPE_WEBHOOK_SECRET="whsec_…" \
  STRIPE_TEAM_PRICE_ID="price_…" \
  STRIPE_LIVE_MODE=false
```

Then run the smoke against staging:

```bash
STRIPE_LIVE_SECRET_KEY="sk_test_…" \
STRIPE_TEAM_PRICE_ID="price_…" \
SF_API_BASE="https://api-staging.splatforge.dev" \
SF_SMOKE_EMAIL="ops+smoke@splatforge.dev" \
tasks/scripts/stripe-smoke-live.sh
```

Note: `stripe-smoke-live.sh` enforces `sk_live_` and `cs_live_`
prefixes by design — for test-mode dry runs, comment out the two
prefix checks at the top of the script (or skim its 5 contract
assertions and verify them manually). The test-mode dry-run is
optional; its purpose is to give the operator muscle memory before
the live run.

### 3. Live mode bootstrap

```bash
# Both scripts gate behind an interactive 'PROVISION LIVE' prompt.
STRIPE_LIVE_SECRET_KEY="sk_live_…" \
  tasks/scripts/stripe-bootstrap.sh --mode live

STRIPE_LIVE_SECRET_KEY="sk_live_…" \
  tasks/scripts/stripe-bootstrap-team-tier.sh --mode live
```

Capture the resulting live-mode `STRIPE_TEAM_PRICE_ID` — it is
NOT the same id as the test-mode one.

### 4. Wire the Stripe webhook endpoint

In the Stripe dashboard (LIVE mode):

* Endpoint: `https://api.splatforge.dev/v1/checkout/webhook`
* Events: `checkout.session.completed`
* Copy the `whsec_…` signing secret — this is `STRIPE_WEBHOOK_SECRET`.

If you also have a separate endpoint for the usage-meter webhook
(`/v1/stripe/webhook`), repeat with its own `whsec_…` and stash
it as `STRIPE_CHECKOUT_WEBHOOK_SECRET` (override; defaults to
`STRIPE_WEBHOOK_SECRET`).

### 5. Set Fly secrets for the live deploy

```bash
fly secrets set --app splatforge-api \
  STRIPE_SECRET_KEY="sk_live_…" \
  STRIPE_WEBHOOK_SECRET="whsec_…" \
  STRIPE_TEAM_PRICE_ID="price_…" \
  STRIPE_LIVE_MODE=true \
  SPLATFORGE_PUBLIC_SITE_URL="https://splatforge.dev"
```

Fly will trigger a rolling deploy. Wait for it to finish.

### 6. Live mode smoke (gates the rollout)

```bash
STRIPE_LIVE_SECRET_KEY="sk_live_…" \
STRIPE_TEAM_PRICE_ID="price_…" \
SF_API_BASE="https://api.splatforge.dev" \
SF_SMOKE_EMAIL="ops+smoke@splatforge.dev" \
tasks/scripts/stripe-smoke-live.sh
```

All 5 contract checks must pass. If any fail, roll back via
`fly secrets set STRIPE_LIVE_MODE=false` (the create-session
endpoint will return 503 again and no card will be charged).

The smoke creates a Checkout *session*; Stripe does not bill for
unused sessions and they expire in 24h. No card is charged by the
smoke itself.

### 7. Confirm the public surface

* `/pricing` shows the Team CTA.
* `POST /v1/checkout/create-session` returns a `cs_live_…` session id
  (the smoke already verified this).
* A real-card test buy from a personal account hits `/welcome` and
  the reveal endpoint returns the plaintext once. Save the key, then
  refund the test charge in the Stripe dashboard if you don't want
  the row in production.

### What "live-mode is wired" means for checkout.rs

`apps/api/src/checkout.rs` reads three env vars at boot:

| Var                     | Live-mode value     | What goes wrong if missing                              |
|-------------------------|---------------------|---------------------------------------------------------|
| `STRIPE_SECRET_KEY`     | `sk_live_…`         | `is_live()` returns false → create-session returns 503  |
| `STRIPE_LIVE_MODE`      | `true`              | `is_live()` returns false even with sk_live_ → 503      |
| `STRIPE_TEAM_PRICE_ID`  | `price_…` (live)    | `create_session` returns `PriceNotConfigured`           |

Stripe API base URL is identical for test + live mode
(`https://api.stripe.com`); the secret key alone determines which
account/mode is touched. Webhook signature verification
(`billing::verify_webhook`) is also mode-agnostic — it HMACs the
raw body against the configured `STRIPE_WEBHOOK_SECRET`. No
test-mode-only code paths exist in `checkout.rs`; the audit
covering this is in `feat/stripe-live-mode-prep`.
