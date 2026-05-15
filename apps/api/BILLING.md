# SplatForge Cloud — Stripe-metered billing

Operational guide for the usage-based billing scaffold attached to
`POST /v1/jobs/:id/repack`.

The billing surface lives in `apps/api/src/billing.rs`. It posts to
Stripe's **Billing Meter Events API** (the 2024+ usage-based pattern,
NOT the legacy subscription-item usage records API). Two SKUs are
emitted per successful repack run:

| SKU                          | Unit          | Reported by                       |
|------------------------------|---------------|-----------------------------------|
| `splatforge_repack_runs`     | 1 per call    | Synchronous `/repack` dispatch    |
| `splatforge_repack_seconds`  | seconds       | Modal callback (terminal `done`)  |

A repack call therefore produces one `runs` event at dispatch time and
one `seconds` event when the worker reports back with elapsed time.
Both are idempotent on `(job_id, sku)` — see "Double-charge safety"
below.

---

## Bootstrap

```bash
# 1. Install + login to Stripe CLI (one time).
brew install stripe/stripe-cli/stripe
stripe login

# 2. Create the meters + prices + products in test mode.
./tasks/scripts/stripe-bootstrap.sh

# The script is idempotent: re-running it skips anything already created
# and prints the env-var lines you need on Fly:
#
#   STRIPE_SECRET_KEY=sk_test_...
#   STRIPE_WEBHOOK_SECRET=whsec_...
#   SPLATFORGE_KEY_CUSTOMERS=key1:cus_xxx,key2:cus_yyy
```

The script creates two billing meters whose `event_name` exactly
matches the SKU constants in `billing.rs`:

* `splatforge_repack_runs`     — defaults to `sum` aggregation on `value`
* `splatforge_repack_seconds`  — same, summing seconds across the period

A meter without a price attached collects events but doesn't charge —
that's the right default for the closed-beta stage. Attach prices in
the Stripe dashboard once usage shape is settled (recommended starting
point: $0.05 per `_runs` event + $0.0008 per `_seconds` unit, putting
the bonsai reference repack at ~$0.064).

---

## Mapping a bearer key to a Stripe customer

Customers are linked by API key via a static env var:

```
SPLATFORGE_KEY_CUSTOMERS=key_alpha:cus_NciAYcXfLnqBoz,key_beta:cus_OZx...
```

Format: comma-separated `key:customer_id` pairs. Whitespace around
either side is trimmed. Unknown keys (any key not in this map) fall
through to the **no-customer code path** — the paid pipeline still
runs, but no Stripe event is emitted. This is intentional for the
closed-beta stage where the operator may invoice manually.

To enable billing for a new customer:

1. Create the customer in Stripe (`stripe customers create --email …`).
   Capture the `cus_xxx` id.
2. Append `<existing-paid-key>:cus_xxx` to `SPLATFORGE_KEY_CUSTOMERS`
   on Fly: `fly secrets set SPLATFORGE_KEY_CUSTOMERS="…"`.
3. Restart the API: `fly deploy` (or `fly machine restart`). The map
   is parsed once at startup; live reload is not wired.

The customer id is stamped onto the `Job.customer_id` column at
creation time. The repack handler additionally re-resolves the key
at repack time, so a key that was added to the map *after* the
original `/v1/jobs` call still bills correctly.

---

## Local-dev mode (no Stripe credentials)

When `STRIPE_SECRET_KEY` is unset, `BillingClient::from_env` returns
**dry-run mode**:

```
INFO splatforge_api::billing: billing dry-run: would post meter event
  job_id=11111111-… sku=splatforge_repack_runs units=1
```

The ledger row is still claimed (so retries are still deduped), but
no network call to Stripe is made. This is the default `cargo run`
behavior and the right configuration for CI.

A `sk_live_` key with `STRIPE_LIVE_MODE` unset (or not `"true"`) is
demoted to dry-run mode with a `WARN` log — this is a safety belt so
a misconfigured env var can't ship real charges.

---

## Production mode

```bash
fly secrets set \
  STRIPE_SECRET_KEY="sk_test_..." \
  STRIPE_WEBHOOK_SECRET="whsec_..." \
  SPLATFORGE_KEY_CUSTOMERS="key1:cus_xxx" \
  STRIPE_LIVE_MODE=false
```

Flip `STRIPE_LIVE_MODE=true` *and* swap to `sk_live_...` /
`whsec_...` (live) only when you're ready to bill real cards. The two
gates are deliberately separate: changing one without the other
falls back to dry-run, not real charges.

### Webhook endpoint

Stripe webhooks land at `POST /v1/stripe/webhook`. The handler:

| Event                                | Action                                                                 |
|--------------------------------------|------------------------------------------------------------------------|
| `customer.subscription.created`      | Logs status; manual tier reconciliation                                |
| `customer.subscription.updated`      | Logs status (this is where tier *upgrades* land)                       |
| `customer.subscription.deleted`      | Logs; downgrade target                                                 |
| `invoice.payment_failed`             | **WARN** log; downgrade target                                         |
| `invoice.payment_succeeded`          | Logs for observability                                                 |
| (everything else)                    | 200 OK so Stripe stops retrying                                        |

Automatic tier flipping is **not** wired up: the key→customer map is
a static env var, and a webhook silently revoking a key would be a
support nightmare. The handler emits structured logs; the operator
reconciles. A future control-plane DB swaps the static map for a row
in `keys` and makes the flips automatic.

Configure the webhook secret in the Stripe dashboard under
**Developers → Webhooks → Add endpoint** for
`https://splatforge-api.fly.dev/v1/stripe/webhook`, select the events
above, and copy the resulting `whsec_...` to
`STRIPE_WEBHOOK_SECRET`. For local-dev use `stripe listen --forward-to
http://localhost:8080/v1/stripe/webhook` — it prints a different
`whsec_...` for the tunnel.

---

## Double-charge safety

This is the load-bearing invariant.

A repack run's lifecycle:

1. Client POSTs `/v1/jobs/:id/repack`.
2. API enqueues to Modal.
3. **API records `splatforge_repack_runs` (1 unit) → Stripe.**
4. Modal A100 runs, posts back to `/v1/jobs/:id/result` (with
   `compute_seconds` on `done`).
5. **API records `splatforge_repack_seconds` (N units) → Stripe.**

Steps 3 and 5 each pass through two dedupe gates:

* **Local ledger** (`billing_events` table, `UNIQUE(job_id, sku)`).
  The first `INSERT … ON CONFLICT DO NOTHING` wins; losers skip the
  Stripe call entirely. SQLite serializes the conflict so concurrent
  callers can't both win.
* **Stripe `identifier`** (`Idempotency-Key` header + meter event
  `identifier` field). Both are
  `sf_<sku>_<sha256(job_id || ":" || sku || ":billing")>`, deterministic
  per `(job_id, sku)`. Even if our ledger is wiped, Stripe still
  rejects the duplicate.

The Modal callback can fire twice (flaky webhooks, retries, machine
restarts). That's the single biggest billing-correctness risk in the
system — and it's exactly what this two-layer gate exists to neutralize.

Test coverage:

* `tests/billing.rs::billing_is_idempotent_across_retries` — 3 calls,
  asserts exactly 2 Stripe POSTs.
* `tests/billing.rs::billing_free_tier_emits_no_events` — `customer_id =
  None` produces 0 calls.
* `src/billing.rs::tests::no_double_charge_invariant` — direct ledger
  check.

---

## Free is free

Free-tier jobs **never** emit billing events. The contract is enforced
in three places:

1. `record_repack_job` short-circuits when `customer_id.is_none()`.
2. The callback path only bills when `tier == Tier::Paid`.
3. Only the paid `/repack` handler calls `record_repack_job` in the
   synchronous path. The free `/upload` + `/v1/jobs` paths never touch
   the billing module.

If a paid customer accidentally hits the free pipeline (e.g. they
forget to call `/repack`), no charge is emitted. They got CPU time
for free; that's the deal.
