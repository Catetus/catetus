# SplatForge Cloud — Checkout + live-meter operator runbook

Sister document to [`BILLING.md`](./BILLING.md). Where `BILLING.md`
covers the metered side of the equation (what the API emits per
repack run), this file is the operator's runbook for the steps the
Stripe MCP can't fire on its own — most importantly, **creating the
two live billing meters** that the `splatforge_repack_runs` and
`splatforge_repack_seconds` events count against.

The MCP can mint products and prices, but the
[`/v1/billing/meters`](https://docs.stripe.com/api/billing/meter/create)
endpoint isn't surfaced by the Claude MCP today. You have to `curl` it
yourself with a live secret key. The exact commands are below — fill
in the `sk_live_…` placeholder after rotation and paste.

> **⚠️  Read this end-to-end before running anything.** The two meters
> created below are what every Team-tier customer's invoice rolls up
> against. A typo in `event_name` and the events vanish into thin air
> (Stripe silently drops events for unknown meters).

---

## Prereqs

1. The live Stripe secret key, **post-rotation**. The previous live
   key may have been exposed during the PR #1 / MCP dance; rotate in
   the Stripe Dashboard → Developers → API keys before running these
   commands. Substitute `sk_live_REPLACE_AFTER_ROTATION` below.

2. `STRIPE_LIVE_MODE=true` set on Fly. Without this, the API refuses
   to use the `sk_live_` key and falls back to dry-run (see
   `apps/api/src/billing.rs::BillingClient::from_env`).

3. `STRIPE_TEAM_PRICE_ID` set on Fly to the `$99/seat/mo` recurring
   price already provisioned in the SplatForge Stripe account
   (`acct_1TXM7fIZCvZOU40b`, product `prod_UWP01FtK3Czpyl`, price
   `price_1TXMD8IZCvZOU40bzQUA2OWX`).

---

## Step 1 — create `splatforge_repack_runs` meter

The flat per-job meter. One unit per call to `/v1/jobs/:id/repack`.

```bash
curl https://api.stripe.com/v1/billing/meters \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -d "display_name=SplatForge repack runs" \
  -d "event_name=splatforge_repack_runs" \
  -d "default_aggregation[formula]=sum" \
  -d "customer_mapping[event_payload_key]=stripe_customer_id" \
  -d "customer_mapping[type]=by_id" \
  -d "value_settings[event_payload_key]=value"
```

Expected `200`:
```json
{
  "id": "mtr_…",
  "event_name": "splatforge_repack_runs",
  "status": "active",
  …
}
```

Capture the `mtr_…` id; you'll attach a price to it in step 3.

---

## Step 2 — create `splatforge_repack_seconds` meter

The per-second compute meter. Reported on the Modal callback when
the worker knows the elapsed wall-clock time.

```bash
curl https://api.stripe.com/v1/billing/meters \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -d "display_name=SplatForge repack compute seconds" \
  -d "event_name=splatforge_repack_seconds" \
  -d "default_aggregation[formula]=sum" \
  -d "customer_mapping[event_payload_key]=stripe_customer_id" \
  -d "customer_mapping[type]=by_id" \
  -d "value_settings[event_payload_key]=value"
```

`event_name` MUST match the constant `SKU_REPACK_SECONDS` in
`apps/api/src/billing.rs` byte-for-byte — Stripe silently drops
events whose `event_name` doesn't resolve to an active meter.

---

## Step 3 — attach prices

The v0.1 published rate card (also encoded in
`apps/api/src/pricing.rs::PER_JOB_FLAT_CENTS` and
`PER_COMPUTE_SECOND_CENTS`):

```
$0.01 per `splatforge_repack_runs` event
$0.001 per `splatforge_repack_seconds` unit (= $3.60/hr compute)
```

These two prices already exist if PR #1 / the stripe-bootstrap script
ran. If not, create them — replace `mtr_RUNS_FROM_STEP_1` and
`mtr_SECONDS_FROM_STEP_2`:

```bash
# $0.01/run
curl https://api.stripe.com/v1/prices \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -d "currency=usd" \
  -d "product=prod_UWP01FtK3Czpyl" \
  -d "recurring[usage_type]=metered" \
  -d "recurring[interval]=month" \
  -d "recurring[meter]=mtr_RUNS_FROM_STEP_1" \
  -d "billing_scheme=per_unit" \
  -d "unit_amount=1"

# $0.001/compute-second
curl https://api.stripe.com/v1/prices \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -d "currency=usd" \
  -d "product=prod_UWP01FtK3Czpyl" \
  -d "recurring[usage_type]=metered" \
  -d "recurring[interval]=month" \
  -d "recurring[meter]=mtr_SECONDS_FROM_STEP_2" \
  -d "billing_scheme=per_unit_decimal" \
  -d "unit_amount_decimal=0.1"
```

Capture both `price_…` ids and add them as line items on the Team
Seat subscription (or update the Team checkout product config to
include them as add-on metered components).

---

## Step 4 — sanity-check with a test event

After the meters land, send a manual event from the operator
workstation to verify wiring before any real customer touches the
metered path:

```bash
curl https://api.stripe.com/v1/billing/meter_events \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -d "event_name=splatforge_repack_runs" \
  -d "payload[stripe_customer_id]=cus_REAL_OPERATOR_TEST_CUSTOMER" \
  -d "payload[value]=1" \
  -d "identifier=sf_manual_smoke_$(date +%s)"
```

Then in the dashboard: Billing → Meters → `splatforge_repack_runs` →
should show a single ingested event within ~30s. If it doesn't, the
event name probably has a typo — re-check against
`apps/api/src/billing.rs::SKU_REPACK_RUNS`.

---

## Step 5 — flip Fly env

```bash
fly secrets set \
  STRIPE_SECRET_KEY=sk_live_REPLACE_AFTER_ROTATION \
  STRIPE_LIVE_MODE=true \
  STRIPE_TEAM_PRICE_ID=price_1TXMD8IZCvZOU40bzQUA2OWX \
  STRIPE_WEBHOOK_SECRET=whsec_… \
  STRIPE_CHECKOUT_WEBHOOK_SECRET=whsec_…
```

The two webhook secrets can be the same value if you've configured a
single Stripe webhook endpoint that subscribes to both event
categories (`customer.subscription.*` and `checkout.session.*`). Two
endpoints is fine too — `STRIPE_CHECKOUT_WEBHOOK_SECRET` lets the
checkout route use a distinct secret per Stripe's per-endpoint
signing-secret model.

After `fly deploy` (or `fly machine restart`), confirm the API
banner logs `billing client initialized mode=live`.

---

## Step 6 — first real customer

1. Buyer hits `https://splatforge.dev/pricing`, enters email, clicks
   Continue.
2. Browser POSTs `/v1/checkout/create-session`; the API hands back a
   `cs_live_…` URL.
3. Buyer pays. Stripe POSTs `checkout.session.completed` to
   `/v1/checkout/webhook`. The API mints `sf_live_<24>`, hashes it,
   stashes the plaintext in memory.
4. Stripe redirects to `/welcome?session_id=cs_live_…&token=…`.
   `/welcome` calls `/v1/checkout/reveal` which displays the plaintext
   **exactly once** and flips `key_revealed_at`.
5. Operator: keep an eye on the `team_signups` table for the first 5
   customers — if `key_revealed_at` is still NULL 30 min after a
   `checkout.session.completed`, the buyer never landed on
   `/welcome`. Email them with a rotated key from the WorkOS-branch
   admin API (or operator's existing manual flow).

---

## Step 7 — first metered repack

Once the first customer is provisioned and is hitting the paid
`/repack` endpoint:

1. Watch the API logs for `stripe meter event posted job_id=… sku=…`.
2. Dashboard → Billing → Meters → both meters should show the same
   timestamp.
3. End of billing period: the invoice should show one line item per
   meter, each at the v0.1 rate. If it shows `$0.00`, the meter→price
   wiring in step 3 didn't land — re-check the price's `recurring[meter]`.

---

## Rollback

If the live meters need to be torn down (e.g. wrong `event_name`
typo), Stripe doesn't allow deletion — only deactivation:

```bash
curl https://api.stripe.com/v1/billing/meters/mtr_BAD_METER_ID/deactivate \
  -u "sk_live_REPLACE_AFTER_ROTATION:" \
  -X POST
```

Then create a fresh one with the corrected name and update
`apps/api/src/billing.rs::SKU_REPACK_*` to match if needed (and ship
a release).

---

## Related surfaces

* `apps/api/src/billing.rs` — meter-event poster (the side that
  CALLS the meters created above).
* `apps/api/src/checkout.rs` — `checkout.session.completed` webhook
  that mints the `sf_live_` key after a successful $99/mo signup.
* `apps/api/src/pricing.rs` — the per-job rate card (`v0.1` constants)
  and the SDK licensing surface. The numbers here MUST agree with the
  Stripe prices created in step 3 — both are versioned via the
  `pricing_version` field in `/v1/pricing/preview` responses.
* `apps/web/src/pages/pricing.astro` — the customer-facing calculator
  that calls `/v1/pricing/preview` for cents-accurate quotes.
* `apps/web/src/pages/sdk.astro` — SDK MAU pricing + license-flow doc.
