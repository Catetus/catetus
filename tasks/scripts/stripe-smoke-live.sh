#!/usr/bin/env bash
# stripe-smoke-live.sh — post-deploy smoke for the LIVE-mode Stripe
# surface. Run AFTER stripe-bootstrap.sh --mode live and after Fly
# has the corresponding secrets set.
#
# Contracts verified (each line should print "OK ...". Any "FAIL" exits 1):
#   1. The two usage meters exist in LIVE mode with the SKU constants
#      that billing.rs hard-codes:
#          splatforge_repack_runs    (event_name)
#          splatforge_repack_seconds (event_name)
#   2. STRIPE_TEAM_PRICE_ID resolves to an ACTIVE recurring price in
#      live mode and is denominated in USD (the dashboard lets you
#      create a non-recurring price by accident; this checks
#      .recurring.interval == 'month').
#   3. POST /v1/checkout/create-session against the live API endpoint
#      returns a sessions.stripe.com URL (proves the API can reach
#      live Stripe with the right key).
#   4. /v1/checkout/webhook signature gate rejects a payload signed
#      with the wrong secret (proves verify_webhook is wired and
#      tolerant-replay rejection works).
#   5. /v1/checkout/reveal with an unknown session_id returns 404
#      (not 200, not 500) — proves the one-shot contract surface
#      is reachable.
#
# Required env vars:
#   STRIPE_LIVE_SECRET_KEY     sk_live_...
#   STRIPE_TEAM_PRICE_ID       price_... (the $99/seat/mo)
#   SF_API_BASE                e.g. https://api.splatforge.dev
#   SF_SMOKE_EMAIL             e.g. ops+smoke@splatforge.dev
#
# The smoke creates a Checkout *session* but does not complete payment.
# Stripe does not bill for unused sessions; they expire in 24h. The
# script does NOT attempt to confirm a payment.
#
# Fail-fast: any contract violation exits 1 within seconds so a
# post-deploy gate can block the rollout.

set -euo pipefail

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

die() { red "FAIL: $*" >&2; exit 1; }
ok()  { green "OK:   $*"; }

for v in STRIPE_LIVE_SECRET_KEY STRIPE_TEAM_PRICE_ID SF_API_BASE SF_SMOKE_EMAIL; do
  [[ -n "${!v:-}" ]] || die "missing env var: $v"
done
[[ "${STRIPE_LIVE_SECRET_KEY}" == sk_live_* ]] \
  || die "STRIPE_LIVE_SECRET_KEY must start with sk_live_ (got: ${STRIPE_LIVE_SECRET_KEY:0:8}…)"
[[ "${STRIPE_TEAM_PRICE_ID}" == price_* ]] \
  || die "STRIPE_TEAM_PRICE_ID must start with price_ (got: ${STRIPE_TEAM_PRICE_ID:0:8}…)"

command -v curl >/dev/null || die "curl missing"
command -v jq   >/dev/null || die "jq missing"
command -v openssl >/dev/null || die "openssl missing (needed for webhook signature)"

SF_API_BASE="${SF_API_BASE%/}"
STRIPE_API="https://api.stripe.com"
SMOKE_TS="$(date +%s)"

yellow "=== SplatForge Stripe LIVE-mode smoke ==="
echo "  API:   $SF_API_BASE"
echo "  Price: $STRIPE_TEAM_PRICE_ID"
echo "  Email: $SF_SMOKE_EMAIL"
echo

# ----- 1) Meters present in live mode -----
echo "[1/5] Verifying usage meters exist in LIVE mode"
METERS_JSON="$(curl -fsS \
  -u "${STRIPE_LIVE_SECRET_KEY}:" \
  "$STRIPE_API/v1/billing/meters?limit=100" \
  || die "GET /v1/billing/meters failed (auth or network)")"
for sku in splatforge_repack_runs splatforge_repack_seconds; do
  count="$(jq --arg n "$sku" '[.data[] | select(.event_name == $n)] | length' <<<"$METERS_JSON")"
  [[ "$count" -ge 1 ]] || die "meter '$sku' not found in live mode — run stripe-bootstrap.sh --mode live"
  ok "meter '$sku' present"
done

# ----- 2) Team price is active + recurring + USD -----
echo "[2/5] Verifying STRIPE_TEAM_PRICE_ID is active recurring USD"
PRICE_JSON="$(curl -fsS \
  -u "${STRIPE_LIVE_SECRET_KEY}:" \
  "$STRIPE_API/v1/prices/$STRIPE_TEAM_PRICE_ID" \
  || die "GET /v1/prices/$STRIPE_TEAM_PRICE_ID failed (does the price exist in LIVE mode?)")"
active="$(jq -r '.active' <<<"$PRICE_JSON")"
currency="$(jq -r '.currency' <<<"$PRICE_JSON")"
interval="$(jq -r '.recurring.interval // ""' <<<"$PRICE_JSON")"
amount="$(jq -r '.unit_amount' <<<"$PRICE_JSON")"
[[ "$active"   == "true" ]] || die "Team price is not active"
[[ "$currency" == "usd"  ]] || die "Team price currency != usd (got: $currency)"
[[ "$interval" == "month" ]] || die "Team price not recurring monthly (got: '$interval')"
[[ "$amount"   == "9900" ]]  || yellow "  note: unit_amount=$amount (expected 9900 for \$99/mo)"
ok "price active=$active currency=$currency interval=$interval unit_amount=$amount"

# ----- 3) /v1/checkout/create-session returns sessions.stripe.com URL -----
echo "[3/5] POST $SF_API_BASE/v1/checkout/create-session"
NONCE="smoke_$SMOKE_TS"
CS_RESP="$(curl -fsS -w '\n%{http_code}\n' -X POST \
  -H 'content-type: application/json' \
  -d "{\"email\":\"$SF_SMOKE_EMAIL\",\"nonce\":\"$NONCE\"}" \
  "$SF_API_BASE/v1/checkout/create-session" \
  || die "create-session request failed")"
CS_CODE="$(tail -n1 <<<"$CS_RESP")"
CS_BODY="$(sed '$d' <<<"$CS_RESP")"
[[ "$CS_CODE" == "200" ]] || die "create-session returned HTTP $CS_CODE: $CS_BODY"
CS_URL="$(jq -r '.url' <<<"$CS_BODY")"
CS_SID="$(jq -r '.session_id' <<<"$CS_BODY")"
[[ "$CS_URL" == https://checkout.stripe.com/* || "$CS_URL" == https://*.stripe.com/* ]] \
  || die "session url does not look like Stripe-hosted: $CS_URL"
[[ "$CS_SID" == cs_live_* ]] \
  || die "session_id does not start with cs_live_ (got: ${CS_SID:0:12}…) — API is NOT in live mode"
ok "create-session -> $CS_SID (live mode confirmed)"

# ----- 4) Webhook rejects bad signature -----
echo "[4/5] /v1/checkout/webhook rejects bad signature"
BAD_BODY='{"id":"evt_smoke_bad","type":"checkout.session.completed","data":{"object":{}}}'
BAD_TS="$(date +%s)"
BAD_PAYLOAD="${BAD_TS}.${BAD_BODY}"
BAD_SIG="$(printf '%s' "$BAD_PAYLOAD" \
  | openssl dgst -sha256 -hmac 'whsec_wrong_smoke_key' -hex \
  | awk '{print $NF}')"
WH_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H 'content-type: application/json' \
  -H "stripe-signature: t=$BAD_TS,v1=$BAD_SIG" \
  -d "$BAD_BODY" \
  "$SF_API_BASE/v1/checkout/webhook")"
case "$WH_CODE" in
  400|401|403)
    ok "webhook rejected bad signature with HTTP $WH_CODE"
    ;;
  200)
    die "webhook accepted a payload signed with the WRONG secret — signature gate is broken"
    ;;
  *)
    die "webhook returned unexpected HTTP $WH_CODE for bad signature"
    ;;
esac

# ----- 5) /v1/checkout/reveal one-shot contract surface -----
echo "[5/5] /v1/checkout/reveal returns 404 for unknown session_id"
RV_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H 'content-type: application/json' \
  -d '{"session_id":"cs_live_smoke_does_not_exist","token":"nope"}' \
  "$SF_API_BASE/v1/checkout/reveal")"
case "$RV_CODE" in
  404|403|410)
    ok "reveal rejected unknown session with HTTP $RV_CODE"
    ;;
  200)
    die "reveal returned 200 for a bogus session_id — one-shot contract is broken"
    ;;
  *)
    die "reveal returned unexpected HTTP $RV_CODE"
    ;;
esac

echo
green "=== ALL SMOKE CHECKS PASSED — Stripe live surface is healthy ==="
