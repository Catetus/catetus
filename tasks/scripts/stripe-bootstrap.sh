#!/usr/bin/env bash
# stripe-bootstrap.sh — provision the Stripe products + prices + meters
# that SplatForge Cloud's paid tier expects.
#
# Idempotent. Safe to run repeatedly; existing resources are detected by
# matching on the meter `event_name` and product `lookup_key`, so a
# second run does NOT create duplicates.
#
# Usage:
#   tasks/scripts/stripe-bootstrap.sh [--mode test|live]
#
#   --mode test  (default) — provisions against test mode. If
#                STRIPE_TEST_SECRET_KEY (or STRIPE_SECRET_KEY starting
#                with sk_test_) is set, that key is exported as
#                STRIPE_API_KEY for the CLI; otherwise the CLI's
#                logged-in session is used.
#   --mode live  — Targets LIVE mode. Requires STRIPE_LIVE_SECRET_KEY
#                (or STRIPE_SECRET_KEY starting with sk_live_).
#                Adds an interactive "PROVISION LIVE" confirmation
#                gate and passes --live to every stripe CLI call.
#
# Requirements:
#   * stripe CLI installed and authenticated (`stripe login`).
#   * jq available (Stripe CLI emits JSON; we grep with jq).
#
# Output: prints the env-var block ready to paste into
# `fly secrets set` (or your .env). Nothing is written to disk.

set -euo pipefail

MODE="test"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      shift
      MODE="${1:-}"
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)
      echo "error: unknown arg '$1' (use --mode test|live)" >&2
      exit 2
      ;;
  esac
  shift || true
done

case "$MODE" in
  test|live) ;;
  *)
    echo "error: --mode must be 'test' or 'live' (got '$MODE')" >&2
    exit 2
    ;;
esac

if ! command -v stripe >/dev/null 2>&1; then
  echo "error: stripe CLI not on PATH. Install via: brew install stripe/stripe-cli/stripe" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq not on PATH. Install via: brew install jq" >&2
  exit 2
fi

# Pick the right secret + CLI flag for the requested mode.
STRIPE_CLI_FLAG=""
if [[ "$MODE" == "live" ]]; then
  STRIPE_CLI_FLAG="--live"
  EXPECTED_SECRET="${STRIPE_LIVE_SECRET_KEY:-${STRIPE_SECRET_KEY:-}}"
  if [[ -z "$EXPECTED_SECRET" ]]; then
    echo "error: --mode live requires STRIPE_LIVE_SECRET_KEY (or STRIPE_SECRET_KEY)" >&2
    exit 2
  fi
  if [[ "$EXPECTED_SECRET" != sk_live_* ]]; then
    echo "error: --mode live but the configured secret does NOT start with sk_live_" >&2
    echo "       (refusing to run; pass an actual sk_live_ key)" >&2
    exit 2
  fi
  echo "" >&2
  echo "============================================================" >&2
  echo "  ABOUT TO PROVISION RESOURCES IN STRIPE LIVE MODE" >&2
  echo "============================================================" >&2
  echo "  This creates meters + products in your live-mode Stripe" >&2
  echo "  account. Idempotent (skipped if lookup_key / event_name" >&2
  echo "  already exists). No charges are created by THIS script." >&2
  echo "" >&2
  read -rp "Type 'PROVISION LIVE' to continue: " confirm
  if [[ "$confirm" != "PROVISION LIVE" ]]; then
    echo "aborted." >&2
    exit 1
  fi
  export STRIPE_API_KEY="$EXPECTED_SECRET"
else
  EXPECTED_SECRET="${STRIPE_TEST_SECRET_KEY:-${STRIPE_SECRET_KEY:-}}"
  if [[ -n "$EXPECTED_SECRET" ]]; then
    if [[ "$EXPECTED_SECRET" == sk_live_* ]]; then
      echo "error: --mode test but configured secret starts with sk_live_" >&2
      echo "       (refusing to run; pass a sk_test_ key or use --mode live)" >&2
      exit 2
    fi
    export STRIPE_API_KEY="$EXPECTED_SECRET"
  fi
  # Belt-and-braces: refuse to silently run against a live-mode CLI session.
  if [[ -z "${STRIPE_API_KEY:-}" ]] \
      && stripe config --list 2>/dev/null | grep -qi '^live_mode_api_key'; then
    read -rp "WARNING: stripe CLI appears to have a live key configured. Continue in TEST mode? [yN] " ans
    case "${ans:-n}" in
      y|Y|yes) ;;
      *) echo "aborting." >&2; exit 1 ;;
    esac
  fi
fi

# Thin wrapper so every stripe call carries --live when MODE=live.
stripe_cmd() {
  if [[ -n "$STRIPE_CLI_FLAG" ]]; then
    stripe "$STRIPE_CLI_FLAG" "$@"
  else
    stripe "$@"
  fi
}

# These constants MUST match billing.rs SKU_REPACK_* constants. If you
# rename one there, rename the corresponding event_name here.
RUNS_SKU="splatforge_repack_runs"
SECS_SKU="splatforge_repack_seconds"

find_meter_by_event_name() {
  local event_name="$1"
  stripe_cmd billing meters list --limit 100 --format json 2>/dev/null \
    | jq -r --arg n "$event_name" '.data[] | select(.event_name == $n) | .id' \
    | head -n1
}

ensure_meter() {
  local event_name="$1"
  local display_name="$2"
  local existing
  existing="$(find_meter_by_event_name "$event_name")"
  if [[ -n "$existing" ]]; then
    echo "  - meter '$event_name' exists: $existing" >&2
    echo "$existing"
    return
  fi
  echo "  - creating meter '$event_name'" >&2
  stripe_cmd billing meters create \
    --display-name "$display_name" \
    --event-name "$event_name" \
    --default-aggregation.formula "sum" \
    --value-settings.event-payload-key "value" \
    --customer-mapping.type "by_id" \
    --customer-mapping.event-payload-key "stripe_customer_id" \
    --format json \
    | jq -r '.id'
}

find_product_by_lookup_key() {
  local lk="$1"
  stripe_cmd products list --limit 100 --format json 2>/dev/null \
    | jq -r --arg k "$lk" '.data[] | select(.metadata.lookup_key == $k or .id == $k) | .id' \
    | head -n1
}

ensure_product() {
  local lookup_key="$1"
  local name="$2"
  local description="$3"
  local existing
  existing="$(find_product_by_lookup_key "$lookup_key")"
  if [[ -n "$existing" ]]; then
    echo "  - product '$lookup_key' exists: $existing" >&2
    echo "$existing"
    return
  fi
  echo "  - creating product '$lookup_key'" >&2
  stripe_cmd products create \
    --name "$name" \
    --description "$description" \
    -d "metadata[lookup_key]=$lookup_key" \
    --format json \
    | jq -r '.id'
}

echo "==> Provisioning SplatForge billing meters + products (mode: $MODE)" >&2

RUNS_METER_ID="$(ensure_meter "$RUNS_SKU" "SplatForge repack runs")"
SECS_METER_ID="$(ensure_meter "$SECS_SKU" "SplatForge repack compute seconds")"

RUNS_PRODUCT_ID="$(ensure_product \
  "splatforge_repack_runs_product" \
  "SplatForge Cloud — Repack Runs" \
  "Per-job flat fee for paid-tier differentiable repack on A100. \
Recommended starting price: \$0.05/run. Attach a usage-based price to \
meter $RUNS_METER_ID in the Stripe dashboard." \
)"
SECS_PRODUCT_ID="$(ensure_product \
  "splatforge_repack_seconds_product" \
  "SplatForge Cloud — Repack Compute Seconds" \
  "Per-second A100 compute cost for paid-tier repack. Recommended \
starting price: \$0.0008/second. Attach a usage-based price to meter \
$SECS_METER_ID in the Stripe dashboard." \
)"

echo "" >&2
echo "==> Done. Resources (mode=$MODE):" >&2
echo "    meters:" >&2
echo "      runs    -> $RUNS_METER_ID  (event_name=$RUNS_SKU)" >&2
echo "      seconds -> $SECS_METER_ID  (event_name=$SECS_SKU)" >&2
echo "    products:" >&2
echo "      runs    -> $RUNS_PRODUCT_ID" >&2
echo "      seconds -> $SECS_PRODUCT_ID" >&2
echo "" >&2
echo "==> Next steps:" >&2
echo "    1. In the Stripe dashboard, attach a usage-based price to each" >&2
echo "       meter (the script intentionally does NOT set prices)." >&2
echo "    2. Provision the recurring Team-tier price via:" >&2
echo "         tasks/scripts/stripe-bootstrap-team-tier.sh --mode $MODE" >&2
echo "    3. Create the webhook endpoint:" >&2
echo "         stripe listen --forward-to https://splatforge-api.fly.dev/v1/stripe/webhook" >&2
echo "       and copy the printed whsec_... below." >&2
echo "    4. Map each beta customer to their bearer key in" >&2
echo "       SPLATFORGE_KEY_CUSTOMERS (see apps/api/BILLING.md)." >&2
echo "" >&2

if [[ "$MODE" == "live" ]]; then
  ENV_SECRET_HINT="sk_live_REPLACE_ME"
  LIVE_FLAG="true"
else
  ENV_SECRET_HINT="sk_test_REPLACE_ME"
  LIVE_FLAG="false"
fi
cat <<EOF
# Paste into Fly: fly secrets set ...
STRIPE_SECRET_KEY=$ENV_SECRET_HINT
STRIPE_WEBHOOK_SECRET=whsec_REPLACE_ME
STRIPE_TEAM_PRICE_ID=price_REPLACE_ME
STRIPE_LIVE_MODE=$LIVE_FLAG
SPLATFORGE_KEY_CUSTOMERS=key1:cus_REPLACE_ME

# Reference (informational only — billing.rs uses the SKU constants):
#   splatforge_repack_runs    meter -> $RUNS_METER_ID
#   splatforge_repack_seconds meter -> $SECS_METER_ID
EOF
