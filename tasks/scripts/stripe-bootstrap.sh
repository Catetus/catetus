#!/usr/bin/env bash
# stripe-bootstrap.sh — provision the Stripe products + prices + meters
# that SplatForge Cloud's paid tier expects.
#
# Idempotent. Safe to run repeatedly; existing resources are detected by
# matching on the meter `event_name` and product `lookup_key`, so a
# second run does NOT create duplicates.
#
# Requirements:
#   * stripe CLI installed and authenticated against your TEST-MODE
#     account (`stripe login`).
#   * jq available (Stripe CLI emits JSON; we grep with jq).
#
# Output: prints the env-var block ready to paste into
# `fly secrets set` (or your .env). Nothing is written to disk.

set -euo pipefail

if ! command -v stripe >/dev/null 2>&1; then
  echo "error: stripe CLI not on PATH. Install via: brew install stripe/stripe-cli/stripe" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq not on PATH. Install via: brew install jq" >&2
  exit 2
fi

# Belt-and-braces: refuse to run against a live-mode key. The script
# does not pass `--live` anywhere, so the CLI's configured key decides;
# this is purely a "did the operator type the wrong command" guard.
if stripe config --list 2>/dev/null | grep -qi '^live_mode_api_key'; then
  read -rp "WARNING: stripe CLI appears to have a live key configured. Continue? [yN] " ans
  case "${ans:-n}" in
    y|Y|yes) ;;
    *) echo "aborting." >&2; exit 1 ;;
  esac
fi

# These constants MUST match billing.rs SKU_REPACK_* constants. If you
# rename one there, rename the corresponding event_name here.
RUNS_SKU="splatforge_repack_runs"
SECS_SKU="splatforge_repack_seconds"

# Returns the meter id whose event_name matches $1, or empty string.
# Stripe paginates at 10 by default; we ask for 100 because we're
# unlikely to have more than two SplatForge meters in practice.
find_meter_by_event_name() {
  local event_name="$1"
  stripe billing meters list --limit 100 --format json 2>/dev/null \
    | jq -r --arg n "$event_name" '.data[] | select(.event_name == $n) | .id' \
    | head -n1
}

# Creates a meter if it doesn't exist. Echoes the meter id.
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
  # default_aggregation.formula=sum + value_settings.event_payload_key=value
  # is the Stripe-recommended shape for usage-based pricing where the
  # event payload carries the unit count (1 per run for _runs, elapsed
  # seconds for _seconds).
  stripe billing meters create \
    --display-name "$display_name" \
    --event-name "$event_name" \
    --default-aggregation.formula "sum" \
    --value-settings.event-payload-key "value" \
    --customer-mapping.type "by_id" \
    --customer-mapping.event-payload-key "stripe_customer_id" \
    --format json \
    | jq -r '.id'
}

# Returns the product id whose lookup_key matches $1, or empty string.
find_product_by_lookup_key() {
  local lk="$1"
  stripe products list --limit 100 --format json 2>/dev/null \
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
  stripe products create \
    --name "$name" \
    --description "$description" \
    -d "metadata[lookup_key]=$lookup_key" \
    --format json \
    | jq -r '.id'
}

echo "==> Provisioning SplatForge billing meters + products (test mode)" >&2

RUNS_METER_ID="$(ensure_meter "$RUNS_SKU" "SplatForge repack runs")"
SECS_METER_ID="$(ensure_meter "$SECS_SKU" "SplatForge repack compute seconds")"

# Products are organizational labels in the Stripe dashboard. Prices
# (the actual $ amounts) are *not* created here — pricing is a
# commercial decision that lives in the dashboard, not a script. The
# meters above will collect events with $0 attached until you wire a
# usage-based price to each in the dashboard.
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
echo "==> Done. Resources:" >&2
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
echo "    2. Create the webhook endpoint:" >&2
echo "         stripe listen --forward-to https://splatforge-api.fly.dev/v1/stripe/webhook" >&2
echo "       and copy the printed whsec_... below." >&2
echo "    3. Map each beta customer to their bearer key in" >&2
echo "       SPLATFORGE_KEY_CUSTOMERS (see apps/api/BILLING.md)." >&2
echo "" >&2

# Print the env-var template last so the operator can pipe stdout
# straight into a `.env.staging` file. The $-prefixed values are
# placeholders; everything not derived from this script's run is
# left for the operator to fill in.
cat <<EOF
# Paste into Fly: fly secrets set ...
STRIPE_SECRET_KEY=sk_test_REPLACE_ME
STRIPE_WEBHOOK_SECRET=whsec_REPLACE_ME
STRIPE_LIVE_MODE=false
SPLATFORGE_KEY_CUSTOMERS=key1:cus_REPLACE_ME

# Reference (informational only — billing.rs uses the SKU constants):
#   splatforge_repack_runs    meter -> $RUNS_METER_ID
#   splatforge_repack_seconds meter -> $SECS_METER_ID
EOF
