#!/usr/bin/env bash
# stripe-bootstrap-team-tier.sh — provision the recurring Team-tier
# product + $99/seat/mo USD price for SplatForge Cloud's self-serve
# Checkout flow.
#
# This script is INTENTIONALLY separate from stripe-bootstrap.sh:
#   * stripe-bootstrap.sh handles usage-meters (per-run / per-second
#     billing for paid-tier repack jobs).
#   * THIS script handles the recurring subscription price ($99/seat/mo)
#     that apps/api/src/checkout.rs references via STRIPE_TEAM_PRICE_ID.
#
# Idempotent. Detection is by `metadata.lookup_key` on both product and
# price, so re-running prints the existing IDs without duplication.
#
# Usage:
#   tasks/scripts/stripe-bootstrap-team-tier.sh [--mode test|live]
#
# On success prints:
#   STRIPE_TEAM_PRICE_ID=price_...
# ready to paste into `fly secrets set`.

set -euo pipefail

MODE="test"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)  shift; MODE="${1:-}";;
    --mode=*) MODE="${1#--mode=}";;
    -h|--help) sed -n '1,22p' "$0"; exit 0;;
    *) echo "error: unknown arg '$1' (use --mode test|live)" >&2; exit 2;;
  esac
  shift || true
done

case "$MODE" in
  test|live) ;;
  *) echo "error: --mode must be 'test' or 'live' (got '$MODE')" >&2; exit 2;;
esac

if ! command -v stripe >/dev/null 2>&1; then
  echo "error: stripe CLI not on PATH" >&2; exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq not on PATH" >&2; exit 2
fi

STRIPE_CLI_FLAG=""
if [[ "$MODE" == "live" ]]; then
  STRIPE_CLI_FLAG="--live"
  EXPECTED_SECRET="${STRIPE_LIVE_SECRET_KEY:-${STRIPE_SECRET_KEY:-}}"
  if [[ -z "$EXPECTED_SECRET" || "$EXPECTED_SECRET" != sk_live_* ]]; then
    echo "error: --mode live requires a sk_live_ key in STRIPE_LIVE_SECRET_KEY or STRIPE_SECRET_KEY" >&2
    exit 2
  fi
  echo "" >&2
  echo "============================================================" >&2
  echo "  ABOUT TO CREATE A LIVE-MODE \$99/seat/mo RECURRING PRICE" >&2
  echo "============================================================" >&2
  echo "  Idempotent — if a price already exists with metadata" >&2
  echo "  lookup_key=splatforge_team_seat_monthly_usd this is a no-op." >&2
  echo "" >&2
  read -rp "Type 'PROVISION LIVE' to continue: " confirm
  [[ "$confirm" == "PROVISION LIVE" ]] || { echo "aborted." >&2; exit 1; }
  export STRIPE_API_KEY="$EXPECTED_SECRET"
else
  EXPECTED_SECRET="${STRIPE_TEST_SECRET_KEY:-${STRIPE_SECRET_KEY:-}}"
  if [[ -n "$EXPECTED_SECRET" && "$EXPECTED_SECRET" == sk_live_* ]]; then
    echo "error: --mode test but secret is sk_live_" >&2; exit 2
  fi
  [[ -n "$EXPECTED_SECRET" ]] && export STRIPE_API_KEY="$EXPECTED_SECRET"
fi

stripe_cmd() {
  if [[ -n "$STRIPE_CLI_FLAG" ]]; then stripe "$STRIPE_CLI_FLAG" "$@"; else stripe "$@"; fi
}

# These constants are the wire-contract with apps/api/src/checkout.rs.
# Renaming the lookup_key here breaks idempotency for already-provisioned
# accounts; coordinate with the operator before changing.
PRODUCT_LOOKUP="splatforge_team_seat_monthly_usd"
PRICE_LOOKUP="splatforge_team_seat_monthly_usd"
PRICE_AMOUNT_USD_CENTS=9900
PRICE_CURRENCY="usd"
PRICE_INTERVAL="month"

find_product() {
  stripe_cmd products list --limit 100 --format json 2>/dev/null \
    | jq -r --arg k "$PRODUCT_LOOKUP" \
        '.data[] | select(.metadata.lookup_key == $k) | .id' \
    | head -n1
}

find_price() {
  local product_id="$1"
  stripe_cmd prices list --product "$product_id" --limit 100 --format json 2>/dev/null \
    | jq -r --arg k "$PRICE_LOOKUP" \
        '.data[] | select(.metadata.lookup_key == $k and .active == true) | .id' \
    | head -n1
}

echo "==> Provisioning SplatForge Team-tier product + price (mode: $MODE)" >&2

PRODUCT_ID="$(find_product)"
if [[ -z "$PRODUCT_ID" ]]; then
  echo "  - creating product '$PRODUCT_LOOKUP'" >&2
  PRODUCT_ID="$(stripe_cmd products create \
    --name "SplatForge Cloud — Team" \
    --description "Team tier: \$99/seat/mo. Self-serve via Stripe Checkout. \
Wire STRIPE_TEAM_PRICE_ID to the resulting price id; checkout.rs's create_session \
references it as line_items[0][price]." \
    -d "metadata[lookup_key]=$PRODUCT_LOOKUP" \
    --format json | jq -r '.id')"
else
  echo "  - product '$PRODUCT_LOOKUP' exists: $PRODUCT_ID" >&2
fi

PRICE_ID="$(find_price "$PRODUCT_ID")"
if [[ -z "$PRICE_ID" ]]; then
  echo "  - creating recurring price (\$$((PRICE_AMOUNT_USD_CENTS/100))/seat/mo $PRICE_CURRENCY)" >&2
  PRICE_ID="$(stripe_cmd prices create \
    --product "$PRODUCT_ID" \
    --currency "$PRICE_CURRENCY" \
    --unit-amount "$PRICE_AMOUNT_USD_CENTS" \
    --recurring.interval "$PRICE_INTERVAL" \
    --recurring.usage-type "licensed" \
    --nickname "SplatForge Team — \$99/seat/mo" \
    -d "metadata[lookup_key]=$PRICE_LOOKUP" \
    --format json | jq -r '.id')"
else
  echo "  - price '$PRICE_LOOKUP' exists: $PRICE_ID" >&2
fi

echo "" >&2
echo "==> Done. Team-tier provisioning (mode=$MODE):" >&2
echo "      product -> $PRODUCT_ID" >&2
echo "      price   -> $PRICE_ID" >&2
echo "" >&2
echo "==> Set the env var on Fly:" >&2
echo "      fly secrets set STRIPE_TEAM_PRICE_ID=$PRICE_ID" >&2
echo "" >&2

cat <<EOF
# Paste into Fly: fly secrets set ...
STRIPE_TEAM_PRICE_ID=$PRICE_ID
EOF
