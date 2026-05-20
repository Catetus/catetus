#!/usr/bin/env bash
#
# scripts/usdc-roundtrip.sh — prove that catetus's USDC writer round-trips
# bit-exact-as-USDA against Pixar's `usdcat`.
#
# Pipeline (per fixture):
#   1. catetus convert FIXTURE.usda /tmp/<name>.usdc   (our binary writer)
#   2. usdcat /tmp/<name>.usdc -o /tmp/<name>.via_usdcat.usda  (Pixar reads it)
#   3. catetus convert /tmp/<name>.via_usdcat.usda /tmp/<name>.recovered.usda
#      (round-trip through our IR — should match the original semantic content)
#   4. Compare the attribute arrays of the original and the recovered scene.
#
# We do not require *byte equality* of the USDC binaries — multiple encodings
# are valid. We require usdcat to accept what we wrote and produce a USDA
# that decodes back to the same IR.
#
# Requires:
#   * cargo (in this repo)
#   * Pixar `usdcat` on PATH (Apple USD Tools on macOS ships /usr/bin/usdcat).
#     Install: brew install usd-tools  OR  download from openusd.org.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$REPO_ROOT/crates/catetus-usd/fixtures"
SCRATCH="${SCRATCH_DIR:-$(mktemp -d -t usdc-roundtrip-XXXX)}"

if ! command -v usdcat >/dev/null 2>&1; then
  echo "FATAL: usdcat not on PATH. Install Pixar USD tools." >&2
  echo "  macOS:    brew install usd-tools" >&2
  echo "  Source:   https://github.com/PixarAnimationStudios/OpenUSD" >&2
  exit 2
fi

USDCAT_VERSION="$(usdcat --version 2>&1 | head -1 || true)"
echo "Using usdcat: $USDCAT_VERSION"

# Build the CLI in release mode so the run is fast.
(cd "$REPO_ROOT" && cargo build --release -p catetus-cli >/dev/null)
CATETUS_BIN="$REPO_ROOT/target/release/catetus"

FIXTURES_LIST=(
  "minimal.usda"
  "particle_field.usda"
  "dense.usda"
)

passed=0
failed=0
for fname in "${FIXTURES_LIST[@]}"; do
  src="$FIXTURES/$fname"
  name="${fname%.usda}"
  binary="$SCRATCH/$name.usdc"
  reformat="$SCRATCH/$name.via_usdcat.usda"
  recovered="$SCRATCH/$name.recovered.usda"

  echo
  echo "=== $fname ==="

  "$CATETUS_BIN" convert "$src" --to usdc -o "$binary"
  echo "  [ok] wrote $binary ($(stat -f %z "$binary" 2>/dev/null || stat -c %s "$binary") bytes)"

  if ! usdcat "$binary" -o "$reformat" 2>"$SCRATCH/$name.usdcat.err"; then
    echo "  [FAIL] usdcat rejected our USDC:" >&2
    cat "$SCRATCH/$name.usdcat.err" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "  [ok] usdcat accepted; reformat = $reformat"

  # Round-trip the reformat through our own IR (proves Pixar's USDA is
  # semantically equivalent to the original at the IR level).
  "$CATETUS_BIN" convert "$reformat" --to usda -o "$recovered"

  # Compare per-attribute by parsing both with a tiny python helper.
  if python3 "$REPO_ROOT/scripts/_usda_diff.py" "$src" "$recovered"; then
    echo "  [PASS] $fname round-tripped bit-exact-as-USDA"
    passed=$((passed + 1))
  else
    echo "  [FAIL] $fname did not round-trip" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "==========================================="
echo "  PASS: $passed   FAIL: $failed"
echo "  Scratch directory: $SCRATCH"
echo "==========================================="

[ "$failed" -eq 0 ]
