#!/usr/bin/env bash
# Compare `splatforge analyze` output against the golden expected report.
#
# The "hash" field is allowed to differ when the golden report still carries
# the "blake3:PLACEHOLDER_REGENERATE" sentinel — useful while the analyzer
# is in flux. Once the hash is real, drop that allowance.

set -euo pipefail

BIN=${SPLATFORGE_BIN:-./target/release/splatforge}
FIXTURE=fixtures/tiny/basic_binary.ply
GOLDEN=fixtures/golden/expected_reports/basic_binary.analyze.json

if [[ ! -x "$BIN" ]]; then
  echo "error: splatforge binary not found at $BIN" >&2
  exit 127
fi

for cmd in jq diff; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required tool '$cmd' not on PATH" >&2
    exit 127
  fi
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

ACTUAL="$WORK/actual.json"
EXPECTED="$WORK/expected.json"

"$BIN" analyze "$FIXTURE" > "$ACTUAL"

# If the golden hash is still the placeholder, drop hash on both sides before
# diffing. Otherwise compare every key.
GOLDEN_HASH=$(jq -r '.hash' "$GOLDEN")
if [[ "$GOLDEN_HASH" == "blake3:PLACEHOLDER_REGENERATE" ]]; then
  echo "note: golden hash is placeholder — comparing without 'hash' field"
  jq -S 'del(.hash)' "$ACTUAL" > "$WORK/a.json"
  jq -S 'del(.hash)' "$GOLDEN" > "$WORK/e.json"
else
  jq -S '.' "$ACTUAL" > "$WORK/a.json"
  jq -S '.' "$GOLDEN" > "$WORK/e.json"
fi

if ! diff -u "$WORK/e.json" "$WORK/a.json"; then
  echo "golden diff FAILED" >&2
  echo "  expected: $GOLDEN" >&2
  echo "  actual:   $ACTUAL  (preserved at $ACTUAL)" >&2
  trap - EXIT
  exit 1
fi

echo OK
