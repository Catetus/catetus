#!/usr/bin/env bash
# Bench runner for PlayCanvas @playcanvas/splat-transform.
#
# Contract (see ../README.md):
#   INPUT_PLY=$1      absolute path to the source .ply
#   OUTPUT_DIR=$2     directory to write output.* and meta.json into
#
# We invoke `splat-transform convert` to produce SOG (PlayCanvas Self-
# Organizing Gaussians, 20× compression claim). The output format is chosen
# to match what splat-transform users would ship — SOG, not glTF — so the
# comparison column on /bench reflects the real product, not a forced
# format mismatch.

set -euo pipefail

INPUT_PLY="${1:?usage: run.sh INPUT_PLY OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run.sh INPUT_PLY OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

# Pin to a specific splat-transform version so the column is reproducible.
# Update the pin deliberately; the meta.json captures whatever is resolved.
SPLAT_TRANSFORM_VERSION="${SPLAT_TRANSFORM_VERSION:-latest}"

# splat-transform CLI shape: `splat-transform [GLOBAL] INPUT [ACTIONS] OUTPUT`.
# Last positional is the output path; no --output flag. We add
# `--morton-order` because Catetus's pipeline includes MortonSort by
# default — comparing without it would force splat-transform to ship
# unsorted output, an unfair handicap.
output_file="$OUTPUT_DIR/output.sog"
log_file="$OUTPUT_DIR/.log"

# Remove any prior run's artifact before re-running; splat-transform errors
# out on existing outputs unless -w is passed (we want the explicit-delete
# semantic so we never silently overwrite something the caller cared about).
rm -f "$output_file" "$log_file"

start=$(date +%s.%N)
if ! npx --yes "@playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION}" \
        "$INPUT_PLY" --morton-order "$output_file" \
        > "$log_file" 2>&1; then
    echo "splat-transform failed; see $log_file" >&2
    exit 1
fi
end=$(date +%s.%N)
wall=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')

# Resolve installed version from the tool's banner — splat-transform's
# first stdout line is always `splat-transform v<x.y.z> (<sha>)`.
resolved_version=$(awk 'NR==1 && $1=="splat-transform" { sub(/^v/,"",$2); print $2; exit }' "$log_file" 2>/dev/null)
[[ -z "$resolved_version" ]] && resolved_version="$SPLAT_TRANSFORM_VERSION"

output_bytes=$(wc -c < "$output_file" | tr -d ' ')

cat > "$OUTPUT_DIR/meta.json" <<JSON
{
  "encoder": "splat-transform",
  "version": "$resolved_version",
  "output_path": "$output_file",
  "output_bytes": $output_bytes,
  "wall_seconds": $wall,
  "command": "splat-transform <in> --morton-order <out>"
}
JSON

echo "splat-transform: ${output_bytes} bytes in ${wall}s → $output_file"
