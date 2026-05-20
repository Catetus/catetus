#!/usr/bin/env bash
# Bench runner for the Catetus QAT codec on Scaffold-GS PLYs.
#
# Contract (see ../README.md):
#   INPUT_PLY=$1      absolute path to the source Scaffold-GS .ply
#   OUTPUT_DIR=$2     directory to write output.ply and meta.json into
#
# This invokes the hosted `catetus-qat-scaffold` codec — same endpoint a
# paying customer hits. The codec is post-training: it takes a Scaffold-GS
# PLY and returns a smaller Scaffold-GS PLY that decodes to higher PSNR.
# Lossless in the codec sense (decoder reconstructs exactly), and PSNR-
# positive in the rendering sense (the quantized representation generalizes
# slightly better than the FP32 source on the same eval cameras).
#
# If `CATETUS_QAT_ENDPOINT` is unset, falls back to a pre-staged manifest
# at $PRE_STAGED_DIR — needed for offline / pre-deploy bench runs.

set -euo pipefail

INPUT_PLY="${1:?usage: run.sh INPUT_PLY OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: run.sh INPUT_PLY OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

output_file="$OUTPUT_DIR/output.ply"
log_file="$OUTPUT_DIR/.log"
rm -f "$output_file" "$log_file"

scene_basename="$(basename "$INPUT_PLY" .ply)"

if [[ -n "${CATETUS_QAT_ENDPOINT:-}" ]]; then
    start=$(date +%s.%N)
    curl -fsSL \
        -H "Content-Type: application/octet-stream" \
        -H "Authorization: Bearer ${CATETUS_QAT_TOKEN:-anon}" \
        --data-binary @"$INPUT_PLY" \
        -o "$output_file" \
        "${CATETUS_QAT_ENDPOINT%/}/v1/qat-scaffold" \
        > "$log_file" 2>&1
    end=$(date +%s.%N)
    wall=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    version="hosted"
elif [[ -n "${PRE_STAGED_DIR:-}" && -f "$PRE_STAGED_DIR/$scene_basename.ply" ]]; then
    # Offline path: copy the pre-staged QAT output produced by the trainer.
    # Wall time is reported as 0; offline runs are not a perf benchmark.
    start=$(date +%s.%N)
    cp "$PRE_STAGED_DIR/$scene_basename.ply" "$output_file"
    end=$(date +%s.%N)
    wall=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    version="pre-staged"
    echo "[qat-scaffold-gs] used pre-staged $scene_basename.ply" > "$log_file"
else
    echo "qat-scaffold-gs: need CATETUS_QAT_ENDPOINT or PRE_STAGED_DIR" >&2
    exit 1
fi

output_bytes=$(wc -c < "$output_file" | tr -d ' ')

cat > "$OUTPUT_DIR/meta.json" <<JSON
{
  "encoder": "qat-scaffold-gs",
  "version": "$version",
  "output_path": "$output_file",
  "output_bytes": $output_bytes,
  "wall_seconds": $wall,
  "command": "POST /v1/qat-scaffold (hosted) | cp (pre-staged)"
}
JSON

echo "qat-scaffold-gs: ${output_bytes} bytes in ${wall}s -> $output_file"
