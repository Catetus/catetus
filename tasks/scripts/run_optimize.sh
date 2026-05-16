#!/bin/bash
# Run optimize for one (scene, preset) pair. Output: GLB self-contained.
# Args: SCENE_ID INPUT_PLY PRESET OUT_DIR
set -e
SCENE_ID="$1"
INPUT_PLY="$2"
PRESET="$3"
OUT_DIR="$4"
LOG_FILE="$5"

mkdir -p "$OUT_DIR"
CLI="/Users/montabano1/Desktop/SplatForge/target/release/splatforge"

echo "[$(date +%H:%M:%S)] START $SCENE_ID/$PRESET" >> "$LOG_FILE"
START=$(date +%s)

# Use --chunked so the viewer's manifest loader can stream the scene.
# scene.gltf is JSON + sidecar buffers under buffers/.
# We report the SUM of (scene.gltf + scene.json + buffers/**) as the
# "preset size on disk" — the honest answer for a customer who'd serve
# the directory behind a CDN.
if "$CLI" optimize "$INPUT_PLY" \
    --preset "$PRESET" \
    --chunked \
    -o "$OUT_DIR/scene.gltf" \
    >> "$LOG_FILE" 2>&1; then
  END=$(date +%s)
  SIZE=$(du -sk "$OUT_DIR" 2>/dev/null | awk '{print $1 * 1024}')
  SIZE=${SIZE:-0}
  echo "[$(date +%H:%M:%S)] OK $SCENE_ID/$PRESET $((END-START))s bytes=$SIZE (gltf+buffers dir total)" >> "$LOG_FILE"
  echo "$SCENE_ID,$PRESET,ok,$SIZE,$((END-START))" >> "$LOG_FILE.csv"
else
  END=$(date +%s)
  echo "[$(date +%H:%M:%S)] FAIL $SCENE_ID/$PRESET $((END-START))s" >> "$LOG_FILE"
  echo "$SCENE_ID,$PRESET,fail,0,$((END-START))" >> "$LOG_FILE.csv"
fi
