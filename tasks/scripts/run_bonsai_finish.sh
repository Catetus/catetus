#!/bin/bash
# Finish bonsai presets while the orchestrator is paused.
# Just bonsai web-desktop (already running) and quality-max.
set +e
WT="/Users/montabano1/Desktop/SplatForge/.wt-bench-visual"
BR="$WT/apps/web/public/bench-renders"
LOG="$WT/tasks/logs/optimize.log"
SH="$WT/tasks/scripts/run_optimize.sh"

PLY=/Users/montabano1/Desktop/SplatForge/benches/scenes/real/bonsai_iter7000.ply

# Wait until current bonsai web-desktop finishes before launching quality-max.
while pgrep -fl "splatforge optimize.*bonsai_iter7000.ply.*--preset web-desktop" >/dev/null 2>&1; do
  sleep 10
done

# quality-max
GLTF="$BR/bonsai/quality-max/scene.gltf"
if [ ! -f "$GLTF" ] || [ ! -s "$GLTF" ]; then
  bash "$SH" "bonsai" "$PLY" "quality-max" "$BR/bonsai/quality-max" "$LOG"
fi

echo "BONSAI ALL DONE" >> "$LOG"
