#!/bin/bash
# Run all (scene, preset) combinations serially, smallest-PLY first
# so the bench gallery has the most coverage if time runs out.
set +e

WT="/Users/montabano1/Desktop/SplatForge/.wt-bench-visual"
BR="$WT/apps/web/public/bench-renders"
LOG="$WT/tasks/logs/optimize.log"
SH="$WT/tasks/scripts/run_optimize.sh"
chmod +x "$SH"

# Order: small synthetic first, then bonsai (287MB), then stump (944MB).
# Bicycle (896MB) skipped — too slow under shared-host contention and we
# already have stump for the "large outdoor scene" visual.
SCENES=(
  "splatbench_portrait|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_portrait_proxy.ply"
  "splatbench_specular|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_specular_proxy.ply"
  "splatbench_foliage|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_foliage_proxy.ply"
  "splatbench_transparency|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_transparency_proxy.ply"
  "splatbench_indoor|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_indoor_proxy.ply"
  "splatbench_outdoor|/Users/montabano1/Desktop/SplatForge/benches/scenes/splatbench_outdoor_proxy.ply"
  "bonsai|/Users/montabano1/Desktop/SplatForge/benches/scenes/real/bonsai_iter7000.ply"
  "stump|/Users/montabano1/Desktop/SplatForge/benches/scenes/real/stump_iter7000.ply"
)

PRESETS=("web-mobile" "web-desktop" "quality-max")

for s in "${SCENES[@]}"; do
  ID="${s%%|*}"
  PLY="${s##*|}"
  if [ ! -f "$PLY" ]; then
    echo "SKIP $ID (no ply)" >> "$LOG"
    continue
  fi
  for p in "${PRESETS[@]}"; do
    GLTF="$BR/$ID/$p/scene.gltf"
    if [ -f "$GLTF" ] && [ -s "$GLTF" ]; then
      echo "[$(date +%H:%M:%S)] SKIP $ID/$p (exists)" >> "$LOG"
      SIZE=$(du -sk "$BR/$ID/$p" 2>/dev/null | awk '{print $1 * 1024}')
      echo "$ID,$p,skip,${SIZE:-0},0" >> "$LOG.csv"
      continue
    fi
    bash "$SH" "$ID" "$PLY" "$p" "$BR/$ID/$p" "$LOG"
  done
done

echo "ALL DONE" >> "$LOG"
