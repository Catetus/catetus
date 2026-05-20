#!/usr/bin/env bash
set -euo pipefail
# Run on 4090 WSL.
# 1. Decode codec-gs-mixed CRF=28 for bonsai/bicycle/stump.
# 2. Run psnr_v2 on the matrix:
#       codec-gs-mixed-crf28 x {bonsai, bicycle, stump}
#       mesonpp-k256          x {bonsai, bicycle}                # have decoded
# 3. Emit a combined summary.

cd $HOME/sf-crf-sweep
source $HOME/catetus/.venv/bin/activate

SCENES_SOURCE=(
  "bonsai_iter7000:$HOME/catetus/saliency_v1_real/bonsai_iter7000.ply"
  "bicycle_iter7000:$HOME/catetus/scenes/bicycle.ply"
  "stump_iter7000:$HOME/catetus/saliency_v1_real/stump_iter7000.ply"
)

OUT=$HOME/sf-crf-sweep/sanity-out
mkdir -p "$OUT/decoded"

echo "=== STEP 1: decode codec-gs-mixed CRF=28 PLYs ==="
for entry in "${SCENES_SOURCE[@]}"; do
  name="${entry%%:*}"
  src="${entry##*:}"
  run_dir="$HOME/sf-crf-sweep/runs/${name}_crf28"
  decoded="$OUT/decoded/${name}_codecgs_crf28.ply"
  if [[ -f "$decoded" ]]; then
    echo "  [skip] $decoded exists"
    continue
  fi
  echo "  decoding $name -> $decoded"
  python3 - <<PY
import sys
sys.path.insert(0, "$HOME/sf-crf-sweep/codec_gs_lib")
import codec_gs_decode_ply as dec
from pathlib import Path
dec.decode_to_ply(
    Path("$run_dir"),
    Path("$src"),
    Path("$decoded"),
)
print("  decoded OK ->", "$decoded")
PY
done

echo "=== STEP 2: run psnr_v2 matrix ==="
for entry in "${SCENES_SOURCE[@]}"; do
  name="${entry%%:*}"
  src="${entry##*:}"
  decoded="$OUT/decoded/${name}_codecgs_crf28.ply"
  out_json="$OUT/${name}__codec-gs-mixed-crf28.json"
  python3 "$OUT/psnr_v2.py" \
    --source "$src" \
    --decoded "$decoded" \
    --scene "$name" \
    --label "codec-gs-mixed-crf28" \
    --out "$out_json" \
    --n-cams 8
done

echo "=== STEP 3: run psnr_v2 on MesonGS++ K=256 (anchors) ==="
# bonsai_iter7000 decoded ply exists
python3 "$OUT/psnr_v2.py" \
  --source "$HOME/catetus/saliency_v1_real/bonsai_iter7000.ply" \
  --decoded "$HOME/Catetus/.bench-scenes/meson-validate/bonsai_iter7000_decoded.ply" \
  --scene "bonsai_iter7000" \
  --label "mesonpp-k256" \
  --out "$OUT/bonsai_iter7000__mesonpp-k256.json" \
  --n-cams 8

# bicycle decoded ply exists; source was $HOME/catetus/scenes/bicycle.ply
python3 "$OUT/psnr_v2.py" \
  --source "$HOME/catetus/scenes/bicycle.ply" \
  --decoded "$HOME/Catetus/.bench-scenes/meson-validate/bicycle_decoded.ply" \
  --scene "bicycle_iter7000" \
  --label "mesonpp-k256" \
  --out "$OUT/bicycle_iter7000__mesonpp-k256.json" \
  --n-cams 8

echo "=== DONE ==="
ls -lh "$OUT"/*.json
