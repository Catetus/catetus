#!/usr/bin/env bash
# build-hero.sh — reusable rebuild pipeline for the Catetus homepage hero
# asset. Wraps tools/build-hero.py (preprocess) and `catetus optimize`
# (web-mobile preset, chunked glTF). See tools/build-hero.md for context.
#
# Usage:
#   tools/build-hero.sh <source.ply> [--decimate 150000] [--axis-pct 0.03,0.97] \
#                                    [--out apps/web/public/hero-scene]
#
# Defaults match the current production hero build.

set -euo pipefail

DECIMATE=150000
AXIS_PCT="0.03,0.97"
OUT="apps/web/public/hero-scene"
SRC=""

err() { printf 'build-hero: error: %s\n' "$*" >&2; }
info() { printf 'build-hero: %s\n' "$*"; }

usage() {
  cat >&2 <<EOF
Usage: $0 <source.ply> [--decimate N] [--axis-pct LO,HI] [--out DIR]
  source.ply       Vanilla Inria 3DGS PLY (NOT Scaffold-GS).
  --decimate N     Target splat count after importance decimation.
                   Default: ${DECIMATE}.
  --axis-pct LO,HI Per-axis percentile bbox crop, measured against
                   high-opacity splats only. Default: ${AXIS_PCT}.
  --out DIR        Output directory for scene.gltf + buffers/.
                   Default: ${OUT}.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --decimate) DECIMATE="$2"; shift 2 ;;
    --axis-pct) AXIS_PCT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --) shift; break ;;
    -*) err "unknown flag: $1"; usage; exit 2 ;;
    *)
      if [[ -z "$SRC" ]]; then
        SRC="$1"; shift
      else
        err "unexpected positional arg: $1"; usage; exit 2
      fi
      ;;
  esac
done

if [[ -z "$SRC" ]]; then
  err "missing <source.ply>"
  usage
  exit 2
fi
if [[ ! -f "$SRC" ]]; then
  err "source PLY not found: $SRC"
  exit 2
fi

# Resolve repo root from the script location so we work regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If --out is relative, anchor to repo root.
case "$OUT" in
  /*) ABS_OUT="$OUT" ;;
  *)  ABS_OUT="$REPO_ROOT/$OUT" ;;
esac

BIN="$REPO_ROOT/target/release/catetus"
if [[ ! -x "$BIN" ]]; then
  err "catetus release binary not found at $BIN"
  err "build it with: cargo build --release -p catetus-cli"
  exit 3
fi

PY="$SCRIPT_DIR/build-hero.py"
if [[ ! -f "$PY" ]]; then
  err "preprocessor missing: $PY"
  exit 3
fi

# Resolve absolute path of SRC for log clarity.
ABS_SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"

info "src       : $ABS_SRC"
info "out       : $ABS_OUT"
info "decimate  : $DECIMATE"
info "axis-pct  : $AXIS_PCT"

mkdir -p "$ABS_OUT"

# Stale-chunk hygiene: blow away any prior buffers/ and scene.* so we don't
# leave orphaned chunk_*.bin behind when the new build produces fewer chunks.
if [[ -d "$ABS_OUT/buffers" ]]; then
  info "cleaning stale chunks in $ABS_OUT/buffers/"
  rm -f "$ABS_OUT/buffers/chunk_"*.bin "$ABS_OUT/buffers/chunk_"*.bin.zst 2>/dev/null || true
  # Remove buffers dir if empty so optimize re-creates it cleanly.
  rmdir "$ABS_OUT/buffers" 2>/dev/null || true
fi
rm -f "$ABS_OUT/scene.gltf" "$ABS_OUT/scene.json" 2>/dev/null || true

TMP_DIR="$(mktemp -d -t catetus-hero.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
PREP_PLY="$TMP_DIR/preprocessed.ply"

IFS=',' read -r AX_LO AX_HI <<<"$AXIS_PCT"

info "step 1/4: preprocess (per-axis crop + importance decimation)"
python3 "$PY" "$ABS_SRC" "$PREP_PLY" \
  --axis-pct "$AX_LO,$AX_HI" \
  --decimate "$DECIMATE"

info "step 2/4: catetus optimize --preset web-mobile --chunked"
"$BIN" optimize "$PREP_PLY" \
  --preset web-mobile \
  --chunked \
  --out "$ABS_OUT/scene.gltf"

info "step 3/4: verify gltf parses + sanity check splatCount"
python3 - "$ABS_OUT/scene.gltf" <<'PY'
import json, os, sys
gltf_path = sys.argv[1]
try:
    with open(gltf_path) as f:
        doc = json.load(f)
except Exception as e:
    print(f"error: failed to parse {gltf_path}: {e}", file=sys.stderr)
    sys.exit(1)

# catetus optimize writes the splat count to the sidecar scene.json
# as `splats_after`. The gltf itself carries the count implicitly via
# accessor[0].count (positions), which is the source of truth.
splat_count = None
accessors = doc.get("accessors", [])
if accessors:
    splat_count = accessors[0].get("count")

sidecar_count = None
sidecar = os.path.join(os.path.dirname(gltf_path), "scene.json")
if os.path.exists(sidecar):
    with open(sidecar) as f:
        side = json.load(f)
    sidecar_count = side.get("splats_after") or side.get("splatCount")

if splat_count is None:
    splat_count = sidecar_count

if splat_count is None:
    print("error: could not determine splat count from gltf accessors or scene.json", file=sys.stderr)
    sys.exit(1)
if splat_count <= 50_000:
    print(f"error: splatCount={splat_count} is too low (<=50000)", file=sys.stderr)
    sys.exit(1)

# bbox extent — read accessor[0] (positions) min/max.
bbox_min = accessors[0].get("min") if accessors else None
bbox_max = accessors[0].get("max") if accessors else None
print(f"  splatCount: {splat_count}", end="")
if sidecar_count and sidecar_count != splat_count:
    print(f" (accessor) / {sidecar_count} (scene.json)")
else:
    print()
if bbox_min is not None and bbox_max is not None and len(bbox_min) >= 3:
    extents = [bbox_max[i] - bbox_min[i] for i in range(3)]
    print(f"  bbox min  : [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
    print(f"  bbox max  : [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
    print(f"  extent    : [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
else:
    print(f"  bbox      : (no min/max on accessor[0])")
PY

info "step 4/4: report total bytes"
TOTAL_BYTES=$(find "$ABS_OUT" -type f \( -name '*.gltf' -o -name '*.bin' -o -name '*.json' \) -exec wc -c {} + 2>/dev/null \
  | awk 'END { print $1 }')
if [[ -z "$TOTAL_BYTES" ]]; then
  TOTAL_BYTES="(unknown)"
fi
info "  total bytes: $TOTAL_BYTES"

CHUNK_COUNT=$(find "$ABS_OUT/buffers" -maxdepth 1 -name 'chunk_*.bin' 2>/dev/null | wc -l | tr -d ' ')
info "  chunk files: $CHUNK_COUNT"

info "done. Output: $ABS_OUT"
