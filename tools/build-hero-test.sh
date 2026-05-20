#!/usr/bin/env bash
# build-hero-test.sh — end-to-end smoke test for tools/build-hero.sh.
#
# 1. Synthesizes a tiny vanilla-3DGS-layout PLY (~80k splats centered on a
#    cube with a sparse floater halo padding the bbox).
# 2. Runs tools/build-hero.sh against it with low decimation so the pipeline
#    finishes in seconds.
# 3. Asserts: scene.gltf parses + splatCount > 50000 + at least one
#    chunk_*.bin emitted.
#
# Target runtime: <30s. Stands alone; no pytest hook in this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN="$REPO_ROOT/target/release/catetus"
if [[ ! -x "$BIN" ]]; then
  echo "build-hero-test: catetus release binary missing at $BIN" >&2
  echo "build-hero-test: run 'cargo build --release -p catetus-cli' first" >&2
  exit 3
fi

TMP_DIR="$(mktemp -d -t catetus-hero-test.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

SRC_PLY="$TMP_DIR/synth.ply"
OUT_DIR="$TMP_DIR/out"

echo "build-hero-test: synthesizing tiny vanilla 3DGS PLY at $SRC_PLY"
python3 - "$SRC_PLY" <<'PY'
import struct, sys
import numpy as np

dst = sys.argv[1]
rng = np.random.default_rng(0xBADCAFE)

# 80k dense subject splats inside a [-1,1]^3 cube + 8k floater halo at
# radius ~10 to make the bbox crop step actually do work.
n_dense = 80_000
n_halo = 8_000
n = n_dense + n_halo

xyz = np.empty((n, 3), dtype=np.float32)
xyz[:n_dense] = rng.uniform(-1.0, 1.0, size=(n_dense, 3)).astype(np.float32)
# Halo: random direction * radius in [8, 12]
dirs = rng.normal(size=(n_halo, 3)).astype(np.float32)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
radii = rng.uniform(8.0, 12.0, size=(n_halo,)).astype(np.float32)
xyz[n_dense:] = dirs * radii[:, None]

# scale (log-space) — small for both populations
scales = rng.uniform(-4.0, -2.0, size=(n, 3)).astype(np.float32)
# rotation — identity quaternion w/ small jitter
rots = np.zeros((n, 4), dtype=np.float32)
rots[:, 0] = 1.0
rots += rng.normal(scale=0.01, size=rots.shape).astype(np.float32)
# opacity — dense splats high logit (visible), halo low logit
opacity = np.empty((n,), dtype=np.float32)
opacity[:n_dense] = rng.uniform(2.0, 5.0, size=(n_dense,)).astype(np.float32)
opacity[n_dense:] = rng.uniform(-1.0, 0.5, size=(n_halo,)).astype(np.float32)
# f_dc_0..2 — uniform-ish color
fdc = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)

# No f_rest (SH degree 0) — keeps PLY small and parser happy.
header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property float f_dc_0\n"
    "property float f_dc_1\n"
    "property float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\n"
    "property float scale_1\n"
    "property float scale_2\n"
    "property float rot_0\n"
    "property float rot_1\n"
    "property float rot_2\n"
    "property float rot_3\n"
    "end_header\n"
)

# Pack matching the header property order.
row_dtype = np.dtype([
    ("x", np.float32), ("y", np.float32), ("z", np.float32),
    ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
    ("opacity", np.float32),
    ("scale_0", np.float32), ("scale_1", np.float32), ("scale_2", np.float32),
    ("rot_0", np.float32), ("rot_1", np.float32), ("rot_2", np.float32), ("rot_3", np.float32),
])
rows = np.empty(n, dtype=row_dtype)
rows["x"] = xyz[:, 0]; rows["y"] = xyz[:, 1]; rows["z"] = xyz[:, 2]
rows["f_dc_0"] = fdc[:, 0]; rows["f_dc_1"] = fdc[:, 1]; rows["f_dc_2"] = fdc[:, 2]
rows["opacity"] = opacity
rows["scale_0"] = scales[:, 0]; rows["scale_1"] = scales[:, 1]; rows["scale_2"] = scales[:, 2]
rows["rot_0"] = rots[:, 0]; rows["rot_1"] = rots[:, 1]; rows["rot_2"] = rots[:, 2]; rows["rot_3"] = rots[:, 3]

with open(dst, "wb") as f:
    f.write(header.encode("latin-1"))
    f.write(rows.tobytes())
print(f"  wrote {dst} ({n} splats)")
PY

echo "build-hero-test: running tools/build-hero.sh"
bash "$SCRIPT_DIR/build-hero.sh" "$SRC_PLY" \
  --decimate 60000 \
  --axis-pct 0.02,0.98 \
  --out "$OUT_DIR"

echo "build-hero-test: asserting output"
python3 - "$OUT_DIR/scene.gltf" <<'PY'
import json, os, sys, glob
gltf = sys.argv[1]
assert os.path.exists(gltf), f"missing {gltf}"
with open(gltf) as f:
    doc = json.load(f)
splat_count = None
accessors = doc.get("accessors", [])
if accessors:
    splat_count = accessors[0].get("count")
if splat_count is None:
    side = os.path.join(os.path.dirname(gltf), "scene.json")
    if os.path.exists(side):
        with open(side) as f:
            splat_count = json.load(f).get("splats_after")
assert splat_count is not None, "no splat count found"
assert splat_count > 50_000, f"splatCount={splat_count} <= 50000"
chunks = glob.glob(os.path.join(os.path.dirname(gltf), "buffers", "chunk_*.bin"))
assert len(chunks) >= 1, f"no chunk_*.bin produced (found {chunks})"
print(f"  OK splatCount={splat_count} chunks={len(chunks)}")
PY

echo "build-hero-test: PASS"
