#!/usr/bin/env bash
# Catetus end-to-end CLI happy path.
#
# Walks the four primary commands (analyze, inspect, convert, optimize) on the
# tiny binary PLY fixture. Designed for CI: hermetic, no network, no GPU.
#
# Usage:
#   ./tests/integration/cli.sh
#   CATETUS_BIN=/path/to/catetus ./tests/integration/cli.sh

set -euo pipefail

BIN=${CATETUS_BIN:-./target/release/catetus}
FIXTURE=fixtures/tiny/basic_binary.ply

if [[ ! -x "$BIN" ]]; then
  echo "error: catetus binary not found at $BIN" >&2
  echo "       build it first: cargo build --release -p catetus-cli" >&2
  exit 127
fi

if [[ ! -f "$FIXTURE" ]]; then
  echo "error: fixture missing: $FIXTURE" >&2
  echo "       run: python3 fixtures/build.py" >&2
  exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[1/5] analyze ..."
"$BIN" analyze "$FIXTURE" > "$WORK/analyze.json"

echo "[2/5] inspect (ply) ..."
"$BIN" inspect "$FIXTURE"

echo "[3/5] convert ply -> gltf ..."
"$BIN" convert "$FIXTURE" --to gltf --out "$WORK/scene.gltf"

echo "[4/5] inspect (gltf) ..."
"$BIN" inspect "$WORK/scene.gltf"

echo "[5/5] optimize --preset web-mobile ..."
"$BIN" optimize "$FIXTURE" --preset web-mobile --out "$WORK/opt.gltf"
"$BIN" inspect "$WORK/opt.gltf"

echo OK
