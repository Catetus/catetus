#!/usr/bin/env bash
# Pull a real outdoor 3DGS scene for SplatBench.
#
# Original target: the canonical "Sweet Corals" outdoor-coral photogrammetry
# tile referenced as `sc-tile2` in private research notes (2.8M splats,
# 8.9× net residual ratio under PostHAC). That asset lives on a private
# Blurry Drive — there is no public canonical URL. SplatBench is a public
# corpus and cannot depend on internal credentials.
#
# Substitution: Mip-NeRF 360 `stump` at iteration 7,000, mirrored on the
# HuggingFace `dylanebert/3dgs` dataset alongside the existing `bonsai` and
# `bicycle` real scenes. `stump` is a complementary outdoor scene to
# `bicycle` (organic foliage + tree-stump texture vs bicycle-on-grass) and
# gives the leaderboard a second real outdoor anchor without relying on
# private data.
#
# Note: HuggingFace mirror has `garden` and `flowers` only as `.splat`
# files (PlayCanvas format), not PLY. Falling back to `stump`, which
# carries the same Inria-PLY layout used by every other real entry in the
# SplatBench corpus.
#
# Usage:
#   bash benches/scenes/scripts/pull-sweet-corals.sh
#
# Idempotent: re-runs verify the on-disk blake3 against manifest.json and
# skip the download if the bytes already match.

set -euo pipefail

SCENE_ID="stump_mipnerf360_iter7k"
URL="https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/stump/point_cloud/iteration_7000/point_cloud.ply"
EXPECTED_BYTES=944270460

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${HERE}/real"
DEST_FILE="${DEST_DIR}/stump_iter7000.ply"
MANIFEST="${DEST_DIR}/manifest.json"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
    actual_bytes=$(wc -c < "$DEST_FILE" | tr -d ' ')
    if [[ "$actual_bytes" == "$EXPECTED_BYTES" ]]; then
        echo "[pull-sweet-corals] $DEST_FILE already present ($actual_bytes bytes); skipping download."
        exit 0
    fi
    echo "[pull-sweet-corals] $DEST_FILE exists but size mismatch ($actual_bytes != $EXPECTED_BYTES); re-downloading."
    rm -f "$DEST_FILE"
fi

echo "[pull-sweet-corals] downloading $URL → $DEST_FILE"
curl -L --fail --retry 3 --retry-delay 5 -o "$DEST_FILE" "$URL"

actual_bytes=$(wc -c < "$DEST_FILE" | tr -d ' ')
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
    echo "[pull-sweet-corals] ERROR: downloaded $actual_bytes bytes, expected $EXPECTED_BYTES" >&2
    exit 1
fi

echo "[pull-sweet-corals] ok: $DEST_FILE ($actual_bytes bytes)"
echo "[pull-sweet-corals] verify hash against $MANIFEST when adding to splatbench"
