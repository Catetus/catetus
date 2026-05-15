#!/usr/bin/env bash
# Pull the cluster_fly LOD ladder (S → XXL, 5 scenes) for SplatBench.
#
# Source: Dany Bittel (https://www.danybittel.ch / https://x.com/DanyBittel)
# Subject: cluster fly (Pollenia / Herbstfliege), found South of Switzerland,
#          indoor, September 2025.
# License: CC-BY-4.0 — https://creativecommons.org/licenses/by/4.0/
# Attribution requirement: "Dany Bittel (CC-BY-4.0) — www.danybittel.ch"
#
# The 5 LODs (25k → 3.5M splats) give SplatBench its first **indoor
# close-up macro** capture, complementing the existing Mip-NeRF 360 anchors
# (bonsai indoor-real-estate, bicycle/stump outdoor-scene) and the
# synthetic proxy ladder. The shared subject across the 5 LODs lets the
# leaderboard measure how compression ratios scale with splat count on a
# single capture.
#
# The original assets live on a private host (Dany's hand-delivered drop
# under cluster.fly/). When the canonical public URL is decided this script
# will be updated to curl from there; until then the script verifies bytes
# against the manifest and prints the expected location for manual drop.
#
# Usage:
#   bash benches/scenes/scripts/pull-cluster-fly.sh [SRC_DIR]
#
# SRC_DIR defaults to $HOME/Downloads/cluster.fly. The script copies the
# 5 spaced-name PLYs into benches/scenes/real/cluster_fly_{s,m,l,xl,xxl}.ply
# and stages symlinks under benches/scenes/ for the harness.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
REAL_DIR="${HERE}/real"
STAGE_DIR="${HERE}"
SRC_DIR="${1:-${HOME}/Downloads/cluster.fly}"

mkdir -p "$REAL_DIR"

# Map: LOD → (source-filename, expected-bytes).
declare -a LODS=(s m l xl xxl)
declare -A SRC_NAME=(
    [s]="cluster fly S.ply"
    [m]="cluster fly M.ply"
    [l]="cluster fly L.ply"
    [xl]="cluster fly XL.ply"
    [xxl]="cluster fly XXL.ply"
)
declare -A EXPECTED=(
    [s]=6049505
    [m]=34367146
    [l]=71263622
    [xl]=147308014
    [xxl]=827606099
)

for lod in "${LODS[@]}"; do
    dst="${REAL_DIR}/cluster_fly_${lod}.ply"
    if [[ -f "$dst" ]]; then
        actual=$(wc -c < "$dst" | tr -d ' ')
        if [[ "$actual" == "${EXPECTED[$lod]}" ]]; then
            echo "[pull-cluster-fly] $dst already present (${actual} bytes); skipping."
        else
            echo "[pull-cluster-fly] $dst size mismatch (${actual} != ${EXPECTED[$lod]}); re-copying."
            rm -f "$dst"
        fi
    fi
    if [[ ! -f "$dst" ]]; then
        src="${SRC_DIR}/${SRC_NAME[$lod]}"
        if [[ ! -f "$src" ]]; then
            echo "[pull-cluster-fly] ERROR: source not found: $src" >&2
            echo "[pull-cluster-fly] Provide SRC_DIR pointing at the cluster.fly drop." >&2
            exit 1
        fi
        cp "$src" "$dst"
        actual=$(wc -c < "$dst" | tr -d ' ')
        if [[ "$actual" != "${EXPECTED[$lod]}" ]]; then
            echo "[pull-cluster-fly] ERROR: copied $actual bytes, expected ${EXPECTED[$lod]}" >&2
            exit 1
        fi
        echo "[pull-cluster-fly] copied $dst (${actual} bytes)"
    fi
    # Stage symlink for the bench harness (benches/scenes/<id>.ply).
    ln -sf "real/cluster_fly_${lod}.ply" "${STAGE_DIR}/cluster_fly_${lod}.ply"
done

echo "[pull-cluster-fly] ok — 5 LODs ready under ${REAL_DIR}/"
echo "[pull-cluster-fly] attribution: Dany Bittel (CC-BY-4.0) — www.danybittel.ch"
