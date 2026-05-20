#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Operator script: re-run the WebGPU viewer bench on Monte's home 4090 box
# over Tailscale. Used to fill in the PENDING cells in
# packages/viewer/COMPUTE-DECODE.md after the 8-bit radix + subgroup
# histogram changes land. The Darwin CI shell has no real WebGPU adapter
# and SwiftShader skews numbers 1.3-1.8x vs native, so we ship the code
# with PENDING perf cells rather than fake numbers from the wrong adapter.
#
# Pre-conditions:
#   - Tailscale up and `montespc` reachable (`tailscale status | grep montespc`).
#   - Catetus cloned on the 4090 box at $REMOTE_REPO (default
#     ~/Catetus). The branch with the perf changes must be checked out
#     and `pnpm install` done at least once.
#   - Chromium with WebGPU available on the remote box (playwright-core is
#     already a dev dep of tests/visual, so `pnpm install` is enough).
#
# Output:
#   - Local file: packages/viewer/bench/results-4090.json
#   - Stdout: a copy of the same JSON for quick eyeballing.
#
# Usage:
#   scripts/run-bench-on-4090.sh
#   scripts/run-bench-on-4090.sh --branch feat/subgroup-histogram

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-montespc}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/Catetus}"
BRANCH="${BRANCH:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --host)   REMOTE_HOST="$2"; shift 2 ;;
    --repo)   REMOTE_REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if ! tailscale status >/dev/null 2>&1; then
  echo "tailscale not running locally; aborting" >&2
  exit 1
fi

if ! tailscale status | grep -q "$REMOTE_HOST"; then
  echo "host $REMOTE_HOST not in tailscale status; aborting" >&2
  exit 1
fi

if tailscale status | grep "$REMOTE_HOST" | grep -q offline; then
  echo "host $REMOTE_HOST is offline; bench will fail. Wake the box and retry." >&2
  exit 1
fi

REMOTE_CMD="cd $REMOTE_REPO"
if [ -n "$BRANCH" ]; then
  REMOTE_CMD="$REMOTE_CMD && git fetch && git checkout $BRANCH && git pull --ff-only"
fi
REMOTE_CMD="$REMOTE_CMD && pnpm --filter @catetus/viewer run bench"

echo "running on $REMOTE_HOST: $REMOTE_CMD"
ssh "$REMOTE_HOST" "$REMOTE_CMD" | tee /dev/stderr | \
  awk '/^{/{flag=1} flag{print}' > packages/viewer/bench/results-4090.json

echo
echo "wrote packages/viewer/bench/results-4090.json"
echo "fill in the After columns in packages/viewer/COMPUTE-DECODE.md from this JSON."
