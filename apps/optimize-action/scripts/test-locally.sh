#!/usr/bin/env bash
# Local smoke test for splatforge/optimize-action.
#
# Builds an ephemeral git repo, drops a small bonsai sample .ply into it as a
# "new file in PR", invokes src/index.js with the same env vars Actions would
# set, and asserts:
#   1. Process exits 0
#   2. compression-ratio is written to $GITHUB_OUTPUT
#   3. An output .glb URL is produced
#
# Hits the LIVE Fly API. Requires SPLATFORGE_API_KEY in env.
#
# Usage:
#   SPLATFORGE_API_KEY=sk_... ./scripts/test-locally.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

: "${SPLATFORGE_API_KEY:?SPLATFORGE_API_KEY must be set in env}"
: "${API_URL:=https://splatforge-api.fly.dev}"
: "${SOURCE_URL:=https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/iteration_7000/point_cloud.ply}"
: "${PRESET:=web-mobile}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die()   { red "FAIL: $*"; exit 1; }

bold "→ smoke-testing optimize-action against $API_URL"

# 1. Health check the API first so we get a fast, clear failure if it's down.
curl -sSf "$API_URL/healthz" >/dev/null || die "API healthcheck failed at $API_URL/healthz"
green "✓ API healthy"

# 2. Materialize a tiny test repo with a real PLY committed at HEAD~1, then
#    "modify" it at HEAD so `git diff HEAD~1..HEAD` lists it. This is what
#    discoverSplats() sees in a push-event scenario.
WORK=$(mktemp -d -t splatforge-action-smoke.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
bold "→ workdir: $WORK"

cd "$WORK"
git init -q
git config user.email "smoke@splatforge.local"
git config user.name "smoke"
mkdir -p scenes
# Initial commit: empty placeholder scene.
printf 'ply\nformat ascii 1.0\nend_header\n' > scenes/scene.ply
git add scenes/scene.ply
git commit -q -m "init"

# Now download the real bonsai sample as the "PR change".
bold "→ downloading bonsai sample from $SOURCE_URL"
curl -sSL "$SOURCE_URL" -o scenes/scene.ply
ls -lh scenes/scene.ply
git add scenes/scene.ply
git commit -q -m "swap in real splat"

HEAD_SHA=$(git rev-parse HEAD)
BASE_SHA=$(git rev-parse HEAD~1)

# 3. Synthesize the env Actions provides. No GITHUB_TOKEN ⇒ no PR comment,
#    but the optimize flow still exercises everything else end-to-end.
OUT_FILE=$(mktemp -t splatforge-output.XXXXXX)
trap 'rm -rf "$WORK" "$OUT_FILE"' EXIT

env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  GITHUB_WORKSPACE="$WORK" \
  GITHUB_SHA="$HEAD_SHA" \
  GITHUB_EVENT_NAME="push" \
  GITHUB_OUTPUT="$OUT_FILE" \
  GITHUB_REPOSITORY="splatforge/smoke" \
  INPUT_API_URL="$API_URL" \
  INPUT_API_KEY="$SPLATFORGE_API_KEY" \
  INPUT_PRESET="$PRESET" \
  INPUT_REGRESSION_THRESHOLD="1.0" \
  INPUT_COMMENT="false" \
  INPUT_TIMEOUT_SECONDS="300" \
  node "$ROOT/src/index.js"
status=$?

if [[ $status -ne 0 ]]; then
    die "action exited with status $status"
fi

bold "→ verifying outputs"
cat "$OUT_FILE"

grep -q '^compression-ratio<<' "$OUT_FILE" || die "missing compression-ratio output"
grep -q '^output-url<<' "$OUT_FILE"        || die "missing output-url output"
# Extract the JSON array from output-url.
url_block=$(awk '/^output-url<<EOF_/{flag=1; delim=$1; sub("output-url<<","",delim); next} flag && $0==delim {flag=0} flag {print}' "$OUT_FILE")
echo "output-url=$url_block"
[[ "$url_block" == \[*\]* ]] || die "output-url is not a JSON array: $url_block"
[[ "$url_block" != "[]" ]]    || die "output-url is empty — no splat was optimized"

green "✓ smoke test passed against $API_URL"
