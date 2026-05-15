#!/usr/bin/env bash
# Update distrib/homebrew/splatforge.rb with the version + SHA-256 hashes for
# a given release tag. Run AFTER the GitHub release workflow has finished
# publishing all five archives.
#
# Usage:
#   scripts/release/update-homebrew-formula.sh v0.2.0
#
# What it does:
#   1. Downloads SHASUMS256.txt from the release assets via `gh release`.
#   2. Parses the per-target SHA values for macOS and Linux archives.
#   3. Rewrites the `version "..."` line + the four `sha256 "..."` lines in
#      distrib/homebrew/splatforge.rb in-place.
#   4. Leaves the working tree dirty so the operator can commit + copy the
#      file into the homebrew-tap repo (see distrib/RELEASE.md).
#
# Documented mitigation for the project's biggest dist risk: stale SHAs in
# the Homebrew formula. Do not skip this script on release.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <tag>   (e.g. v0.2.0)" >&2
  exit 2
fi
TAG="$1"
VERSION="${TAG#v}"
REPO="splatforge/splatforge"
FORMULA_PATH="$(cd "$(dirname "$0")/../.." && pwd)/distrib/homebrew/splatforge.rb"

if ! command -v gh >/dev/null; then
  echo "ERROR: gh CLI required" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo ">> fetching SHASUMS256.txt for $TAG"
gh release download "$TAG" --repo "$REPO" \
  --pattern "SHASUMS256.txt" --dir "$tmp"

# Map our four Homebrew-relevant targets to their archive filenames.
declare -A targets=(
  [arm_mac]="splatforge-${TAG}-aarch64-apple-darwin.tar.gz"
  [x86_mac]="splatforge-${TAG}-x86_64-apple-darwin.tar.gz"
  [x86_lin]="splatforge-${TAG}-x86_64-unknown-linux-gnu.tar.gz"
  [arm_lin]="splatforge-${TAG}-aarch64-unknown-linux-gnu.tar.gz"
)

declare -A shas
for key in "${!targets[@]}"; do
  fname="${targets[$key]}"
  sha=$(awk -v f="$fname" '$2 == f { print $1 }' "$tmp/SHASUMS256.txt" || true)
  if [[ -z "$sha" || ${#sha} -ne 64 ]]; then
    echo "ERROR: no sha256 found for $fname in SHASUMS256.txt" >&2
    exit 1
  fi
  shas[$key]="$sha"
done

# In-place rewrite. Python helper to avoid sed-on-mac BSD/GNU divergence.
python3 - "$FORMULA_PATH" \
  "$VERSION" "${shas[arm_mac]}" "${shas[x86_mac]}" \
  "${shas[x86_lin]}" "${shas[arm_lin]}" <<'PY'
import re, sys, pathlib
path, version, arm_mac, x86_mac, x86_lin, arm_lin = sys.argv[1:]
text = pathlib.Path(path).read_text()
text = re.sub(r'version "[^"]+"', f'version "{version}"', text, count=1)

# Replace the four `sha256 "..."` lines in source order: arm-mac, x86-mac,
# x86-linux, arm-linux. The formula file is hand-written to keep that order
# stable; if you reshuffle it, update this list too.
order = [arm_mac, x86_mac, x86_lin, arm_lin]
def sub(m, _i=[0]):
    val = order[_i[0]]; _i[0] += 1
    return f'sha256  "{val}"'
text = re.sub(r'sha256\s+"[0-9a-fA-F]{64}"', sub, text, count=4)

pathlib.Path(path).write_text(text)
print(f"updated {path} -> version={version}")
PY

echo "Done. Next steps (see distrib/RELEASE.md):"
echo "  1. Review the diff: git diff distrib/homebrew/splatforge.rb"
echo "  2. Commit it to the main repo."
echo "  3. Copy the file into splatforge/homebrew-tap and push."
