#!/usr/bin/env bash
# Composite build: produces a single deployable `packages/website/dist/` that
# contains BOTH the landing page (at /) and the interactive viewer-app
# (at /viewer/). Same origin → the website's <iframe src="/viewer/"> works
# without CORS or COOP/COEP gymnastics.
#
# Idempotent. Safe to re-run.
#
# Used by:
#   - packages/website/vercel.json   (buildCommand)
#   - packages/website/wrangler.toml (build.command)
#   - local verification before a deploy day
#
# Stealth: this script never runs `vercel`, `wrangler`, `gh-pages`, or any
# other publish command. It only builds + composites.

set -euo pipefail

# --- locate repo root ------------------------------------------------------
# This script lives at packages/website/build-composite.sh. Resolve repo root
# relative to the script so it works from any cwd (CI, IDE, manual).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WEBSITE_DIR="${REPO_ROOT}/packages/website"
VIEWER_DIR="${REPO_ROOT}/packages/viewer-app"
WEBSITE_DIST="${WEBSITE_DIR}/dist"
VIEWER_DIST="${VIEWER_DIR}/dist"

echo "==> repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

# --- pnpm install (idempotent) --------------------------------------------
# CI provides --frozen-lockfile via installCommand. Locally we don't force it
# because contributors may have edited package.json without committing lock.
if [[ "${CI:-}" == "true" ]] || [[ "${VERCEL:-}" == "1" ]] || [[ "${CF_PAGES:-}" == "1" ]]; then
  echo "==> pnpm install --frozen-lockfile (CI)"
  pnpm install --frozen-lockfile
else
  if [[ ! -d "${REPO_ROOT}/node_modules" ]]; then
    echo "==> pnpm install (local, no node_modules yet)"
    pnpm install
  else
    echo "==> skipping pnpm install (local, node_modules present)"
  fi
fi

# --- build viewer-app first -----------------------------------------------
echo "==> building @catetus/viewer-app"
pnpm -C "${VIEWER_DIR}" build

if [[ ! -f "${VIEWER_DIST}/index.html" ]]; then
  echo "ERROR: viewer-app build did not produce ${VIEWER_DIST}/index.html" >&2
  exit 1
fi

# --- build website --------------------------------------------------------
echo "==> building @catetus/website"
pnpm -C "${WEBSITE_DIR}" build

if [[ ! -f "${WEBSITE_DIST}/index.html" ]]; then
  echo "ERROR: website build did not produce ${WEBSITE_DIST}/index.html" >&2
  exit 1
fi

# --- composite: viewer-app into website's dist under /viewer/ -------------
# Wipe any prior /viewer/ to avoid stale files from previous composites.
echo "==> compositing viewer-app into ${WEBSITE_DIST}/viewer/"
rm -rf "${WEBSITE_DIST}/viewer"
mkdir -p "${WEBSITE_DIST}/viewer"

# Prefer rsync for speed + perms; fall back to cp -R if rsync absent.
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${VIEWER_DIST}/" "${WEBSITE_DIST}/viewer/"
else
  cp -R "${VIEWER_DIST}/." "${WEBSITE_DIST}/viewer/"
fi

# --- verification ---------------------------------------------------------
echo "==> verifying composite layout"
fail=0
for path in \
  "${WEBSITE_DIST}/index.html" \
  "${WEBSITE_DIST}/viewer/index.html" \
  "${WEBSITE_DIST}/demos/splatbench_lowlight.glb"; do
  if [[ -e "${path}" ]]; then
    size="$(wc -c < "${path}" | tr -d ' ')"
    echo "  OK  ${path#${REPO_ROOT}/} (${size} bytes)"
  else
    echo "  MISSING  ${path#${REPO_ROOT}/}" >&2
    fail=1
  fi
done

if [[ ${fail} -ne 0 ]]; then
  echo "ERROR: composite verification failed" >&2
  exit 1
fi

# Show top-level layout so deploy logs are debuggable.
echo "==> dist/ top-level:"
ls -la "${WEBSITE_DIST}" | sed 's/^/    /'
echo "==> dist/viewer/ top-level:"
ls -la "${WEBSITE_DIST}/viewer" | sed 's/^/    /'

# Total dist size (helps stay under Cloudflare Pages's 25 MB per file / overall
# project quotas, and Vercel's 100 MB per deployment soft limits).
total_kb="$(du -sk "${WEBSITE_DIST}" | awk '{print $1}')"
echo "==> total dist size: ${total_kb} KB"

echo "==> composite build OK"
