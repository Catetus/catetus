#!/usr/bin/env bash
# Configure this clone to use Catetus's repo-tracked githooks.
#
# Idempotent — safe to run repeatedly. Opt-in: not invoked automatically by
# setup.sh / bootstrap because there are legitimate flows (e.g. cleaning up
# leaked-doc history) where you need to bypass the hook for the same audit-
# cleanup work that motivated installing it.
#
# Run once after cloning:
#     ./scripts/install-githooks.sh

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "install-githooks: not inside a git repo" >&2
  exit 1
}

cd "${repo_root}"

if [ ! -d .githooks ]; then
  echo "install-githooks: .githooks/ directory missing — is this the right repo?" >&2
  exit 1
fi

# Make sure hooks are executable (covers checkouts that lost the +x bit,
# e.g. Windows / fresh tar extracts).
chmod +x .githooks/* 2>/dev/null || true

current="$(git config --get core.hooksPath 2>/dev/null || echo '')"
if [ "${current}" = ".githooks" ]; then
  echo "install-githooks: already configured (core.hooksPath=.githooks)"
  exit 0
fi

git config core.hooksPath .githooks
echo "install-githooks: set core.hooksPath=.githooks"
echo "install-githooks: installed hooks:"
ls -1 .githooks | sed 's/^/  - /'
