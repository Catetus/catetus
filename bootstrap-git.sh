#!/usr/bin/env bash
# bootstrap-git.sh — initialize git, stage the working tree, and create the
# first SplatForge commit. Idempotent and safe to re-run while WIP.
#
# This script intentionally does NOT push. The push step requires GitHub
# authentication, which lives in your shell session.
#
# Usage:
#   cd ~/Desktop/SplatForge
#   ./bootstrap-git.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BLUE=$(printf '\033[34m'); GREEN=$(printf '\033[32m'); YELLOW=$(printf '\033[33m'); RED=$(printf '\033[31m'); RESET=$(printf '\033[0m')
say() { printf "${BLUE}==>${RESET} %s\n" "$*"; }
ok()  { printf "${GREEN}✓${RESET}  %s\n" "$*"; }
warn(){ printf "${YELLOW}!${RESET}  %s\n" "$*"; }
die() { printf "${RED}✗ %s${RESET}\n" "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git not found — install the Xcode Command Line Tools first: xcode-select --install"

# ---------------------------------------------------------------------------
# 1. Init (or no-op if already a repo)
# ---------------------------------------------------------------------------
if [ -d .git ]; then
  ok "git repo already initialized"
else
  say "git init -b main"
  git init -b main
  ok "initialized"
fi

# ---------------------------------------------------------------------------
# 2. Local git identity (uses your global config if set; otherwise sets a sane default)
# ---------------------------------------------------------------------------
if ! git config user.name >/dev/null 2>&1; then
  git config user.name "Monte"
  ok "set local user.name = Monte"
fi
if ! git config user.email >/dev/null 2>&1; then
  git config user.email "monte@recruitplan.ai"
  ok "set local user.email = monte@recruitplan.ai"
fi
say "committing as $(git config user.name) <$(git config user.email)>"

# ---------------------------------------------------------------------------
# 3. Tidy — remove anything that shouldn't be committed
# ---------------------------------------------------------------------------
if [ -f bin/splatforge ]; then
  rm -f bin/splatforge
  ok "removed stale linux binary at bin/splatforge"
fi

# ---------------------------------------------------------------------------
# 4. Stage everything
# ---------------------------------------------------------------------------
say "staging changes"
git add -A

# Show a brief summary of what's staged
STAGED=$(git diff --cached --shortstat 2>/dev/null || true)
if [ -z "$STAGED" ]; then
  warn "nothing to commit — already up to date"
  echo
  say "remote setup (re-run if you change the URL):"
  echo "    git remote add origin https://github.com/montabano1/SplatForge.git"
  echo "    git push -u origin main"
  exit 0
fi
ok "staged: $STAGED"

# ---------------------------------------------------------------------------
# 5. Commit
# ---------------------------------------------------------------------------
COMMIT_MSG=$(cat <<'EOF'
feat: initial SplatForge v0.1.0 — Phase 0–2 + SplatBench v0

What's in this commit:

  * 10 SpecDD specs (SPEC-0001 .. SPEC-0010) covering IR, PLY, SPZ,
    glTF KHR Gaussian Splatting, analyze report, optimization passes,
    spatial streaming index, viewer SDK, visual diff, viewer parity.

  * Rust workspace (7 crates):
      splatforge-core      — canonical SplatIR + deterministic JSON report
      splatforge-ply       — Inria 3DGS binary-LE + ASCII reader/writer
      splatforge-spz       — v2 SPZ codec, round-trip parity tests
      splatforge-gltf      — glTF 2.0 + KHR_gaussian_splatting + GLB writer,
                             SF_spatial_streaming_index vendor ext
      splatforge-optimize  — 10-pass framework with 8 named presets
      splatforge-bench     — corpus runner
      splatforge-cli       — `splatforge` binary, 8 subcommands

  * TypeScript packages:
      @splatforge/viewer    — WebGPU + WebGL2 instanced-quad Gaussian
                              renderer with EWA fragment shading and
                              progressive chunked loading
      @splatforge/report-ui — visual-diff + parity HTML templates

  * Visual harness (tests/visual) — Playwright 4-renderer parity matrix
    and end-to-end `splatforge diff` Node helper.

  * Fixtures — 17 tiny PLY/SPZ/glTF assets including invalid variants
    (missing rotation, NaN, floater clusters, truncated payloads,
    unsupported KHR version).

  * SplatBench v0 — 7-scene benchmark corpus (2 real Mip-NeRF360
    anchors + 5 deterministic synthetic scenes) with an interactive
    HTML leaderboard. Median compression 21.75× (web-mobile) /
    24.24× (size-min) across the corpus.

  * Real-world demo — full pipeline on the canonical Inria 3DGS
    bonsai scene: 1.16M splats / 273 MB → 12 MB at 22.81× in 730 ms.

  * 40 Rust tests passing across the workspace.

  * CI workflows, project meta (CHANGELOG, CONTRIBUTING, INSTALL,
    CODE_OF_CONDUCT, GitHub issue/PR templates, .editorconfig).

Phase 3 (hosted API + OpenUSD) and Phase 4+ are explicit non-goals
for this commit; tracked under SPEC-0011, SPEC-0012, and the
roadmap section of the engineering plan.

Signed-off-by: Monte <monte@recruitplan.ai>
EOF
)

say "committing"
git commit --allow-empty-message -m "$COMMIT_MSG"
ok "$(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"

# ---------------------------------------------------------------------------
# 6. Remote + push hint
# ---------------------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  ok "remote 'origin' already set to: $(git remote get-url origin)"
else
  say "adding remote: git remote add origin https://github.com/montabano1/SplatForge.git"
  git remote add origin https://github.com/montabano1/SplatForge.git
  ok "remote added"
fi

cat <<EOF

${GREEN}Local repo is ready.${RESET}

Next, push to GitHub:

    cd ~/Desktop/SplatForge
    git push -u origin main

If you get an auth prompt:
  • HTTPS + Personal Access Token: paste the PAT when prompted for password
  • HTTPS + GitHub CLI:  brew install gh && gh auth login
  • SSH (cleaner):
        git remote set-url origin git@github.com:montabano1/SplatForge.git
        git push -u origin main

Then visit: https://github.com/montabano1/SplatForge
EOF
