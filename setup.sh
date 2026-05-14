#!/usr/bin/env bash
# SplatForge first-time setup. Idempotent — safe to re-run.
#
# Usage:
#   ./setup.sh             # full setup (Rust + Node + build)
#   ./setup.sh --no-js     # skip the JS workspace
#   ./setup.sh --quick     # skip the release build (cargo check only)
#
set -euo pipefail

SKIP_JS=0
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --no-js)  SKIP_JS=1 ;;
    --quick)  QUICK=1 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0 ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2 ;;
  esac
done

BLUE=$(printf '\033[34m'); GREEN=$(printf '\033[32m'); YELLOW=$(printf '\033[33m'); RESET=$(printf '\033[0m')
say() { printf "${BLUE}==>${RESET} %s\n" "$*"; }
ok()  { printf "${GREEN}✓${RESET}  %s\n" "$*"; }
warn(){ printf "${YELLOW}!${RESET}  %s\n" "$*"; }

# ----------------------------------------------------------------------------
# Rust
# ----------------------------------------------------------------------------
say "Checking Rust..."
if command -v cargo >/dev/null 2>&1; then
  ok "cargo $(cargo --version | awk '{print $2}')"
else
  warn "cargo not found — installing via rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  ok "cargo $(cargo --version | awk '{print $2}')"
fi

# ----------------------------------------------------------------------------
# Node + pnpm
# ----------------------------------------------------------------------------
if [ "$SKIP_JS" -eq 0 ]; then
  say "Checking Node..."
  if command -v node >/dev/null 2>&1; then
    NODE_MAJOR=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 20 ]; then
      ok "node $(node --version)"
    else
      warn "node $(node --version) is older than 20. Please upgrade."
      exit 1
    fi
  else
    warn "node not found. Install Node 20+ from https://nodejs.org/ or via brew/apt."
    warn "Re-run ./setup.sh after installing, or pass --no-js to skip."
    exit 1
  fi

  say "Checking pnpm..."
  if command -v pnpm >/dev/null 2>&1; then
    ok "pnpm $(pnpm --version)"
  else
    warn "pnpm not found — installing"
    npm install -g pnpm@9
    ok "pnpm $(pnpm --version)"
  fi
fi

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
say "Building splatforge CLI..."
if [ "$QUICK" -eq 1 ]; then
  cargo check --workspace
  ok "cargo check passed"
else
  cargo build --release -p splatforge-cli
  ok "built target/release/splatforge"
fi

if [ "$SKIP_JS" -eq 0 ]; then
  say "Installing JS dependencies..."
  pnpm install
  say "Building JS packages..."
  pnpm -r --if-present run build
  ok "JS packages built"
fi

# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if [ "$QUICK" -eq 0 ]; then
  say "Smoke testing CLI..."
  ./target/release/splatforge analyze fixtures/tiny/basic_binary.ply --pretty > /tmp/_sf_smoke.json
  if grep -q '"splatCount": 3' /tmp/_sf_smoke.json; then
    ok "splatforge analyze works"
  else
    warn "smoke test produced unexpected output. See /tmp/_sf_smoke.json"
    exit 1
  fi
fi

cat <<EOF

${GREEN}SplatForge is ready.${RESET}

Try:
  ./target/release/splatforge analyze fixtures/tiny/basic_binary.ply --pretty
  ./target/release/splatforge optimize fixtures/tiny/basic_binary.ply --preset web-mobile --out /tmp/scene.gltf
  ./target/release/splatforge inspect /tmp/scene.gltf
  make help

Docs: README.md, INSTALL.md, docs/getting-started.md
SplatBench v0 leaderboard: open benches/reports/splatbench-v0.html in your browser

EOF
