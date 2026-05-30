#!/usr/bin/env bash
# Build the Catetus .mcpb bundle for Claude Desktop one-click install.
#
# What this produces:
#   dist-bundle/catetus-<version>.mcpb   — zip-archive MCPB bundle ready for distribution
#   dist-bundle/build/                    — staging directory (regenerated each run)
#
# Prerequisites:
#   * Node 20+ on PATH
#   * The MCP server built (packages/mcp/dist/index.js must exist).
#     Run `npm run build` in packages/mcp/ first, OR pass --build to do it here.
#   * @anthropic-ai/mcpb CLI available via npx (fetched on demand).
#
# Usage:
#   bash scripts/build-mcpb.sh              # build bundle from existing dist/
#   bash scripts/build-mcpb.sh --build      # run `npm run build` first
#   bash scripts/build-mcpb.sh --sign       # also sign the .mcpb (requires Apple cert)
#   bash scripts/build-mcpb.sh --validate   # validate the manifest, no pack

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_BUNDLE="$PKG_DIR/dist-bundle"
BUILD_DIR="$DIST_BUNDLE/build"
MANIFEST_SRC="$PKG_DIR/mcpb-manifest.json"

# ── Flags ───────────────────────────────────────────────────────────────
DO_BUILD=0
DO_SIGN=0
DO_VALIDATE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --build)        DO_BUILD=1 ;;
    --sign)         DO_SIGN=1 ;;
    --validate)     DO_VALIDATE_ONLY=1 ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

# ── Sanity ──────────────────────────────────────────────────────────────
if [[ ! -f "$MANIFEST_SRC" ]]; then
  echo "ERROR: $MANIFEST_SRC not found." >&2
  exit 1
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "==> Building MCP server (npm run build in $PKG_DIR)"
  ( cd "$PKG_DIR" && npm run build )
fi

if [[ ! -f "$PKG_DIR/dist/server-stdio.js" && "$DO_VALIDATE_ONLY" -eq 0 ]]; then
  echo "ERROR: $PKG_DIR/dist/server-stdio.js not found. Run with --build, or run 'npm run build' first." >&2
  exit 1
fi

# ── Version from manifest ───────────────────────────────────────────────
VERSION=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$MANIFEST_SRC','utf8')).version)")
BUNDLE_NAME="catetus-${VERSION}.mcpb"
echo "==> Catetus MCP v${VERSION} → ${BUNDLE_NAME}"

# ── Validate manifest ───────────────────────────────────────────────────
echo "==> Validating manifest against MCPB schema"
( cd "$PKG_DIR" && npx -y @anthropic-ai/mcpb validate "$MANIFEST_SRC" )

if [[ "$DO_VALIDATE_ONLY" -eq 1 ]]; then
  echo "Validate-only mode; exiting after schema check."
  exit 0
fi

# ── Stage ───────────────────────────────────────────────────────────────
echo "==> Staging $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/server"

# Manifest must live at the bundle root and be named manifest.json
cp "$MANIFEST_SRC" "$BUILD_DIR/manifest.json"

# Copy built JS + sourcemaps
cp -R "$PKG_DIR/dist/." "$BUILD_DIR/server/"

# Optional icon (provide one if you have it; mcpb tolerates missing icons)
if [[ -f "$PKG_DIR/icon.png" ]]; then
  cp "$PKG_DIR/icon.png" "$BUILD_DIR/icon.png"
else
  echo "    (no icon.png at $PKG_DIR/icon.png — bundle will install without one)"
fi

# Bundle the production node_modules so Claude Desktop's bundled Node can resolve them.
# We don't have a package.json at MCPB-build time in this dir; copy from packages/mcp/node_modules
# if present, else instruct the user.
if [[ -d "$PKG_DIR/node_modules" ]]; then
  echo "==> Copying node_modules into bundle (this is the safest approach)"
  # Use cp -RL to dereference symlinks (pnpm/yarn workspaces use symlinks)
  cp -RL "$PKG_DIR/node_modules" "$BUILD_DIR/server/node_modules" || \
    cp -R  "$PKG_DIR/node_modules" "$BUILD_DIR/server/node_modules"
else
  echo "WARN: $PKG_DIR/node_modules missing. Either:"
  echo "      (a) cd $PKG_DIR && npm ci --omit=dev    (preferred)"
  echo "      (b) configure tsup to bundle deps into dist/ (then this step is a no-op)"
fi

# ── Pack ────────────────────────────────────────────────────────────────
echo "==> Packing → $DIST_BUNDLE/$BUNDLE_NAME"
( cd "$BUILD_DIR" && npx -y @anthropic-ai/mcpb pack . "$DIST_BUNDLE/$BUNDLE_NAME" )

# ── Sign (optional) ─────────────────────────────────────────────────────
if [[ "$DO_SIGN" -eq 1 ]]; then
  echo "==> Signing"
  npx -y @anthropic-ai/mcpb sign "$DIST_BUNDLE/$BUNDLE_NAME"
fi

# ── Report ──────────────────────────────────────────────────────────────
SIZE=$(du -h "$DIST_BUNDLE/$BUNDLE_NAME" | cut -f1)
echo ""
echo "✓ Built $DIST_BUNDLE/$BUNDLE_NAME (${SIZE})"
echo ""
echo "Install on Claude Desktop: drag the .mcpb file onto the Claude Desktop window."
echo "Distribute: upload to https://github.com/catetus/catetus-mcp/releases/tag/v${VERSION}"
