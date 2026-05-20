#!/usr/bin/env bash
# Regenerate the C header from the Rust FFI surface.
#
# Output: ios/Sources/CatetusViewerC/include/catetus_viewer_mobile.h
#
# Requires: `cargo install cbindgen`
set -euo pipefail
CORE="$(cd "$(dirname "$0")/../core" && pwd)"
OUT="$(cd "$(dirname "$0")/.." && pwd)/ios/Sources/CatetusViewerC/include/catetus_viewer_mobile.h"
cd "$CORE"
cbindgen --config cbindgen/cbindgen.toml --crate catetus-viewer-mobile --output "$OUT"
echo "wrote $OUT"
