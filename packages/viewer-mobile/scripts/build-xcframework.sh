#!/usr/bin/env bash
# Build the iOS XCFramework that ships the Rust staticlib + C header.
#
# Required toolchains (NOT available on the SplatForge CI host — this is a
# local / future-Buildkite step):
#   - Xcode 15+
#   - rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
#
# Output: packages/viewer-mobile/ios/SplatforgeViewerCore.xcframework
#
# Status: PENDING — wire-up depends on macOS+Xcode in CI.
set -euo pipefail

CORE="$(cd "$(dirname "$0")/../core" && pwd)"
OUT="$(cd "$(dirname "$0")/.." && pwd)/ios/SplatforgeViewerCore.xcframework"

for TARGET in aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios; do
  (cd "$CORE" && cargo build --release --target "$TARGET")
done

DEVICE="$CORE/../../../target/aarch64-apple-ios/release/libsplatforge_viewer_mobile.a"
SIM_ARM="$CORE/../../../target/aarch64-apple-ios-sim/release/libsplatforge_viewer_mobile.a"
SIM_X86="$CORE/../../../target/x86_64-apple-ios/release/libsplatforge_viewer_mobile.a"

mkdir -p "$(dirname "$SIM_ARM")"
SIM_FAT="$(dirname "$SIM_ARM")/libsplatforge_viewer_mobile_simfat.a"
lipo -create "$SIM_ARM" "$SIM_X86" -output "$SIM_FAT"

rm -rf "$OUT"
xcodebuild -create-xcframework \
  -library "$DEVICE"  -headers "$(dirname "$0")/../ios/Sources/SplatforgeViewerC/include" \
  -library "$SIM_FAT" -headers "$(dirname "$0")/../ios/Sources/SplatforgeViewerC/include" \
  -output "$OUT"
echo "XCFramework: $OUT"
