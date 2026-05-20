#!/usr/bin/env bash
# Build the XCFramework that ships the Rust staticlib + C header.
#
# Required toolchains:
#   - Xcode 15+
#   - rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios \
#                       aarch64-apple-darwin x86_64-apple-darwin
#
# Output: packages/viewer-mobile/ios/CatetusViewerCore.xcframework
#
# Slices included:
#   - ios-arm64                       (iPhone / iPad device)
#   - ios-arm64_x86_64-simulator      (iOS Simulator, Apple Silicon + Intel host)
#   - macos-arm64_x86_64              (host macOS for `swift build` + `swift test`)
set -euo pipefail

CORE="$(cd "$(dirname "$0")/../core" && pwd)"
OUT="$(cd "$(dirname "$0")/.." && pwd)/ios/CatetusViewerCore.xcframework"
HEADERS="$(dirname "$0")/../ios/Sources/CatetusViewerC/include"

for TARGET in aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios aarch64-apple-darwin x86_64-apple-darwin; do
  (cd "$CORE" && cargo build --release --target "$TARGET")
done

TARGET_DIR="$CORE/../../../target"
DEVICE="$TARGET_DIR/aarch64-apple-ios/release/libcatetus_viewer_mobile.a"
SIM_ARM="$TARGET_DIR/aarch64-apple-ios-sim/release/libcatetus_viewer_mobile.a"
SIM_X86="$TARGET_DIR/x86_64-apple-ios/release/libcatetus_viewer_mobile.a"
MAC_ARM="$TARGET_DIR/aarch64-apple-darwin/release/libcatetus_viewer_mobile.a"
MAC_X86="$TARGET_DIR/x86_64-apple-darwin/release/libcatetus_viewer_mobile.a"

mkdir -p "$(dirname "$SIM_ARM")"
SIM_FAT="$(dirname "$SIM_ARM")/libcatetus_viewer_mobile_simfat.a"
lipo -create "$SIM_ARM" "$SIM_X86" -output "$SIM_FAT"

mkdir -p "$(dirname "$MAC_ARM")"
MAC_FAT="$(dirname "$MAC_ARM")/libcatetus_viewer_mobile_macfat.a"
lipo -create "$MAC_ARM" "$MAC_X86" -output "$MAC_FAT"

rm -rf "$OUT"
xcodebuild -create-xcframework \
  -library "$DEVICE"  -headers "$HEADERS" \
  -library "$SIM_FAT" -headers "$HEADERS" \
  -library "$MAC_FAT" -headers "$HEADERS" \
  -output "$OUT"
echo "XCFramework: $OUT"
