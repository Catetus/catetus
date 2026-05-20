#!/usr/bin/env bash
# Build the Android .so files for all four ABIs and copy them into the AAR
# source tree at `android/catetus/src/main/jniLibs/<abi>/`.
#
# Required toolchains (NOT available on the Catetus CI host):
#   - Android NDK r26+
#   - `cargo install cargo-ndk`
#   - `rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android`
#
# Status: PENDING — requires NDK in CI.
set -euo pipefail

CORE="$(cd "$(dirname "$0")/../core" && pwd)"
LIBS="$(cd "$(dirname "$0")/.." && pwd)/android/catetus/src/main/jniLibs"

cd "$CORE"
cargo ndk \
  -t arm64-v8a \
  -t armeabi-v7a \
  -t x86_64 \
  -o "$LIBS" \
  build --release
echo "jniLibs: $LIBS"
