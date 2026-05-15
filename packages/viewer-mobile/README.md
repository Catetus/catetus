# SplatForge Mobile Viewer SDK

Native iOS + Android SDKs for rendering SplatForge `.glb` / `.spz` Gaussian-splat
assets without a WKWebView / Chrome WebView. Both wrappers are thin shells
around a shared Rust core (`splatforge-viewer-mobile`) that ports the
decode + sort + project math from the WebGPU viewer.

## Layout

```
packages/viewer-mobile/
├─ core/                       # Rust crate, C ABI (cbindgen)
│  ├─ src/{lib,decode,vertex,camera,math,sort,ffi}.rs
│  ├─ tests/glb_roundtrip.rs   # 3 tests pass against synthetic + bonsai
│  └─ cbindgen/cbindgen.toml
├─ ios/                        # SwiftPM package: SplatforgeViewer
│  ├─ Package.swift
│  ├─ Sources/SplatforgeViewer/{SplatforgeViewer,SplatforgeRenderer}.swift
│  ├─ Sources/SplatforgeViewer/Shaders/{SplatPointSprite,RadixSort,ProjectCovariance}.metal
│  └─ Sources/SplatforgeViewerC/include/splatforge_viewer_mobile.h
├─ android/splatforge/         # Kotlin AAR module: com.splatforge.viewer
│  ├─ build.gradle.kts
│  └─ src/main/{java,res/raw,jni,AndroidManifest.xml}
├─ examples/iOS-Demo/          # SwiftUI demo wiring SplatforgeViewer to bonsai-7k.glb
├─ examples/android-demo/      # Single-Activity demo bundling bonsai-7k.glb
└─ scripts/                    # build-xcframework, build-android-jniLibs, regen-headers
```

## What compiles in this PR

| Component                                              | Status                  |
|--------------------------------------------------------|-------------------------|
| Rust core (`cargo build -p splatforge-viewer-mobile`)  | builds, clippy clean    |
| Round-trip tests (`cargo test -p splatforge-viewer-mobile`) | 3/3 pass           |
| iOS Swift Package (`swift build` on macOS host)        | builds                  |
| iOS / iPadOS device build (Xcode + iOS SDK)            | PENDING — requires Xcode|
| iOS XCFramework (`scripts/build-xcframework.sh`)       | PENDING — requires Xcode|
| Kotlin source syntax check (regex + brace balance)     | OK                      |
| Android AAR build (`./gradlew :splatforge:assemble`)   | PENDING — requires NDK  |
| GLSL ES shaders compile (`glslangValidator`)           | PENDING                 |
| Compute kernels (radix sort + 2D-cov projection)       | STUBBED — follow-up PR  |
| Visual screenshots (iOS / Android device)              | PENDING — requires physical device |

## Cross-compile flow (run locally, NOT in this CI)

### iOS
```
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
packages/viewer-mobile/scripts/regen-headers.sh
packages/viewer-mobile/scripts/build-xcframework.sh
# then add the XCFramework as a binary target in Package.swift
```

### Android
```
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
cargo install cargo-ndk
packages/viewer-mobile/scripts/build-android-jniLibs.sh
(cd packages/viewer-mobile/android && ./gradlew :splatforge:assemble)
```

## Renderer Phase Plan

- **Phase 1 (this PR):** CPU decode + sort, GPU draws one instanced quad per
  splat with a soft-circle alpha. Looks correct enough to demo on stage; not
  yet ellipsoidal.
- **Phase 2:** Port `radix_sort.wgsl` → `RadixSort.metal` / `radix_sort.glsl`,
  move the sort onto the GPU.
- **Phase 3:** Port `projectCovariance2D` to a compute kernel, switch the
  vertex stage to project the 2x2 screen-space covariance for anisotropic
  splats. This brings parity with the WebGPU viewer.

## Asset note

The bonsai fixture (`bonsai-7k.glb`, ~22 MB) is not vendored into git.
Both demos ship a `.placeholder` you replace with the real asset. The
core's optional `decode_bonsai_when_available` test reads `$BONSAI_GLB`.
