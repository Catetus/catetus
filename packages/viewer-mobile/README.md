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
| iOS MSL compute kernels (`xcrun metal -c *.metal`)     | 5/5 green (macOS + iphoneos SDK) |
| iOS kernel-parity tests (`swift test`)                 | radix sort: 8/8 fixtures pass |
| iOS / iPadOS device build (Xcode + iOS SDK)            | PENDING — requires Xcode|
| iOS XCFramework (`scripts/build-xcframework.sh`)       | PENDING — requires Xcode|
| Kotlin source syntax check (regex + brace balance)     | OK                      |
| Android AAR build (`./gradlew :splatforge:assemble`)   | PENDING — requires NDK  |
| GLSL ES shaders compile (`glslangValidator`)           | PENDING                 |
| Compute kernels (decode / project / sort / scan)       | PORTED — see iOS section below |
| Visual screenshots (iOS / Android device)              | PENDING — requires physical device |

### iOS MSL compute kernels

The 5 critical compute paths from the WebGPU viewer are ported to MSL under
`ios/Sources/SplatforgeShaders/Shaders/`:

| File                       | Source                              | Entry points                                |
|----------------------------|-------------------------------------|---------------------------------------------|
| `SplatDecode.metal`        | `viewer/src/webgpu/decode.wgsl`     | `cs_decode`                                  |
| `RadixSort.metal`          | `viewer/src/webgpu/radix_sort.wgsl` | `cs_histogram`, `cs_scan`, `cs_scatter`      |
| `HistogramSubgroup.metal`  | (planned WGSL — `simd_sum` variant) | `cs_histogram_subgroup`                      |
| `ScanMultiblock.metal`     | (planned WGSL — 3-kernel chained)   | `cs_scan_reduce`, `cs_scan_spine`, `cs_scan_downsweep` |
| `ProjectGather.metal`      | `viewer/src/webgpu/decode.wgsl` (project pass) | `cs_project`, `cs_gather`         |

Two of those (`HistogramSubgroup`, `ScanMultiblock`) are upstream-planned
WGSL kernels that have not landed in `packages/viewer/src/webgpu/` yet —
the MSL versions here ship the canonical algorithm so the iOS path is not
gated on the WebGPU follow-up. The eventual WGSL ports should mirror them.

### Compile + test locally

```
xcrun -sdk iphoneos metal -c packages/viewer-mobile/ios/Sources/SplatforgeShaders/Shaders/SplatDecode.metal -o /tmp/SplatDecode.air
xcrun -sdk iphoneos metal -c packages/viewer-mobile/ios/Sources/SplatforgeShaders/Shaders/RadixSort.metal -o /tmp/RadixSort.air
xcrun -sdk iphoneos metal -c packages/viewer-mobile/ios/Sources/SplatforgeShaders/Shaders/HistogramSubgroup.metal -o /tmp/HistogramSubgroup.air
xcrun -sdk iphoneos metal -c packages/viewer-mobile/ios/Sources/SplatforgeShaders/Shaders/ScanMultiblock.metal -o /tmp/ScanMultiblock.air
xcrun -sdk iphoneos metal -c packages/viewer-mobile/ios/Sources/SplatforgeShaders/Shaders/ProjectGather.metal -o /tmp/ProjectGather.air

(cd packages/viewer-mobile/ios && swift test)
```

Requires Xcode + the Metal Toolchain asset (`xcodebuild -downloadComponent
MetalToolchain` if `metal -c` reports "missing Metal Toolchain"). Tests
auto-skip on hosts without a Metal device.

### Stability fix vs the WGSL source

The WGSL scatter relies on intra-workgroup `atomicAdd` "winner" order being
deterministic — true on Chrome/WebKit WebGPU but NOT on raw Apple Metal.
The MSL port replaces the atomic with a deterministic predecessor-count
scan in threadgroup memory so LSD radix stability holds. See the comment
block in `RadixSort.metal` and the assertion in `KernelParityTests.swift`.

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
