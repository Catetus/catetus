# CatetusViewer (iOS / iPadOS / macOS Swift Package)

Native SwiftUI splat viewer. Renders Catetus `.glb` / `.spz` Gaussian-splat
assets with Metal — no WKWebView, no Chrome WebView.

## Targets

| Target              | Role                                                          |
|---------------------|---------------------------------------------------------------|
| `CatetusShaders` | Resource-only Swift target hosting the MSL compute kernels.   |
| `CatetusViewerC` | Cbindgen-generated C ABI shim onto the Rust `catetus-viewer-mobile` staticlib. |
| `CatetusViewer`  | SwiftUI view + `MTKView` renderer.                            |
| `CatetusViewerTests` | Kernel-parity tests for the MSL compute kernels.          |

## Shaders

Compute kernels live in `Sources/CatetusShaders/Shaders/`:

- `SplatDecode.metal` — `cs_decode` — port of `decode.wgsl`. 8-bit canonical IR → float `DecodedSplat[]`. One thread per splat, tg=256.
- `HistogramSubgroup.metal` — `cs_histogram_subgroup` — SIMD-group-accelerated histogram using `simd_sum`.
- `ScanMultiblock.metal` — `cs_scan_reduce` / `cs_scan_spine` / `cs_scan_downsweep` — 3-kernel chained exclusive prefix sum.
- `RadixSort.metal` — `cs_histogram` / `cs_scan` / `cs_scatter` — 4-bit LSD radix sort (8 passes). Per-pass stability via deterministic intra-tg predecessor count (see file header for why we don't use `atomic_fetch_add` here).
- `ProjectGather.metal` — `cs_project` / `cs_gather` — port of the project pass from `decode.wgsl`; `cs_gather` reorders the per-instance buffer using the sorted index buffer.
- `SplatPointSprite.metal` — Phase-1 vertex + fragment for point-sprite renderer (already shipped).

## Validation

### Compile each kernel against the iOS SDK

```
for f in SplatDecode RadixSort HistogramSubgroup ScanMultiblock ProjectGather; do
  xcrun -sdk iphoneos metal -c Sources/CatetusShaders/Shaders/$f.metal -o /tmp/$f.air
done
```

If `metal` reports "missing Metal Toolchain", run
`xcodebuild -downloadComponent MetalToolchain` once and retry. The asset is
~700 MB.

### Run the kernel-parity tests

```
swift test
```

`KernelParityTests.testRadixSortParityAcrossFixtures` compiles
`RadixSort.metal` at runtime, dispatches it over 8 fixture key arrays
(tiny / dense_random / nearly_sorted / reverse_sorted / heavy_duplicates /
sparse / large_uniform / small_edge), and asserts the GPU output matches a
Swift `Array.sorted` oracle key-by-key AND payload-by-payload. The oracle
mirrors the back-to-front ordering produced by `core/src/sort.rs` after the
`cs_project` bit-flip.

On a host with no Metal device the tests `XCTSkip` rather than fail.

## XCFramework build (local-only)

`swift build` type-checks against the Rust core's headers but the actual
staticlib is not vendored in this PR; the renderer's symbols
(`ctmv_decode_glb`, etc.) link only via the XCFramework. To produce that:

```
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
../scripts/regen-headers.sh
../scripts/build-xcframework.sh
# then add CatetusViewerCore.xcframework as a binaryTarget in Package.swift
```
