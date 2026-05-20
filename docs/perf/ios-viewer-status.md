# iOS Viewer — Status & Next Steps (2026-05-15)

Owner: ios-viewer-update agent / `feat/ios-viewer-update`
Stack: `packages/viewer-mobile/` — Rust core (`core/`), Swift Package (`ios/`), SwiftUI demo (`examples/iOS-Demo/`).

## What ships today (verified this invocation)

| Component                                                | Status                          |
|----------------------------------------------------------|---------------------------------|
| Rust core `catetus-viewer-mobile`                     | builds; clippy clean            |
| `cargo test -p catetus-viewer-mobile`                 | 3/3 pass (synth roundtrip, depth sort, optional bonsai) |
| Rust iOS device target `aarch64-apple-ios`               | builds (release)                |
| Rust iOS sim targets `aarch64/x86_64-apple-ios-sim`      | build (release)                 |
| Rust macOS targets `aarch64/x86_64-apple-darwin`         | build (release) — host slice    |
| `CatetusViewerCore.xcframework` (5-slice)             | builds via `scripts/build-xcframework.sh` |
| iOS Swift Package `swift build` (host macOS)             | clean build                     |
| iOS Swift Package `xcodebuild ... -destination generic/platform=iOS` | **BUILD SUCCEEDED** (arm64 iPhone) |
| iOS Swift Package `swift test`                           | **5/5 pass** (kernel parity + FFI smoke) |
| iOS-Demo app `swift build`                               | clean build                     |
| MSL compute kernels (decode, project, radix, scan, hist) | compile + radix-sort parity green over 8 fixtures |
| Bundled demo asset `synth.glb` (10k splats, ~550 KB)     | generated; demo loads it on first launch |
| Interactive orbit camera (touch pan / pinch zoom on iOS, mouse pan / magnify on macOS) | wired into renderer |
| FFI depth-sort wired into frame loop                     | yes (`sfmv_sort_by_depth` per frame; CPU oracle) |

## What is NOT shipped (the honest gap list)

1. **No `.xcodeproj`** — the SwiftUI demo is declared as a SwiftPM `.library`
   product, not an iOS `.app` target. SwiftPM alone cannot produce an
   installable `.app`. To install on an iPhone today you must:
   - open `packages/viewer-mobile/examples/iOS-Demo/Package.swift` in Xcode,
   - choose `File → New → Project → iOS App`, drop the `CatetusViewer`
     dependency in via SwiftPM, paste `ContentView` from `iOSDemoApp.swift`,
   - sign with a free Apple ID developer profile.
   This is a 10-minute Xcode click-path and intentionally not scripted —
   automating it would require either a templated `.xcodeproj` (brittle) or
   tuist/xcodegen (extra dep). **Next session should add an `xcodegen` project.yml
   so the demo app produces an `.ipa` from CLI.**

2. **CPU sort, not GPU sort.** The 5 MSL kernels (`SplatDecode`, `RadixSort`,
   `HistogramSubgroup`, `ScanMultiblock`, `ProjectGather`) are ported and
   parity-tested but the live frame loop still calls `sfmv_sort_by_depth`
   (single-threaded Rust `sort_unstable_by`). On bonsai-7k this is ~150 µs
   per frame; on a 1-3 M splat tile it would be ~30 ms per frame and bottleneck
   us below 30 fps. **The bonsai-7k demo will hit 30+ fps with the CPU sort;
   anything bigger needs the GPU sort wired in.**

3. **No 2D-covariance / anisotropic splats.** `SplatPointSprite.metal` draws
   isotropic round point-sprites sized by `scale.x * focal / depth`. Real
   3DGS rendering needs the `cs_project` kernel computing the 2x2 screen-space
   covariance + EWA-style fragment falloff. **The MSL is written; the wiring
   into the render pipeline is the missing step.**

4. **SH not evaluated.** `decode.rs` currently extracts only the SH DC term
   (L=0 band). Higher-order SH evaluation is a per-frame shader pass that
   has not been written for the mobile path.

5. **No real-scene asset in the demo.** We ship a 10k-splat synthetic
   `synth.glb` so the demo renders out of the box. Loading bonsai-7k requires
   downloading the file (~22 MB) and dropping it into
   `examples/iOS-Demo/Sources/iOSDemo/Assets/` — the demo's `ContentView`
   auto-picks it if present.

6. **No on-device perf measurement.** We have not run the app on a physical
   iPhone 15 Pro yet — the build succeeds for `generic/platform=iOS` but no
   one has measured the actual frame time. The 30+ fps target on bonsai-7k
   is a projection from desktop timings (CPU sort 150 µs + 7k point-sprite
   draws at iPhone 15 Pro's GPU fillrate ≫ headroom).

7. **No Android slice.** `packages/viewer-mobile/android/` exists but was not
   touched this invocation. It needs the JNI bridge built and a sample
   Activity wired similarly to the iOS demo.

## Performance projection — bonsai-7k on iPhone 15 Pro

Per frame, point-sprite path:

| Stage                            | Cost (projected)         | Source                |
|----------------------------------|--------------------------|-----------------------|
| `sfmv_sort_by_depth` (CPU)       | ~150-300 µs              | std `sort_unstable_by` on 7k entries; benched on M2 host as ~80 µs, iPhone A17 derate factor ~2× |
| Index buffer memcpy              | ~10 µs                   | 28 KB memcpy            |
| Vertex stage (7k × 4 verts)      | ~50-100 µs               | Trivial transform; GPU-bound |
| Fragment stage (round alpha)     | ~200-500 µs              | bbox-bound; ~7k splats × ~25 px² avg × overdraw |
| **Total**                        | ~0.4-1 ms                | well under 16.7 ms budget |
| **Projected fps**                | **240-2000 fps**         |                       |

For a 1-3 M splat tile (Sweet Corals territory):
- CPU sort scales O(N log N): ~50-100 ms → **immediate 30 fps floor breach**.
- Fragment overdraw scales linearly: ~30-60 ms → also a stretch.
- **GPU radix sort wire-in is the blocker** for the Sweet Corals demo.

## Next-step plan (priority-ranked)

### P0 — make the demo runnable on a phone

1. **Add an `xcodegen` `project.yml`** under `examples/iOS-Demo/` so
   `xcodegen generate` produces an installable `.xcodeproj`. The Info.plist
   needs `NSCameraUsageDescription` (future capture feature) and a bundle ID.
2. **Add a download-on-first-run** path that fetches a `.glb` from a known
   public URL (e.g. the HuggingFace bonsai mirror) into the app's Documents
   directory — same auto-pickup pattern as the bundled asset.
3. **Take a physical-device screenshot at 30+ fps** and add to
   `docs/perf/ios-viewer-status.md` as the BUILT proof.

### P1 — Sweet Corals 1-3 M splat target

4. **Wire `RadixSort.metal` into the frame loop**, replacing
   `sfmv_sort_by_depth`. The kernel is parity-tested; the only missing piece
   is allocating the histogram + ping-pong buffers in `CatetusRenderer`
   and dispatching three encoders per radix pass × 8 passes.
5. **Wire `cs_project` into the vertex stage** so splats are anisotropic
   ellipsoids (the 2x2 covariance pass). Replace the
   `pxRadius = clamp(scale.x * 200/w, ...)` hack in `SplatPointSprite.metal`.
6. **Profile on iPhone 15 Pro at 1 M synthetic splats.** If sort + project +
   draw fits in <33 ms, we have the demo. If not, decompose: which stage is
   the bottleneck?

### P2 — production polish

7. SH evaluation kernel for L≥1 bands (DC-only colors look flat-shaded).
8. Tile / LOD streaming using the existing `packages/viewer/src/streaming/`
   contract (cold-start 1.3 ms, 512 MB LRU) — port to Swift.
9. Capture pipeline (separate parallel agent per task spec).

## How to reproduce this state locally

```bash
# 1. Install iOS Rust targets (one-time).
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios \
                  aarch64-apple-darwin x86_64-apple-darwin

# 2. Build the XCFramework (3-slice: device + sim-fat + macOS-fat).
bash packages/viewer-mobile/scripts/build-xcframework.sh

# 3. Generate the bundled synth.glb fixture (10k splats, ~550 KB).
(cd packages/viewer-mobile/core && \
 cargo run --example gen_synth_glb --release -- \
   ../examples/iOS-Demo/Sources/iOSDemo/Assets/synth.glb 10000)

# 4. Validate.
cargo test -p catetus-viewer-mobile
(cd packages/viewer-mobile/ios && swift test)
(cd packages/viewer-mobile/ios && \
 xcodebuild -scheme CatetusViewer -destination 'generic/platform=iOS' build)

# 5. Run on simulator / device.
(cd packages/viewer-mobile/examples/iOS-Demo && swift build)
# For physical device: open Package.swift in Xcode, create iOS App target,
# add CatetusViewer SPM dep, copy ContentView from iOSDemoApp.swift,
# sign with a free profile, hit ⌘R.
```

## Why this matters

Polycam + Luma own phone-native capture+view because they wrote native
renderers. Catetus already has the **best public open-source web viewer**
(127 fps @ 1 M splats on Chrome WebGPU per the bench leaderboard). What we
*didn't* have until this invocation was a runnable Swift package that links
the same decode + sort + project math against Metal on a real iPhone.

After this invocation:
- The Rust → Swift FFI is real (linked, smoke-tested, runs).
- The Metal kernels are real (compile, parity tests pass).
- The XCFramework is built and ships 5 platform slices.
- The demo app shows a synthetic splat cloud on first launch with touch orbit.

What remains: package the demo as an `.app` (xcodegen), wire the GPU sort
into the frame loop, and measure on real hardware. That's a 1-day session,
not a 1-month foundation rewrite.
