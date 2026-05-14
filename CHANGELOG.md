# Changelog

All notable changes to SplatForge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ML Score column on SplatBench leaderboard.** A new `ML Score (pro)` column
  appears on `benches/reports/splatbench-v0.{md,html,json}` and on the live
  Astro site at `splatforge.vercel.app`. Values are computed by
  `splatforge-pro` — a proprietary build that scores rendered vs. baseline
  frames with a splat-aware perceptual metric tuned for Gaussian-splat failure
  modes (floater ghosting, edge breakdown, specular pop). Values are
  **published**; reproducing them requires the proprietary binary. This is
  the first column on the benchmark that is asymmetrically reproducible —
  corpus + formats + viewer stay open, the differentiated scoring stays
  private. First values: `floater_proxy / size-min` 88.65% ML
  (correctly fingerprints the floater failure ΔE94 also flags),
  `bicycle / *` ~88.6% ML, the rest 99.84-99.97%.
- **`apps/web` leaderboard** now actually renders the fidelity columns from
  the v0.1.1 ΔE94 numbers — the prior wiring referenced a legacy
  `fidelity.deltaE94` shape that never matched what `splatbench-update.mjs`
  emits. `SceneFidelity` aligned to the real `webMobile`/`sizeMin`
  sub-objects, `hasAnyMlScore()` + `fidelityFor()` helpers added.
- **SPEC-0013** (`KHR_mesh_quantization` for splat attributes) implemented. The
  Rust glTF writer now emits POSITION as `UNSIGNED_SHORT` and
  `_SCALE` / `_OPACITY` / `_COLOR_DC` as `UNSIGNED_BYTE` (normalized, with
  per-component `min`/`max`) when `WriteOpts::quantize == true`. `splatforge
  optimize` flips this on for the web-targeted presets (`web-mobile`,
  `web-desktop`, `quest-browser`, `visionos-preview`, `thumbnail-preview`,
  `size-min`); `lossless-repack` and `quality-max` keep f32 accessors so
  byte-identical round-trips remain possible. `KHR_mesh_quantization` is
  advertised in `extensionsUsed` only — never `extensionsRequired` — so
  legacy viewers still load the asset (they just render un-dequantized
  integer values). Expected wire-size impact on the bonsai `web-mobile`
  scene: glTF buffer **59.4 MB → 30.9 MB** (1.9×), closing the gap to SPZ
  from ~5× to ~2.5×.
- **`@splatforge/viewer`** learns to decode the new integer accessors.
  `SoaAttributeSlice` now carries `componentType` / `normalized` / `min` /
  `max`; the SoA decoder dequantizes u16 and u8 attributes against the
  accessor metadata before re-interleaving to the existing `DecodedSplat`
  layout.
- **`apps/api`** ships a `README.md` documenting the dev-mode story — a
  contributor can `cargo run -p splatforge-api` from a fresh checkout and
  hit `/healthz` + `POST /v1/jobs` without provisioning Vercel Blob or
  Modal first (the service degrades to stub backends + an in-memory job
  store when env vars are absent).
- GitHub repo metadata polished: topic tags (`gaussian-splatting`, `webgpu`,
  `3d`, `splat`, `rust`, `gltf`, `splatforge`, `computer-graphics`),
  description, and homepage URL.

## [0.1.1] — 2026-05-14

Closes the v0.1.0 "visual fidelity pending" gap and ships the public landing page.

### Added

- **Visual-fidelity benchmark** — every scene in SplatBench is now measured for
  perceptual degradation (CIE ΔE94 / pixelmatch / per-block SSIM) by rendering
  8 deterministic orbit frames through `@splatforge/viewer` in headless
  Chromium and comparing each non-baseline preset to `lossless-repack`. New
  runner at `tests/visual/scripts/splatbench-fidelity.mjs`, new report at
  `benches/reports/fidelity-v0.json`. The SplatBench leaderboard
  (`benches/reports/splatbench-v0.{md,html,json}`) gains a Fidelity column
  with pass / borderline / fail buckets.
- **`@splatforge/viewer` is now actually runnable** — previously the SDK was
  tsc-clean but had never been exercised against a real splatforge-emitted
  glTF in a browser. v0.1.1 fixes the structural wire-format mismatches:
  the Rust glTF writer now emits top-level `KHR_gaussian_splatting.splatCount`
  + `bbox`, POSITION accessor `min`/`max` (glTF 2.0 §3.6.2.4), and a `uri`
  field on each `SF_spatial_streaming_index` chunk record. The JS viewer
  learns to (a) decode structure-of-arrays attribute buffers (one bufferView
  per attribute) by re-interleaving them at decode time, (b) derive scene
  splatCount + bbox from accessor metadata when the extension lacks them,
  and (c) resolve chunk URIs against the manifest's anchored location.
  The viewer's `cameraPath: 'orbit-8'` mode now actually drives 8 orbit poses
  and emits a per-frame `frameRendered` event, used by the visual harness.
- **`apps/web`** — Astro 4 static landing page at the repo's first deployable
  site. Composes a hero + headline-stat triptych + tabbed install snippet +
  embedded leaderboard + drag-drop placeholder. Reads the SplatBench JSON at
  build time so future fidelity-runner updates land automatically.
- **SPEC-0011** (`OpenUSD ParticleField3DGaussianSplat round-trip`) and
  **SPEC-0012** (`OpenUSD streaming via payload + variant sets`) — draft
  specs covering the v0.2 OpenUSD interop work. Both flagged `Status: Draft`
  pending validation against a real USD toolchain.

### Fixed

- Insertion sort in the WebGL2 renderer's per-frame draw path was O(n²)
  and didn't scale past ~10K splats. Replaced with a stable O(n log n)
  paired-index sort so the renderer handles the SplatBench corpus.
- `tests/visual/scripts/diff-cli.mjs` now stages the `buffers/` directory
  alongside the staged `.gltf` so chunk-URI fetches resolve in the harness.
- Three GitHub Actions workflows (`test.yml`, `visual.yml`, `benchmark.yml`)
  had latent first-run failures: `pnpm -r run lint/test` would invoke the
  Playwright project's `tsc --noEmit` without the viewer dist available, the
  visual workflow never built the viewer before running Playwright, and the
  benchmark workflow wrote results to stdout rather than `benches/reports/`.
  All three patched for green first run.

### Notes

- Fidelity numbers were captured on macOS aarch64 with SwiftShader; relative
  degradation between presets is the load-bearing signal, not absolute pixel
  identity to a hardware GPU.

## [0.1.0] — 2026-05-14

Initial public release. Phase 0 + Phase 1 + Phase 2 of the PRD roadmap, plus
v1 tightening and the first SplatBench leaderboard.

### Added

- **CLI** (`splatforge`) with `analyze`, `inspect`, `convert`, `optimize`,
  `preview`, `diff`, `benchmark`, and `corpus run` subcommands.
- **`splatforge-core`** — canonical `SplatIR` with deterministic serialization,
  stable BLAKE3 hashing, coordinate-system metadata, reserved temporal fields,
  optional semantic labels (SPEC-0001), and the analyze-report struct +
  deterministic JSON emitter (SPEC-0005).
- **`splatforge-ply`** — Inria 3DGS PLY reader (binary little-endian + ASCII)
  with structured errors and writer for round-trip (SPEC-0002).
- **`splatforge-spz`** — v2 SPZ codec (24-bit fixed-point positions,
  smallest-three quaternions, 8-bit scales/colors, zlib payload) with
  round-trip parity tests (SPEC-0003).
- **`splatforge-gltf`** — glTF 2.0 + `KHR_gaussian_splatting` writer with
  external-buffer chunking + `SF_spatial_streaming_index` vendor extension
  (Morton-ordered chunks, per-chunk BLAKE3 checksums) (SPEC-0004, SPEC-0007).
  GLB binary container writer + reader.
- **`splatforge-optimize`** — composable `Pass` framework with 10 passes
  (RemoveInvalidSplats, OpacityPrune, FloaterPrune, QuantizePosition/Scale/
  Rotation, ReduceSHDegree, MortonSort, BuildLOD, ObjectAwarePruneExperimental)
  and 8 named presets (SPEC-0006).
- **`@splatforge/viewer`** — WebGPU primary + WebGL2 fallback Gaussian-splat
  renderer with instanced-quad EWA fragment shading, manifest parser,
  chunk-by-chunk progressive loader, deterministic camera-path mode for tests
  (SPEC-0008).
- **`@splatforge/report-ui`** — visual-diff + viewer-parity HTML templates.
- **Visual harness** (`tests/visual/`) — Playwright-based 8-frame deterministic
  capture per asset across `chrome-webgpu`, `chrome-webgl2`, `firefox-webgl2`,
  `webkit-webgl2` projects (SPEC-0009, SPEC-0010).
- **End-to-end `splatforge diff`** — Rust CLI dispatches to a Node helper that
  drives headless Chromium via `playwright-core` to render and pixel-compare
  before/after frames, emitting a deterministic JSON + HTML report.
- **SplatBench v0** — 7-scene benchmark corpus (2 real Mip-NeRF360 + 5
  deterministic synthetic) with interactive HTML leaderboard. web-mobile
  median 21.75×, size-min median 24.24× compression vs raw PLY across the
  corpus.
- **Real-world demo** — full analyze/optimize/SPZ pipeline run on the canonical
  Inria 3DGS bonsai scene (1.16M splats, 273 MB → 12 MB at 22.8× compression
  in 730 ms).
- **40 Rust tests** passing across 7 crates.
- **Test infrastructure** — 17 fixtures (tiny PLY/SPZ/glTF + invalid + corpus)
  generated by `fixtures/build.py`, 3 GitHub Actions workflows
  (test / visual / benchmark).
- **Docs** — README, getting-started, architecture, all 10 SPEC documents.

### Known Limitations

- glTF accessor types are still f32; `KHR_mesh_quantization`-style integer
  accessors are tracked for v0.2.
- `BuildLOD` produces per-LOD splat-index lists, not yet wired into the chunked
  external-buffer layout (the LOD count is written to the streaming index).
- `splatforge diff` requires `playwright-core + chromium` for the real
  rendering path; otherwise produces a `status: "degraded"` JSON.
- Pre-built binary is Linux aarch64 only — `cargo build` locally for other
  platforms (instructions in [`INSTALL.md`](./INSTALL.md)).

[Unreleased]: https://github.com/montabano1/SplatForge/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/montabano1/SplatForge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/montabano1/SplatForge/releases/tag/v0.1.0
