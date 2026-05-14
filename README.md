# SplatForge

<p>
  <a href="https://github.com/montabano1/SplatForge/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/montabano1/SplatForge/test.yml?branch=main&label=ci&style=flat-square" /></a>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" /></a>
  <a href="./benches/reports/splatbench-v0.md"><img alt="SplatBench v0" src="https://img.shields.io/badge/SplatBench-v0%20%E2%80%94%2021.75%C3%97%20median-7dd3fc?style=flat-square" /></a>
  <a href="./specs/"><img alt="specs" src="https://img.shields.io/badge/spec--driven-yes-34d399?style=flat-square" /></a>
</p>

> **SplatForge makes Gaussian Splats production-ready: optimized, standards-aligned, streamable, benchmarked, and safe to ship.**

FFmpeg + Lighthouse + Cloudinary for Gaussian Splats. Ingest large `.ply`, `.spz`, and glTF Gaussian Splat assets; optimize them for real device budgets; output standards-aligned glTF / SPZ artifacts; and generate reproducible visual / performance reports.

This repository implements **Phase 0 + Phase 1 + Phase 2** of the SplatForge roadmap. See the [engineering plan](./gaussian_splat_prd_eng_plan%20(1).md) for full product context, and [`specs/`](./specs) for per-feature SpecDD documents.

## What you can do today

```bash
# Build (Rust 1.74+ stable, Node 20+)
./setup.sh

# Analyze a real splat
./target/release/splatforge analyze fixtures/tiny/basic_binary.ply --pretty

# Optimize for the web
./target/release/splatforge optimize fixtures/tiny/basic_binary.ply \
    --preset web-mobile --out /tmp/scene.gltf

# Run the SplatBench v0 corpus locally
make bench-splatbench
```

## SplatBench v0 — what the pipeline does on real data

| Scene | Splats | PLY in | SPZ out (web-mobile) | **Ratio** |
| ----- | ---: | ---: | ---: | ---: |
| `bicycle_mipnerf360_iter7k` (real) | **3.62M** | 856 MB | 34 MB | **25.46×** |
| `bonsai_mipnerf360_iter7k` (real)  | 1.16M | 273 MB | 12 MB | **22.81×** |
| `splatbench_dense_proxy` (synth)   | 2.0M  | 474 MB | 22 MB | 21.75× |
| `splatbench_floater_proxy` (synth) | 250K  |  60 MB | 2.3 MB | **25.84×** |
| _full leaderboard →_ | | | | [splatbench-v0.html](./benches/reports/splatbench-v0.html) |

Median compression across 7 scenes: **21.75× (web-mobile)** / **24.24× (size-min)**.
Median analyze wall time on real splats: **~1 µs/splat**.

## Layout

```
splatforge/
  specs/                   # SpecDD spec docs (SPEC-0001 .. SPEC-0010)
  crates/
    splatforge-core/       # SplatIR + canonical types
    splatforge-ply/        # PLY ingest + write
    splatforge-spz/        # SPZ I/O
    splatforge-gltf/       # glTF KHR Gaussian Splatting + GLB
    splatforge-optimize/   # Optimization pass framework
    splatforge-bench/      # Benchmark runner
    splatforge-cli/        # `splatforge` binary
  packages/
    viewer/                # @splatforge/viewer (WebGPU + WebGL2)
    report-ui/             # @splatforge/report-ui
  tests/
    integration/           # CLI end-to-end scripts
    visual/                # Playwright visual-regression tests
  fixtures/                # Tiny, invalid, corpus, golden assets
  benches/
    synth_scenes.py        # Reproducible synthetic SplatBench scenes
    reports/               # SplatBench v0 + bonsai demo writeups
  docs/                    # User docs
  .github/workflows/       # CI gates
```

## CLI surface

| Command | What it does |
| ------- | ------------ |
| `analyze` | Emit deterministic JSON analysis report |
| `inspect` | Validate an asset and print a brief summary |
| `convert` | Convert between PLY, SPZ, glTF, GLB |
| `optimize` | Run a preset (or custom passes); emit chunked glTF + report |
| `preview` | Launch a local WebGPU viewer instance |
| `diff` | Render before/after frames; emit a visual-diff report |
| `benchmark` | Device-profile timings |
| `corpus run` | Run a named SplatBench suite |

## Standards

SplatForge writes **glTF 2.0 with `KHR_gaussian_splatting`** as the primary delivery target, with optional external-buffer chunking and a vendor extension (`SF_spatial_streaming_index`) for Morton-ordered LOD streaming.

SPZ is a first-class compressed format. **No proprietary `.sfz` package format.** When advanced streaming metadata is present, the asset degrades gracefully to baseline glTF behavior in viewers that ignore the vendor extension.

## Docs

- [INSTALL.md](./INSTALL.md) — toolchain + first build
- [docs/getting-started.md](./docs/getting-started.md) — running the CLI end-to-end
- [docs/architecture.md](./docs/architecture.md) — high-level component map
- [specs/](./specs) — 10 SpecDD documents (IR, PLY, SPZ, glTF, analyze, optimize, streaming, viewer, diff, parity)
- [CHANGELOG.md](./CHANGELOG.md) — release notes
- [CONTRIBUTING.md](./CONTRIBUTING.md) — DCO, PR flow, principles

## Status

| Phase | Spec coverage | Status |
| ----- | ------------- | ------ |
| Phase 0 — technical spike + design partners | (PRD §) | ✓ proven on real Inria 3DGS data |
| Phase 1 — CLI alpha + standards I/O | SPEC-0001..0007 | ✓ shipped in v0.1.0 |
| Phase 2 — viewer SDK + visual diff + benchmark runner | SPEC-0008..0010 | ✓ shipped in v0.1.0 |
| Phase 3 — hosted API + OpenUSD + partnerships | SPEC-0011, 0012 | planned |
| Phase 4 — advanced compression + enterprise pipeline | (PRD §) | planned |

## License

Apache-2.0. See [LICENSE](./LICENSE).
