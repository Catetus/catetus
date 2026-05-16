# SplatForge

**Production infrastructure for Gaussian Splats** — compress, validate, and ship splat assets with standards-aligned output and reproducible quality gates.

<p>
  <a href="https://splatforge.com"><img alt="Website" src="https://img.shields.io/badge/website-splatforge.com-0ea5e9?style=flat-square" /></a>
  <a href="https://github.com/montabano1/SplatForge/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/montabano1/SplatForge/test.yml?branch=main&label=ci&style=flat-square" /></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" /></a>
  <a href="https://splatforge.com/bench"><img alt="SplatBench" src="https://img.shields.io/badge/SplatBench-21.9×%20median-7dd3fc?style=flat-square" /></a>
</p>

| | |
| --- | --- |
| **Website** | [splatforge.com](https://splatforge.com) — live demos, leaderboard, docs |
| **Try it** | [Drop a `.ply` in the browser](https://splatforge.com/#try) |
| **Benchmark** | [SplatBench leaderboard](https://splatforge.com/bench) — 16 scenes, open submission |
| **Standards** | [KHR conformance report](https://splatforge.com/khr-conformance) — 23 clauses, 10 fixtures |

---

## What is SplatForge?

Gaussian Splatting is moving from research demos into production web, mobile, and spatial apps. Capture tools export huge `.ply` files; runtimes expect compact, standards-aligned assets; teams need proof that optimization did not destroy visual quality.

**SplatForge is the delivery layer in the middle** — think **FFmpeg + Lighthouse for Gaussian Splats**:

- **Ingest** `.ply`, `.spz`, and glTF Gaussian Splat assets
- **Optimize** for real device byte and fidelity budgets (CLI presets + hosted API)
- **Export** [glTF `KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) and SPZ — no proprietary container format
- **Prove** quality with deterministic visual diff, SplatBench, and conformance suites

The open-source core (this repo) ships the CLI, viewer SDK, benchmark corpus, and Khronos/OpenUSD conformance tooling. Hosted optimize, premium passes, and enterprise deployment live on [splatforge.com](https://splatforge.com).

---

## Why teams choose SplatForge

| Capability | What it means |
| --- | --- |
| **Standards-first output** | glTF KHR Gaussian Splatting + SPZ; assets work in mainstream viewers and DCC pipelines |
| **Reproducible pipeline** | Same input + preset → byte-identical output and stable BLAKE3 scene hashes |
| **Public benchmark moat** | [SplatBench](https://splatforge.com/bench) — 16 scenes (real + synthetic stress tests), open encoder comparison |
| **Quality gates** | Per-scene fidelity (ΔE94, SSIM), visual diff harness, KHR conformance crate in CI |
| **Ship anywhere** | Rust CLI, `@splatforge/viewer` (WebGPU + WebGL2), GitHub Action, REST API |

**Headline numbers (SplatBench v0, `web-mobile` preset):**

| Metric | Value |
| ---: | ---: |
| Median compression (16 scenes) | **21.9×** |
| Real outdoor (`bicycle`, 3.6M splats) | **25.5×** (856 MB → 34 MB) |
| Real indoor (`bonsai`, 1.2M splats) | **22.8×** (274 MB → 12 MB) |
| Fidelity gates passing | **16 / 16** scenes |

Full tables and per-scene breakdown: [leaderboard](https://splatforge.com/bench) · [report](./benches/reports/splatbench-v0.md) · [interactive HTML](./benches/reports/splatbench-v0.html)

---

## Quick start

**Prerequisites:** Rust stable (≥ 1.74), Node.js 20+, pnpm 9+. See [INSTALL.md](./INSTALL.md) for platform notes.

```bash
git clone https://github.com/montabano1/SplatForge.git
cd SplatForge
./setup.sh

# Inspect a splat
./target/release/splatforge analyze fixtures/tiny/basic_binary.ply --pretty

# Optimize for web delivery (glTF + chunked buffers)
./target/release/splatforge optimize fixtures/tiny/basic_binary.ply \
  --preset web-mobile --out /tmp/scene.gltf

# Preview in the browser (WebGPU viewer)
./target/release/splatforge preview /tmp/scene.gltf
```

**Run tests:** `make test` · **Run SplatBench locally:** `make bench-splatbench`

Step-by-step guide: [docs/getting-started.md](./docs/getting-started.md)

---

## CLI commands

| Command | Description |
| --- | --- |
| `analyze` | Deterministic JSON report (size, bounds, attribute stats) |
| `inspect` | Validate an asset; print a short summary |
| `convert` | Convert between PLY, SPZ, glTF, GLB |
| `optimize` | Run a preset (`web-mobile`, `size-min`, `geospatial`, …) |
| `preview` | Local WebGPU viewer |
| `diff` | Before/after visual diff report (Playwright-backed) |
| `benchmark` | Device-profile timings |
| `corpus run` | Run a named SplatBench suite |
| `submit` | Submit a job to the hosted API |
| `spec-check` | Validate against KHR / extension rules |

```bash
splatforge optimize scene.ply --preset web-mobile --out out/
splatforge diff scene.ply out/scene.gltf --threshold 0.03 --out reports/diff/
```

---

## Integrations

| Surface | Link |
| --- | --- |
| **GitHub Action** | [apps/optimize-action](./apps/optimize-action) — PR gate on compression + fidelity badge |
| **Hosted API** | `https://splatforge-api.fly.dev` — job create, upload, status ([apps/api](./apps/api)) |
| **Viewer SDK** | [`@splatforge/viewer`](./packages/viewer) — WebGPU compute decode + streaming LOD |
| **Blender add-on** | [integrations/blender](./integrations/blender) |

---

## Repository layout

```
SplatForge/
  crates/
    splatforge-core/          # SplatIR — canonical internal representation
    splatforge-ply/           # PLY ingest + write
    splatforge-spz/           # SPZ I/O
    splatforge-gltf/          # glTF KHR Gaussian Splatting + GLB
    splatforge-optimize/      # Optimization pass framework
    splatforge-khr-conformance/  # KHR extension validator (23 clauses)
    splatforge-usd/           # OpenUSD writer (draft)
    splatforge-cli/           # `splatforge` binary
  packages/
    viewer/                   # @splatforge/viewer
    report-ui/                # Diff + parity HTML reports
  specs/                      # Feature specs (SPEC-0001 … SPEC-0013)
  benches/                    # SplatBench corpus + reports
  apps/
    web/                      # Marketing site + leaderboard (Astro)
    api/                      # Hosted optimize API (Rust / Axum)
    optimize-action/          # GitHub Action
  docs/                       # User + architecture docs
  tests/                      # Integration + visual regression
```

Architecture overview: [docs/architecture.md](./docs/architecture.md)

---

## Standards & conformance

SplatForge targets **glTF 2.0 + `KHR_gaussian_splatting`** as the primary interchange format, with **SPZ** as a first-class compressed wire format. Advanced streaming uses a vendor extension (`SF_spatial_streaming_index`) that degrades gracefully in viewers that ignore it.

- **KHR suite:** `cargo test -p splatforge-khr-conformance` — [conformance report](./crates/splatforge-khr-conformance/conformance.md) · [live matrix](https://splatforge.com/khr-conformance)
- **KHR SPZ compression extension (draft):** [docs/standards/KHR_gaussian_splatting_compression_spz.md](./docs/standards/KHR_gaussian_splatting_compression_spz.md)
- **OpenUSD:** writer + conformance work in progress ([SPEC-0011](./specs/0011-openusd-roundtrip.md), [SPEC-0012](./specs/0012-openusd-streaming.md))

---

## Documentation

| Doc | Audience |
| --- | --- |
| [INSTALL.md](./INSTALL.md) | First-time setup (macOS, Linux, Windows) |
| [docs/getting-started.md](./docs/getting-started.md) | End-to-end CLI walkthrough |
| [docs/architecture.md](./docs/architecture.md) | System design |
| [specs/](./specs) | Feature specifications |
| [CHANGELOG.md](./CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | How to contribute |

---

## Contributing

Issues and PRs welcome. We are **spec-driven** and **determinism is non-negotiable** — see [CONTRIBUTING.md](./CONTRIBUTING.md) for the workflow, DCO, and review expectations.

- [Report a bug](https://github.com/montabano1/SplatForge/issues/new?template=bug.md)
- [Request a feature](https://github.com/montabano1/SplatForge/issues/new?template=feature.md)
- [Submit a scene to SplatBench](https://github.com/montabano1/SplatForge/issues/new?template=corpus_request.md)

---

## License

Apache-2.0. See [LICENSE](./LICENSE).
