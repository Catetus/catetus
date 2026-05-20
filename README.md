# Catetus

**Production infrastructure for Gaussian Splats** — compress, validate, and ship splat assets with standards-aligned output and reproducible quality gates.

<p>
  <a href="https://catetus.com"><img alt="Website" src="https://img.shields.io/badge/website-catetus.com-0ea5e9?style=flat-square" /></a>
  <a href="https://github.com/Catetus/catetus/actions/workflows/rust-ci.yml?query=branch%3Amain"><img alt="Rust CI" src="https://img.shields.io/github/actions/workflow/status/Catetus/catetus/rust-ci.yml?branch=main&label=rust-ci&style=flat-square" /></a>
  <a href="https://github.com/Catetus/catetus/actions/workflows/ts-ci.yml?query=branch%3Amain"><img alt="TS CI" src="https://img.shields.io/github/actions/workflow/status/Catetus/catetus/ts-ci.yml?branch=main&label=ts-ci&style=flat-square" /></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" /></a>
  <a href="https://catetus.com/bench"><img alt="SplatBench" src="https://img.shields.io/badge/SplatBench-22.7×%20median-7dd3fc?style=flat-square" /></a>
</p>

| | |
| --- | --- |
| **Website** | [catetus.com](https://catetus.com) — live demos, leaderboard, docs |
| **Try it** | [Drop a `.ply` in the browser](https://catetus.com/#try) |
| **Benchmark** | [SplatBench leaderboard](https://catetus.com/bench) — 28 scenes (14 real + 14 synthetic), open submission |
| **Standards** | [KHR conformance report](https://catetus.com/khr-conformance) — 30 clauses, 13 fixtures |

---

## What is Catetus?

Gaussian Splatting is moving from research demos into production web, mobile, and spatial apps. Capture tools export huge `.ply` files; runtimes expect compact, standards-aligned assets; teams need proof that optimization did not destroy visual quality.

**Catetus is the delivery layer in the middle** — think **FFmpeg + Lighthouse for Gaussian Splats**:

- **Ingest** `.ply`, `.spz`, and glTF Gaussian Splat assets
- **Optimize** for real device byte and fidelity budgets (CLI presets + hosted API)
- **Export** [glTF `KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) and SPZ — no proprietary container format
- **Prove** quality with deterministic visual diff, SplatBench, and conformance suites

The open-source core (this repo) ships the CLI, viewer SDK, benchmark corpus, and Khronos/OpenUSD conformance tooling. Hosted optimize, premium passes, and enterprise deployment live on [catetus.com](https://catetus.com).

---

## Why teams choose Catetus

| Capability | What it means |
| --- | --- |
| **Standards-first output** | glTF KHR Gaussian Splatting + SPZ; assets work in mainstream viewers and DCC pipelines |
| **Reproducible pipeline** | Same input + preset → byte-identical output and stable BLAKE3 scene hashes |
| **Public benchmark** | [SplatBench](https://catetus.com/bench) — 28 scenes (14 real + 14 synthetic stress tests), open encoder comparison |
| **Quality gates** | Per-scene fidelity (ΔE94, SSIM), visual diff harness, KHR conformance crate in CI |
| **Ship anywhere** | Rust CLI, `@catetus/viewer` (WebGPU + WebGL2), GitHub Action, REST API |

**Canonical-11 leaderboard** ([Mip-NeRF 360 + Tanks-and-Temples + Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), `wmv-vq45-no-prune-tight` preset):

| Metric | Value |
| ---: | ---: |
| Scenes measured | **11 / 11** |
| Mean compression vs input PLY | **19.8×** (range 16.6× – 21.9×) |
| Mean PSNR (gsplat, 512², orbit-8, SH=3) | **47.45 dB** (median 47.78, min 43.46) |
| Mean SSIM | **0.9991** (min 0.9973) |

Per-scene table: [canonical-11.md](./benches/reports/canonical-11.md) · machine-readable: [canonical-11.json](./benches/reports/canonical-11.json)

**Broader SplatBench v0 corpus** (28 scenes including synthetic stress probes, `web-mobile` preset):

| Metric | Value |
| ---: | ---: |
| Median compression (28 corpus scenes) | **22.7×** |
| Real outdoor (`bicycle`, 3.6M splats) | **25.5×** (855 MB → 34 MB) |
| Real indoor (`bonsai`, 1.2M splats) | **22.8×** (274 MB → 12 MB) [^bonsai] |
| Fidelity gates passing | **16 / 16 fidelity-gated scenes** (`web-mobile` + `size-min` both pass) |

**Coverage scope:** 16 of the 28 corpus scenes have fidelity oracles registered (`bonsai` + `bicycle` real photogrammetry + 14 synthetic stress probes). The remaining 12 are size-benchmarked only: 6 await scaffold-GS render oracles, 5 are LOD-ladder synthetic without ground-truth captures (`cluster_fly_*`), 1 is an under-trained iter7k variant. Coverage expansion tracked on the [SplatBench leaderboard page](https://catetus.com/bench).

Full tables and per-scene breakdown: [leaderboard](https://catetus.com/bench) · [report](./benches/reports/splatbench-v0.md) · [interactive HTML](./benches/reports/splatbench-v0.html)

[^bonsai]: The two `bonsai` rows above measure two different files at two different presets — they are not directly comparable. SplatBench v0 uses a 274 MB iter7k bonsai derivative at the bytes-first `web-mobile` preset; the canonical-11 row uses the official Inria iter30k bonsai (308.7 MB, md5 `ad5377eb…`) at the fidelity-first `wmv-vq45-no-prune-tight` preset. Identical scene names, different inputs and budgets.

---

## Quick start

**Prerequisites:** Rust stable (≥ 1.74), Node.js 20+, pnpm 9+. See [INSTALL.md](./INSTALL.md) for platform notes.

```bash
git clone https://github.com/Catetus/catetus.git
cd Catetus
./scripts/install-githooks.sh   # one-time: enables the partnership-docs pre-push guard (see docs/CONTRIBUTING.md)
./setup.sh

# Inspect a splat
./target/release/catetus analyze fixtures/tiny/basic_binary.ply --pretty

# Optimize for web delivery (glTF + chunked buffers)
./target/release/catetus optimize fixtures/tiny/basic_binary.ply \
  --preset web-mobile --out /tmp/scene.gltf

# Preview in the browser (WebGPU viewer)
./target/release/catetus preview /tmp/scene.gltf
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
catetus optimize scene.ply --preset web-mobile --out out/
catetus diff scene.ply out/scene.gltf --threshold 0.03 --out reports/diff/
```

---

## Integrations

| Surface | Link |
| --- | --- |
| **GitHub Action** | [apps/optimize-action](./apps/optimize-action) — PR gate on compression + fidelity badge |
| **Hosted API** | `https://api.catetus.com` — job create, upload, status, fidelity scoring (server source is private; see [OpenAPI](https://api.catetus.com/docs)) |
| **Viewer SDK** | [`@catetus/viewer`](./packages/viewer) — WebGPU compute decode + streaming LOD |
| **Blender add-on** | [integrations/blender](./integrations/blender) |

---

## Repository layout

```
Catetus/
  crates/
    catetus-core/          # SplatIR — canonical internal representation
    catetus-ply/           # PLY ingest + write
    catetus-spz/           # SPZ I/O
    catetus-gltf/          # glTF KHR Gaussian Splatting + GLB
    catetus-optimize/      # Optimization pass framework
    catetus-khr-conformance/  # KHR extension validator (30 clauses)
    catetus-usd/           # OpenUSD writer (draft)
    catetus-cli/           # `catetus` binary
  packages/
    viewer/                   # @catetus/viewer
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

Catetus targets **glTF 2.0 + `KHR_gaussian_splatting`** as the primary interchange format, with **SPZ** as a first-class compressed wire format. Advanced streaming uses a vendor extension (`CT_spatial_streaming_index`) that degrades gracefully in viewers that ignore it.

- **KHR suite:** `cargo test -p catetus-khr-conformance` — [conformance report](./crates/catetus-khr-conformance/conformance.md) · [live matrix](https://catetus.com/khr-conformance)
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

- [Report a bug](https://github.com/Catetus/catetus/issues/new?template=bug.md)
- [Request a feature](https://github.com/Catetus/catetus/issues/new?template=feature.md)
- [Submit a scene to SplatBench](https://github.com/Catetus/catetus/issues/new?template=corpus_request.md)

---

## License

Apache-2.0. See [LICENSE](./LICENSE).
