# Catetus architecture

## Layered overview

```
                              users / partners
                                      │
                                      v
            +-------------------------+---------------------------+
            |                                                     |
   public landing page                              hosted optimize API
   apps/web (Astro static)                          apps/api (Axum on DO droplet,
   github.com/Catetus                            planned; in-memory store today)
   /Catetus                                              │
                                                            v
                                                  apps/worker (Modal Python)
                                                  pinned catetus CLI
                                                  scales to zero
                                                            │
                                                            v
                                            +-----------------------------+
                                            |       catetus-cli        |
+-------------------------------------------+ analyze inspect convert     |
|                                           |  optimize preview diff      |
|                                           +-+---------------------------+
|     catetus-bench                        |                  |
|     (named corpora)                         v                  v
+---------------------------------+   catetus-optimize   catetus-bench
                                  │  (10 passes, 8 presets)
                                  v
                +----------------------------------------+
                |              SplatIR (core)            |
                | Splat, SplatScene, CoordinateSystem,   |
                | TemporalMode, LodLevel, scene_hash     |
                +----------------------------------------+
                  ^         ^         ^         ^
                  |         |         |         |
            catetus-  catetus-  catetus-  catetus-
              ply         spz         gltf         usd (draft)

                            packages/viewer        packages/report-ui
                            WebGPU + WebGL2        diff + parity HTML
```

## SplatIR

The single source of truth for splat data inside the pipeline. All importers
convert to IR; all optimizers operate on IR; all exporters convert from IR.

IR is deterministic: serialization, hashing, and ordering are stable across
runs and platforms. The BLAKE3 hash of the canonical-form IR is the corpus
identifier used by the public benchmark.

## Optimization pipeline

A `Pipeline` is an ordered list of `Pass` impls. Passes:

* receive `&mut SplatScene` plus a `PassContext` carrying logger + RNG seed
* return `PassStats { removed, modified, duration_ms, notes }`
* must be deterministic given the same input + config + seed

Presets (`lossless-repack`, `web-mobile`, `quest-browser`, `size-min`, …) are
named `Pipeline` configurations. See SPEC-0006.

## Wire formats

Always glTF 2.0 + `KHR_gaussian_splatting` as the primary delivery target.
Optional `SF_spatial_streaming_index` extension (SPEC-0007) for Morton-ordered
LOD streaming. SPZ (SPEC-0003) is first-class for compressed delivery; SPZ is
about 2× smaller than the glTF buffer today but the gap closes once SPEC-0013
(`KHR_mesh_quantization`) lands.

OpenUSD (SPEC-0011 + SPEC-0012) is the v0.2 target — `catetus-usd` already
round-trips USDA on the 3-splat fixture; USDC binary writer is sketched but
not yet bit-exact against `usdcat`.

**No proprietary container formats** as the default output. The folder layout
(`scene.gltf` + `buffers/*.bin`) is a *deployment* layout, not a new format:
removing the vendor extension still leaves a valid glTF.

## Viewer SDK

TypeScript + WebGPU primary, WebGL2 fallback. The renderer interface is the
same — only the backend changes. Deterministic camera-path mode powers
SPEC-0009 / SPEC-0010 tests.

Two consumers today:

1. The visual-fidelity harness (`tests/visual/scripts/splatbench-fidelity.mjs`)
   that produces the SplatBench v0 ΔE94 / pixelmatch / SSIM column.
2. The Astro landing page's hero canvas — currently a placeholder animation
   that *resembles* a Gaussian splat scene; v0.3 will swap in the real viewer
   loading a tiny pre-baked scene.

## Hosted services (Phase 3)

```
  user browser ─────────►  apps/web                            (Astro static, Vercel)
        │                  │
        │                  └─► fetches splatbench-v0.json       (build-time inlined)
        │
        └─► POST /v1/jobs ─►  apps/api  (Axum)                   (DO droplet, planned)
                                │
                                ├─► presigns Vercel Blob upload URL
                                │
                                └─► POST /enqueue ────►  apps/worker (Modal)
                                                                │
                                                                ├─► pulls splat from Blob
                                                                ├─► runs pinned catetus CLI
                                                                └─► POSTs result back to API
```

Boundary choices:

* **apps/api** is light HTTP — auth, job lifecycle, presigning. Stays on the
  DigitalOcean droplet (or any single-region VM) for the predictable public
  IP that Modal can webhook back to.
* **apps/worker** is heavy compute — Modal scales to zero, pins the CLI
  by git tag for reproducibility, runs each optimize in its own container.
* **Vercel Blob** is the canonical splat storage layer. Public read so the
  Modal worker can pull splats with a plain `urllib.request` call; presigned
  write so only authorized clients can upload.
* **apps/web** is purely static — no server-side rendering, no API routes,
  no auth. The leaderboard JSON is inlined at build time.

## SplatBench corpus

The benchmark *is* the moat (see PRD §"Benchmark and moat strategy"). The
corpus lives at:

```
benches/
├── synth_scenes.py            # Deterministic synthetic scene generator
└── reports/
    ├── splatbench-v0.{md,json,html}    # Headline leaderboard
    ├── fidelity-v0.json                # Per-frame ΔE94/pixelmatch/SSIM
    ├── bonsai-real-demo.md             # Single-scene deep dive
    └── frames/                          # (gitignored, reproducible)
```

The corpus is **public** and reproducible end-to-end: same input + same
pinned CLI = byte-identical output. That property is what makes SplatBench
a credible candidate for Khronos `KHR_gaussian_splatting` conformance.

## Determinism guarantees

* **`catetus analyze`** produces byte-identical JSON across runs.
* **`catetus optimize`** is deterministic given `(input, preset, seed)`.
* **The viewer's `cameraPath: 'orbit-8'`** generates 8 pose-identical frames
  across reruns on the same backend.
* **The fidelity runner** writes results in a stable key order.

What's *not* yet deterministic: pixel-exact rendering across different GPUs
or drivers. The fidelity threshold (mean ΔE94 < 3%) is calibrated to tolerate
that variance.

## Crates + packages map

| Path                          | Kind         | Status     | Notes                                |
| ----------------------------- | ------------ | ---------- | ------------------------------------ |
| `crates/catetus-core`      | Rust lib     | shipped    | IR + hash + analyze report           |
| `crates/catetus-ply`       | Rust lib     | shipped    | Inria 3DGS PLY in/out                |
| `crates/catetus-spz`       | Rust lib     | shipped    | SPZ v2 codec                         |
| `crates/catetus-gltf`      | Rust lib     | shipped    | glTF + KHR_gaussian_splatting        |
| `crates/catetus-optimize`  | Rust lib     | shipped    | 10 passes, 8 presets                 |
| `crates/catetus-bench`     | Rust lib     | shipped    | Corpus runner                        |
| `crates/catetus-cli`       | Rust bin     | shipped    | The `catetus` binary              |
| `crates/catetus-usd`       | Rust lib     | draft v0.2 | OpenUSD I/O — SPEC-0011              |
| `apps/api`                    | Rust bin     | scaffolded | Hosted optimize endpoint — not deployed |
| `apps/worker`                 | Modal Python | deployed   | https://api.catetus.com |
| `apps/fidelity-gpu`           | Modal Python | deployed   | One-shot GPU fidelity rerun          |
| `apps/web`                    | Astro static | deployed   | https://catetus-…vercel.app       |
| `packages/viewer`             | TS lib       | shipped    | WebGPU + WebGL2 splat renderer       |
| `packages/report-ui`          | TS lib       | shipped    | Diff + parity HTML templates         |
| `tests/visual/`               | Playwright   | shipped    | Visual harness                       |

## How to add a new format

Worked example — adding `catetus-foo` as a new ingest/export format:

1. Create `crates/catetus-foo/` with `read_foo(path) -> SplatScene` and
   `write_foo(scene, path, opts)`. Mirror the surface of `catetus-ply` for
   ergonomic parity.
2. Add the crate to `Cargo.toml`'s `workspace.members`.
3. Wire ingest detection in `catetus-core::format_from_extension` +
   `format_from_magic`.
4. Wire the CLI subcommand into `crates/catetus-cli/src/main.rs` —
   `convert` is the natural place to hang `--to foo`.
5. Add a SPEC document under `specs/` describing the wire mapping + Gherkin
   acceptance criteria. Make sure determinism + reproducibility are explicit
   in the acceptance bar.
6. Add a round-trip test in `crates/catetus-foo/tests/roundtrip.rs`.

`catetus-usd` was added this way and is the freshest reference.

## How to add a new optimize pass

1. Implement `Pass` in `crates/catetus-optimize/src/passes.rs`.
2. Reference it from one or more presets in `crates/catetus-optimize/src/presets.rs`.
3. Add a unit test in `crates/catetus-optimize/tests/passes.rs` asserting
   the determinism + the expected `PassStats` for a small fixture.
4. If the pass changes the wire format (e.g., new SH coefficient layout),
   update the relevant SPEC doc and the round-trip tests in the affected
   format crate.
