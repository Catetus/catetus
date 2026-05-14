# SplatForge — Cowork-session handoff to Claude Code

> This document captures the state of the project at the moment of handoff from the Cowork-mode session that built Phase 0–2 + SplatBench v0 to a Claude Code session that will continue the work from `~/Desktop/SplatForge` on the user's Mac. Claude Code starts with no chat history; treat this file as the only context you have.

**Handoff date:** 2026-05-14
**Last commit:** `54ff5aa` — `feat: initial SplatForge v0.1.0 — Phase 0–2 + SplatBench v0`
**Owner:** Monte (`monte@recruitplan.ai`)
**Remote:** `https://github.com/montabano1/SplatForge.git`

---

## TL;DR

SplatForge is a Gaussian Splat optimization CLI + viewer SDK + visual-diff harness. The Rust workspace (7 crates, 40 passing tests) and TypeScript packages (`@splatforge/viewer`, `@splatforge/report-ui`) are all wired and compile clean. Real-world demo on the Inria 3DGS bonsai scene: **22.81× compression in 730 ms with deterministic output.**

You are picking up at the **visual-fidelity** step. The SplatBench v0 leaderboard reports compression ratios honestly but does NOT yet report perceptual quality. That's the gap to close.

**Your primary task:** render the SplatBench corpus at three presets through the viewer SDK, compute ΔE94 / SSIM / pixelmatch versus the lossless baseline, regenerate the leaderboard with fidelity columns, and ship v0.1.1.

---

## What's shipped (v0.1.0)

### Specs

10 SpecDD docs in [`specs/`](./specs/) — SPEC-0001 (IR) through SPEC-0010 (viewer parity). Each has Gherkin acceptance criteria. **Read the spec for any feature you touch before changing it.**

### Rust workspace

| Crate | What it does |
| ----- | ------------ |
| `splatforge-core` | Canonical `SplatIR` + deterministic JSON report + BLAKE3 scene hashing |
| `splatforge-ply` | Inria 3DGS binary-LE + ASCII reader and writer |
| `splatforge-spz` | v2 SPZ codec, round-trip parity tests |
| `splatforge-gltf` | glTF 2.0 + `KHR_gaussian_splatting` + GLB writer/reader, `SF_spatial_streaming_index` vendor ext |
| `splatforge-optimize` | 10-pass framework with 8 named presets |
| `splatforge-bench` | Corpus runner |
| `splatforge-cli` | `splatforge` binary, 8 subcommands |

All 40 tests pass. `cargo build --release -p splatforge-cli` produces `target/release/splatforge`. **Build locally — the `bin/splatforge` file in the repo is from a Linux sandbox build and is gitignored.**

### TypeScript packages

- **`@splatforge/viewer`** — WebGPU primary + WebGL2 fallback Gaussian-splat renderer with instanced-quad EWA fragment shading, manifest parser, chunk-by-chunk progressive loader, deterministic camera-path mode. tsc-clean. **Has not been run in a real browser yet** — see Known Risks below.
- **`@splatforge/report-ui`** — visual-diff + viewer-parity HTML templates. tsc-clean.

### Visual harness (`tests/visual/`)

Playwright config for 4 browser projects (`chrome-webgpu`, `chrome-webgl2`, `firefox-webgl2`, `webkit-webgl2`). `harness/page.html` mounts the viewer with `deterministic: true, cameraPath: 'orbit-8'` and exposes `window.__sf.frames` (base64 PNGs). `scripts/diff-cli.mjs` drives the harness end-to-end and is what `splatforge diff` shells out to. `scripts/diff-cli.test.mjs` has 10 self-contained metric-aggregator assertions that pass.

### SplatBench v0

[`benches/reports/splatbench-v0.{md,json,html}`](./benches/reports/) — 7-scene corpus (2 real Mip-NeRF360 anchors + 5 deterministic synthetic scenes). Median compression 21.75× (web-mobile) / 24.24× (size-min). Honest gaps section explicitly calls out the fidelity-not-yet-measured caveat — that's the one this handoff is here to close.

### CI

Three workflows under `.github/workflows/`: `test`, `visual`, `benchmark`. They've never run because the repo was just pushed — verifying they go green is a Phase 1 task.

---

## What's in flight — your first task: visual fidelity (v0.1.1)

### Goal

Regenerate SplatBench v0 with fidelity columns, ship as v0.1.1. Specifically, the leaderboard's "What v0 doesn't yet measure" section currently lists visual fidelity as item #1. Cross it out.

### Acceptance criteria

* For each scene in SplatBench v0 (7 scenes) at each preset in {`lossless-repack`, `web-mobile`, `size-min`}:
  * render 8 deterministic orbit frames via `@splatforge/viewer` in headless chromium
  * compute pixelmatch %, ΔE94, SSIM versus the same 8 frames rendered from `lossless-repack`
* The PRD targets are:
  * **Perceptual degradation: less than 3–5% under the chosen visual metric for standard presets.**
  * No regression at the `lossless-repack` baseline (self-diff should be near zero).
* Regenerated files:
  * `benches/reports/splatbench-v0.json` — gains a `fidelity` block per scene
  * `benches/reports/splatbench-v0.md` — gains a "Fidelity" table column
  * `benches/reports/splatbench-v0.html` — gains a fidelity cell per row with pass/fail color
  * `benches/reports/bonsai-real-demo.md` — gains a "Visual fidelity" section
  * `CHANGELOG.md` — gains a `## [0.1.1]` entry under `[Unreleased]`
* All 40 existing Rust tests still pass; no regressions.

### Concrete plan

1. **Build everything locally.**
   ```bash
   cd ~/Desktop/SplatForge
   ./setup.sh                              # rust + node + build
   make install-playwright                 # tests/visual + chromium
   pnpm -F @splatforge/viewer run build    # if not already
   pnpm -F @splatforge/report-ui run build
   ```

2. **Smoke test the viewer first.** This is the highest-risk step — see Known Risks. Render the tiny 3-splat fixture through the existing `harness/page.html` and look at the captured frame. It should show 3 splats. If it shows a blank canvas, you have a shader/projection bug. Iterate until the tiny fixture renders correctly. **Don't move on to bigger scenes until this works.**
   ```bash
   ./target/release/splatforge convert fixtures/tiny/basic_binary.ply --to gltf --out /tmp/tiny.gltf
   ./target/release/splatforge diff fixtures/tiny/basic_binary.ply /tmp/tiny.gltf --out /tmp/tiny-diff
   open /tmp/tiny-diff/diff.html
   ```
   Both inputs are identical, so the diff should be near-zero. If the HTML shows a `degraded` status, playwright didn't kick in — debug.

3. **Generate the SplatBench corpus.** Same procedure as the Cowork session:
   ```bash
   make bench-splatbench-synth      # generates 5 synthetic scenes
   make bench-splatbench-real       # downloads bonsai + bicycle (~1.13 GB, takes a few minutes)
   ```

4. **Write a fidelity runner.** Probably under `benches/`. For each of 7 scenes:
   * optimize → `lossless-repack` (the baseline) → convert to glTF → render 8 frames → save as `frames/<scene>/lossless-repack/000{0..7}.png`
   * optimize → `web-mobile` → render → save under `frames/<scene>/web-mobile/`
   * optimize → `size-min` → render → save under `frames/<scene>/size-min/`
   * For each non-baseline preset, compute pixelmatch %, ΔE94 (OKLab), SSIM versus the baseline frames; aggregate `{max, mean, p95}` across the 8 frames

5. **Decide thresholds.** Default `pass = (mean ΔE94 < 3% && max ΔE94 < 8%)`. If you have a strong reason to change this, document it in the report.

6. **Regenerate `splatbench-v0.html`.** Add a "Fidelity" column. Color: green `pass`, yellow `borderline (3–5%)`, red `fail`. Re-sort logic stays the same. The leaderboard's JS data array (`DATA = [...]`) needs each entry to gain `fidelity: { mean, p95, max, passed }`.

7. **Regenerate `splatbench-v0.md` + `.json`** identically.

8. **Update `bonsai-real-demo.md`.** Section 5 currently says "what we haven't measured: visual fidelity." Replace with the actual measurement.

9. **CHANGELOG.** Add a `## [0.1.1]` entry under `[Unreleased]` documenting the v0.1.1 fidelity update.

10. **Commit.** Conventional Commits style. DCO sign-off:
    ```
    feat(splatbench): add fidelity measurements (ΔE94 + SSIM + pixelmatch) for v0.1.1
    ```

### Estimated effort

If the renderer works out of the box: ~2–4 hours of focused work (~half of that is just chromium rendering 21 frame sets sequentially).

If the renderer has shader bugs: add 2–6 hours of shader debugging. WebGPU shader debugging in Chromium is painful but tractable — use `console.log` in TS to print pre-shader buffer values, then dump the canvas and compare to ground truth.

---

## Known risks

### Risk 1 (high probability): the WGSL/GLSL shaders are credible but unverified

The viewer's `renderer/webgpu.ts` and `renderer/webgl2.ts` were written by an agent that could only check that they tsc-compile. **The actual rendering pipeline has never run in a real browser.** Algorithmically the code is correct (instanced quads, back-to-front sort, EWA Gaussian fragment shader, premultiplied alpha blend, 2D covariance projection — the standard 3DGS rasterizer). But the chance that the first run produces a clean image is well below 50%.

**Most likely issues:**
- Per-instance attribute layout in WebGL2 wrong size/stride → garbage geometry
- 2D covariance projection has the wrong sign on Y because of left- vs right-handed mismatch → splats mirrored
- Premultiplied alpha not actually premultiplied → splats look transparent
- Quad-corner offset units (pixels vs NDC) inconsistent → tiny dots instead of splats
- Depth-test enabled when it shouldn't be → only one splat visible

**Debug procedure:** open `harness/page.html` in actual Chrome, add `?renderer=webgl2` to force the easier-to-debug backend. Open DevTools. Look at WebGL state. The two renderers share the algorithm; if one works the other's bug is local.

### Risk 2 (medium): chromium-headless-shell vs full chromium difference

Playwright by default installs `chromium-headless-shell`, which uses SwiftShader for WebGPU rather than your real GPU. Numbers may differ slightly from what real users see. For the v0.1.1 numbers to count as "production fidelity" we ideally want the full headed chromium with hardware acceleration. Try `--with-deps` if you have sudo, or run with `headless: 'shell'` and acknowledge the caveat in the report. **Don't block on this** — software-rendered fidelity numbers are still meaningful as long as we disclose.

### Risk 3 (medium): determinism across CI runs

Two runs of the same render on the same machine should produce identical pixels (we've designed for this). But two runs on different machines may not — different GPU drivers, different anti-aliasing, etc. Pin the chromium version (Playwright already does this) and run all the v0.1.1 numbers on the same machine in the same session. Don't mix.

### Risk 4 (low): the `splatforge diff` Node helper doesn't compose well with 21 (scene, preset) pairs

`scripts/diff-cli.mjs` was written for the one-shot case ("diff before vs after"). For SplatBench you want a batch mode. Either extend the helper or write a sibling `scripts/splatbench-fidelity.mjs` that orchestrates the 21 pairs in one process. The latter is probably cleaner.

---

## Working agreements

These are non-negotiable for this repo (and they're in [`CONTRIBUTING.md`](./CONTRIBUTING.md)):

1. **Spec-driven.** Every new feature gets a spec or amends one. Bugfixes don't need a spec but do need a failing test first.
2. **Determinism is sacred.** Same input + same config = byte-identical output. No wall-clock, no unseeded RNG in library code.
3. **Tests before refactors.** Even cosmetic refactors get a failing test first if behavior is changing.
4. **Snapshots are sacred.** Don't update snapshot files unless the spec change requires it.
5. **No proprietary container formats** as the default output. glTF is primary, SPZ is first-class, OpenUSD is the professional target. No `.sfz`.
6. **DCO sign-off** on every commit (`git commit -s`).
7. **`unwrap()` and `panic!()` are forbidden in library code.** Use `?` and typed errors. Tests can unwrap.
8. **Conventional Commits** style (`feat(scope): subject`, `fix(scope): subject`, etc.).
9. **Public API has docs.** Every exported item gets at least a one-line comment.
10. **`cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`** must all be green before any commit lands on `main`.

---

## After v0.1.1 — strategic next moves

In rough priority order based on the Cowork-session conversation. The user should pick; don't decide unilaterally.

1. **Public landing page / shareable demo.** Stitch the SplatBench v0.1 leaderboard, the viewer SDK, and a "try it on your splat" upload form into a static site deployable to GitHub Pages / Vercel / Cloudflare Pages. With fidelity numbers attached this becomes a powerful shareable artifact.

2. **Design-partner outreach kit.** PRD §"Design partner plan" — recruit 5 design partners with real assets. Write the outreach email template, the 10 intake questions, an asset-license memo. The intake form (corpus_request.md template) is already there.

3. **Phase 3 — Hosted API + OpenUSD.** Build `apps/api` (Axum), `apps/worker` (queue + pinned-CLI invocation), `apps/web-demo` (Next.js upload UI), and the SPEC-0012 OpenUSD round-trip spike.

4. **Standards engagement.** Submit the SplatBench corpus as conformance assets to the Khronos `KHR_gaussian_splatting` working group. Open a discussion thread on the OpenUSD `ParticleField3DGaussianSplat` schema.

5. **Polish CI.** First push will trigger the workflows. Address whatever fails. Add status badges to the README.

6. **Performance work.** The Rust quantization passes currently round-trip through f32; promoting them to true integer accessors via `KHR_mesh_quantization` would shrink the glTF buffer to roughly the SPZ payload size and make the glTF-only path competitive.

---

## Quick reference

### Build & test

```bash
make build        # cargo + pnpm builds
make test         # full test suite (Rust + JS)
make demo         # analyze + optimize on tiny fixture
make bench-splatbench
make install-playwright
```

### Useful files

- `README.md` — public entry point
- `INSTALL.md` — toolchain setup, troubleshooting
- `CHANGELOG.md` — release notes (you'll add a `## [0.1.1]` entry)
- `CONTRIBUTING.md` — DCO, principles, commit style
- `docs/architecture.md` — high-level component map
- `specs/` — 10 SpecDD docs, read these before touching any feature
- `benches/synth_scenes.py` — deterministic synthetic scene generator
- `benches/reports/splatbench-v0.{md,json,html}` — current leaderboard (what you're regenerating)
- `tests/visual/scripts/diff-cli.mjs` — Playwright-driven frame capture, may need extending
- `tests/visual/harness/page.html` — headless test page that mounts the viewer

### Critical constants

| Thing | Value |
| ----- | ----- |
| Default fidelity threshold | mean ΔE94 < 3% (PRD target: 3–5%) |
| Frames per asset | 8 (camera path `orbit-8`) |
| Frame size | 512×512 |
| Deterministic seed | 42 (in `harness/page.html`) |
| PLY field order | x, y, z, nx, ny, nz, f_dc_{0..2}, f_rest_{0..44}, opacity, scale_{0..2}, rot_{0..3} |
| Quaternion order on disk | (w, x, y, z) — flipped to (x, y, z, w) on import |

### One-line questions for the user if you get stuck

- Should the fidelity threshold be tighter or looser than mean ΔE94 < 3%?
- Should we run all v0.1.1 numbers on a single machine, or accept cross-machine variance?
- Should bicycle (855 MB) be included in v0.1.1 even though rendering it will take ~5 min × 3 presets?
- After v0.1.1 ships, which of the 6 strategic moves above is the priority?

---

## Final note

The Cowork session pushed hard on breadth: scaffolding, specs, fixtures, real-data demo, SplatBench v0. The numbers we have are honest and the architecture is solid. **The single most valuable thing you can do for this project is close the visual-fidelity gap.** Until that's done, every external claim ("22× compression") lands the same skeptical question. After it's done, design-partner outreach, standards submissions, and a public landing page all become 10× more credible.

Good luck. The code is in good shape — if something feels wrong, trust your instinct and check the spec.

— previous session, signing off
