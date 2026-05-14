# SplatForge — handoff to next Claude Code session

**Handoff date:** 2026-05-14
**Last release tag:** `v0.1.1`
**Owner:** Monte (`montabano1@gmail.com`)
**Repo:** https://github.com/montabano1/SplatForge (**public** since 2026-05-14)
**Live preview:** https://splatforge-montabano1-gmailcoms-projects.vercel.app

---

## TL;DR

SplatForge shipped v0.1.0 (Phase 0–2: CLI + viewer SDK + SplatBench corpus)
yesterday, then v0.1.1 today which closed the "visual fidelity pending"
gap, wired the viewer SDK end-to-end for the first time, shipped the
public Astro landing page, drafted SPEC-0011/0012 (OpenUSD), patched CI,
and stood up the Phase 3 hosted-API + Modal-worker scaffold. Repo is now
public; pushes auto-deploy via Vercel.

**Where to start tomorrow:** SPEC-0013 (`KHR_mesh_quantization`) has three
`#[ignore]`'d acceptance tests waiting; implementing them is the single
highest-leverage piece of work remaining for v0.2. After that, the open
items are: wire Vercel Blob into apps/api, deploy apps/api to the
DigitalOcean droplet at `167.99.231.209` (Monte to provide SSH access),
and run the GPU fidelity rerun once the Modal image build completes.

---

## What shipped in v0.1.1 (already pushed to `main`)

### Visual-fidelity benchmark — the moat-critical piece

Every SplatBench scene now publishes a perceptual-degradation column:

```
benches/reports/
├── fidelity-v0.json              # raw per-frame ΔE94 / pixelmatch / SSIM
├── splatbench-v0.{md,json,html}  # leaderboard with new Fidelity column
└── frames/                       # 168 PNGs (gitignored — reproducible)
```

| Scene                       | web-mobile mean ΔE94 | size-min mean ΔE94 | Bucket             |
| --------------------------- | -------------------: | -----------------: | ------------------ |
| `bonsai_mipnerf360_iter7k`  | **0.60%**            | 0.64%              | pass / pass        |
| `bicycle_mipnerf360_iter7k` | 2.86%                | 2.60%              | borderline / borderline |
| `splatbench_product_proxy`  | 0.02%                | 0.04%              | pass / pass        |
| `splatbench_indoor_proxy`   | 0.03%                | 0.10%              | pass / pass        |
| `splatbench_floater_proxy`  | 0.09%                | **14.48%**         | pass / **fail**    |
| `splatbench_outdoor_proxy`  | 0.02%                | 0.09%              | pass / pass        |
| `splatbench_dense_proxy`    | 0.00%                | 0.02%              | pass / pass        |

Floater's size-min failure is by design (the scene is a "noisy-capture"
proxy with mostly low-opacity splats that size-min prunes); leaving it
on the leaderboard is honest. **All seven web-mobile rows pass the 3%
attentive-observer threshold.**

The runner: `tests/visual/scripts/splatbench-fidelity.mjs`. Honors:

- `SBENCH_SCENES=<csv>` to limit which scenes run
- `SBENCH_RESUME=1` to pick up a previous crash from the JSON
- `SBENCH_RENDERER=webgl2|webgpu`
- `SBENCH_RENDER_TIMEOUT_MS=<ms>` (default 1.8 M — 30 min — for big scenes)
- `SBENCH_CHROME_FLAGS="..."` to force hardware-accel paths (used by the
  Modal GPU rerun; see `apps/fidelity-gpu/`)
- `SBENCH_HEADLESS=0` to debug visually

Caveat baked into every number: SwiftShader. See "GPU rerun" below.

### `@splatforge/viewer` is now actually runnable

Previously the SDK tsc-compiled but had never been exercised in a real
browser. v0.1.1 fixed four structural wire-format mismatches between
the Rust glTF writer and the JS viewer:

1. **Top-level scene metadata.** Rust now emits
   `extensions.KHR_gaussian_splatting.{splatCount, bbox, shDegree}` at the
   asset root + POSITION accessor `min`/`max` (glTF 2.0 §3.6.2.4 requires
   the latter anyway).
2. **Chunk URIs.** Per-chunk `SF_spatial_streaming_index` records now
   carry `uri` (matching the buffer's URI), not just `buffer` (index).
3. **SoA decode in JS.** `splatforge-gltf` emits one bufferView per
   attribute (POSITION → ROTATION → SCALE → OPACITY → COLOR_DC). The
   JS viewer reads these per-attribute slices and re-interleaves into
   the existing `DecodedSplat` AoS layout at decode time.
4. **Camera-path execution.** The viewer's `cameraPath: 'orbit-8'` mode
   now actually drives 8 orbit poses and emits `frameRendered` per pose;
   the visual harness binds to that event for capture.

The smoke test is `splatforge diff fixtures/tiny/basic_binary.ply <gltf> --out /tmp/tiny-diff`
— mean = 0.0, 8 frames captured, splats render as 3 gray blobs.

Also: WebGL2 renderer's per-frame sort was O(n²) insertion sort; replaced
with a stable O(n log n) paired-index sort so it scales past 10 K splats.

### apps/web — Astro static landing page

Live at https://splatforge-montabano1-gmailcoms-projects.vercel.app.
Section layout informed by a Lazyweb research pass across dev-tool,
open-source, and 3D-graphics landing-page references (see
`.lazyweb/design-research/splatforge-landing-2026-05-13/report.html`).

Architecture:

- `apps/web/scripts/sync-data.mjs` copies `benches/reports/*.json` into
  `apps/web/src/data/` at build time so the Astro project stays
  self-contained for Vercel's per-app-dir build.
- `vercel.json` at repo root tells Vercel to use `apps/web/dist/` as
  the output directory after the monorepo `pnpm run build`.
- `src/components/Hero.astro` — canvas-based animated Gaussian-blob
  visual (deterministic xorshift32 seed; not the real viewer SDK — v0.3).
- `src/components/Leaderboard.astro` — interactive preset toggle; reads
  the synced JSON. Fidelity column lights up with pass/borderline/fail
  colour pills automatically when the JSON has a `fidelity` block.
- `src/components/Install.astro` — tabbed Cargo / Homebrew / from-source
  with copy buttons.
- `src/components/TryIt.astro` — drag-drop UI (placeholder; intercepts
  drops and routes to the design-partner GitHub issue).

### CI workflows patched for green first run

The three `.github/workflows/` files had latent first-run failures (over-
broad `pnpm -r` filter, missing viewer build before Playwright, benchmark
report written to stdout not `benches/reports/`). All three surgically
fixed.

---

## What's in flight (uncommitted-but-incomplete, this session is mid-flight)

### Phase 3 hosted API + Modal worker

```
apps/
├── api/                   # Axum endpoint — splatforge-api crate
│   ├── Cargo.toml         # workspace member; routes:
│   ├── src/main.rs        #   POST /v1/jobs        create + presign upload
│   ├── src/store.rs       #   GET  /v1/jobs/:id    poll status + results
│   └── src/modal_client.rs#   POST /v1/jobs/:id/upload  confirm + enqueue
├── worker/                # Modal Python — splatforge-worker app
│   ├── README.md
│   └── worker.py          # /enqueue + /healthz + run_optimize
└── fidelity-gpu/          # Modal Python — one-shot SplatBench rerun on T4
    └── run.py             # writes fidelity-v0-hwaccel.json
```

**Modal worker is LIVE** at:

- `POST https://montabano1--enqueue.modal.run`
- `GET  https://montabano1--healthz.modal.run`

Image bakes `splatforge` CLI from the pinned git tag (`SPLATFORGE_REF=v0.1.1`).
Two-CPU x 4-GB container with `/data` volume for staging.

**apps/api is buildable but NOT deployed.** Three things block it:

1. **No Blob backend wired.** `BlobBackend::presign_upload` returns a
   stub URL today. Need to call Vercel Blob's presign API (Vercel CLI
   `vercel blob store add splatforge-blobs` provisions; token goes in
   `BLOB_READ_WRITE_TOKEN` env). Monte is OK with us provisioning via CLI.
2. **No public host.** Options:
   - **DigitalOcean droplet** at `167.99.231.209` (1 vCPU, 2 GB, NYC1) —
     Monte's preferred. Already paid (sunk $14.40/mo). Need SSH key.
     Already at 38% CPU / 50% memory baseline; some other project lives
     there — coexist on a different port.
   - **Modal `web_endpoint`** — only works for Python; we'd need a thin
     Python proxy around the Rust binary. Less clean.
   - **Vercel Function** — Rust isn't first-class; ruled out.
3. **No CI hook.** Eventually the worker should auto-deploy on tag push
   to keep `SPLATFORGE_REF` in sync.

### OpenUSD crate (SPEC-0011 implementation)

```
crates/splatforge-usd/
├── Cargo.toml
├── src/lib.rs         # write_usda + read_usda + USDA round-trip
└── tests/roundtrip.rs # 3 tests pass on the 3-splat fixture
```

`write_usda` produces canonical USDA emitting `ParticleField3DGaussianSplat`
prims with flipped (w, x, y, z) quaternion order, custom
`splatforge:shCoefficients` for SH-rest, `point3f[]` / `quatf[]` /
`float3[]` / `float[]` / `color3f[]` typed arrays.

`write_usdc` is stubbed — Pixar's Crate binary container needs a custom
writer (or vendoring Pixar's C++ via FFI). Tracked in SPEC-0011 §"Open
questions".

`read_usda` parses the canonical layout back. **Not yet a real-world USD
reader** — tools like Houdini/Maya emit USDA with subtly different
whitespace/attribute ordering. v0.2 scope.

3 tests pass: round-trip, empty-scene rejection, quaternion-order check.
Total workspace test count is now **46 passed + 3 ignored** (the 3 are
the SPEC-0013 specs below).

### SPEC-0013 — KHR_mesh_quantization

`specs/0013-gltf-mesh-quantization.md` is fully drafted. Implementation is
the obvious next slice of work — three `#[ignore]`'d tests in
`crates/splatforge-gltf/tests/mesh_quantization.rs` pin the acceptance
criteria. Removing the ignore flag drives the implementation.

Expected wins on bonsai web-mobile: glTF buffer 59.4 MB → 30.9 MB (1.9×).
Brings the glTF-only delivery path within ~2.5× of the SPZ payload.

### GPU fidelity rerun (running in background as of handoff)

`apps/fidelity-gpu/run.py` is deployed. Currently running `run_corpus(skip_bicycle=True)`
on a T4 — image is mid-build at the time of handoff (CUDA base + Rust +
Chromium + 1.13 GB of pre-staged Mip-NeRF360 scenes; ~10-min cold build).

On success the resulting `fidelity-v0-hwaccel.json` lives in the
`splatforge-fidelity-results` Modal Volume. Pull it down with:

```bash
python3 -m modal volume get splatforge-fidelity-results \
    fidelity-v0-hwaccel.json benches/reports/
```

Then commit and re-run `benches/splatbench-update.mjs` — the leaderboard
updater auto-detects which JSON is present.

---

## Concrete next steps (priority-ordered)

1. **Implement SPEC-0013.** Flip the three `#[ignore]`s in
   `crates/splatforge-gltf/tests/mesh_quantization.rs`, wire integer
   accessors in `splatforge_gltf::pack_chunk`, surface a `quantize: bool`
   on `WriteOpts`, default the relevant presets to `quantize=true` in
   `splatforge-optimize`. ~1 day.

2. **Provision Vercel Blob.**
   ```bash
   vercel blob store add splatforge-blobs --scope montabano1-gmailcoms-projects
   ```
   Wire `BlobBackend::presign_upload` in `apps/api/src/store.rs` against
   the `@vercel/blob` HTTPS presign endpoint. ~half day.

3. **Deploy apps/api to the DigitalOcean droplet.**
   - Get the SSH key onto the droplet (Monte to paste).
   - Cross-compile `splatforge-api` for `x86_64-unknown-linux-gnu`.
   - Install as a systemd unit at `splatforge-api.service`.
   - Front with Caddy (free auto-TLS) for `api.splatforge.dev`.
   ~1 day.

4. **Run the GPU fidelity rerun.** If still building / completed but not
   committed at session start: pull from the Modal Volume, commit the
   resulting JSON, re-run the leaderboard updater.

5. **Khronos KHR_gaussian_splatting conformance submission.** Open a
   discussion thread on the WG repo; offer the SplatBench corpus +
   fidelity numbers as conformance assets. The PRD §"Benchmark and moat
   strategy" makes this the highest-leverage move for the moat.

6. **Design-partner outreach kit.** PRD §"Design partner plan" — recruit
   5 partners with real assets. Email template + intake form + asset-
   license memo.

7. **OpenUSD validation.** Run the USDA we emit through `usdcat` or
   `usdview` in a Docker container; close the SPEC-0011 §"Open questions"
   items one by one.

8. **README polish + GitHub repo metadata.** Now that the repo is public,
   add topic tags (`gaussian-splatting`, `webgpu`, `3d`, `splat`), set the
   GitHub repo description, add the live site URL to the sidebar.

---

## Working agreements (still in force)

These haven't changed from the v0.1.0 handoff; restating for completeness:

1. **Spec-driven.** Every new feature gets a spec or amends one.
2. **Determinism is sacred.** Same input + same config = byte-identical output.
3. **Tests before refactors.** Even cosmetic refactors get a failing test first if behavior changes.
4. **Snapshots are sacred.** Don't update snapshot files unless the spec change requires it.
5. **No proprietary container formats** as the default output.
6. **DCO sign-off** on every commit (`git commit -s`).
7. **`unwrap()` and `panic!()` are forbidden in library code.**
8. **Conventional Commits** style.
9. **Public API has docs.**
10. **`cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`** all green.

**New for v0.2:**

11. **No Claude attribution in GitHub.** Commit messages, PR descriptions,
    issue bodies — strip the `Co-Authored-By: Claude` line and any
    "Generated with Claude Code" tail. Repo is public; history reads as
    Monte's work.
12. **Push autonomy is on.** This session and future ones push to remote
    after tests + build pass without waiting for explicit permission.
    Force-push and history-rewrite operations are also authorized.

---

## Quick reference

### Build & test (verified green at handoff)

```bash
cargo fmt --all -- --check          # clean
cargo clippy --workspace --all-targets -- -D warnings  # clean
cargo test --workspace --release    # 46 passed + 3 ignored
pnpm -F @splatforge/viewer run test # 28 passed
pnpm -F @splatforge/report-ui run test  # 5 passed
pnpm -F @splatforge/web run build   # clean
```

### Useful files

- `apps/web/src/components/*.astro` — landing-page sections
- `apps/api/src/main.rs` — hosted API entrypoint (not yet deployed)
- `apps/worker/worker.py` — Modal worker (deployed)
- `apps/fidelity-gpu/run.py` — Modal GPU rerun (deployed; one run in flight at handoff)
- `crates/splatforge-usd/src/lib.rs` — OpenUSD writer/reader
- `crates/splatforge-gltf/tests/mesh_quantization.rs` — SPEC-0013 specs
- `tests/visual/scripts/splatbench-fidelity.mjs` — fidelity runner
- `benches/splatbench-update.mjs` — leaderboard regen from fidelity JSON
- `docs/fidelity-on-real-gpu.md` — GPU rerun recipe + caveats
- `.lazyweb/design-research/splatforge-landing-2026-05-13/` — landing-page references (gitignored)

### Critical constants

| Thing                          | Value |
| ------------------------------ | ----- |
| Fidelity threshold             | mean ΔE94 < 3% AND max ΔE94 < 8% |
| Frames per asset               | 8 (camera path `orbit-8`)         |
| Frame size                     | 512 × 512                         |
| Deterministic harness seed     | 42                                |
| PLY field order                | x y z nx ny nz f_dc_{0..2} f_rest_{0..44} opacity scale_{0..2} rot_{0..3} |
| Quaternion order on disk (glTF/PLY) | (w, x, y, z) — flipped to (x, y, z, w) on import |
| Quaternion order on disk (USDA) | (w, x, y, z) — flipped to (x, y, z, w) on import |
| Modal spending cap (overnight) | $5 (~$0.60 used at handoff)       |

### One-liners

```bash
# Smoke-test the viewer end-to-end on the tiny fixture:
./target/release/splatforge diff fixtures/tiny/basic_binary.ply fixtures/tiny/basic_binary.ply --out /tmp/tiny-diff && open /tmp/tiny-diff/frames/before/0001.png

# Regenerate the SplatBench corpus + fidelity (from scratch, ~45 min on Mac aarch64):
make bench-splatbench-synth
make bench-splatbench-real
node tests/visual/scripts/splatbench-fidelity.mjs
node benches/splatbench-update.mjs

# Re-deploy the Modal worker after a CLI bump:
SPLATFORGE_REF=v0.1.2 python3 -m modal deploy apps/worker/worker.py

# Local dev for the API (env vars left unset → stubbed Blob + worker):
cargo run -p splatforge-api --release
```

---

## Final note

v0.1.1 closed the single biggest credibility gap the project had — the
"22× compression" claim now lands with a "and here's proof the image
quality holds" footnote. The moat strategy is intact: open-source the
toolchain + corpus, monetize the hosted side (apps/api + worker), keep
standards-leadership posture.

Best next move for v0.2 is SPEC-0013. The three failing tests are the
exec spec; just implement them.
