# build-hero — reusable rebuild pipeline for the homepage hero asset

## What

`apps/web/public/hero-scene/` is the live 3D Gaussian Splat scene rendered
above the fold on **splatforge.dev**. It loads as a chunked glTF
(`scene.gltf` + `buffers/chunk_*.bin`) through the in-tree WebGPU viewer.
Replacing it requires re-running a deterministic preprocess +
`splatforge optimize` pipeline so the output stays small enough to ship,
tight enough to frame on mobile, and rendered without floater halos.

## Why pre-process at all

The source is a vanilla Inria 3DGS PLY (~5M splats at 30k iterations).
Two things go wrong if we feed that directly to `splatforge optimize`:

1. **Floater halos pad the bbox.** Inria 3DGS densification creates
   *clustered* low-opacity floaters far from the subject. The in-binary
   `FloaterPrune` pass (real k-NN since commit `005a311`) catches
   *isolated* floaters but cannot drop a cluster of 50 floaters that all
   support each other. Those clusters bake into the scene bbox, the
   viewer auto-frames the bbox, and the actual subject ends up dim and
   tiny in the center while the camera dollies out to clear the halo.

2. **Splat count exceeds the mobile budget.** 5M @ web-mobile preset
   produces a ~120MB blob. Mobile target is <20MB total, ideally ~10MB
   compressed.

`tools/build-hero.py` solves both:

- **Per-axis percentile bbox crop** measured *against high-opacity splats
  only*. Floater halos are mostly low-opacity, so they do not pull the
  bbox out — we crop at `[0.03, 0.97]` of the high-opacity distribution
  on each axis independently. This is far more aggressive than k-NN
  pruning because it does not care whether the halo is clustered.
- **Importance-rank decimation**: keep the top-N by
  `sigmoid(opacity) * mean(exp(scale))^2` (a screen-space-contribution
  proxy). Stable, deterministic, view-independent.

The result is a clean ~150k-splat PLY that `splatforge optimize` can
quantize to a small chunked glTF without losing visual quality.

## How to rebuild

```bash
# 1. Build the release binary (one time, or after optimize-pass changes).
cargo build --release -p splatforge-cli

# 2. Run the pipeline.
tools/build-hero.sh /tmp/bonsai-30k/inria_3dgs_iter30k.ply
```

That single invocation reproduces the current production hero with
defaults (`--decimate 150000`, `--axis-pct 0.03,0.97`, `--out
apps/web/public/hero-scene`).

Full flags:

```bash
tools/build-hero.sh <source.ply> \
    [--decimate 150000] \
    [--axis-pct 0.03,0.97] \
    [--out apps/web/public/hero-scene]
```

The script:

1. Cleans stale chunk files in the output dir (critical — see failures).
2. Pre-processes the PLY into a tmp file.
3. Runs `splatforge optimize --preset web-mobile --chunked
   --out <out>/scene.gltf`.
4. Parses the gltf and asserts `splatCount > 50_000`.
5. Prints final splat count, bbox extent, and total bytes.

Exits non-zero with a clear error on any failure.

## Common failures

- **Stale chunks not cleaned.** If a previous build produced
  `chunk_0..7.bin` and the new build only writes `chunk_0..3.bin`, the
  orphaned chunks 4..7 ship with the site and the viewer either ignores
  them (wasted bytes) or attempts to load them (404s). `build-hero.sh`
  removes `buffers/chunk_*.bin` and `scene.gltf` before invoking
  `optimize` — do not skip this if you call the steps manually.

- **Wrong PLY layout (Scaffold-GS vs vanilla 3DGS).** The preprocessor
  only handles the vanilla Inria 3DGS property layout
  (`x/y/z/scale_0..2/rot_0..3/opacity/f_dc_*/f_rest_*`). Scaffold-GS PLYs
  embed anchor/offset/feature properties that are not splats — the
  preprocessor will exit with `error: PLY is missing required vanilla
  3DGS properties: ...`. Re-train with vanilla 3DGS, or write a separate
  decoder.

- **Oversized output.** If `splatCount` after decimation is still
  producing a multi-MB scene, drop `--decimate` (try 100000 or 80000).
  Per-axis crop also bounds the splat count — a wider `--axis-pct`
  (e.g. `0.01,0.99`) keeps more periphery and inflates byte count.

- **Dim render due to wide bbox + framing mismatch.** If you see a tiny,
  dim subject in the live page after rebuild, the bbox is too wide:
  the viewer auto-frames the bbox extent, so a halo at radius 12 means
  the camera dollies 12 units away. Tighten `--axis-pct` (e.g.
  `0.05,0.95`) and rebuild. Watch the "extent" line in the script
  output — for the bonsai scene a healthy extent is roughly
  `[3, 4, 3]`.

## Visual verification checklist (operator MUST run before commit)

1. **Build the web app:**
   ```bash
   pnpm --filter @splatforge/web build
   ```
   Build must pass; the gltf is loaded eagerly so a broken scene fails
   the build, not just the runtime.

2. **Serve locally:**
   ```bash
   pnpm --filter @splatforge/web preview
   ```
   Open `http://localhost:4321/`.

3. **Screenshot via Playwright** (or your tool of choice) at desktop
   (1440x900) and mobile (390x844) viewports. Both screenshots must
   show the subject occupying roughly the center 40-60% of the viewport
   — not a tiny dot, not clipping out of frame.

4. **Look at it.** Rotate (the hero auto-rotates); confirm no floater
   halo flares as the camera circles. If you see drifting low-opacity
   dots far from the bonsai, the crop was too generous.

5. Only then `git add apps/web/public/hero-scene/ && git commit`.

## Sample invocation (production hero)

This is what the current production hero was built with — DO NOT run
this on a whim, it overwrites the live asset:

```bash
tools/build-hero.sh /tmp/bonsai-30k/inria_3dgs_iter30k.ply \
    --decimate 150000 \
    --axis-pct 0.03,0.97 \
    --out apps/web/public/hero-scene
```

## Files

- `tools/build-hero.sh` — entry point.
- `tools/build-hero.py` — preprocessor (per-axis crop + importance decimation).
- `tools/build-hero-test.sh` — synthetic-PLY smoke test (<30s).
- `tools/build-hero.md` — this file.

## Test

```bash
cargo build --release -p splatforge-cli  # one time
tools/build-hero-test.sh
```

Synthesizes a fake 88k-splat PLY with a halo, runs the full pipeline at
reduced decimation, and asserts that the gltf parses with
`splatCount > 50000` and at least one chunk file was written.
