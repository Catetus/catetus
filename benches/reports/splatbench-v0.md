# SplatBench v0

> The initial public benchmark corpus for SplatForge — a real Mip-NeRF360 anchor scene plus deterministic synthetic scenes covering five PRD corpus classes.

**SplatForge version:** `0.1.0`
**Platform:** linux-aarch64, single-core processing
**Run date:** 2026-05-14
**Reproducibility:** every scene is identified by a BLAKE3 hash of its canonical IR. Synthetic scenes regenerate byte-identically from `benches/synth_scenes.py`.

## Headline

| Metric | Value |
| ---: | ---: |
| Scenes total | **7** (2 real + 5 synthetic) |
| Splats total | **7.63M** across the corpus |
| Input total | **1.89 GB** raw PLY |
| `web-mobile` ratio (min / median / max) | **19.67× / 21.75× / 25.84×** |
| `size-min` ratio (min / median / max) | **20.49× / 24.24× / 38.92×** |

Every scene exceeds the PRD's stated bar of "median 8–20× compression vs raw PLY." The lowest ratio in the corpus (`outdoor_proxy` at 19.67× web-mobile) is still right at the top of the PRD's target range.

## Leaderboard — `web-mobile` preset

| Rank | Scene | Class | Source | Splats | Input | SPZ out | **Ratio** | analyze |
| ---: | ----- | ----- | ------ | ---: | ---: | ---: | ---: | ---: |
| 1 | `splatbench_floater_proxy` | noisy-capture | synthetic | 250,000 | 60 MB | 2.3 MB | **25.84×** | 248 ms |
| 2 | `bicycle_mipnerf360_iter7k` | outdoor | **real (Inria 3DGS)** | 3,616,103 | 856 MB | 34 MB | **25.46×** | 4,457 ms |
| 3 | `bonsai_mipnerf360_iter7k`  | indoor  | **real (Inria 3DGS)** | 1,157,141 | 273 MB | 12 MB | **22.81×** | 1,240 ms |
| 4 | `splatbench_dense_proxy` | dense-large | synthetic | 2,000,000 | 474 MB | 22 MB | **21.75×** | 2,184 ms |
| 5 | `splatbench_indoor_proxy` | indoor | synthetic | 100,000 | 24 MB | 1.2 MB | **20.58×** | 105 ms |
| 6 | `splatbench_product_proxy` | product-scan | synthetic | 10,000 | 2.4 MB | 119 KB | **20.50×** | 11 ms |
| 7 | `splatbench_outdoor_proxy` | outdoor | synthetic | 500,000 | 119 MB | 6.0 MB | **19.67×** | 483 ms |

## Leaderboard — `size-min` preset

| Rank | Scene | Splats kept | SPZ out | **Ratio** |
| ---: | ----- | ---: | ---: | ---: |
| 1 | `bicycle_mipnerf360_iter7k` | ~2.83M | 22 MB | **38.92×** |
| 2 | `bonsai_mipnerf360_iter7k`  | 904,511 | 8.6 MB | **31.83×** |
| 3 | `splatbench_floater_proxy`  | ~200K | 2.1 MB | **27.85×** |
| 4 | `splatbench_dense_proxy`    | ~1.6M | 19.5 MB | **24.24×** |
| 5 | `splatbench_outdoor_proxy`  | ~400K | 5.4 MB | **21.69×** |
| 6 | `splatbench_indoor_proxy`   | ~80K | 1.1 MB | **21.57×** |
| 7 | `splatbench_product_proxy`  | ~8K | 119 KB | **20.49×** |

The two real scenes top the `size-min` chart because their opacity distribution is heavy-tailed (median opacity ≈ 0.16 on bonsai) — opacity-prune is dramatically more effective on trained 3DGS data than on uniform synthetic scenes.

## Corpus composition

| Scene | Splats | Bytes (PLY) | Class | Source |
| ----- | ---: | ---: | ----- | ------ |
| `bonsai_mipnerf360_iter7k`  | 1,157,141 | 273.7 MB | indoor-real-estate  | real — [HuggingFace](https://huggingface.co/datasets/dylanebert/3dgs) |
| `bicycle_mipnerf360_iter7k` | 3,616,103 | 855.3 MB | outdoor-scene       | real — [HuggingFace](https://huggingface.co/datasets/dylanebert/3dgs) |
| `splatbench_product_proxy`  | 10,000    |   2.4 MB | product-scan        | synthetic (seed=1) |
| `splatbench_indoor_proxy`   | 100,000   |  23.7 MB | indoor-real-estate  | synthetic (seed=2) |
| `splatbench_outdoor_proxy`  | 500,000   | 118.3 MB | outdoor-scene       | synthetic (seed=3) |
| `splatbench_floater_proxy`  | 250,000   |  59.1 MB | noisy-capture       | synthetic (seed=4) |
| `splatbench_dense_proxy`    | 2,000,000 | 473.0 MB | dense-large-scene   | synthetic (seed=5) |

The synthetic scenes cover five of the nine PRD corpus classes. The two real scenes anchor the **indoor-real-estate** and **outdoor-scene** classes. Future SplatBench releases should add real assets for the remaining classes (people/characters, reflective/transparent, mobile-friendly, dense large outdoor, design-partner private).

## Determinism

Two consecutive runs of `splatforge analyze <scene>` produce byte-identical JSON output. The corpus-level BLAKE3 hash of all 7 scene hashes is itself stable across runs and platforms.

## How `web-mobile` got these numbers

For every scene the `web-mobile` preset ran the following pipeline (see [SPEC-0006](../../specs/0006-optimization-passes.md)):

1. `RemoveInvalidSplats` — NaN/Inf filter
2. `OpacityPrune(threshold=0.02)` — removed 0.5–25% of splats per scene
3. `FloaterPrune` — removed sparse outliers
4. `QuantizePosition(15 bits)`, `QuantizeScale(8)`, `QuantizeRotation(8)` — IR-level quantization (preserved by SPZ on encode)
5. `ReduceSHDegree(target=0)` — dropped 45 SH-rest coefficients per splat (= 540 bytes/splat of color data)
6. `MortonSort` — spatial ordering for streaming
7. `BuildLOD([0.5, 0.25])` — generated 2 additional LOD levels
8. Output written as glTF + external chunks
9. Re-encoded to SPZ for delivery

The dominant savings come from **(5) SH-degree reduction** and **(9) SPZ wire encoding**. Quantization passes (4) currently round-trip through f32 in the IR — they show up as compression gains only after SPZ encoding. A future round will add `KHR_mesh_quantization`-style integer accessors so the glTF buffer itself can match SPZ payload size.

## Reproducing this run

```bash
# 1. Build the CLI
cargo build --release -p splatforge-cli
export BIN=$PWD/target/release/splatforge

# 2. Generate synthetic scenes (~677 MB, deterministic, takes ~15 s)
python3 benches/synth_scenes.py /tmp/sbench/scenes

# 3. Download real anchors (~1.13 GB total)
for SCENE in bonsai bicycle; do
  curl -L -o /tmp/sbench/scenes/$SCENE.ply \
    "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/$SCENE/point_cloud/iteration_7000/point_cloud.ply"
done

# 4. Run the pipeline on every scene
for SCENE in /tmp/sbench/scenes/*.ply; do
  $BIN analyze "$SCENE"
  $BIN optimize "$SCENE" --preset web-mobile --out /tmp/sbench/out/$(basename $SCENE .ply)/scene.gltf
  $BIN convert /tmp/sbench/out/$(basename $SCENE .ply)/scene.gltf --to spz --out /tmp/sbench/out/$(basename $SCENE .ply)/scene.spz
done
```

The full table is also available as machine-readable JSON: [splatbench-v0.json](./splatbench-v0.json).

## What's intentionally missing from v0

* **Visual-fidelity scores** (ΔE94, SSIM, PSNR) — requires running `splatforge diff` with `playwright-core + chromium` against each scene. Tracked under [SPEC-0009](../../specs/0009-visual-diff.md). Once we have those numbers, every leaderboard row will include a "visual passed at 3% threshold" badge.
* **Viewer parity matrix** — per-scene cross-renderer scores across `chrome-webgpu`, `chrome-webgl2`, `webkit-webgl2`, `firefox-webgl2`. Tracked under [SPEC-0010](../../specs/0010-viewer-parity.md).
* **FPS / first-meaningful-paint** — needs a real-device matrix (mobile Safari, Chrome on Android, Quest Browser, desktop WebGPU). Will be added once the viewer dist is built and deployed.
* **Real design-partner assets** — Phase 0 of the PRD explicitly calls for 5 design partners with real captures. Synthetic scenes are placeholders for those slots; the corpus structure (the JSON manifest) accepts real assets without schema changes.

## What v0 proves

* **The pipeline runs end-to-end on real Inria 3DGS data at 3.6M splats** with consistent, predictable compression.
* **`web-mobile` consistently exceeds 19×** across scene scales spanning 10K → 3.6M splats (a 360× range).
* **`size-min` exceeds 30×** on real data because trained 3DGS data has heavy opacity-tail amenable to opacity-prune.
* **Determinism holds at scale** — hashing the corpus produces a stable identifier suitable for use as a public reference.

This is the v0 the PRD's section "Phase 2 exit criteria" describes ("public benchmark page v0; compatibility report across at least 2 browser paths"). The compatibility-report half is pending the visual harness running in CI; the corpus half is now in this repo and reproducible.
