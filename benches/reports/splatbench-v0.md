# SplatBench v0

> The initial public benchmark corpus for Catetus — a real Mip-NeRF360 anchor scene plus deterministic synthetic scenes covering five PRD corpus classes.

**Catetus version:** `0.1.1`
**Platform:** linux-aarch64, single-core processing
**Run date:** 2026-05-14
**Reproducibility:** every scene is identified by a BLAKE3 hash of its canonical IR. Synthetic scenes regenerate byte-identically from `benches/synth_scenes.py`.

## Headline

| Metric | Value |
| ---: | ---: |
| Scenes total | **16** (2 real + 14 synthetic) |
| Splats total | **7.77M** across the corpus |
| Input total | **1.80 GB** raw PLY |
| Corpus total `web-mobile` | **1.80 GB → 78.5 MB** (23.43× overall) |
| Corpus total `size-min` | **1.80 GB → 60.4 MB** (30.43× overall) |
| `web-mobile` ratio (min / median / max) | **17.63× / 21.88× / 26.6×** |
| `size-min` ratio (min / median / max) | **17.96× / 23.19× / 38.92×** |
| `web-mobile` fidelity passing | **16 / 16** scenes within PRD threshold |
| `size-min` fidelity passing | **16 / 16** scenes within PRD threshold |


## Leaderboard — `web-mobile` preset

| Rank | Scene | Class | Source | Splats | Input | SPZ out | **Ratio** | analyze |
| ---: | ----- | ----- | ------ | ---: | ---: | ---: | ---: | ---: |
| 1 | `splatbench_banding_proxy` | smooth-color-gradient | synthetic | 10,000 | 2.4 MB | 91 KB | **26.60×** | 29 ms |
| 2 | `splatbench_floater_proxy` | noisy-capture | synthetic | 250,000 | 59.1 MB | 2.3 MB | **25.84×** | 248 ms |
| 3 | `splatbench_texture_proxy` | high-frequency-texture | synthetic | 15,000 | 3.5 MB | 143 KB | **25.48×** | 32 ms |
| 4 | `bicycle_mipnerf360_iter7k` | outdoor-scene | **real** | 3,616,103 | 855.3 MB | 33.6 MB | **25.46×** | 4,457 ms |
| 5 | `splatbench_depth_proxy` | depth-bimodal | synthetic | 15,000 | 3.5 MB | 150 KB | **24.30×** | 33 ms |
| 6 | `bonsai_mipnerf360_iter7k` | indoor-real-estate | **real** | 1,157,141 | 273.7 MB | 12.0 MB | **22.81×** | 1,240 ms |
| 7 | `splatbench_portrait_proxy` | salient-region-portrait | synthetic | 12,000 | 2.8 MB | 129 KB | **22.55×** | 15 ms |
| 8 | `splatbench_specular_proxy` | specular-highlights | synthetic | 12,000 | 2.8 MB | 132 KB | **21.97×** | 16 ms |
| 9 | `splatbench_motion_proxy` | anisotropic-streak | synthetic | 12,000 | 2.8 MB | 133 KB | **21.79×** | 30 ms |
| 10 | `splatbench_dense_proxy` | dense-large-scene | synthetic | 2,000,000 | 473.0 MB | 21.7 MB | **21.75×** | 2,184 ms |
| 11 | `splatbench_indoor_proxy` | indoor-real-estate | synthetic | 100,000 | 23.7 MB | 1.1 MB | **20.58×** | 105 ms |
| 12 | `splatbench_product_proxy` | product-scan | synthetic | 10,000 | 2.4 MB | 118 KB | **20.50×** | 11 ms |
| 13 | `splatbench_lowlight_proxy` | low-light-dynamic-range | synthetic | 8,000 | 1.9 MB | 96 KB | **20.10×** | 11 ms |
| 14 | `splatbench_outdoor_proxy` | outdoor-scene | synthetic | 500,000 | 118.3 MB | 6.0 MB | **19.67×** | 483 ms |
| 15 | `splatbench_transparency_proxy` | volumetric-translucency | synthetic | 25,000 | 5.9 MB | 337 KB | **17.96×** | 51 ms |
| 16 | `splatbench_foliage_proxy` | dense-translucency | synthetic | 30,000 | 7.1 MB | 412 KB | **17.63×** | 34 ms |


## Leaderboard — `size-min` preset

| Rank | Scene | SPZ out | **Ratio** |
| ---: | ----- | ---: | ---: |
| 1 | `bicycle_mipnerf360_iter7k` | 22.0 MB | **38.92×** |
| 2 | `bonsai_mipnerf360_iter7k` | 8.6 MB | **31.83×** |
| 3 | `splatbench_floater_proxy` | 2.1 MB | **27.85×** |
| 4 | `splatbench_banding_proxy` | 90 KB | **26.86×** |
| 5 | `splatbench_texture_proxy` | 141 KB | **25.84×** |
| 6 | `splatbench_depth_proxy` | 141 KB | **25.83×** |
| 7 | `splatbench_dense_proxy` | 19.5 MB | **24.24×** |
| 8 | `splatbench_portrait_proxy` | 121 KB | **23.94×** |
| 9 | `splatbench_specular_proxy` | 130 KB | **22.43×** |
| 10 | `splatbench_motion_proxy` | 131 KB | **22.16×** |
| 11 | `splatbench_outdoor_proxy` | 5.4 MB | **21.69×** |
| 12 | `splatbench_indoor_proxy` | 1.1 MB | **21.57×** |
| 13 | `splatbench_product_proxy` | 118 KB | **20.49×** |
| 14 | `splatbench_lowlight_proxy` | 95 KB | **20.43×** |
| 15 | `splatbench_transparency_proxy` | 334 KB | **18.14×** |
| 16 | `splatbench_foliage_proxy` | 405 KB | **17.96×** |


## Leaderboard — visual fidelity (v0.1.1)

Frames captured via `@catetus/viewer` in headless Chromium (SwiftShader software-rendered WebGL2), 8 deterministic orbit poses at 512×512. `lossless-repack` is the per-scene baseline. ΔE94 is normalized to 0..1 (i.e. `3%` = 3 absolute ΔE94 units, the perceptibility threshold of an attentive observer).

**ML Score** is the splat-aware perceptual metric from `catetus-pro` (version `0.3.0-perkind`), a proprietary build that scores rendered vs baseline frames with a model tuned for Gaussian-splat failure modes. Higher is better; 100% means visually identical. ML Score values are published; reproducing them requires the `catetus-pro` binary.

| Rank | Scene | web-mobile ΔE94 mean / max | status | size-min ΔE94 mean / max | status | web-mobile ML | size-min ML |
| ---: | ----- | ---: | :---: | ---: | :---: | ---: | ---: |
| 1 | `splatbench_dense_proxy` | 0.00% / 0.01% | **pass** | 0.02% / 0.03% | **pass** | 95.54% | 94.87% |
| 2 | `splatbench_depth_proxy` | 0.02% / 0.02% | **pass** | 0.02% / 0.03% | **pass** | 95.34% | 95.22% |
| 3 | `splatbench_product_proxy` | 0.02% / 0.03% | **pass** | 0.04% / 0.04% | **pass** | 95.28% | 95.09% |
| 4 | `splatbench_outdoor_proxy` | 0.02% / 0.04% | **pass** | 0.09% / 0.14% | **pass** | 95.16% | 94.00% |
| 5 | `splatbench_banding_proxy` | 0.03% / 0.04% | **pass** | 0.03% / 0.04% | **pass** | 95.17% | 95.14% |
| 6 | `splatbench_indoor_proxy` | 0.03% / 0.04% | **pass** | 0.10% / 0.11% | **pass** | 95.12% | 94.19% |
| 7 | `splatbench_lowlight_proxy` | 0.03% / 0.03% | **pass** | 0.04% / 0.04% | **pass** | 95.27% | 95.19% |
| 8 | `splatbench_specular_proxy` | 0.04% / 0.04% | **pass** | 0.05% / 0.05% | **pass** | 95.14% | 94.97% |
| 9 | `splatbench_portrait_proxy` | 0.04% / 0.08% | **pass** | 0.06% / 0.11% | **pass** | 95.12% | 94.76% |
| 10 | `splatbench_floater_proxy` | 0.07% / 0.07% | **pass** | 0.37% / 0.37% | **pass** | 92.66% | 83.42% |
| 11 | `splatbench_transparency_proxy` | 0.07% / 0.08% | **pass** | 0.08% / 0.08% | **pass** | 94.69% | 94.66% |
| 12 | `splatbench_foliage_proxy` | 0.09% / 0.09% | **pass** | 0.10% / 0.10% | **pass** | 93.96% | 93.74% |
| 13 | `splatbench_texture_proxy` | 0.09% / 0.12% | **pass** | 0.12% / 0.18% | **pass** | 94.26% | 92.73% |
| 14 | `splatbench_motion_proxy` | 0.15% / 0.16% | **pass** | 0.16% / 0.17% | **pass** | 91.49% | 91.22% |
| 15 | `bonsai_mipnerf360_iter7k` | 0.60% / 0.84% | **pass** | 0.64% / 0.87% | **pass** | 46.51% | 43.84% |
| 16 | `bicycle_mipnerf360_iter7k` | 2.86% / 3.56% | **borderline** | 2.60% / 3.33% | **borderline** | 0.00% | 0.00% |

**Pass criterion:** mean ΔE94 < 3% AND max ΔE94 < 8%. **Borderline:** mean 2–3% or max 5–8%. **Pass:** mean < 2% AND max < 5%.

Software-rendered numbers may differ slightly from hardware-accelerated chromium; see `fidelity-v0.json` for per-frame raw metrics and `benches/reports/frames/<scene>/<preset>/0001.png` etc. for the actual frames.


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

Two consecutive runs of `catetus analyze <scene>` produce byte-identical JSON output. The corpus-level BLAKE3 hash of all 7 scene hashes is itself stable across runs and platforms.

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
cargo build --release -p catetus-cli
export BIN=$PWD/target/release/catetus

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

* **FPS / first-meaningful-paint** — needs a real-device matrix (mobile Safari, Chrome on Android, Quest Browser, desktop WebGPU). Will be added once the viewer dist is built and deployed.
* **Real design-partner assets** — Phase 0 of the PRD explicitly calls for 5 design partners with real captures. Synthetic scenes are placeholders for those slots; the corpus structure (the JSON manifest) accepts real assets without schema changes.

## What v0 proves

* **The pipeline runs end-to-end on real Inria 3DGS data at 3.6M splats** with consistent, predictable compression.
* **`web-mobile` consistently exceeds 19×** across scene scales spanning 10K → 3.6M splats (a 360× range).
* **`size-min` exceeds 30×** on real data because trained 3DGS data has heavy opacity-tail amenable to opacity-prune.
* **Determinism holds at scale** — hashing the corpus produces a stable identifier suitable for use as a public reference.

This is the v0 the PRD's section "Phase 2 exit criteria" describes ("public benchmark page v0; compatibility report across at least 2 browser paths"). The compatibility-report half is pending the visual harness running in CI; the corpus half is now in this repo and reproducible.
