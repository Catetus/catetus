# Canonical-11 leaderboard — SF baseline

**Preset:** `wmv-vq45-no-prune-tight`  ·  **Encoder:** `catetus 0.1.0`  ·  **Generated:** 2026-05-20T17:54:14Z

Canonical 11-scene corpus used by the Inria 3D Gaussian Splatting paper (Mip-NeRF 360 + Tanks-and-Temples + Deep Blending), with input PLYs taken from the [official 30k-iteration pretrained release](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip).

## Summary

| Metric | Value |
| ---: | ---: |
| Scenes | **11** |
| PSNR mean | **47.45 dB** (median 47.78, range 43.46–50.59) |
| SSIM mean | **0.9991** (median 0.9995) |
| Compression mean | **19.8×** (median 19.8×, range 16.6×–21.9×) |

## Per-scene results

| Scene | Splats | Input | Output | Ratio | PSNR | SSIM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 6,131,954 | 1520.7 MB | 69.50 MB | **21.9×** | 49.87 | 0.9998 |
| truck | 2,541,226 | 630.2 MB | 29.84 MB | **21.1×** | 47.78 | 0.9996 |
| garden | 5,834,784 | 1447.0 MB | 69.70 MB | **20.8×** | 48.38 | 0.9997 |
| stump | 4,961,797 | 1230.5 MB | 60.21 MB | **20.4×** | 48.85 | 0.9998 |
| room | 1,593,376 | 395.2 MB | 19.83 MB | **19.9×** | 50.59 | 0.9996 |
| kitchen | 1,852,335 | 459.4 MB | 23.24 MB | **19.8×** | 48.30 | 0.9995 |
| drjohnson | 3,405,153 | 844.5 MB | 43.06 MB | **19.6×** | 47.24 | 0.9988 |
| playroom | 2,546,116 | 631.4 MB | 32.24 MB | **19.6×** | 43.46 | 0.9973 |
| train | 1,026,508 | 254.6 MB | 13.36 MB | **19.1×** | 45.51 | 0.9986 |
| counter | 1,222,956 | 303.3 MB | 16.30 MB | **18.6×** | 44.93 | 0.9987 |
| bonsai | 1,244,819 | 308.7 MB | 18.65 MB | **16.6×** | 47.07 | 0.9991 |

## Methodology

- **Input:** Inria 3DGS pretrained, iteration 30000; PLYs verified by MD5 against `benches/scenes/canonical-11/MANIFEST.md`.
- **Encoder:** `catetus optimize --preset wmv-vq45-no-prune-tight --target glb`.
- **Renderer:** gsplat (PyTorch, SH degree 3) at 512x512, 8 frames per scene along an orbit camera path derived from the input bounding box.
- **PSNR:** RGB MSE → 10·log10(255²/MSE).
- **SSIM:** 8x8 non-overlapping blocks on luminance, Bessel-corrected variance.

Per-frame PSNR/SSIM and decoder reproducibility detail in [`canonical-11-sf.json`](./canonical-11-sf.json).

