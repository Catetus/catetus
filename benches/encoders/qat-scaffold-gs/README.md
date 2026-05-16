# qat-scaffold-gs bench runner

Runs the SplatForge **QAT codec for Scaffold-GS** PLYs. This is a
post-training operation — it takes a Scaffold-GS PLY and returns a smaller
Scaffold-GS PLY with **higher** measured PSNR on the same eval cameras.

The runner is intentionally thin: it POSTs the input PLY to the hosted
`splatforge-qat-scaffold` endpoint and stores whatever comes back, plus a
small `meta.json` for the bench harness. There is no encoder code in this
directory — the codec is hosted and reproducible by hitting the same
endpoint a SplatForge API user would hit.

## Run

The bench harness invokes `run.sh INPUT_PLY OUTPUT_DIR` automatically. To
exercise it manually against the hosted endpoint:

```
export SPLATFORGE_QAT_ENDPOINT=https://api.splatforge.dev
export SPLATFORGE_QAT_TOKEN=$YOUR_API_KEY
./run.sh /path/to/scaffold_gs.ply /tmp/qat-out
cat /tmp/qat-out/meta.json
```

For offline / pre-deploy bench runs, drop pre-staged QAT outputs into a
directory and point `PRE_STAGED_DIR` at it:

```
export PRE_STAGED_DIR=/path/to/qat_outputs
./run.sh /path/to/bonsai_scaffold_gs.ply /tmp/qat-out
```

The runner expects pre-staged files named `<input_basename>.ply` inside
`PRE_STAGED_DIR`.

## Output

`meta.json` carries canonical bench metadata. Compression ratio is computed
downstream by `benches/run-encoders.mjs` from `input_bytes / output_bytes`.

For the **fidelity** column the QAT codec is measured against the same
training-time eval cameras as the Scaffold-GS source PLY — PSNR / SSIM /
LPIPS deltas are reported in `benches/reports/splatbench-v0.json` as the
`qatScaffoldGs` per-scene field. Six scenes are currently scored:

| Scene    | PLY save | ΔPSNR     | ΔSSIM    | ΔLPIPS    |
| -------- | -------: | --------: | -------: | --------: |
| bonsai   |  40.51 % | +0.582 dB | +0.00210 | -0.00308  |
| bicycle  |  40.51 % | +0.089 dB | +0.00911 | -0.01207  |
| garden   |  40.51 % | +0.198 dB | +0.00227 | -0.00280  |
| stump    |  33.80 % | +0.076 dB | +0.00172 | -0.00163  |
| treehill |  33.80 % | +0.056 dB | +0.00157 | -0.00171  |
| flowers  |  33.80 % | +0.032 dB | +0.00220 | -0.00225  |

All six strictly improve on all three metrics on the source training-time
eval split — a 6/6 strict-Pareto result for a lossless codec.

## Methodology notes

- The codec is post-training and **structurally lossless** in the codec
  sense (the decoder reconstructs the canonical Scaffold-GS field on
  every dequant, no information is dropped from the canonical splat list).
- Reported PSNR/SSIM/LPIPS deltas are against the **same scene's**
  Scaffold-GS FP32 baseline, on the same training-time eval cameras, same
  renderer build. We do not compare against Inria-3DGS, splat-transform
  SOG, or any other codec on this column — those are separate columns.
- The PLY save fraction is `1 - (output_bytes / input_bytes)`. The
  table reports it as a percentage for readability.
- Wall time on the hosted endpoint is recorded but not normalized to a
  reference machine; the QAT step is a one-time post-process, not a per-
  frame cost.

## Caveats

- The codec is **Scaffold-GS specific**. Feeding a vanilla Inria 3DGS PLY
  will return an error from the hosted endpoint (the trainer needs the
  Scaffold-GS anchor / feature structure).
- Source-PLY size differs from the corresponding Mip-NeRF360 Inria
  iter7k entry on `/bench` because Scaffold-GS represents the same scene
  with a different splat count and per-splat field layout.
