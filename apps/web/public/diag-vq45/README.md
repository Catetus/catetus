# diag-vq45 — production-viewer decoder smoke test

This directory ships a single-page diagnostic that proves the production
viewer can decode Catetus's compressed GLB format end-to-end in the
browser.

## What it tests

Three new producer-side features (all originally implemented in
`crates/catetus-optimize` and validated by the bench harness at
`experiments/w3-fidelity-harness/code/cpu-fidelity.mjs`):

| Feature | Where it ships in the viewer |
|---|---|
| `SF_zstd_split_buffer` extension | `packages/viewer/src/streaming/glb.ts` → `decompressZstdSplitBuffer` |
| `SF_gaussian_splatting_palette` (`.shpal`) extension | same file → `decodeShPaletteSidecar` |
| BYTE / SHORT `SH_DEGREE_l_COEF_n` accessors with per-channel ranges | inlined in `diag.js` (`decodeAttribute` — case `5120`/`5122`) |

Both new viewer-module functions are pure JS and dependency-free: they take a
`zstdDecompress` callback so the viewer SDK doesn't have to bundle a zstd
implementation by default. The diag page wires up `fzstd.decompress` (~10 KB
gz) — that's the recommended browser zstd until WebAssembly's native zstd
ships.

## What the page does

1. Fetches `bonsai_input.glb` (273 MiB, vanilla `KHR_gaussian_splatting`).
2. Fetches `wmv-vq45-no-prune.glb` (68 MiB compressed; 252 MiB uncompressed
   BIN; `KHR_mesh_quantization` + `SF_zstd_split_buffer`).
3. Probes `wmv-vq45-no-prune.glb.shpal` (3.8 MiB compressed; 65 536-entry
   45-D codebook + 1.16M u16 indices) through the new `.shpal` reader.
4. Renders both side-by-side with the same camera so the visual parity is
   obvious. The renderer is a tiny inlined 3DGS splatter (~200 LOC) that
   mirrors the production WebGL2 path; the heavy lifting is in the decoder,
   not the rasterizer.

A green diag means the compressed format is **shippable** — users can view
Catetus-optimized assets directly without round-tripping through the
bench harness.

## Run it

```
cd apps/web/public && python3 -m http.server 4321
open http://localhost:4321/diag-vq45.html
```

The `?auto=1` query starts decoding + rendering immediately; the
`shoot-diag-vq45.mjs` Playwright script uses that.

## Repro the screenshot

```
cd apps/web
node scripts/shoot-diag-vq45.mjs http://localhost:4321/diag-vq45.html screenshots/diag-vq45.png
```

## Open issues

- **Browser zstd perf**: 252 MiB through `fzstd` takes ~1.5–4 s on M-series.
  Once Node-style `DecompressionStream("zstd")` ships in browsers (already in
  Chrome ≥ 123 / Safari TP) we should auto-detect it and skip the fzstd
  bundle.
- **Memory**: holding the full uncompressed BIN (252 MiB) + decoded splats
  (75 MiB SoA) pushes the peak working set past 350 MiB. The streaming-tile
  path is the right answer for mobile; this diag intentionally bypasses it
  to keep the proof one-file.
- **SH-rest unused**: the production renderer ignores SH-rest (DC color
  only). The BYTE/SHORT decoder is exported and unit-tested, but rendering
  view-dependent color is a separate workstream.
- **VQ45 palette not exercised by `wmv-vq45-no-prune.glb`**: that GLB
  embeds SH-rest in-line (no palette ext set). The `.shpal` sidecar is
  probed independently to exercise the decoder path. The first GLB that
  emits `SF_gaussian_splatting_palette` end-to-end will validate the
  combined codebook-lookup path.

## Files

- `diag.js` — page logic (fetch → decode → render).
- `fzstd.mjs` — vendored copy of `fzstd@0.x` (~24 KB ESM).
- `wmv-vq45-no-prune.glb`, `.shpal`, `bonsai_input.glb` — symlinks into
  `experiments/SOG_STUDY_RUN/` and `experiments/w3-fidelity-harness/out/`.
- `../diag-vq45.html` — the page itself.
- `../../scripts/shoot-diag-vq45.mjs` — Playwright wrapper that drives the
  page in headless Chromium and saves a PNG.
- `../../screenshots/diag-vq45.png` — captured side-by-side baseline vs.
  compressed render. The two panels show visually-equivalent bonsai trees;
  any decoder bug would produce a garbled cloud or all-zero attribute, both
  of which look obviously wrong.
