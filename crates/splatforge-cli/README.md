# `splatforge` CLI

Command-line front end for the SplatForge Gaussian-splat pipeline.

```
splatforge <command> [options]
```

## Install

```bash
# Homebrew (macOS, Linux)
brew tap splatforge/tap && brew install splatforge

# npm (any platform with Node 18+ on $PATH)
npm install -g @splatforge/cli

# From source (requires a Rust toolchain)
cargo install --git https://github.com/splatforge/splatforge \
  --locked splatforge-cli
```

All three install paths land the same three binaries on your `$PATH`:
`splatforge`, `splatforge-khr-validate`, `splatforge-usd-validate`. The
release runbook (`distrib/RELEASE.md`) documents how releases are cut.

## Commands

| Command | Summary |
| --- | --- |
| `analyze`   | Parse a splat file and print a JSON diagnostic report. |
| `inspect`   | Validate a splat file; non-zero exit on failure. |
| `convert`   | Convert between `ply` / `spz` / `gltf` / `glb`. |
| `optimize`  | Run a named optimization preset. |
| `preview`   | Serve a tiny static viewer that loads the asset over HTTP. |
| `diff`      | Compare two splat files visually. |
| `benchmark` | Microbenchmark `analyze` on a file. |
| `corpus`    | Run a named benchmark suite. |

Run `splatforge <command> --help` for the full flag list.

## `optimize` presets

| Preset | What it does | Output shape |
| --- | --- | --- |
| `lossless-repack`   | RemoveInvalid + MortonSort. Byte-stable round-trip. | single `.gltf` |
| `quality-max`       | Alias for `lossless-repack`. | single `.gltf` |
| `web-mobile`        | Aggressive prune + quantize + 2-LOD pyramid for phones. | single `.gltf` |
| `web-desktop`       | Quantize + Morton; keeps SH degree 1. | single `.gltf` |
| `quest-browser`     | Phone-mobile-ish but tuned for Quest browser. | single `.gltf` |
| `visionos-preview`  | Lossless-leaning, visionOS preview tuning. | single `.gltf` |
| `thumbnail-preview` | Smallest preview — heavy prune + quantize. | single `.gltf` |
| `size-min`          | Smallest possible bytes, including LOD pyramid. | single `.gltf` |
| **`geospatial`**    | **Cesium 3D Tiles 1.1 + KHR_gaussian_splatting with 4-LOD pyramid.** | **multi-file tileset** |

## `--preset geospatial`

Produces a Cesium ion–compatible tileset rather than a single glTF. The pipeline composes the
`web-mobile` baseline (prune + quantize + Morton sort) with a four-level LOD pyramid where each
LOD halves the splat count of the previous one (LOD0 = full, LOD1 = ½, LOD2 = ¼, LOD3 = ⅛).
Output is a directory containing `tileset.json` and one `.glb` per LOD.

```bash
splatforge optimize \
  --preset geospatial \
  --output-dir my-asset/ \
  capture.ply
```

The directory after the run:

```
my-asset/
├── tileset.json
├── lod0.glb         # leaf — full resolution
├── lod1.glb         # 1/2
├── lod2.glb         # 1/4
├── lod3.glb         # 1/8  ← root (coarsest tile)
└── optimize-report.json
```

The `tileset.json` declares `asset.version = "1.1"`, lists `KHR_gaussian_splatting` in
`extensionsRequired`, and chains the four LODs with `refine: "REPLACE"` so Cesium picks finer tiles
as the viewer approaches. `geometricError` halves at each refinement step from a root error sized
to half the scene bounding-box diagonal — see the module docs in
`crates/splatforge-optimize/src/tileset.rs` for the full heuristic.

For an upload-to-ion walkthrough see [`docs/cesium-ion-quickstart.md`](../../docs/cesium-ion-quickstart.md).

`--output-dir` and `--out` are mutually exclusive; `--output-dir` is required for `geospatial` and
rejected for every other preset.

## Determinism

Every preset is byte-deterministic: identical input + identical preset = identical output bytes,
including the `tileset.json` JSON ordering for `geospatial`. This is a hard project constraint —
see `docs/architecture.md`.
