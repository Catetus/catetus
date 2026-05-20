# Catetus Rust crates

Implementation of the Rust workspace described by `specs/0001..0007`. Phase 0/1
scope. Each crate is self-contained and depends on `catetus-core` for the
canonical IR.

| Crate | Purpose | Key public items |
| ----- | ------- | ---------------- |
| `catetus-core` | Canonical `SplatScene` IR, BLAKE3 hashing, coordinate-system helpers, deterministic `AnalyzeReport` types. | `Splat`, `Color`, `SplatScene`, `scene_hash`, `AnalyzeReport`, `format_from_extension`, `format_from_magic` |
| `catetus-ply`  | Inria-style 3DGS PLY ingest (binary little-endian + ASCII). | `read_ply`, `read_ply_bytes`, `PlyError` |
| `catetus-spz`  | Minimal SPZ v2 writer/reader with 24-bit fixed-point positions, smallest-three quaternions, and zlib payload. | `read_spz`, `write_spz`, `encode_spz`, `SpzError` |
| `catetus-gltf` | glTF 2.0 + `KHR_gaussian_splatting` writer/reader plus the `SF_spatial_streaming_index` vendor extension. | `write_gltf`, `read_gltf`, `inspect_gltf`, `WriteOpts` |
| `catetus-optimize` | Pass framework + 8 Phase 1 passes (RemoveInvalidSplats, OpacityPrune, FloaterPrune, QuantizePosition/Scale/Rotation, ReduceSHDegree, MortonSort) and all 8 named presets from SPEC-0006. BuildLOD/ObjectAware are stubs that record a note. | `Pass`, `Pipeline`, `preset`, individual pass types |
| `catetus-bench` | Tiny benchmark harness that times analyze + optimize over a fixtures directory. | `BenchSuite`, `run_smoke`, `run_named` |
| `catetus-cli`  | The `catetus` binary: `analyze`, `inspect`, `convert`, `optimize`, `preview`, `diff`, `benchmark`, `corpus run`. | `catetus` binary |

## Building

```sh
cargo check --workspace
cargo test  --workspace
```

## Determinism

All hashing, reports, and pipelines are deterministic for the same input —
there is no wall-clock or RNG outside `catetus-cli benchmark` and the
`catetus-bench` crate, which time individual runs.
