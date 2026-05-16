# splatforge-ply — Changelog

All notable changes to the Inria-style 3DGS PLY ingest crate. The crate's
public surface (`read_ply`, `read_ply_bytes`, `write_ply`, `write_ply_bytes`,
the `Mgs2*` progressive bitstream re-exports, and the `PlyError` enum)
remains unchanged across these versions — the entries below are all
backend-only changes unless explicitly noted.

## Unreleased

### Changed
- Replaced the byte-stream binary-body decoder with a hoisted-offset
  walker. The old path constructed an `io::Cursor` for every scalar field,
  ran an `O(properties²)` name lookup per row, and allocated a scratch
  `Vec<f32>` per splat. The new path resolves every required field's
  byte-offset + scalar type once into a `VertexLayout`, then iterates the
  body as a flat byte slice with direct `from_le_bytes` reads.
- File reads now use `memmap2::Mmap` instead of `fs::read`. This avoids
  materialising a multi-gigabyte heap buffer for Sweet-Corals-class PLYs
  (~30 GiB) and lets the OS page in the body on demand. A buffered-read
  fallback is retained for filesystems where mmap fails (pipes,
  zero-length, some network mounts).
- Large vertex counts (> 256 K splats) are now sharded across the global
  rayon pool. Each shard decodes a contiguous slice of vertex records and
  produces a `Vec<Splat>`; the chunks are concatenated in shard-index
  order, preserving the input ordering required by the IR contract.

### Performance — `ply-read-bench` (`cargo run --release --bin ply-read-bench`)

Machine: M-series MacBook (aarch64), default release profile
(`lto = "thin"`, `codegen-units = 1`), no `RUSTFLAGS` overrides.

| Scene                  | Splats     | Old (warm, s) | New (warm, median, s) | Speedup |
| ---------------------- | ---------- | ------------- | ---------------------- | ------- |
| bonsai_iter7000        | 1,157,141  | 0.389         | 0.030                  | **13.0×** |
| bicycle_iter7000       | 3,616,103  | 1.246         | 0.080                  | **15.6×** |
| stump_iter7000         | 3,807,536  | ~1.27 (est.)  | 0.123                  | **~10.3×** |

The cold-cache numbers on bicycle improved from 1.45 s to 0.16 s
(NVMe-bound page-in for ~855 MiB), giving a ~9× cold-start speedup. The
30 GiB Sweet-Corals merged PLY was not benchmarked locally per the
project's 5 GiB disk cap; extrapolating from the bicycle warm rate
(≈11 GB/s decoded) the parse should land in the 3–8 s range on
M-class hardware, dominated by sequential page-in.

### Tests
- Added `tests/fast_decoder_equivalence.rs`: synthesises six PLY field
  orderings (minimal Inria, Inria + normals + SH degree-1, shuffled
  DC/f_rest, 300 K-splat large-count crossing the parallel threshold,
  truncated body, missing-required-field), parses each with the fast
  path, and asserts bit-exact equivalence against an independent
  reference decoder that walks the body one scalar at a time. The
  large-count test exercises both the single-threaded prologue and the
  rayon-sharded path within a single file.
- All five pre-existing `ply_roundtrip` tests and both
  `write_roundtrip` tests continue to pass unchanged.

### Dependencies
- Added `memmap2 = "0.9"` for the file-read path.
- Added workspace `rayon` for the parallel decoder.
