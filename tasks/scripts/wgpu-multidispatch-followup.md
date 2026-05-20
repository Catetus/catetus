# Followup: Radix-sort + scan multi-dispatch (Stage 5 of feat/wgpu-multidispatch)

## Status (landed in feat/wgpu-multidispatch)

The per-splat *compute* kernels are now chunked via
`packages/viewer/src/webgpu/multi-dispatch.ts::dispatchPerSplat`. They tolerate
splat counts above the WebGPU 1.0 `dispatchWorkgroups` ceiling
(65 535 × 256 = 16 776 960):

- `cs_decode`              (decode.wgsl)
- `cs_project`             (decode.wgsl)
- `cs_keygen`              (cs_project_gather.wgsl)
- `cs_project_gather`      (cs_project_gather.wgsl)
- `cs_gather`              (inline GATHER_WGSL in webgpu/index.ts)
- `cs_cull`                (cs_cull.wgsl)
- `cs_compact`             (cs_cull.wgsl)
- `cs_project_cmpct`       (cs_cull.wgsl)
- `cs_lod_blend`           (cs_lod_blend.wgsl)
- `cs_lod_alpha_reset`     (cs_lod_blend.wgsl)
- `cs_tile_bin`            (cs_tile_bin.wgsl)
- `cs_wsr_accumulate`      (cs_wsr_accumulate.wgsl)

## Remaining: radix-sort + scan_multiblock

`packages/viewer/src/webgpu/radix_sort.ts` and `scan_multiblock.wgsl` are NOT
chunked. They dispatch at `numWgs = ceil(splat_count / 256)` for histogram,
per-wg scan, scan-block-sums, scan-add-block-sums, and scatter — every one
of which exceeds 65 535 once `splat_count > 16 776 960`.

This is the gate that still keeps LODGE L1 (~54 M) and L0 (~119 M) from
running. The bench (`real-scene-lodge.bench.ts`) explicitly enforces the
`dispatchCap` for that reason.

### Approach (b) — preferred: per-pass chunked global histogram + scan

For each of the 8 4-bit radix passes:

1. **Chunked histogram**: dispatch the histogram kernel in 1..N chunks of
   <= 65 535 workgroups. Each chunk atomically increments a *single shared*
   16-bin global histogram (storage buffer with `atomic<u32>`). Net effect:
   one global 16-bucket count after all chunks, identical to the unchunked
   path.

2. **Single-block scan** of the 16-bucket global histogram (fits in one
   workgroup; no chunking needed).

3. **Chunked scatter**: dispatch the scatter kernel in 1..N chunks. Each
   thread reads its `(key, value)` at `gid + chunk_offset`, computes its
   global rank as
   `bin_offset[bucket] + atomicAdd(&local_bin_count[bucket], 1u)`, and
   writes to the dst buffer. The atomic counter is shared across chunks
   so per-bucket scatter slots are dense.

### Approach (a) — fallback: per-chunk sort + N-way merge

If (b) proves too invasive: chunk the input into <= 16 776 960 slices,
radix-sort each slice independently (the existing pipeline handles
sub-cap counts), then run an N-way merge kernel (one workgroup per
output slot, picking the smallest among the N current chunk heads).

Worse asymptotic constants (extra merge pass, extra read of all keys),
but a much smaller patch surface.

### Why not in this PR

`radix_sort.ts` is 600 lines and `scan_multiblock.wgsl` carries
inter-block scan state — the chunking has to be co-designed with the
block-sum buffer layout to remain deterministic. Hard to land safely in
the same invocation as the per-splat retrofit. Splitting the work also
keeps the diffs reviewable.

### Acceptance criteria

- `bench/real-scene-lodge.bench.ts` no longer caps via `dispatchCap`
  (drop the `Math.min(dispatchCap, bufferCap)` in the same patch).
- `bench/results-4090-lodge.json` shows L1 numbers on the 4090 (54 M
  splats; expected fps in single digits given DRAM bandwidth ceiling).
- `__tests__/webgpu/fused_project_gather.test.ts` still passes (parity
  vs the unchunked path on small N).
- Add `__tests__/webgpu/radix_sort_chunked.test.ts` that sorts > 16.7 M
  randomized keys and asserts the output is fully sorted.

### After this lands

The `maxBufferSize` ceiling becomes the next bottleneck for L0 (~119 M
splats × 64 B/splat ≈ 7.6 GB in a single buffer). That's a separate
followup — split decoded splats into multiple storage buffers and bind
the active subset per-chunk.
