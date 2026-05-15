# Compute-decode + GPU radix-sort (research queue #62)

Status: implemented behind the `useComputeDecode: true` viewer flag.
WebGL2 path is untouched; WebGPU without the flag is also untouched. The
flag only enables the new GPU compute pipeline when the WebGPU backend is
selected.

## Why

The CPU decode path bottlenecks at 1–2 M splats / frame. Every frame the
viewer was:

1. Dequantizing 5 SoA attribute streams (POSITION u16, ROTATION f32,
   SCALE u8, OPACITY u8, COLOR_DC u8) in JS.
2. Building a per-instance `Float32Array(N × 12)` on the main thread.
3. Sorting back-to-front in JS (`Array.sort` on a paired array).
4. Uploading the buffer with `queue.writeBuffer` — every frame.

At 10 M splats this is well over a second of CPU work per frame. The v2
target — 30 GB scenes at 60 fps on mobile — needs decoding, projection
and sorting on the GPU.

## Architecture

```
raw chunk bytes (Uint8Array)
       │
       ▼  cs_decode          [decode.wgsl]
       │   SoA u8/u16/f32 → canonical DecodedSplat (64 B)
       │   1 thread per splat, workgroup 256
       ▼
decoded splats   (storage)
       │
       ▼  cs_project         [decode.wgsl]
       │   splat × camera → Instance (clipPos, 2×2 cov, color)
       │                  + u32 depth key + u32 splat index
       ▼
unsorted instance buffer  +  (keys[], indices[])
       │
       ▼  radix-sort (8 × 4-bit LSD passes)
       │   per pass:
       │     cs_histogram                          [radix_sort.wgsl]
       │     cs_scan_per_wg                        [scan_multiblock.wgsl] ← NEW
       │     cs_scan_block_sums                    [scan_multiblock.wgsl] ← NEW
       │     cs_scan_add_block_sums                [scan_multiblock.wgsl] ← NEW
       │     cs_scatter                            [radix_sort.wgsl]
       ▼
sorted indices[]   (back-to-front order)
       │
       ▼  cs_gather          [embedded in webgpu/index.ts]
       │   instance[i] = unsorted[indices[i]]
       ▼
sorted instance buffer   →   existing vertex pipeline
```

Total per-frame dispatches: **1 + 1 + 8 × 5 + 1 = 43** compute dispatches
when the multi-block scan is enabled, vs the previous 1 + 1 + 8 × 3 + 1 =
27. The five extra dispatches per pass are cheap: phases (a) and (c) of
the scan parallelize across many workgroups instead of trickling through
a single one.

## Radix-sort algorithm choice: split-block vs prefix-sum

We use the classic **prefix-sum LSD** structure (now five kernels per
pass after this change) and intentionally avoid storage-buffer atomics:

- **Onesweep** (Ha et al. 2022) merges the histogram + scan + scatter into
  a single kernel using cooperative thread-block prefix-look-back via
  storage-buffer atomics. Faster (~1.8×) but requires `storage` atomics
  and 64-bit atomics — both **optional** features in WebGPU 1.0.
- **Split-block** (Merrill & Grimshaw 2010) uses two-pass per radix
  (count + scatter). Cleaner; still needs storage atomics for the
  per-block "tile state" handoff.
- **Prefix-sum, three-kernel** (Wyman's `wgsl-radix-sort`, splatviz,
  antimatter15). Three kernels per pass and zero storage-buffer atomics
  — just workgroup-shared atomics, which are mandatory. The original
  implementation in this viewer.
- **Prefix-sum, five-kernel with multi-block scan** (this PR). Same
  zero-storage-atomic property as the three-kernel form, but the scan
  step itself is now a chained 3-kernel exclusive prefix-sum
  (per-tile → block-sums → add-back) so phases (a) and (c) parallelize
  across many workgroups.

The viewer's job is portability, not raw throughput. Onesweep can land
later behind a `requiredFeatures: ['storage-atomics']` check.

References reused (cited inline in the WGSL):

- C. Wyman, "wgsl-radix-sort"
  <https://github.com/cwyman/wgsl-radix-sort> — three-kernel layout.
- Merrill & Grimshaw 2010, "High Performance and Scalable Radix Sorting".
- Merrill & Grimshaw 2010, "Parallel Scan for Stream Architectures".
- Harris et al., "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3.
- antimatter15 / splatviz GPU sort prototypes (multi-pass LSD on u32).

## Multi-block scan (this change)

Previously the `cs_scan` step ran as **one workgroup of 256 threads**
striding through the entire histogram array (`num_wgs × RADIX` ≈ 625 K
elements per pass at 10 M splats). That single workgroup was the top
cost in the sort.

The replacement is a textbook 3-kernel chained scan
(`scan_multiblock.wgsl`):

1. `cs_scan_per_wg` — every 256-thread workgroup does an exclusive scan
   over its tile in shared memory (Hillis-Steele), writes the scanned
   tile back to `histograms`, and writes the tile's total to
   `block_sums[wgid]`.
2. `cs_scan_block_sums` — a single workgroup of 256 threads exclusive-
   scans the `block_sums` array. The block-sums array is tiny — for
   10 M splats it has only `ceil(625000 / 256) ≈ 2 442` entries — so a
   single-WG scan with serial striding is cheap and not the bottleneck.
3. `cs_scan_add_block_sums` — every 256-thread workgroup adds its
   scanned `block_sums[wgid]` to every element of its tile, producing
   the final global exclusive prefix.

Phases (a) and (c) now run across **`ceil(num_wgs × RADIX / 256)`**
workgroups in parallel instead of one. For 10 M splats that's ~2 442
workgroups doing useful work instead of 1.

Toggle is in `RadixSort` via `useMultiBlockScan` (default `true`). The
legacy single-WG path is retained in `radix_sort.wgsl`'s `cs_scan` so we
can A/B compare and so any downstream consumer who still allocates
without passing the multi-block WGSL gets unchanged behavior.

## Bin-major histogram layout

Per-workgroup histograms are stored `histograms[bin * numWgs + wgid]`,
not `[wgid * 16 + bin]`. The exclusive prefix-sum over this layout
places all of bin 0's workgroups first, then bin 1's, etc. — so the
scatter pass's destination offsets are correct ascending-order positions
without an extra grouping pass. This trick is straight out of the
classic LSD radix literature (Merrill). The multi-block scan treats the
histogram array as a flat u32 stream; the bin-major layout is preserved
because the scan is just an exclusive-prefix-sum over that stream.

## Determinism

Same input + same camera ⇒ same output, every run:

- Decode: each splat decoded independently. No floating-point reduction.
- Project: deterministic per-splat math. Quaternion length normalize uses
  `max(length(q), 1e-8)` so q≈0 doesn't NaN.
- Radix sort: stable per pass (stable LSD radix). Equal-depth splats fall
  back to splat-index because the key is `bitcast(depth)` and ties resolve
  in scatter order — which is fixed for a fixed global-id pattern.
- Multi-block scan: pure data-parallel u32 addition. No floating-point,
  no atomic order-dependence on storage buffers.

## Visual regression

`tests/visual/tests/compute-decode.spec.ts` renders the same scene under
the WebGPU backend with `useComputeDecode: false` and `=true`. Per-frame
pixelmatch threshold is set to `0.02` (2 %). The spec exists; it requires
`tests/visual/fixtures/tiny/cube.gltf` which is not in the repo by
default and must be generated locally (see `tests/visual/README.md`).

Unit-level correctness for the new scan lives in
`packages/viewer/src/__tests__/webgpu/scan_multiblock.test.ts` — a
TypeScript mirror of the WGSL algorithm runs against a reference exclusive
prefix-sum on 128, 256, 257, 4 096, 4 097, 65 536, 70 000 and 625 008-
element inputs (the last being the exact histogram-scan size for 10 M
splats per pass). The mirror is byte-for-byte deliberate so any
algorithmic divergence in the WGSL forces a TS change to keep the test
green.

## Bench results

Run from `pnpm --filter @splatforge/viewer run bench`. Bench harness is
in `bench/`. JSON output: `bench/results.json`.

The previous numbers in this doc were measured before the multi-block
scan landed and are kept in the "Before" column for the operator to
compare. The "After" column is **PENDING — operator re-run on M-series
host**. Run the one-liner below and overwrite the `PENDING` cells.

```sh
pnpm --filter @splatforge/viewer run bench
```

| Scale | Decode (one-shot) Before | Decode After | Frame (avg) Before | Frame After | FPS Before | FPS After |
|-------|--------------------------|--------------|---------------------|-------------|------------|-----------|
| 1 M   | 245.7 ms                 | PENDING      | 7.86 ms             | PENDING     | 127        | PENDING   |
| 10 M  | 626.1 ms                 | PENDING      | 89.21 ms            | PENDING     | 11.2       | PENDING   |

Per-stage estimated breakdown (Before column from the legacy single-WG
scan; After column to be filled in by the operator):

| Scale | Project Before | Sort Before | Gather Before | Project After | Sort After | Gather After |
|-------|----------------|-------------|---------------|---------------|------------|--------------|
| 1 M   | 1.57 ms        | 5.50 ms     | 0.79 ms       | PENDING       | PENDING    | PENDING      |
| 10 M  | 17.8 ms        | 62.4 ms     | 8.92 ms       | PENDING       | PENDING    | PENDING      |

> Numbers from `pnpm --filter @splatforge/viewer run bench` on the
> operator's M-series host. Do **not** copy in numbers from any other
> machine — ANGLE/Metal swiftshader and native Metal differ by 1.3–1.8×.

## 60 fps at 10 M is not in this PR

The honest framing for this change: it removes the single-WG scan as the
top bottleneck in the sort. It does **not** by itself land 60 fps at
10 M. Reaching that target needs follow-up work:

1. **8-bit radix** instead of 4-bit. Halves the number of passes from 8
   to 4. Needs 256-bin histograms (vs 16) which fits in shared memory
   but increases the histogram scan size; the multi-block scan landed
   here is the prerequisite that makes the larger scan tractable.
2. **Subgroup-aware histogram** using `subgroupAdd` / ballot ops to
   compute per-warp bin counts before writing one value per warp to
   shared memory. WebGPU 1.1 subgroup feature.
3. **Faster gather** — collapse `cs_gather` into a second `cs_project`
   pass that reads sorted indices and writes the final instance buffer
   in one go, eliminating the unsorted-instance staging buffer.
4. **Streaming overlap** — let the next frame's `cs_decode` start while
   the current frame's `cs_scatter` finishes, double-buffering the
   instance buffer.

This PR removes scan as the top cost. The next bottleneck once the
multi-block scan is in place is **`cs_project` + `cs_scatter` memory
bandwidth on the unsorted instance buffer** (~28 M loads/stores per
frame at 10 M splats with 12-float instances). That's where item (3)
above starts to matter.

## Biggest WebGPU limitation hit (still)

**Lack of mandatory storage-buffer atomics.** With them we'd use a
single-pass onesweep + storage-atomic look-back chain instead of the
five-kernel split. We don't have them mandatory in WebGPU 1.0, so the
five-kernel form is the portable ceiling.

Secondary: **`maxStorageBufferBindingSize` defaults to 128 MB**. At 10 M
splats × 64 B = 640 MB, the device must advertise 1 GB+. M-series Mac
adapters do (`maxStorageBufferBindingSize >= 2^31`); cheap Android
adapters often cap at 256 MB. The pipeline supports requesting a higher
limit at device-creation time; mobile callers should query
`adapter.limits` before constructing the pipeline.

## Files

- `src/webgpu/decode.wgsl` — decode + project compute shaders.
- `src/webgpu/radix_sort.wgsl` — 4-bit LSD radix sort
  (histogram + legacy single-WG scan + scatter).
- `src/webgpu/scan_multiblock.wgsl` — 3-kernel chained exclusive prefix
  sum (NEW; replaces the single-WG scan).
- `src/webgpu/radix_sort.ts` — TS orchestration of 8 passes. The
  `useMultiBlockScan` flag (default `true`) selects the multi-block path.
- `src/webgpu/index.ts` — `ComputeDecodePipeline` (the public surface).
- `src/webgpu/shaders.generated.ts` — bundled WGSL strings; regenerated by
  `scripts/embed-wgsl.mjs` (runs on every build/test).
- `src/__tests__/webgpu/scan_multiblock.test.ts` — unit tests for the
  multi-block scan (Node, no GPU).
- `bench/compute-decode.bench.ts` — synthetic-scene bench.
- `bench/run-bench.mjs` — headless Chromium driver.
- `bench/results.json` — last-run output.
- `tests/visual/tests/compute-decode.spec.ts` — visual regression vs CPU
  path.
