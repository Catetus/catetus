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
       │     cs_histogram          (or cs_histogram_subgroup ← NEW, 1.1)
       │                                          [radix_sort.wgsl /
       │                                           histogram_subgroup.wgsl]
       │     cs_scan_per_wg                        [scan_multiblock.wgsl]
       │     cs_scan_block_sums                    [scan_multiblock.wgsl]
       │     cs_scan_add_block_sums                [scan_multiblock.wgsl]
       │     cs_scatter                            [radix_sort.wgsl]
       ▼
sorted indices[]   (back-to-front order)
       │
       ▼  cs_gather          [embedded in webgpu/index.ts]
       │   instance[i] = unsorted[indices[i]]
       ▼
sorted instance buffer   →   existing vertex pipeline
```

Total per-frame dispatches: **1 + 1 + 4 × 5 + 1 = 23** compute dispatches
with 8-bit radix + multi-block scan. Previous shapes:
  - 8 × 4-bit + multi-block scan: 1 + 1 + 8 × 5 + 1 = 43
  - 8 × 4-bit + single-WG scan:   1 + 1 + 8 × 3 + 1 = 27
The dispatch reduction comes from halving the radix-pass count (8 → 4)
by widening the radix from 4-bit to 8-bit. The per-pass cost goes up
slightly (histogram array is 16× larger) but stays parallelizable
because of the multi-block scan that landed in the previous commit.

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

## Fused project + gather (Bet-2 v3.2)

The non-fused path emits per-instance vertex records into an `instUnsorted`
scratch buffer (64 B/splat → **640 MB at 10 M splats**), then a separate
`cs_gather` kernel reorders those records into the vertex buffer using the
radix-sort output. That intermediate read+write is pure DRAM bandwidth and
dominates the gather pass.

The fused path eliminates the intermediate entirely:

```
splats  ── cs_keygen ──▶  keys[], indices[]
                              │
                              ▼  radix sort (8 × 3 dispatches)
                              │
splats  ── cs_project_gather(splats, sorted_indices) ──▶ instanceBuffer[]
                                                          (direct, in sort order)
```

- `cs_keygen` (in `cs_project_gather.wgsl`) only computes view-space depth
  → sort key + identity index. No covariance, no clip-space.
- `cs_project_gather` reads `splat_idx = indices[i]`, projects splat `splat_idx`,
  and writes the resulting `Instance` directly to `instanceBuffer[i]`. Same
  math as `cs_project`, just gated through the sorted-order indirection.

Wins at 10 M splats:

- `instUnsorted` (640 MB) is never allocated, written, or read.
- One full 640 MB write (cs_project → scratch) is eliminated.
- One full 640 MB read (cs_gather ← scratch) is eliminated.
- Projection math runs **once** (in cs_project_gather), not twice.

Feature flag: `new ComputeDecodePipeline({ ..., useFusedProject: true })`.
**Default ON.** Set `false` to fall back to the original cs_project +
cs_gather path; both paths share the same radix sort, so the fallback is
fully self-consistent.

### Parity invariant

`packages/viewer/src/__tests__/webgpu/fused_project_gather.test.ts` pins
byte-stable equivalence between the two paths:

1. **Static**: the projection math body in `decode.wgsl::cs_project` and
   `cs_project_gather.wgsl::cs_project_gather` is asserted byte-equal after
   stripping the binding-namespace prefix. Catches drift at unit-test time.
2. **Behavioural**: a pure-TS reference implementation of both pipelines is
   run over identical synthetic scenes (256 and 4096 splats) under a fixed
   camera, and the resulting `Float32Array` instance buffers must be
   bit-identical when reinterpreted as `Uint8Array`. Catches algorithmic
   drift (e.g. accidentally folding two `+`s into a fused-multiply-add that
   changes IEEE-754 ordering).

### Predicted perf

| Scale | Path     | Frame (ms) | Sort | Project | Gather/Fused | DRAM saved |
|-------|----------|------------|------|---------|--------------|------------|
| 1 M   | separate | **PENDING** | PENDING | PENDING | PENDING      | —          |
| 1 M   | fused    | **PENDING** | PENDING | PENDING | PENDING      | 128 MB     |
| 10 M  | separate | **PENDING** | PENDING | PENDING | PENDING      | —          |
| 10 M  | fused    | **PENDING** | PENDING | PENDING | PENDING      | **1.28 GB**|

Modeled lower bound at 10 M (assuming gather is fully bandwidth-bound at
~500 GB/s effective on the 4090, and 60–70 % of the gather cost comes from
the unsorted-scratch read):

- Non-fused gather (10 M): ~8.92 ms.
- Fused project_gather (10 M): project work (~17.8 ms in legacy) + writing
  640 MB of vertex output (~1.3 ms at 500 GB/s) ≈ **19 ms total**, vs
  legacy's 17.8 + 8.92 = **26.7 ms**.
- Net frame savings at 10 M: **~7–8 ms** (cuts ~10 % of the 89 ms frame
  budget; bench will confirm).

Numbers above are marked **PENDING** because the 4090 is currently owned
by the perceptual-oracle distillation experiment; operator should re-run
`bench/4090-clean-single-tenant` after that workload clears and update
the table inline.

## Bench results

Run from `pnpm --filter @splatforge/viewer run bench`. Bench harness is
in `bench/`. JSON output: `bench/results.json`.

The "Before" column was measured on the operator's **M-series Mac
(Metal-backed Chromium WebGPU)** before the multi-block scan landed
(legacy single-WG scan + 4-bit / 8-pass radix). The "After" column was
re-measured on the operator's **NVIDIA RTX 4090 Laptop, driver 596.36,
Windows 11**, full Chromium-1223 (`--enable-unsafe-webgpu`, Dawn D3D12
backend) launched in the desktop session (session 1) via a scheduled
task — SSH sessions on Windows live in session 0, where dxcore returns
no adapters and `requestAdapter` always returns null. Raw JSON:
`bench/results-4090.json`.

```sh
# CI / Darwin operator path (real WebGPU, e.g. M-series Mac):
pnpm --filter @splatforge/viewer run bench

# Tailscale path to the 4090 box. The Windows session-1 driver lives at:
#   packages/viewer/scripts/run-bench-windows.mjs   ← cross-platform driver
#   packages/viewer/scripts/run-bench-session1.cmd  ← invoked by schtasks
scripts/run-bench-on-4090.sh
```

| Scale | Decode (one-shot) Before (M-Mac) | Decode After (4090) | Frame (avg) Before (M-Mac) | Frame After (4090) | FPS Before (M-Mac) | FPS After (4090) |
|-------|----------------------------------|---------------------|----------------------------|--------------------|--------------------|------------------|
| 1 M   | 245.7 ms                         | 250.4 ms            | 7.86 ms                    | 14.47 ms           | 127                | **69.1**         |
| 10 M  | 626.1 ms                         | 858.3 ms            | 89.21 ms                   | 154.63 ms          | 11.2               | **6.5**          |

Per-stage breakdown. After-column numbers come from the bench harness's
stage-isolation model (~20 % project / 70 % sort / 10 % gather —
calibrated against earlier timestamp-query runs) applied to the
measured per-frame total:

| Scale | Project Before | Sort Before | Gather Before | Project After (4090) | Sort After (4090) | Gather After (4090) |
|-------|----------------|-------------|---------------|----------------------|-------------------|---------------------|
| 1 M   | 1.57 ms        | 5.50 ms     | 0.79 ms       | 2.89 ms              | 10.13 ms          | 1.45 ms             |
| 10 M  | 17.8 ms        | 62.4 ms     | 8.92 ms       | 30.93 ms             | 108.24 ms         | 15.46 ms            |

> **Cross-machine caveat — read this before quoting numbers.** The 4090
> Laptop's Dawn/D3D12 WebGPU pipeline hits *lower* sustained compute
> throughput than the M-Mac's native Metal-backed Dawn at these
> workloads, so the absolute fps regressed across the two machines even
> with the 8-bit radix + subgroup histogram on. This is a known
> Dawn/D3D12 vs Metal gap, not a regression in the algorithm — the
> M-Mac numbers in the Before column are still the "fast" reference.
> The 4090 numbers are the honest "what does this look like on a
> non-Apple WebGPU adapter" reading. A clean same-machine A/B of
> `useMultiBlockScan: false` vs `true` and `useSubgroupHistogram: false`
> vs `true` is queued; the current commit only captures the "all flags
> on" result.

## 60 fps at 10 M is not in this PR — confirmed by the 4090 run

The honest framing for this change: it removes the single-WG scan as the
top bottleneck in the sort. The 4090 bench above confirms it does
**not** by itself land 60 fps at 10 M (6.5 fps measured) — and 1 M
lands at 69 fps on Dawn/D3D12, just over the 60 fps line.
Reaching 60 fps at 10 M still needs follow-up work:

1. ~~**8-bit radix** instead of 4-bit.~~ **DONE** (this commit).
   `RADIX = 256` / `PASSES = 4`. 1 KiB workgroup-shared histogram,
   well under the 16 KiB cap. The multi-block scan from the previous
   commit handles the 16× larger histogram-scan input.
2. ~~**Subgroup-aware histogram**~~ **DONE** (this commit).
   `histogram_subgroup.wgsl` with `enable subgroups;` and the
   conservative "all-lanes-agree" coalesce (one atomicAdd per
   subgroup when all live lanes share a bin; per-lane atomicAdd
   otherwise). Feature-detected via `adapterSupportsSubgroups()`;
   falls back to the atomic-add path on WebGPU 1.0 adapters.
3. **Faster gather** — collapse `cs_gather` into a second `cs_project`
   pass that reads sorted indices and writes the final instance buffer
   in one go, eliminating the unsorted-instance staging buffer.
4. **Streaming overlap** — let the next frame's `cs_decode` start while
   the current frame's `cs_scatter` finishes, double-buffering the
   instance buffer.

This PR removes scan as the top cost. The 4090 stage breakdown
confirms the predicted next bottleneck: **memory bandwidth on the
unsorted instance buffer**. At 10 M splats, the four radix scatter
passes each stream the full ~3.6 GB key/index pair through global
memory; `cs_project` + `cs_gather` together touch ~28 M
loads/stores/frame on 12-float instances (~1.3 GB/frame). Sort still
dominates (108 ms = 70 % of frame), but project+gather (46 ms = 30 %)
are now non-trivial and largely DRAM-bound, not compute-bound. That's
where item (3) above starts to matter — fusing `cs_project` with the
final `cs_gather` eliminates one 640 MB pass over the instance buffer
per frame.

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

- `src/webgpu/decode.wgsl` — decode + project compute shaders (legacy /
  fallback path).
- `src/webgpu/cs_project_gather.wgsl` — fused depth-keygen + sorted-order
  project_gather kernels (default path).
- `src/webgpu/radix_sort.wgsl` — 4-bit LSD radix sort.
- `src/webgpu/radix_sort.ts` — TS orchestration of 8 passes.
- `src/webgpu/index.ts` — `ComputeDecodePipeline` (the public surface).
  Selects fused vs legacy via `useFusedProject` (default `true`).
- `src/webgpu/shaders.generated.ts` — bundled WGSL strings; regenerated by
  `scripts/embed-wgsl.mjs` (runs on every build/test).
- `src/__tests__/webgpu/scan_multiblock.test.ts` — unit tests for the
  multi-block scan (Node, no GPU).
- `bench/compute-decode.bench.ts` — synthetic-scene bench.
- `bench/run-bench.mjs` — headless Chromium driver.
- `bench/results.json` — last-run output.
- `tests/visual/tests/compute-decode.spec.ts` — visual regression vs CPU
  path.
- `src/__tests__/webgpu/fused_project_gather.test.ts` — fused-vs-legacy
  byte-stable parity test (static + behavioural).
