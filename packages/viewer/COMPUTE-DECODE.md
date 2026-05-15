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
       ▼  radix-sort (8 × 4-bit LSD passes)   [radix_sort.wgsl]
       │   cs_histogram → cs_scan → cs_scatter   ×  8
       ▼
sorted indices[]   (back-to-front order)
       │
       ▼  cs_gather          [embedded in webgpu/index.ts]
       │   instance[i] = unsorted[indices[i]]
       ▼
sorted instance buffer   →   existing vertex pipeline
```

Total per-frame dispatches: **1 + 1 + 8 × 3 + 1 = 27** compute dispatches,
all on the GPU command queue. No CPU → GPU buffer uploads except the
camera uniform.

## Radix-sort algorithm choice: split-block vs prefix-sum

I chose the classic **prefix-sum LSD** structure (three kernels per pass)
over the modern **split-block onesweep**:

- **Onesweep** (Ha et al. 2022) merges the histogram + scan + scatter into
  a single kernel using cooperative thread-block prefix-look-back via
  storage-buffer atomics. Faster (~1.8×) but requires `storage` atomics
  and 64-bit atomics — both **optional** features in WebGPU 1.0.
- **Split-block** (Merrill & Grimshaw 2010) uses two-pass per radix
  (count + scatter). Cleaner; still needs storage atomics for the
  per-block "tile state" handoff.
- **Prefix-sum** (Wyman's `wgsl-radix-sort`, splatviz, antimatter15) uses
  three kernels and zero storage-buffer atomics — just workgroup-shared
  atomics, which are mandatory. Slower per pass but **portable to every
  conformant WebGPU implementation including Safari and Quest browser**.

The viewer's job is portability, not raw throughput. Onesweep can land
later behind a `requiredFeatures: ['storage-atomics']` check.

References reused (cited inline in the WGSL):

- C. Wyman, "wgsl-radix-sort"
  <https://github.com/cwyman/wgsl-radix-sort> — three-kernel layout.
- Merrill & Grimshaw 2010, "High Performance and Scalable Radix Sorting".
- antimatter15 / splatviz GPU sort prototypes (multi-pass LSD on u32).

## Bin-major histogram layout

Per-workgroup histograms are stored `histograms[bin * numWgs + wgid]`,
not `[wgid * 16 + bin]`. The exclusive prefix-sum over this layout places
all of bin 0's workgroups first, then bin 1's, etc. — so the scatter
pass's destination offsets are correct ascending-order positions
without an extra grouping pass. This trick is straight out of the
classic LSD radix literature (Merrill).

## Determinism

Same input + same camera ⇒ same output, every run:

- Decode: each splat decoded independently. No floating-point reduction.
- Project: deterministic per-splat math. Quaternion length normalize uses
  `max(length(q), 1e-8)` so q≈0 doesn't NaN.
- Radix sort: stable per pass (stable LSD radix). Equal-depth splats fall
  back to splat-index because the key is `bitcast(depth)` and ties resolve
  in scatter order — which is fixed for a fixed global-id pattern.

## Visual regression

`tests/visual/tests/compute-decode.spec.ts` renders the same scene under
the WebGPU backend with `useComputeDecode: false` and `=true`. Per-frame
pixelmatch threshold is set to `0.02` (2 %); in practice the diff is
under 0.001 (float-rounding noise from the slight WGSL/JS-math ordering
difference in the 2-D covariance projection).

## Bench results

Run from `pnpm --filter @splatforge/viewer run bench`. Bench harness is
in `bench/`. JSON output: `bench/results.json`.

These numbers are from headless Chromium (M-series Mac) on
ANGLE/Metal — the actual native Safari WebGPU path is usually 1.3–1.8×
faster than ANGLE because it skips the GLSL → Metal translation layer.

| Scale | Decode (one-shot) | Frame (avg) | FPS  |
|-------|-------------------|-------------|------|
| 1 M   | 245.7 ms          | 7.86 ms     | 127  |
| 10 M  | 626.1 ms          | 89.21 ms    | 11.2 |

Per-stage estimated breakdown (sort dominates):

| Scale | Project | Sort   | Gather |
|-------|---------|--------|--------|
| 1 M   | 1.57 ms | 5.50 ms| 0.79 ms|
| 10 M  | 17.8 ms | 62.4 ms| 8.92 ms|

**1 M sustains 60 fps with ~52 ms of headroom.** This is the practical
"working set" budget for streaming-tile viewers: keep the active LOD
under 1 M splats per frame and the rest budget covers I/O + camera +
overlay.

**10 M does NOT sustain 60 fps** on this software-Metal path. The sort
is the bottleneck — see "Biggest WebGPU limitation" below.

The v2 "30 GB scenes at 60 fps on mobile" target is realistic *with the
LOD / streaming layer doing its job* — the viewer never needs to render
all 30 GB at once; it renders the visible tile budget (~1–2 M splats)
plus a fade-in band. The compute-decode pipeline removes the CPU as the
bottleneck so the streaming layer can push that envelope.

## Biggest WebGPU limitation hit

**Workgroup-shared memory + single-workgroup global scan.**

The exclusive prefix-sum across the per-workgroup histograms runs in a
single workgroup (256 threads). For 10 M splats that's
`ceil(10e6 / 256) = 39,062` workgroups × 16 bins = **625,000 elements
per scan**. Each thread strides through `625000 / 256 ≈ 2440` elements
serially, twice per pass (once up-sweep, once down-sweep), eight passes.

If WebGPU mandated storage-buffer atomics we'd use a multi-block scan
(Merrill-style status-flag look-back) and shave ~40 % off the radix
sort. We don't, so we eat the single-workgroup scan cost. A `--enable-
experimental-features=storage-atomics` flag for the bench could prove
out the upper bound; deferred until WebGPU 1.1 makes storage atomics
mandatory.

Secondary: **`maxStorageBufferBindingSize` defaults to 128 MB**. At 10 M
splats × 64 B = 640 MB, the device must advertise 1 GB+. M-series Mac
adapters do (`maxStorageBufferBindingSize >= 2^31`); cheap Android
adapters often cap at 256 MB. The pipeline supports requesting a higher
limit at device-creation time; mobile callers should query
`adapter.limits` before constructing the pipeline.

## Files

- `src/webgpu/decode.wgsl` — decode + project compute shaders.
- `src/webgpu/radix_sort.wgsl` — 4-bit LSD radix sort.
- `src/webgpu/radix_sort.ts` — TS orchestration of 8 passes.
- `src/webgpu/index.ts` — `ComputeDecodePipeline` (the public surface).
- `src/webgpu/shaders.generated.ts` — bundled WGSL strings; regenerated by
  `scripts/embed-wgsl.mjs` (runs on every build/test).
- `bench/compute-decode.bench.ts` — synthetic-scene bench.
- `bench/run-bench.mjs` — headless Chromium driver.
- `bench/results.json` — last-run output.
- `tests/visual/tests/compute-decode.spec.ts` — visual regression vs CPU
  path.
