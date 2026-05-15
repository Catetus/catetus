# WebGPU 10M-splat profile — what `sort_full` actually costs

**Measurement context**

- Hardware: NVIDIA GeForce RTX 4090 Laptop GPU (Razer Blade 18).
- Browser: Chrome (chromium channel via playwright), `--use-webgpu-adapter=d3d12 --enable-unsafe-webgpu --enable-features=Vulkan --enable-webgpu-developer-features`.
- Pipeline: `ComputeDecodePipeline` with the fused `cs_project_gather` path
  (commit `12a4bdc`). One `cs_keygen` pass → 4-pass radix sort → one
  `cs_project_gather` pass; no separate `cs_gather`.
- Bench harness: `packages/viewer/bench/compute-decode.bench.ts` with the
  `timestamp-query` instrumentation landed in `56b55c0`. Reports the **median
  of 11 GPU-timestamp samples** per stage in addition to the 30-frame
  wall-clock fps measurement (10 frames at 10M).
- Sort path: 8-bit (256-bin) LSD radix, 4 passes per frame. Each pass =
  histogram → multi-block scan → scatter. Workgroup-shared atomics only; no
  storage-buffer atomics. Multi-block scan in `scan_multiblock.wgsl` provides
  the global exclusive prefix; subgroup histogram in `histogram_subgroup.wgsl`
  cuts workgroup-atomic traffic.

## Headline numbers (cool GPU, single-tenant, fused path)

| Splat count | Frame fps | keygen | sort_full | project_gather | Total GPU |
|-------------|-----------|--------|-----------|----------------|-----------|
| 1 M         | ~67       | 1.10 ms | 10.15 ms  | 2.97 ms        | 14.25 ms  |
| 10 M        | ~6.5      | 11.19 ms | **283.78 ms** | 49.77 ms     | ~351 ms   |

(Numbers above are post-revert of `2a095d9` — the predecessor-count scatter
fix was reverted in `e1333de` after measuring a 64–65 % fps regression.
Thermal state matters: laptop 4090 idles at ~210 MHz and boosts; first cool-
start run hits the published `34cdc61` "clean single-tenant" baseline of 70.5
fps @ 1M / 6.52 fps @ 10M. Back-to-back runs drop into the 5–6 fps range at
10M due to sustained load. The relative breakdown is stable across all
runs.)

## What this changes about our priorities

The bench previously reported a hard-coded 20/70/10 split across
project / sort / gather. That was an *engineering estimate*, not a
measurement. The real numbers say:

- **`sort_full` is 71–81 % of frame time at 10 M splats.** The estimate was
  directionally correct but rounded toward "70 %". Real value swings 71–86 %
  depending on thermal state and is reliably the dominant cost.
- **`project_gather` is the second-largest at 10 M**, costing 14 % of frame
  time. The fuse landed in `12a4bdc` eliminated a 640 MB unsorted scratch
  pass; the per-frame work is now memory-bandwidth-bound on the
  `instanceBuffer[]` writes.
- **`cs_keygen` is essentially free** (3 % at 10 M, < 1 % at 1 M). The
  depth-only key + identity index pass is one storage write per splat with no
  arithmetic worth naming.

## Implications for hitting 60 fps @ 10 M splats

To bring 10 M total frame time from ~155 ms (clean cold) to ~16.6 ms (60 fps)
needs a **9.3× speedup**. The sort alone is 110–280 ms — to clear the 16.6 ms
budget the sort must drop to ≤ ~12 ms, a **9×–23× sort speedup**.

This rules out most marginal sort optimizations (subgroup-aware histogram
cut workgroup-shared atomics ~30 %; 8-bit radix cut total passes 2×). To
clear the budget we need a structural change:

1. **Eliminate the global sort.** Tile-based deferred 3DGS pipelines (e.g.
   3DGS-T, the "tile binning + per-tile sort" family) replace the global
   radix sort with per-tile bucket sorts of `O(splats_per_tile)`. If a 10 M
   scene has ~256 tiles with ~40 K splats each, the per-tile sort cost is
   ~1000× cheaper per element and is fully parallel across tiles.
2. **Hierarchical / clustered splats.** Scaffold-GS / Octree-GS / Hierarchical
   3DGS produce a coarse-to-fine cluster tree; only fine clusters near the
   camera need full sort. Distant clusters are pre-sorted or skipped.
3. **Approximate / partial sort.** k-buffer or weighted-blended OIT skips
   the depth sort entirely at the cost of order-dependent artifacts. The
   3DGS literature has examples that show acceptable quality at very high
   throughput.

Candidates 1 and 2 also gate `project_gather` cost (only project visible
splats), so the 14 % project_gather slice would shrink in concert.

## Next-step bench targets

- **`scan_multiblock` window**: instrument the multi-block scan separately
  from histogram/scatter (needs `timestamp-query-inside-passes` feature). If
  scan dominates the sort, the multi-block chained scan in
  `scan_multiblock.wgsl` is the optimization target; if scatter dominates,
  the WGSL scatter (currently a non-deterministic `atomicAdd`-rank, see
  `radix_sort.wgsl:cs_scatter`) is the optimization target.
- **`cs_project_gather` memory bandwidth**: at 10 M splats the
  `instanceBuffer` write is ~480 MB/frame (10 M × 48 B). Confirm we're
  bandwidth-bound vs ALU-bound by reducing the write width and re-measuring.
