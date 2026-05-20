// SPDX-License-Identifier: Apache-2.0
//
// WebGPU radix sort over u32 keys, carrying a u32 payload (the splat index).
//
// Algorithm: classic 8-bit (256-bin) LSD radix sort. Each 32-bit key is sorted
// in 4 sequential passes. Each pass consists of three kernels:
//
//   1. cs_histogram  — per-workgroup local histogram, written out to
//                      `histograms[bin * num_wgs + wgid]`. WG_SIZE = 256, so
//                      each thread owns exactly one bin during init/writeout.
//   2. cs_scan       — exclusive prefix sum over the entire histogram array
//                      (length numWorkgroups * 256). At 10 M splats that's
//                      ~10 M entries — far beyond the single-WG scan's
//                      working set, so the multi-block chained scan in
//                      `scan_multiblock.wgsl` is the only viable path here.
//   3. cs_scatter    — read scanned offsets, scatter each (key, index) pair
//                      into its bucket position.
//
// 8-bit (256-bin) vs the previous 4-bit (16-bin) layout:
//   - Halves the number of passes (8 -> 4). At the per-pass dispatch
//     granularity for 10 M splats, that's a near-2x reduction in radix-sort
//     compute work (modulo the larger scan-input size).
//   - Grows the per-pass histogram array 16x (numWgs * 16 -> numWgs * 256),
//     i.e. ~625 K -> ~10 M u32s per pass at 10 M splats. The multi-block
//     scan in `scan_multiblock.wgsl` (already chained, parallelized across
//     all available workgroups) handles this size; the legacy single-WG
//     `cs_scan` would not.
//   - Workgroup-shared histogram (`wg_hist[RADIX]`) grows 16x to 1 KiB,
//     well under the 16 KiB workgroup-shared cap on every WebGPU device.
//   - `wg_offsets[RADIX]` likewise grows to 1 KiB.
//
// We deliberately avoid storage-buffer atomics: each workgroup writes its own
// histogram slot, the scan runs across many workgroups (see
// `scan_multiblock.wgsl`) using only workgroup-shared atomics in the
// histogram kernel, and the scatter uses a per-workgroup local prefix scan
// to compute intra-workgroup destination offsets without atomicAdd on
// storage. Atomics on storage buffers are still optional in WebGPU 1.0 and
// not on the mandatory feature set.
//
// Inspired by:
//   - Wyman, "wgsl-radix-sort" (github.com/cwyman/wgsl-radix-sort) - overall
//     three-pass split radix structure.
//   - Merrill & Grimshaw 2010, "High Performance and Scalable Radix Sorting".
//   - antimatter15 / splatviz GPU sort prototypes (multi-pass LSD on u32).

const WG_SIZE : u32 = 256u;
const RADIX   : u32 = 256u;      // 8-bit radix -> 256 bins per pass.
const RADIX_MASK : u32 = 0xffu;  // RADIX - 1
const PASSES  : u32 = 4u;        // 32 bits / 8 bits

struct Uniforms {
  count: u32,
  bit_shift: u32,    // 0, 8, 16, 24
  num_wgs: u32,      // ceil(count / WG_SIZE) — chunk-local at large N
  // Per-chunk sort base offset (Stage 5): when the caller chunks a sort that
  // exceeds the WebGPU 1.0 dispatch cap (16.7M splats), the orchestrator
  // sorts each chunk in-place at [chunk_offset_splats, chunk_offset_splats +
  // count). Reads of `keys_in/values_in` and writes of `keys_out/values_out`
  // both add this base. The histogram array is still chunk-local
  // (size num_wgs * RADIX) because per-chunk sort recomputes it.
  // For non-chunked sorts (count <= 16.7M) chunk_offset_splats = 0 and the
  // shader behaves identically to the original code.
  chunk_offset_splats: u32,
};

@group(0) @binding(0) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(1) var<storage, read>       values_in  : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms : array<u32>;
@group(0) @binding(5) var<uniform>             u : Uniforms;

// Workgroup-shared per-bin counters. With 8-bit radix, RADIX == WG_SIZE so
// every thread initializes / reads exactly one bin during init/writeout —
// no `if (lid.x < RADIX)` guard needed.
var<workgroup> wg_hist : array<atomic<u32>, RADIX>;
// Workgroup-shared offsets table used in the scatter pass.
var<workgroup> wg_offsets : array<u32, RADIX>;
// Workgroup-shared scan scratch (single-workgroup scan over per-thread totals).
var<workgroup> scan_scratch : array<u32, WG_SIZE>;
// Per-thread bin assignment in lid order — needed for the deterministic
// scatter rank computation (LSD radix sort requires STABLE scatter; the
// previous `atomicAdd` returned ranks in an undefined thread order and
// silently broke multi-pass sort correctness — splats were sorted by upper
// bits only, leaving fine-grained depth order randomized — visible as the
// "WebGPU vs viewer-app smear" regression).
var<workgroup> wg_bins   : array<u32, WG_SIZE>;
var<workgroup> wg_ranks  : array<u32, WG_SIZE>;

// ---------------------------------------------------------------------------
// Pass 1: histogram. One workgroup of 256 threads handles 256 keys; each
// thread atomically increments the bucket for its key. The atomics here are
// *workgroup*-scoped — WebGPU mandates `atomic<u32>` in `var<workgroup>`.
// (Storage-buffer atomics are NOT used anywhere.)
//
// With RADIX == WG_SIZE the init / writeout loop collapses to one statement
// per thread.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_histogram(
  @builtin(global_invocation_id)  gid : vec3<u32>,
  @builtin(local_invocation_id)   lid : vec3<u32>,
  @builtin(workgroup_id)          wgid : vec3<u32>,
) {
  atomicStore(&wg_hist[lid.x], 0u);
  workgroupBarrier();

  let i = gid.x;
  if (i < u.count) {
    let k = keys_in[i + u.chunk_offset_splats];
    let bin = (k >> u.bit_shift) & RADIX_MASK;
    atomicAdd(&wg_hist[bin], 1u);
  }
  workgroupBarrier();

  // Each thread writes one bin to the global histogram table at slot
  // [bin * num_wgs + wgid]. This bin-major layout means the global exclusive
  // scan naturally places all of bin 0's workgroups first, then bin 1's,
  // etc. — i.e. ascending sort order without an extra grouping pass after
  // the scan. The histogram is chunk-local; the orchestrator reuses the
  // same buffer across chunks but reset/scanned per chunk.
  let h = atomicLoad(&wg_hist[lid.x]);
  histograms[lid.x * u.num_wgs + wgid.x] = h;
}

// ---------------------------------------------------------------------------
// Pass 2 (legacy single-WG scan): exclusive prefix-sum over the entire
// histogram array. This shader runs as a single workgroup (256 threads) and
// strides through the entire `histograms` array. Retained for
// non-multi-block-scan callers (e.g. older test setups); at 8-bit radix the
// histogram is ~10 M u32s for 10 M splats, which is far too large for a
// single workgroup to scan in any reasonable time. Real callers MUST use
// `scan_multiblock.wgsl` — the orchestration in `radix_sort.ts` enables it
// by default and treats the multi-block scan as a hard requirement when
// RADIX == 256.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_scan(@builtin(local_invocation_id) lid : vec3<u32>) {
  let total = u.num_wgs * RADIX;
  // Step 1: each thread sums its strided slice.
  var tsum: u32 = 0u;
  var idx: u32 = lid.x;
  loop {
    if (idx >= total) { break; }
    tsum = tsum + histograms[idx];
    idx = idx + WG_SIZE;
  }
  scan_scratch[lid.x] = tsum;
  workgroupBarrier();

  // Step 2: in-shared exclusive scan over WG_SIZE entries. Hillis-Steele.
  for (var offset: u32 = 1u; offset < WG_SIZE; offset = offset << 1u) {
    var v: u32 = 0u;
    if (lid.x >= offset) {
      v = scan_scratch[lid.x - offset];
    }
    workgroupBarrier();
    scan_scratch[lid.x] = scan_scratch[lid.x] + v;
    workgroupBarrier();
  }
  // Convert inclusive -> exclusive scan.
  let prefix_total = scan_scratch[lid.x];
  let my_block_exclusive = prefix_total - tsum;
  workgroupBarrier();

  // Step 3: walk strided slice again, writing exclusive prefix.
  var running: u32 = my_block_exclusive;
  var idx2: u32 = lid.x;
  loop {
    if (idx2 >= total) { break; }
    let v = histograms[idx2];
    histograms[idx2] = running;
    running = running + v;
    idx2 = idx2 + WG_SIZE;
  }
}

// ---------------------------------------------------------------------------
// Pass 3: scatter. Each workgroup recomputes its local histogram (cheaper
// than persisting per-element bin assignments to a side buffer), then derives
// each element's destination index as:
//
//   dst[i] = histograms[bin_of(i) * num_wgs + wgid]  (global exclusive prefix)
//          + local_position_of(i, bin_of(i))
//
// We compute `local_position_of` by re-scanning bins inside the workgroup:
// thread 0 of each bin atomically adds, then we use the returned old value
// as the local rank. Order within a bin is the natural intra-workgroup
// dispatch order — which is deterministic given a fixed global-id pattern.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_scatter(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  // Read the global exclusive offset for this (bin, workgroup) pair from
  // the bin-major scanned histogram. RADIX == WG_SIZE so every thread
  // reads exactly one bin's offset.
  wg_offsets[lid.x] = histograms[lid.x * u.num_wgs + wgid.x];

  let i = gid.x;
  var bin: u32 = 0xffffffffu;   // sentinel = "dead lane"
  var key: u32 = 0u;
  var val: u32 = 0u;
  if (i < u.count) {
    key = keys_in[i + u.chunk_offset_splats];
    val = values_in[i + u.chunk_offset_splats];
    bin = (key >> u.bit_shift) & RADIX_MASK;
  }
  // Stash per-thread bin in lid order so the rank scan below is deterministic.
  wg_bins[lid.x] = bin;
  workgroupBarrier();

  // Deterministic per-bin exclusive rank: thread lid.x scans wg_bins[0..lid.x)
  // and counts matches with its own bin. This preserves input order within a
  // bin (LSD radix sort requires a STABLE scatter). The previous atomicAdd
  // returned ranks in an undefined thread order, silently breaking the
  // multi-pass sort — splats ended up grouped by their upper key bits but
  // jumbled within those groups, producing the WebGPU smear regression.
  // O(WG_SIZE) per thread, O(WG_SIZE²) per workgroup; at 256 threads / 4500
  // workgroups / 4 passes that's ~1.2e9 ops per sort — small vs the project
  // and gather kernels at the same scale.
  var local_rank: u32 = 0u;
  if (bin != 0xffffffffu) {
    for (var t: u32 = 0u; t < lid.x; t = t + 1u) {
      if (wg_bins[t] == bin) {
        local_rank = local_rank + 1u;
      }
    }
  }

  if (bin != 0xffffffffu) {
    // wg_offsets[bin] is the chunk-local exclusive prefix from the scan
    // over the chunk's histogram. Add chunk_offset_splats to land the
    // sorted output in the correct chunk slot of the output buffer.
    let dst = wg_offsets[bin] + local_rank + u.chunk_offset_splats;
    keys_out[dst]   = key;
    values_out[dst] = val;
  }
}
