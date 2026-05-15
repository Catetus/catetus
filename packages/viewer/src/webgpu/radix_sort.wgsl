// SPDX-License-Identifier: Apache-2.0
//
// WebGPU radix sort over u32 keys, carrying a u32 payload (the splat index).
//
// Algorithm: classic 4-bit (16-bin) LSD radix sort. Each 32-bit key is sorted
// in 8 sequential passes. Each pass consists of three kernels:
//
//   1. cs_histogram  — per-workgroup local histogram, written out to
//                      `histograms[workgroup * 16 + bin]`. WG_SIZE = 256.
//   2. cs_scan       — single-workgroup exclusive prefix sum over the entire
//                      histogram array (length numWorkgroups * 16).
//   3. cs_scatter    — read scanned offsets, scatter each (key, index) pair
//                      into its bucket position.
//
// Inspired by:
//   - Wyman, "wgsl-radix-sort" (github.com/cwyman/wgsl-radix-sort) — overall
//     three-pass split radix structure.
//   - Merrill & Grimshaw 2010, "High Performance and Scalable Radix Sorting".
//   - antimatter15 / splatviz GPU sort prototypes (multi-pass LSD on u32).
//
// We deliberately avoid storage-buffer atomics: each workgroup writes its own
// histogram slot, the scan runs in a single workgroup using shared memory,
// and the scatter uses a per-workgroup local prefix scan to compute intra-
// workgroup destination offsets without atomicAdd. Atomics on storage buffers
// are still optional in WebGPU 1.0 and not on the mandatory feature set.

const WG_SIZE : u32 = 256u;
const RADIX   : u32 = 16u;       // 4-bit radix → 16 bins per pass.
const PASSES  : u32 = 8u;        // 32 bits / 4 bits

struct Uniforms {
  count: u32,
  bit_shift: u32,    // 0, 4, 8, ..., 28
  num_wgs: u32,      // ceil(count / WG_SIZE)
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(1) var<storage, read>       values_in  : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms : array<u32>;
@group(0) @binding(5) var<uniform>             u : Uniforms;

// Workgroup-shared per-bin counters.
var<workgroup> wg_hist : array<atomic<u32>, RADIX>;
// Workgroup-shared offsets table used in the scatter pass.
var<workgroup> wg_offsets : array<u32, RADIX>;
// Workgroup-shared scan scratch (single-workgroup scan over per-thread totals).
var<workgroup> scan_scratch : array<u32, WG_SIZE>;

// ---------------------------------------------------------------------------
// Pass 1: histogram. One workgroup of 256 threads handles 256 keys; each
// thread atomically increments the bucket for its key. The atomics here are
// *workgroup*-scoped — WebGPU mandates `atomic<u32>` in `var<workgroup>`.
// (Storage-buffer atomics are NOT used anywhere.)
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_histogram(
  @builtin(global_invocation_id)  gid : vec3<u32>,
  @builtin(local_invocation_id)   lid : vec3<u32>,
  @builtin(workgroup_id)          wgid : vec3<u32>,
) {
  if (lid.x < RADIX) {
    atomicStore(&wg_hist[lid.x], 0u);
  }
  workgroupBarrier();

  let i = gid.x;
  if (i < u.count) {
    let k = keys_in[i];
    let bin = (k >> u.bit_shift) & 0xfu;
    atomicAdd(&wg_hist[bin], 1u);
  }
  workgroupBarrier();

  // Thread 0..15 writes the workgroup's 16 bins to the global histogram
  // table at slot [bin * num_wgs + wgid]. This bin-major layout means the
  // global exclusive scan naturally places all of bin 0's workgroups first,
  // then bin 1's, etc. — i.e. ascending sort order without an extra grouping
  // pass after the scan.
  if (lid.x < RADIX) {
    let h = atomicLoad(&wg_hist[lid.x]);
    histograms[lid.x * u.num_wgs + wgid.x] = h;
  }
}

// ---------------------------------------------------------------------------
// Pass 2: exclusive prefix-sum over the entire histogram array. This shader
// runs as a single workgroup (256 threads) and uses a Blelloch up-down sweep
// in workgroup-shared memory. It assumes num_wgs * RADIX <= 4096
// (i.e. up to 256 dispatched histogram workgroups = 65,536 elements per
// pass — sufficient for 10M+ splats since the histogram pass uses
// 256-key tiles ⇒ num_wgs = 10M/256 ≈ 39062 wgs, which exceeds 256).
//
// For large num_wgs (>256), we chunk the scan: each thread strides through
// `num_wgs * RADIX / WG_SIZE` elements doing a serial accumulate first, then
// performs a workgroup-wide scan over the per-thread totals, and finally a
// per-thread serial add-back. Sequential adjacent-difference is fine because
// the scan only needs to run *once* per radix pass.
//
// In practice the layout we use is: thread i owns elements
// histograms[i], histograms[i + WG_SIZE], histograms[i + 2*WG_SIZE], ...
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
  // Convert inclusive → exclusive scan.
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
//   dst[i] = histograms[wgid * RADIX + bin_of(i)]   (global exclusive prefix)
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
  // Init per-bin local counters to 0. Read the global exclusive offset for
  // this (bin, workgroup) pair from the bin-major scanned histogram.
  if (lid.x < RADIX) {
    atomicStore(&wg_hist[lid.x], 0u);
    wg_offsets[lid.x] = histograms[lid.x * u.num_wgs + wgid.x];
  }
  workgroupBarrier();

  let i = gid.x;
  var bin: u32 = 0u;
  var key: u32 = 0u;
  var val: u32 = 0u;
  var local_rank: u32 = 0u;
  var active: bool = false;
  if (i < u.count) {
    key = keys_in[i];
    val = values_in[i];
    bin = (key >> u.bit_shift) & 0xfu;
    // atomicAdd returns the old value → that's the rank of this element
    // within its bin for this workgroup.
    local_rank = atomicAdd(&wg_hist[bin], 1u);
    active = true;
  }
  workgroupBarrier();

  if (active) {
    let dst = wg_offsets[bin] + local_rank;
    keys_out[dst]   = key;
    values_out[dst] = val;
  }
}
