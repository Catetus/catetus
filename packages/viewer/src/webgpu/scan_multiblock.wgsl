// SPDX-License-Identifier: Apache-2.0
//
// Multi-block exclusive prefix-sum (3-kernel chained scan).
//
// Replaces the single-workgroup `cs_scan` step inside the radix sort. The
// original implementation had a single workgroup of 256 threads stride
// through the entire histogram array (`num_wgs * RADIX` elements). For
// 10 M splats that's ~625 K elements scanned by 256 threads, eight times
// per sort — the dominant cost in the sort.
//
// This file implements the classic chained scan from Merrill & Grimshaw 2010
// using zero storage-buffer atomics (we don't have them mandatory in
// WebGPU 1.0). Three kernels per scan:
//
//   1. cs_scan_per_wg          — each workgroup of `WG_SIZE` threads does an
//                                exclusive scan over its tile, writes the
//                                scan back in-place, and writes the tile
//                                total to `block_sums[wgid]`.
//   2. cs_scan_block_sums      — a single workgroup does an exclusive scan
//                                over the `block_sums` array. For up to
//                                `2 * WG_SIZE * WG_SIZE` = 131 072 tiles
//                                (which is 33 M-element tiles' worth of
//                                block sums; far more than we ever need
//                                — 10 M splats yield ~2 442 tiles) the
//                                single-WG scan with serial striding is
//                                trivial and not the bottleneck.
//   3. cs_scan_add_block_sums  — each workgroup adds its scanned
//                                `block_sums[wgid]` to every element in its
//                                tile, producing the final global exclusive
//                                prefix.
//
// We intentionally keep the bind-group layout identical to the existing
// radix-sort layout (binding 4 = histograms / data, binding 5 = uniforms),
// but add a sixth binding for `block_sums`. The radix-sort orchestration
// rebinds for these kernels because the meaning of bindings 0..3 differs.
//
// Inspired by:
//   - Merrill & Grimshaw 2010, "Parallel Scan for Stream Architectures".
//   - Harris et al., "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3.
//   - Wyman, "wgsl-radix-sort", for the WGSL idiom (workgroup-shared
//     Hillis-Steele).

const WG_SIZE : u32 = 256u;

struct ScanUniforms {
  total:        u32,   // total elements to scan == num_wgs_radix * RADIX
  num_scan_wgs: u32,   // ceil(total / WG_SIZE)
  _pad0:        u32,
  _pad1:        u32,
};

@group(0) @binding(0) var<storage, read_write> data       : array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums : array<u32>;
@group(0) @binding(2) var<uniform>             us         : ScanUniforms;

// Workgroup-shared scan scratch.
var<workgroup> ss_scratch : array<u32, WG_SIZE>;

// ---------------------------------------------------------------------------
// Phase A: per-workgroup exclusive scan over a `WG_SIZE` tile.
//
// Each workgroup loads its tile into shared memory, performs a Hillis-Steele
// scan (inclusive → exclusive), writes the exclusive prefix back to `data`,
// and stores the tile total in `block_sums[wgid]`.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_scan_per_wg(
  @builtin(global_invocation_id) gid  : vec3<u32>,
  @builtin(local_invocation_id)  lid  : vec3<u32>,
  @builtin(workgroup_id)         wgid : vec3<u32>,
) {
  let i = gid.x;
  var v: u32 = 0u;
  if (i < us.total) {
    v = data[i];
  }
  ss_scratch[lid.x] = v;
  workgroupBarrier();

  // Hillis-Steele inclusive scan over WG_SIZE entries.
  for (var offset: u32 = 1u; offset < WG_SIZE; offset = offset << 1u) {
    var add: u32 = 0u;
    if (lid.x >= offset) {
      add = ss_scratch[lid.x - offset];
    }
    workgroupBarrier();
    ss_scratch[lid.x] = ss_scratch[lid.x] + add;
    workgroupBarrier();
  }

  // Inclusive[lid] - original[lid] = exclusive[lid].
  let inclusive = ss_scratch[lid.x];
  let exclusive = inclusive - v;
  if (i < us.total) {
    data[i] = exclusive;
  }

  // Last live thread in this tile writes the block total.
  // For a fully-populated tile this is lid.x == WG_SIZE - 1.
  // For the tail tile (i >= us.total beyond some point) we still want the
  // last-loaded element's inclusive prefix, which is the tile's total over
  // the valid prefix only (v was zero for OOB lanes).
  if (lid.x == WG_SIZE - 1u) {
    block_sums[wgid.x] = inclusive;
  }
}

// ---------------------------------------------------------------------------
// Phase B: single-workgroup exclusive scan over `block_sums`.
//
// `num_scan_wgs` is small (≤ a few thousand for 10 M splats per pass), so a
// single workgroup of WG_SIZE threads striding through the array is fine.
// This kernel mirrors the original `cs_scan` body but operates on
// `block_sums` instead of `histograms`.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_scan_block_sums(@builtin(local_invocation_id) lid : vec3<u32>) {
  let total = us.num_scan_wgs;
  // Contiguous per-thread chunks. Thread t owns positions [t*chunk, (t+1)*chunk).
  // We CEIL-divide so threads at the tail cover any remainder. The strided
  // layout used previously was incorrect: after the per-thread tsum + scan,
  // `my_block_exclusive[lid]` is the sum over ALL elements owned by threads
  // 0..lid-1 (a strided union, not a contiguous prefix), so Step 3 wrote
  // the wrong exclusive prefix at positions other than `lid`. That broke
  // the radix-sort scan for arrays > WG_SIZE = 256, leaving the sort
  // partially randomized — observable in
  // experiments/webgpu-quality-regression as the "smear / X-ray" regression.
  let chunk = (total + WG_SIZE - 1u) / WG_SIZE;
  let start = lid.x * chunk;
  let end_excl = min(start + chunk, total);

  // Step 1: each thread sums its contiguous slice.
  var tsum: u32 = 0u;
  for (var idx: u32 = start; idx < end_excl; idx = idx + 1u) {
    tsum = tsum + block_sums[idx];
  }
  ss_scratch[lid.x] = tsum;
  workgroupBarrier();

  // Step 2: shared inclusive scan over WG_SIZE per-thread totals.
  for (var offset: u32 = 1u; offset < WG_SIZE; offset = offset << 1u) {
    var v: u32 = 0u;
    if (lid.x >= offset) {
      v = ss_scratch[lid.x - offset];
    }
    workgroupBarrier();
    ss_scratch[lid.x] = ss_scratch[lid.x] + v;
    workgroupBarrier();
  }
  let prefix_total = ss_scratch[lid.x];
  let my_block_exclusive = prefix_total - tsum;
  workgroupBarrier();

  // Step 3: re-walk slice in order, writing exclusive prefix back.
  var running: u32 = my_block_exclusive;
  for (var idx2: u32 = start; idx2 < end_excl; idx2 = idx2 + 1u) {
    let v = block_sums[idx2];
    block_sums[idx2] = running;
    running = running + v;
  }
}

// ---------------------------------------------------------------------------
// Phase C: each workgroup adds its exclusive `block_sums[wgid]` to every
// element of its tile, producing the global exclusive prefix sum.
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG_SIZE)
fn cs_scan_add_block_sums(
  @builtin(global_invocation_id) gid  : vec3<u32>,
  @builtin(workgroup_id)         wgid : vec3<u32>,
) {
  let i = gid.x;
  if (i >= us.total) { return; }
  let bias = block_sums[wgid.x];
  data[i] = data[i] + bias;
}
