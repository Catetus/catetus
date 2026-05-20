// SPDX-License-Identifier: Apache-2.0
//
// Pairwise stable merge over two adjacent sorted runs of (u32 key, u32 value).
//
// Used by the chunked radix-sort path (`RadixSort.encode` when
// `count > SPLAT_DISPATCH_CAP`): the input is split into K <= 8 chunks each
// of which is sorted independently with the existing 4-pass radix kernels.
// We then run log2(K) merge passes, each merging adjacent pairs of runs into
// runs of double the length.
//
// One workgroup per chunk-of-output (workgroup_size = 256 = WG_SIZE). Each
// thread is responsible for ONE output slot at index
// `out_idx = u.merge_out_base + u.chunk_offset_splats + gid.x`. It performs
// a Merge-Path binary search across the two runs to find which input owns
// that slot, then writes the (key, value).
//
// Stability: when keys tie between runs A and B, the merge picks from A
// first. Since A's run starts before B's run in the input and the per-chunk
// radix sort is stable within a chunk, the overall sort is stable.
//
// Uniform layout (one merge invocation merges ONE adjacent pair):
//   count                — total elements in the dst buffer (for guard).
//   chunk_offset_splats  — splat-index offset added inside the kernel so the
//                          same dispatch can be chunked across multiple
//                          <= 65535 wg sub-dispatches. The kernel-visible
//                          output slot is gid.x + chunk_offset_splats.
//   merge_out_base       — splat-index where this merge pair's output
//                          starts (== run_a_start).
//   merge_out_len        — total length of the merged run (== a_len + b_len).
//   run_a_start          — input start of run A (sorted).
//   run_a_len            — length of run A.
//   run_b_start          — input start of run B (sorted). Adjacent: ==
//                          run_a_start + run_a_len, BUT we pass it
//                          explicitly so this kernel can also be used to
//                          collapse a tail run (set run_b_len = 0).
//   run_b_len            — length of run B. May be 0 for odd-count tails.
//
// Guard: if (gid.x + chunk_offset_splats) >= merge_out_len, the thread
// returns immediately. This handles the tail of the last chunked
// sub-dispatch as well as the per-pass shrinking output set.

const WG_SIZE : u32 = 256u;

struct MergeUniforms {
  count:               u32,
  chunk_offset_splats: u32,
  merge_out_base:      u32,
  merge_out_len:       u32,
  run_a_start:         u32,
  run_a_len:           u32,
  run_b_start:         u32,
  run_b_len:           u32,
};

@group(0) @binding(0) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(1) var<storage, read>       values_in  : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(4) var<uniform>             u          : MergeUniforms;

// Merge-Path: for output slot k (within the merged run), find the (i, j)
// split such that i + j == k AND A[i-1] <= B[j] AND B[j-1] < A[i] (stable,
// favoring A on ties). We binary-search the diagonal i + j = k.
//
// Returns vec2<u32>(i, j) — the count of A and B elements that go before
// slot k. The element at slot k is then:
//   if (i < a_len && (j == b_len || A[i] <= B[j])) take A[i]
//   else                                            take B[j]
fn merge_path(k: u32) -> vec2<u32> {
  let a_len = u.run_a_len;
  let b_len = u.run_b_len;
  // Search range for i: max(0, k - b_len) .. min(a_len, k).
  var lo: u32 = 0u;
  if (k > b_len) { lo = k - b_len; }
  var hi: u32 = a_len;
  if (k < hi) { hi = k; }

  loop {
    if (lo >= hi) { break; }
    let i = (lo + hi) >> 1u;
    let j = k - i;
    // We want: A[i-1] <= B[j] (when i > 0 and j < b_len) AND
    //          B[j-1] < A[i]  (when j > 0 and i < a_len).
    // If A[i] <= B[j-1], i is too small — search right.
    var a_too_small = false;
    if (j > 0u && i < a_len) {
      let a_i = keys_in[u.run_a_start + i];
      let b_jm1 = keys_in[u.run_b_start + (j - 1u)];
      // Stable: B[j-1] must be strictly less than A[i] for split to be valid.
      // If A[i] <= B[j-1], A side is too small.
      if (a_i <= b_jm1) { a_too_small = true; }
    }
    if (a_too_small) {
      lo = i + 1u;
    } else {
      hi = i;
    }
  }
  let i_final = lo;
  let j_final = k - i_final;
  return vec2<u32>(i_final, j_final);
}

@compute @workgroup_size(WG_SIZE)
fn cs_radix_merge(@builtin(global_invocation_id) gid : vec3<u32>) {
  let local_k = gid.x + u.chunk_offset_splats;
  if (local_k >= u.merge_out_len) { return; }

  // Degenerate cases (one side empty): straight copy.
  if (u.run_b_len == 0u) {
    let src = u.run_a_start + local_k;
    let dst = u.merge_out_base + local_k;
    keys_out[dst]   = keys_in[src];
    values_out[dst] = values_in[src];
    return;
  }
  if (u.run_a_len == 0u) {
    let src = u.run_b_start + local_k;
    let dst = u.merge_out_base + local_k;
    keys_out[dst]   = keys_in[src];
    values_out[dst] = values_in[src];
    return;
  }

  let ij = merge_path(local_k);
  let i = ij.x;
  let j = ij.y;
  let dst = u.merge_out_base + local_k;

  // Pick from A if A still has elements AND (B exhausted OR A[i] <= B[j]).
  // Equal keys tie-break to A for stability.
  var pick_a: bool = false;
  if (i < u.run_a_len) {
    if (j >= u.run_b_len) {
      pick_a = true;
    } else {
      let a_i = keys_in[u.run_a_start + i];
      let b_j = keys_in[u.run_b_start + j];
      if (a_i <= b_j) { pick_a = true; }
    }
  }
  if (pick_a) {
    let src = u.run_a_start + i;
    keys_out[dst]   = keys_in[src];
    values_out[dst] = values_in[src];
  } else {
    let src = u.run_b_start + j;
    keys_out[dst]   = keys_in[src];
    values_out[dst] = values_in[src];
  }
}
