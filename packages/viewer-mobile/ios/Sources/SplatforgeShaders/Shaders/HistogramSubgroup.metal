// SPDX-License-Identifier: Apache-2.0
//
// HistogramSubgroup.metal — SIMD-group-accelerated histogram kernel.
//
// This is a faster alternative to `cs_histogram` in RadixSort.metal. Instead
// of bouncing every key through a threadgroup `atomic_uint`, each SIMD-group
// (Apple GPUs use 32-lane subgroups on A11+ / iOS 14+) folds its 32 keys
// together with `simd_sum` so only RADIX = 16 atomic adds happen per SIMD-
// group rather than one per lane. On a 256-wide threadgroup that's 8× fewer
// atomic ops on the threadgroup hist.
//
// Output buffer layout matches the regular histogram:
//   histograms[bin * num_wgs + wgid]
// so this kernel is a drop-in replacement for `cs_histogram`.
//
// The WGSL viewer hasn't shipped a `histogram_subgroup.wgsl` yet (the hand-off
// note in `viewer-mobile/README.md` lists it as planned post-PR-1); the
// algorithm here is the one we intend to upstream. Parity test asserts this
// kernel's `histograms[]` matches the scalar `cs_histogram` output for the
// same input on every fixture.

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

constant uint RADIX_SG = 16u;

struct HistUniforms {
    uint count;
    uint bit_shift;
    uint num_wgs;
    uint _pad;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void cs_histogram_subgroup(
    const device uint        *keys_in    [[buffer(0)]],
    device uint              *histograms [[buffer(4)]],
    constant HistUniforms    &u          [[buffer(5)]],
    uint                      gid        [[thread_position_in_grid]],
    uint                      lid        [[thread_position_in_threadgroup]],
    uint                      wgid       [[threadgroup_position_in_grid]],
    uint                      simd_lane  [[thread_index_in_simdgroup]])
{
    threadgroup atomic_uint wg_hist[16];

    if (lid < RADIX_SG) {
        atomic_store_explicit(&wg_hist[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread reads its key (if in range) and computes its bin.
    bool live = (gid < u.count);
    uint bin  = 0u;
    if (live) {
        bin = (keys_in[gid] >> u.bit_shift) & 0xfu;
    }

    // SIMD-group histogram fold. For each of the 16 bins, every lane
    // contributes (lane_in_bin && live) ? 1 : 0 and we reduce with simd_sum
    // (lane 0 of the SIMD-group then does the single atomicAdd for that bin).
    //
    // This trades 16 simd_sum ops for ~32 atomic adds when bins are diverse,
    // and a much larger savings when many lanes land in the same bin (the
    // common case for sort-bit histograms over already-near-sorted data).
    for (uint b = 0u; b < RADIX_SG; ++b) {
        uint contrib    = (live && bin == b) ? 1u : 0u;
        uint bin_total  = simd_sum(contrib);
        if (simd_lane == 0u && bin_total != 0u) {
            atomic_fetch_add_explicit(&wg_hist[b], bin_total, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < RADIX_SG) {
        uint h = atomic_load_explicit(&wg_hist[lid], memory_order_relaxed);
        histograms[lid * u.num_wgs + wgid] = h;
    }
}
