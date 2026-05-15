// SPDX-License-Identifier: Apache-2.0
//
// ScanMultiblock.metal — 3-kernel chained exclusive prefix sum (scan) over an
// arbitrarily-large u32 buffer, intended to replace the single-workgroup
// `cs_scan` in RadixSort.metal once the histogram array exceeds 4096 entries
// (i.e. num_wgs > 256 on a 4-bit radix, num_wgs > 128 on an 8-bit radix).
//
// The WGSL viewer's `scan_multiblock.wgsl` is planned post-PR-1; this MSL
// port follows the standard Blelloch / chained-scan pattern used by CUB,
// Onesweep, and wgsl-radix-sort's multi-block scan:
//
//   Pass A (`cs_scan_reduce`)    — each threadgroup of WG_SIZE threads reduces
//                                  its TG_TILE = WG_SIZE*ITEMS_PER_THREAD
//                                  contiguous elements into a single block
//                                  sum, written to block_sums[wgid].
//   Pass B (`cs_scan_spine`)     — single threadgroup performs an exclusive
//                                  scan over block_sums (which has at most
//                                  WG_SIZE entries since we pick TG_TILE so
//                                  that num_blocks <= WG_SIZE).
//   Pass C (`cs_scan_downsweep`) — each threadgroup recomputes its tile's
//                                  exclusive scan and adds block_sums[wgid]
//                                  to every element, writing the final
//                                  exclusive prefix in place.
//
// We keep ITEMS_PER_THREAD = 4 so one threadgroup covers 1024 elements; that
// supports up to 256 * 1024 = 262,144-element scans in a single dispatch
// without recursion. For SplatForge today this comfortably covers num_wgs ≤
// 16,384 (i.e. 4.2M splats at 256-key tiles).

#include <metal_stdlib>
using namespace metal;

constant uint SCAN_WG_SIZE         = 256u;
constant uint SCAN_ITEMS_PER_THREAD = 4u;
constant uint SCAN_TG_TILE         = 1024u; // WG_SIZE * ITEMS_PER_THREAD

struct ScanUniforms {
    uint count;
    uint num_blocks;   // ceil(count / SCAN_TG_TILE)
    uint _pad0;
    uint _pad1;
};

// ---------------------------------------------------------------------------
// Pass A: per-block reduce.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_scan_reduce(
    const device uint        *data       [[buffer(0)]],
    device uint              *block_sums [[buffer(1)]],
    constant ScanUniforms    &u          [[buffer(2)]],
    uint                      lid        [[thread_position_in_threadgroup]],
    uint                      wgid       [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[256];

    uint base = wgid * SCAN_TG_TILE;
    uint sum  = 0u;
    for (uint k = 0u; k < SCAN_ITEMS_PER_THREAD; ++k) {
        uint idx = base + k * SCAN_WG_SIZE + lid;
        if (idx < u.count) {
            sum += data[idx];
        }
    }
    scratch[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two reduction.
    for (uint offset = SCAN_WG_SIZE >> 1u; offset > 0u; offset >>= 1u) {
        if (lid < offset) {
            scratch[lid] = scratch[lid] + scratch[lid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0u) {
        block_sums[wgid] = scratch[0];
    }
}

// ---------------------------------------------------------------------------
// Pass B: single-threadgroup exclusive scan over block_sums (≤ SCAN_WG_SIZE
// entries by construction). Hillis-Steele.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_scan_spine(
    device uint              *block_sums [[buffer(1)]],
    constant ScanUniforms    &u          [[buffer(2)]],
    uint                      lid        [[thread_position_in_threadgroup]])
{
    threadgroup uint scratch[256];

    uint v = (lid < u.num_blocks) ? block_sums[lid] : 0u;
    scratch[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1u; offset < SCAN_WG_SIZE; offset <<= 1u) {
        uint add = 0u;
        if (lid >= offset) {
            add = scratch[lid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] = scratch[lid] + add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Inclusive → exclusive.
    uint exclusive = scratch[lid] - v;
    if (lid < u.num_blocks) {
        block_sums[lid] = exclusive;
    }
}

// ---------------------------------------------------------------------------
// Pass C: downsweep. Re-scan the tile locally (exclusive), then add the
// block's prefix from `block_sums[wgid]` to every output element.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_scan_downsweep(
    device uint              *data       [[buffer(0)]],
    const device uint        *block_sums [[buffer(1)]],
    constant ScanUniforms    &u          [[buffer(2)]],
    uint                      lid        [[thread_position_in_threadgroup]],
    uint                      wgid       [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[1024];

    uint base = wgid * SCAN_TG_TILE;

    // Load tile into shared (zero-pad tail).
    for (uint k = 0u; k < SCAN_ITEMS_PER_THREAD; ++k) {
        uint local_idx  = k * SCAN_WG_SIZE + lid;
        uint global_idx = base + local_idx;
        scratch[local_idx] = (global_idx < u.count) ? data[global_idx] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan over SCAN_TG_TILE entries.
    // Each thread participates by handling its ITEMS_PER_THREAD strided slots.
    for (uint offset = 1u; offset < SCAN_TG_TILE; offset <<= 1u) {
        uint local_vals[4]; // ITEMS_PER_THREAD = 4
        for (uint k = 0u; k < SCAN_ITEMS_PER_THREAD; ++k) {
            uint slot = k * SCAN_WG_SIZE + lid;
            local_vals[k] = (slot >= offset) ? scratch[slot - offset] : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < SCAN_ITEMS_PER_THREAD; ++k) {
            uint slot = k * SCAN_WG_SIZE + lid;
            scratch[slot] = scratch[slot] + local_vals[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint block_prefix = block_sums[wgid];

    // Inclusive → exclusive: write (scratch[slot] - data_orig[slot] + block_prefix).
    // We re-read original data to avoid keeping it in shared (would double mem).
    for (uint k = 0u; k < SCAN_ITEMS_PER_THREAD; ++k) {
        uint local_idx  = k * SCAN_WG_SIZE + lid;
        uint global_idx = base + local_idx;
        if (global_idx < u.count) {
            uint orig      = data[global_idx];
            uint inclusive = scratch[local_idx];
            data[global_idx] = inclusive - orig + block_prefix;
        }
    }
}
