// SPDX-License-Identifier: Apache-2.0
//
// RadixSort.metal — port of `packages/viewer/src/webgpu/radix_sort.wgsl`.
//
// Classic LSD radix sort over u32 keys carrying a u32 payload (splat index).
// The WGSL uses a 4-bit radix (16 bins, 8 passes); we preserve that for byte-
// for-byte parity with the WebGPU viewer. Each pass = 3 chained kernels:
//
//   1. cs_histogram — per-threadgroup local histogram, written out to
//                     `histograms[bin * num_wgs + wgid]`. TG_SIZE = 256.
//   2. cs_scan      — single-threadgroup exclusive prefix sum over the entire
//                     histogram array (length num_wgs * 16).
//   3. cs_scatter   — read scanned offsets, scatter each (key, idx) pair into
//                     its bucket position.
//
// We avoid device-buffer atomics: each threadgroup writes its own histogram
// slot, the scan runs in a single threadgroup using threadgroup-shared memory,
// and the scatter uses a deterministic per-thread predecessor count in
// threadgroup memory to compute local rank — preserving per-pass stability
// (which LSD radix REQUIRES) without depending on atomic-add winner order
// (which is undefined on Apple GPUs).

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint WG_SIZE = 256u;
constant uint RADIX   = 16u;        // 4-bit radix → 16 bins per pass.

struct RadixUniforms {
    uint count;
    uint bit_shift;    // 0, 4, 8, ..., 28
    uint num_wgs;      // ceil(count / WG_SIZE)
    uint _pad;
};

// ---------------------------------------------------------------------------
// Pass 1: histogram. One threadgroup of 256 threads handles 256 keys; each
// thread atomically increments the bucket for its key in threadgroup memory.
// The atomic here is THREADGROUP-scoped (Metal `threadgroup atomic_uint`);
// the COUNT is order-independent so atomic_fetch_add is fine here, only the
// scatter step requires deterministic ordering.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_histogram(
    const device uint            *keys_in    [[buffer(0)]],
    device uint                  *histograms [[buffer(4)]],
    constant RadixUniforms       &u          [[buffer(5)]],
    uint                          gid        [[thread_position_in_grid]],
    uint                          lid        [[thread_position_in_threadgroup]],
    uint                          wgid       [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint wg_hist[16];

    if (lid < RADIX) {
        atomic_store_explicit(&wg_hist[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid;
    if (i < u.count) {
        uint k   = keys_in[i];
        uint bin = (k >> u.bit_shift) & 0xfu;
        atomic_fetch_add_explicit(&wg_hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0..15 write the 16 bins to the global histogram table at
    // [bin * num_wgs + wgid] (bin-major). The subsequent exclusive scan then
    // naturally lays bin 0's workgroups first, bin 1's next, etc.
    if (lid < RADIX) {
        uint h = atomic_load_explicit(&wg_hist[lid], memory_order_relaxed);
        histograms[lid * u.num_wgs + wgid] = h;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: exclusive prefix-sum over the entire histogram array. Single
// threadgroup of 256 threads. Hillis-Steele over per-thread strided sums then
// serial add-back, matching the WGSL implementation.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_scan(
    device uint                  *histograms [[buffer(4)]],
    constant RadixUniforms       &u          [[buffer(5)]],
    uint                          lid        [[thread_position_in_threadgroup]])
{
    threadgroup uint scan_scratch[256];

    uint total = u.num_wgs * RADIX;

    // Step 1: each thread sums its strided slice.
    uint tsum = 0u;
    {
        uint idx = lid;
        while (idx < total) {
            tsum += histograms[idx];
            idx  += WG_SIZE;
        }
    }
    scan_scratch[lid] = tsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Hillis-Steele inclusive scan over WG_SIZE entries.
    for (uint offset = 1u; offset < WG_SIZE; offset <<= 1u) {
        uint v = 0u;
        if (lid >= offset) {
            v = scan_scratch[lid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_scratch[lid] = scan_scratch[lid] + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Inclusive → exclusive.
    uint prefix_total       = scan_scratch[lid];
    uint my_block_exclusive = prefix_total - tsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: walk strided slice again, writing exclusive prefix in place.
    uint running = my_block_exclusive;
    uint idx2    = lid;
    while (idx2 < total) {
        uint v = histograms[idx2];
        histograms[idx2] = running;
        running = running + v;
        idx2 += WG_SIZE;
    }
}

// ---------------------------------------------------------------------------
// Pass 3: scatter. Each element's destination index is:
//   dst[i] = histograms[bin * num_wgs + wgid] + local_rank_in_bin
//
// `local_rank_in_bin` MUST equal the count of threads `j < lid` in this
// threadgroup whose bin equals our bin — that's what makes the per-pass sort
// stable, and stability is REQUIRED for LSD radix to be correct end-to-end.
//
// The WGSL source computes this via `atomicAdd(&wg_hist[bin], 1u)` and a
// comment claims "intra-workgroup dispatch order is deterministic given a
// fixed global-id pattern". That holds on the WebGPU implementations the
// viewer ships against (Chrome / WebKit Metal-backed) but does NOT hold on
// raw Apple GPU Metal: atomic-add winner order within a SIMD-group is
// undefined and was observed to corrupt sort output in our parity tests.
//
// We replace the atomic with a deterministic O(WG_SIZE) intra-threadgroup
// predecessor count. Cost: ~64K threadgroup-memory reads per WG, fully
// cached. The parity test in `KernelParityTests.swift` covers this.
// ---------------------------------------------------------------------------
[[max_total_threads_per_threadgroup(256)]]
kernel void cs_scatter(
    const device uint            *keys_in    [[buffer(0)]],
    const device uint            *values_in  [[buffer(1)]],
    device uint                  *keys_out   [[buffer(2)]],
    device uint                  *values_out [[buffer(3)]],
    const device uint            *histograms [[buffer(4)]],
    constant RadixUniforms       &u          [[buffer(5)]],
    uint                          gid        [[thread_position_in_grid]],
    uint                          lid        [[thread_position_in_threadgroup]],
    uint                          wgid       [[threadgroup_position_in_grid]])
{
    threadgroup uint tg_bins[256];     // per-thread bin (0xff = dead lane)
    threadgroup uint wg_offsets[16];

    if (lid < RADIX) {
        wg_offsets[lid] = histograms[lid * u.num_wgs + wgid];
    }

    uint i   = gid;
    uint bin = 0xffu;
    uint key = 0u, val = 0u;
    bool live = false;
    if (i < u.count) {
        key  = keys_in[i];
        val  = values_in[i];
        bin  = (key >> u.bit_shift) & 0xfu;
        live = true;
    }
    tg_bins[lid] = bin;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (live) {
        uint local_rank = 0u;
        for (uint j = 0u; j < lid; ++j) {
            if (tg_bins[j] == bin) { local_rank += 1u; }
        }
        uint dst = wg_offsets[bin] + local_rank;
        keys_out[dst]   = key;
        values_out[dst] = val;
    }
}
