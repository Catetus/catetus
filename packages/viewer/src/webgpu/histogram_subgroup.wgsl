// SPDX-License-Identifier: Apache-2.0
//
// Subgroup-aware histogram kernel for the radix sort.
//
// Functionally equivalent to `cs_histogram` in `radix_sort.wgsl`, but uses
// WebGPU 1.1 subgroup ops to cut the workgroup-shared atomic-add traffic
// when several lanes in the same subgroup hit the same bin (a common case
// once the keys are partially sorted - i.e. all passes after the first).
//
// Strategy (small, conservative - one workgroup-shared atomic per
// (subgroup, unique-bin) pair instead of per-lane):
//
//   - Each lane computes its own `bin` (or RADIX = "dead lane").
//   - If every live lane in the subgroup has the same bin (cheap
//     `subgroupBroadcastFirst` + `subgroupAll` check), the elected lane
//     issues a single `atomicAdd(&wg_hist[bin], subgroup_size)`.
//   - Otherwise, each lane falls back to its own atomicAdd. Atomics on
//     `var<workgroup>` are mandatory in WebGPU 1.0 and are typically backed
//     by LDS atomics on real GPUs, so the fallback is still fine.
//
// Why this pattern and not per-bin ballot+popcount: with RADIX == 256
// (8-bit), a ballot-per-bin loop is 256 iterations per lane, which dwarfs
// the gain. The "all-lanes-agree" shortcut is O(1) per lane and is the
// dominant case for partially-sorted keys (passes 2..4 of a 4-pass sort).
//
// Bind-group layout is *identical* to `radix_sort.wgsl`s `cs_histogram`
// (same group/binding indices, same struct ordering). The orchestration in
// `radix_sort.ts` swaps between the two kernels at compile time based on
// `adapter.features.has('subgroups')`.

enable subgroups;

const WG_SIZE : u32 = 256u;
const RADIX   : u32 = 256u;
const RADIX_MASK : u32 = 0xffu;

struct Uniforms {
  count: u32,
  bit_shift: u32,
  num_wgs: u32,
  // Per-chunk sort base offset (Stage 5): kept in lockstep with
  // `radix_sort.wgsl::Uniforms`. See that file for the contract.
  chunk_offset_splats: u32,
};

@group(0) @binding(0) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(1) var<storage, read>       values_in  : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms : array<u32>;
@group(0) @binding(5) var<uniform>             u : Uniforms;

var<workgroup> wg_hist : array<atomic<u32>, RADIX>;

@compute @workgroup_size(WG_SIZE)
fn cs_histogram_subgroup(
  @builtin(global_invocation_id)   gid : vec3<u32>,
  @builtin(local_invocation_id)    lid : vec3<u32>,
  @builtin(workgroup_id)           wgid : vec3<u32>,
  @builtin(subgroup_size)          sg_size : u32,
  @builtin(subgroup_invocation_id) sg_lane : u32,
) {
  atomicStore(&wg_hist[lid.x], 0u);
  workgroupBarrier();

  let i = gid.x;
  let live = i < u.count;
  // Dead lanes use a sentinel bin (RADIX) so the "all agree" check below
  // never coalesces a dead lane with a live one. RADIX is out of bounds
  // for wg_hist, so the sentinel never reaches atomicAdd.
  var bin: u32 = RADIX;
  if (live) {
    let k = keys_in[i + u.chunk_offset_splats];
    bin = (k >> u.bit_shift) & RADIX_MASK;
  }

  // Cheap subgroup-wide coalesce: if every lane in this subgroup has the
  // same bin (and is live), one atomicAdd per subgroup instead of `sg_size`.
  let leader_bin = subgroupBroadcastFirst(bin);
  let all_same = subgroupAll(bin == leader_bin);
  if (all_same && leader_bin != RADIX) {
    // Whole subgroup hits the same live bin. One atomicAdd of sg_size.
    if (sg_lane == 0u) {
      atomicAdd(&wg_hist[leader_bin], sg_size);
    }
  } else {
    // Mixed bins (or partially-dead subgroup at the tail). Each live lane
    // does its own atomicAdd; dead lanes (bin == RADIX) do nothing.
    if (live) {
      atomicAdd(&wg_hist[bin], 1u);
    }
  }
  workgroupBarrier();

  let h = atomicLoad(&wg_hist[lid.x]);
  histograms[lid.x * u.num_wgs + wgid.x] = h;
}
