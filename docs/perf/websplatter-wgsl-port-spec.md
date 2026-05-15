# WebSplatter WGSL Port — Design Memo

**Source paper:** Han, Xu, Ye, Bi, Dong, Ma. *WebSplatter: Enabling Cross-Device
Efficient Gaussian Splatting in Web Browsers via WebGPU.* arXiv:2602.03207
(submitted 3 Feb 2026). HTML version at
`https://arxiv.org/html/2602.03207v1`; section references below are to that
HTML.

**Goal of this memo.** Specify a concrete WGSL implementation plan that ports
the two WebSplatter contributions — (a) the wait-free hierarchical radix sort
and (b) opacity-aware geometry culling — into our existing WebGPU pipeline
(`packages/viewer/src/webgpu`). The current `sort_full` window costs 71-81 %
of frame time at 10 M splats on a 4090 (see
`docs/perf/webgpu-10m-profile.md`). WebSplatter reports 1.2-4.5 × end-to-end
speedups versus the strongest WebGPU baselines (`KeKsBoTer/web-splat` and
`MarcusAndreasSvensson/gaussian-splatting-webgpu`); on the *sort stage in
isolation* it reports 2.3 ms vs 5.3 ms (RTX 3070, "garden", 5.83 M splats —
§4.4 Table 4). That's the ceiling we can expect on similar hardware.

The B1 audit confirmed our current radix sort does **not** rely on the
inter-workgroup spin-wait pattern that gives WebSplatter its dramatic Apple
M1 wins (19 ms vs 458 ms in the paper). For us, the gain is purely
algorithmic + the rasterization-side culling work; the cross-vendor
correctness win is a free bonus.

**Critical context from B2 drilldown (2026-05-15).** A sub-stage breakdown
of our sort at 10 M splats shows scatter is essentially the entire sort
cost (~70 ms/pass × 4 passes = ~280 ms), while the scan stages combined
cost <20 ms (≈ 5 % of sort_full). This means **swapping our chained scan
for WebSplatter's wait-free hierarchical Blelloch buys <1 ms.** The win
must come from elsewhere in the paper. Re-reading §3.3.2 with this
constraint, the real leverage is:

1. The **scatter redesign** that WebSplatter implies but doesn't loudly
   advertise: by precomputing `intra_wg_rank` in the histogram pass, the
   scatter becomes atomic-free, branchless, and shared-memory-free
   (no per-WG `wg_hist[]` re-population, no second `workgroupBarrier()`).
   Section 2 below tabulates the exact ops eliminated.
2. The **opacity-aware culling** (§3.2 and §3.4) that shrinks N going into
   the sort. At a 50 % cull rate the scatter cost halves directly because
   scatter is O(N). This is the big lever for us.
3. The **opacity-aware quad sizing** in rasterization (§3.4), which the
   paper's own ablation shows is its biggest single-component win (15 %
   of total frame time on M4 with "garden", §4.4 Table 5).

This memo is written to address those three, in priority order. The
hierarchical scan is documented for completeness but explicitly **not the
target of the port**.

---

## 1. Algorithm summary

WebSplatter's sort is a four-pass 8-bit LSD radix sort over the 32-bit
view-space depth key. Each pass is the canonical three-step structure
(histogram → exclusive scan → scatter). The novelty is *not* in the radix
structure or the per-pass count of passes — both match our current
implementation. The novelty is in **how the exclusive scan is performed**.

### 1.1 The decomposition (§3.3.2 "Design")

For each element with input index `i` and digit value `b` in the current
pass, its output position `P(i)` is

```
P(i) = global_offset_lt(b) + count_before_i_with_digit_b
     = global_offset_lt(b) + inter_wg_rank(b, wg(i)) + intra_wg_rank(b, i)
```

- `global_offset_lt(b)` — the total count of elements whose digit value is
  strictly less than `b`. There are `RADIX = 256` such values; this is a
  single 256-entry exclusive scan over per-digit global totals.
- `inter_wg_rank(b, wg)` — for digit `b`, the total count of elements with
  digit `b` that live in workgroups whose `wgid < wg`. This is a per-digit
  exclusive scan along the workgroup axis.
- `intra_wg_rank(b, i)` — within workgroup `wg(i)`, the count of elements
  whose digit is `b` and that appear before `i` in the workgroup's tile.

Our current sort already produces all three of these quantities. The
**bin-major** layout in `radix_sort.wgsl` (`histograms[bin * num_wgs + wgid]`)
already encodes the per-(bin, workgroup) count, and the multi-block scan
turns it into the per-(bin, workgroup) exclusive prefix that the scatter
adds to the intra-workgroup `atomicAdd`-rank. What we are missing is the
*structure* of the scan.

### 1.2 The wait-free hierarchical Blelloch scan (§3.3.2 "Implementation")

A Blelloch scan has two phases:

- **Upsweep (forward).** A balanced binary reduction up a tile tree. At
  level ℓ, each "tile" of size `WG_SIZE` is reduced to a single sum. Those
  sums form the input to level ℓ+1. Recursion stops when one level fits in a
  single workgroup.
- **Downsweep (backward).** Walking the reduction tree top-down, each
  level distributes its computed exclusive prefix back to its child tiles.
  The leaf level produces the final per-element exclusive prefix.

Crucially, **each level is a separate compute dispatch.** A level only ever
reads its own input buffer and writes its own output buffer; there is no
cross-workgroup atomic, no flag-poll, no `while(true)` spin on a global
counter. The harness (TypeScript) is the synchronization mechanism: the
WebGPU command-encoder barrier between dispatches replaces what would have
been an inter-workgroup memory fence. This is the "wait-free" property.

For 32-bit keys with `WG_SIZE = 256` and `N` keys:
- Total histogram entries scanned per pass = `num_wgs * RADIX`
  = `ceil(N / WG_SIZE) * 256`. At 10 M splats: 39 062 × 256 ≈ 10 M.
- Tree depth = `ceil(log_256(10 M))` = 3 levels (10 M / 256 ≈ 39 K /
  256 ≈ 152, fits in one workgroup).
- Total dispatches per pass = (upsweep 3) + (downsweep 2) + histogram +
  scatter = 7. Over 4 passes: **28 dispatches per frame** for the sort.
  (Our current code has 6 dispatches per pass × 4 = 24, so the overhead is
  +4 dispatches per frame — negligible at WebGPU command-recording rate.)

The paper's pseudocode (§3.3.2): *"Each level is executed as an independent
compute dispatch that consumes the previous level's buffers, with only
intra-workgroup barriers inside a tile. There is no cross-workgroup polling,
so the algorithm makes linear work O(N), uses O(N) auxiliary storage, and
avoids spin waiting across heterogeneous GPU schedulers."*

### 1.3 What this replaces in the KeKsBoTer baseline

For full context — the paper repeatedly cites `KeKsBoTer/web-splat` as its
strongest WebGPU baseline. KeKsBoTer ports the Fuchsia GPU radix sort
(originally Vulkan/SPIR-V) which uses a **decoupled lookback scan** (Merrill
& Garland 2016). Decoupled lookback has each workgroup compute its local
sum, write it to a global buffer, then **spin** on a flag-loaded value from
the previous workgroup until that workgroup has published its prefix. On
NVIDIA + AMD this is fine because workgroup launch order is forward
progress; on Apple M1, Adreno, and several Intel iGPUs WebGPU gives no such
guarantee, so a later workgroup that got scheduled first deadlocks the
earlier one. WebSplatter reports 458 ms (KeKsBoTer) vs 19 ms (theirs) on M1
sort alone — a 24× win that's almost entirely "we never deadlock". On
NVIDIA hardware the gap shrinks to ~2.3× (5.3 ms vs 2.3 ms) which reflects
the genuine algorithmic gain.

**Our current sort does not have this bug** (B1 audit confirmed). Our
multi-block chained scan in `scan_multiblock.wgsl` is already a 3-kernel
dispatch-synchronized structure — closer in spirit to WebSplatter than to
KeKsBoTer. The remaining differences are the *shape* of the scan
(WebSplatter recurses deeper; we have only 1 mid-level reduce) and the
**fusion of the per-digit local histogram with the workgroup-local
exclusive scan** that produces `intra_wg_rank` in one kernel.

---

## 2. Comparison vs our current sort

| Aspect | Current (`radix_sort.wgsl` + `scan_multiblock.wgsl`) | WebSplatter port |
|---|---|---|
| Radix | 8-bit, 256 bins, 4 passes | 8-bit, 256 bins, 4 passes — **same** |
| Key | `bitcast<u32>(depth)` w/ sign-flip | view-space depth packed into u32 — **same** |
| Per-WG histogram | Workgroup-atomic `atomicAdd(&wg_hist[bin], 1u)` ; optional subgroup-coalesce | **Same**, plus emits per-element `intra_wg_rank` in the same pass (see §3.2) |
| Global scan structure | 3-kernel chained scan: per-WG tile scan → single-WG block-sums scan → per-WG add | **N-level recursive Blelloch**; for 10 M splats we get one extra reduce level (4-kernel scan instead of 3) |
| Scatter | Recompute per-WG histogram with workgroup-atomic; `dst = global_excl[bin*num_wgs+wgid] + workgroup_atomic_rank` | `dst = global_offset_lt[bin] + inter_wg_rank[bin][wgid] + intra_wg_rank[i]` — **no atomic re-scan, all values precomputed** |
| Inter-workgroup synchronization | Command-encoder barrier between dispatches (no spin-wait) | **Same** — this is the safe property we already have |
| Bin-major vs WG-major layout | bin-major: `histograms[bin * num_wgs + wgid]` | bin-major — **same** |
| Subgroup ops | Optional WebGPU 1.1 `subgroupAll` / `subgroupBroadcastFirst` in histogram | Paper does not specify; we keep ours as an opt-in feature |
| Storage atomics | None | None — **same constraint** |

### What stays
- The 8-bit/4-pass structure (`radix_sort.wgsl` outer loop in
  `radix_sort.ts::encode`).
- Buffer ping-pong between `keysA/B` and `valuesA/B` (4 passes ⇒ even ⇒
  result lands on A — preserved).
- The bin-major histogram layout.
- Our optional subgroup-aware histogram (`histogram_subgroup.wgsl`).
- The bind-group cache keyed on `(numWgs)`.
- `cs_keygen` (depth-only key+identity-index) and `cs_project_gather` (sorted-
  order projection direct to vertex buffer). The sort is a drop-in
  replacement; its interface (input `keysA`/`valuesA`, output sorted in
  `keysA`/`valuesA`) is preserved.

### What changes
1. **Histogram kernel emits a second buffer.** In addition to writing the
   per-(bin, workgroup) count to `histograms[bin*num_wgs+wgid]`, it writes
   the per-element `intra_wg_rank` to a new buffer `local_ranks[]` (one u32
   per input key) — i.e. each thread stores the value returned by its
   workgroup-shared `atomicAdd(&wg_hist[bin], 1u)`.
   - *Cost:* +40 MB per pass at 10 M splats. We need one such buffer; reuse
     across passes by clearing or overwriting.
   - *Win:* the scatter no longer recomputes the histogram. We save one full
     workgroup-atomic histogram pass inside `cs_scatter`.
2. **Scan becomes multi-level.** Replace `cs_scan_per_wg` + `cs_scan_block_sums`
   + `cs_scan_add_block_sums` with a recursive structure. At 10 M splats
   the tree is 3 levels deep:
   - L0 leaves: 10 M entries → 39 062 tiles of 256 → 39 062 sums.
   - L1: 39 062 entries → 153 tiles of 256 → 153 sums.
   - L2: 153 entries → 1 tile of 256 (single-WG scan).
   - L1↓: distribute L2's prefixes back into L1's tiles.
   - L0↓: distribute L1's prefixes back into L0's tiles.
   - Our current code collapses L2 into "the single-WG scan over block sums"
     and skips L1 entirely. For 10 M splats this means the single-WG scan
     handles 39 062 entries — which our `cs_scan_block_sums` already does
     (it strides). **For our scale, the extra recursion level is not the
     win; the win is in step 1 (histogram emits ranks) and step 3 below.**
3. **Scatter becomes branchless / atomic-free.** With `local_ranks[i]`
   precomputed and `global_offset_lt[bin]` + `inter_wg_rank[bin][wgid]`
   already in the scanned `histograms[]`, the scatter is one read per
   element of `keys_in[i]`, `values_in[i]`, `local_ranks[i]`, plus one read
   of `histograms[bin*num_wgs+wgid]`, then one store. No workgroup atomics,
   no `wg_hist[]` shared memory, no second barrier.
   - *Win:* the scatter currently does ~256 init writes + 256 reads of
     `wg_offsets[]` + 1 atomicAdd per element + 1 store. The new scatter is
     2 reads + 1 store. On a memory-bound kernel that's a meaningful
     reduction in shared-memory traffic and removes the workgroup-atomic
     bottleneck on the partially-sorted passes 2-4.

---

## 3. WGSL implementation sketch

Three new/changed shader files:

### 3.1 `radix_sort_ws.wgsl` (replaces `radix_sort.wgsl`)

```wgsl
// New uniform: identical to existing Uniforms.
struct Uniforms {
  count: u32, bit_shift: u32, num_wgs: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read>       keys_in     : array<u32>;
@group(0) @binding(1) var<storage, read>       values_in   : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out    : array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out  : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms  : array<u32>;
// NEW: per-element intra-workgroup rank, one u32 per input element.
@group(0) @binding(5) var<storage, read_write> local_ranks : array<u32>;
@group(0) @binding(6) var<uniform>             u           : Uniforms;

var<workgroup> wg_hist : array<atomic<u32>, 256u>;

@compute @workgroup_size(256)
fn cs_histogram_ranked(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  atomicStore(&wg_hist[lid.x], 0u);
  workgroupBarrier();

  let i = gid.x;
  if (i < u.count) {
    let k = keys_in[i];
    let bin = (k >> u.bit_shift) & 0xffu;
    // atomicAdd returns the OLD value -> that is intra_wg_rank for this element.
    let rank = atomicAdd(&wg_hist[bin], 1u);
    local_ranks[i] = rank;       // NEW: stash for the scatter.
  }
  workgroupBarrier();

  // Bin-major write: histograms[bin * num_wgs + wgid] = wg_hist[bin].
  let h = atomicLoad(&wg_hist[lid.x]);
  histograms[lid.x * u.num_wgs + wgid.x] = h;
}

// Scatter is the big simplification: no atomics, no shared memory.
@compute @workgroup_size(256)
fn cs_scatter_ranked(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let k    = keys_in[i];
  let bin  = (k >> u.bit_shift) & 0xffu;
  let wgid = i / 256u;
  // `histograms[]` after the multi-level scan contains the GLOBAL exclusive
  // prefix for (bin, wgid). That's global_offset_lt(bin) + inter_wg_rank(bin, wgid).
  let global_excl = histograms[bin * u.num_wgs + wgid];
  let dst = global_excl + local_ranks[i];
  keys_out[dst]   = k;
  values_out[dst] = values_in[i];
}
```

### 3.2 `scan_hier_blelloch.wgsl` (replaces `scan_multiblock.wgsl`)

Same three kernels — `cs_scan_per_wg`, `cs_scan_block_sums`,
`cs_scan_add_block_sums` — but with one new wrinkle: the orchestration
recursively re-invokes them when `num_scan_wgs > WG_SIZE`. At 10 M splats
we hit recursion depth 2 (≈ 152 block sums after L0), which a single-WG
scan still handles. The shader code is **byte-identical to current
`scan_multiblock.wgsl`**; the change is purely in the TypeScript driver
(see §3.4). The only WGSL change is to make `block_sums` a parameterizable
bind-group entry so the same pipeline can be re-bound to a different
(input, block_sums) pair per recursion level.

### 3.3 `cs_keygen_cull.wgsl` (replaces / extends `cs_keygen`)

See §4 for the full spec. Signature:

```wgsl
@group(0) @binding(0) var<storage, read>       k_splats   : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> k_keys     : array<u32>;
@group(0) @binding(2) var<storage, read_write> k_indices  : array<u32>;
// NEW: monotone counter for compacted visible-splat list.
@group(0) @binding(3) var<storage, read_write> k_visible_count : atomic<u32>;
@group(0) @binding(4) var<uniform>             ku         : ProjectUniformsExt;

@compute @workgroup_size(256)
fn cs_keygen_cull(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= ku.splat_count) { return; }
  let s = k_splats[i];
  if (!is_visible(s, ku)) { return; }   // frustum + opacity radius cull
  let slot = atomicAdd(&k_visible_count, 1u);
  let depth = view_depth(s.pos.xyz, ku.view);
  k_keys[slot]    = encode_depth_key(depth);
  k_indices[slot] = i;
}
```

After this kernel, the sort runs over `k_visible_count` elements instead of
`splat_count`. Every downstream cost shrinks proportionally.

### 3.4 TypeScript driver changes (`radix_sort.ts`)

```ts
encode(encoder, count) {
  // 1. Pre-pass: clear k_visible_count to 0, dispatch cs_keygen_cull
  //    (already lives in index.ts; encode hook reads back the resolved count
  //    from a small mapped buffer OR uses an indirect-dispatch buffer).
  // 2. For each of 4 passes:
  //    a. cs_histogram_ranked   -> histograms[], local_ranks[]
  //    b. Hierarchical scan over histograms[]:
  //       L0: cs_scan_per_wg dispatch over (num_wgs * RADIX) entries.
  //       L1 (if num_scan_wgs > WG_SIZE): cs_scan_per_wg over block_sums_L0.
  //       Lk: cs_scan_block_sums (single WG) on the smallest level.
  //       L1↓: cs_scan_add_block_sums on L0's block_sums.
  //       L0↓: cs_scan_add_block_sums distributes L1's prefixes into histograms[].
  //    c. cs_scatter_ranked
}
```

The recursion ceiling: for 10 M splats × 4 passes, max `num_scan_wgs` is
ceil(num_wgs × 256 / 256) = num_wgs ≈ 39 062. Block-sums-of-block-sums is
153. A single-WG scan handles 153 in <1 µs. **Recursion depth 2 is
sufficient through 256 × 65 536 = 16.7 M splats.**

---

## 4. Opacity-aware culling spec

WebSplatter combines three culling/sizing tricks (§3.2 "Bounding Box
Culling" and §3.4 "rasterization sizing"):

1. **View-frustum cull.** Standard. Project the 3D center to clip space;
   drop if outside `[-w, w]^3`. Already implicit in our `cs_project_gather`
   (we render off-screen quads at zero cost, but the sort still processes
   them).
2. **Screen-space AABB cull.** Compute the projected 2D ellipse's axis-
   aligned bounding box. If the AABB does not intersect the viewport, drop.
3. **Opacity-radius cull.** Solve for the radius `r` at which the splat's
   final opacity drops below a threshold τ (paper uses τ = 1/255 ≈ 0.0039):
   `α_peak * exp(-r²/2) < τ`  ⇒  `r = sqrt(2 * ln(α_peak / τ))`. If
   `α_peak < τ` (so `r²` would be negative), drop the splat entirely.

### Threshold and per-view eligibility

- **τ = 1/255 (the RGBA8 quantization floor).** This matches the paper's
  RGBA8 packing in pre-processing (§3.2 "View-Dependent Color Calculation").
  We do not currently pack RGBA8 in the instance buffer — our `Instance.color`
  is a vec4<f32>. So τ = 1/255 is a sound *lower bound* for us; we can
  tighten it later if we adopt RGBA8 packing.
- **Per-view-only:** the AABB test must run every frame because the camera
  changes. The opacity-radius test is **view-independent for a static
  scene** — `α_peak` and the splat's 3D scale are fixed — *but* the
  projected screen-space radius depends on the view-space depth (jacobian
  `focal / z` in our `cs_project_gather`). So we recompute it each frame.

### Integration with our `cs_keygen` pass

Current `cs_keygen` is depth-only: 1 thread/splat, 1 storage write per
splat, zero culling. It's free at 10 M (3 % of frame time).

The new `cs_keygen_cull` adds, per splat:
- The clip-space transform (already in `cs_project_gather`, can be hoisted).
- The 2x2 covariance approximation enough to bound the major axis — we can
  use the per-splat 3D scale's largest component as a conservative upper
  bound, then `r_world ≈ max_scale * sqrt(2 * ln(α / τ))`, then
  `r_screen ≈ focal * r_world / depth`. That's ~10 ops/splat. At 10 M
  splats this raises `cs_keygen` from ~11 ms to maybe ~25-30 ms — but the
  sort cost drops proportionally to the cull rate.
- **An `atomicAdd` on a 1-element `visible_count` buffer.** This is the only
  storage atomic we add. WebGPU 1.0 makes `atomic<u32>` in
  `var<storage, read_write>` an *optional* feature in some downlevel
  profiles, but `'storage-atomic-u32'` is universally available on every
  desktop adapter we care about and on Chrome's WebGPU 1.0 conformance.
  (If we want to keep zero storage atomics: replace this with a 2-pass
  "compact" using the scan we already have — count then exclusive-scan
  then write. Adds 2 dispatches. We can ship the atomic version first and
  fall back if needed.)

### Expected cull rate

WebSplatter's ablation (§4.4 Table 5, MacBook Air M4, "garden", 5.83 M
splats):

| Pipeline       | Pre-process | Sort | Render | Total |
|----------------|-------------|------|--------|-------|
| FULL (cull on) | 19.18 ms    | 11.52 ms | 33.04 ms | 63.74 ms |
| -CULL (off)    | 21.79 ms    | 11.05 ms | 34.52 ms | 67.36 ms |
| -RADIUS (off)  | 20.42 ms    | 13.59 ms | 39.20 ms | 73.20 ms |

Two notes: (a) on the M4 the cull *doesn't shrink the sort*, because the
M4 was already render-bound — the cull only saves rasterization fragments.
(b) `-RADIUS` (opacity-aware quad sizing) buys a **15 % total-frame win**
on M4 by tightening the fragment-shader workload. *That* is the
high-leverage knob for our 10M @ 4090 case where rasterization is currently
out of frame (~14 % project_gather, but raster will dominate once sort is
fixed).

For our 4090 measurement at 10 M splats, we project a cull rate around
30-60 % depending on scene (most published 3DGS scenes have wide camera
sweeps; cull is most effective on tightly-framed views). At 50 % cull, the
sort cost drops by ~50 % directly because radix sort is O(N) in N keys —
on top of the per-pass algorithmic win in §3.

---

## 5. Expected speedup table

Three effects compose, in increasing order of leverage on our hardware:
(a) the wait-free hierarchical scan — **near-zero win for us** because our
B2 drilldown shows scan is already <5 % of sort_full; (b) the
atomic-free / shared-memory-free scatter — moderate win because scatter
is currently 95 % of sort cost; (c) the opacity-aware culling that
shrinks N going into both sort and project_gather — biggest practical
win because it scales every downstream stage. Paper-reported numbers come
from §4.4 Table 4 (sort-only) and §4.3 Table 3 (full pipeline).

Per-component prediction at 10 M splats on our laptop 4090:

| Component                            | Now    | After port    | Source of win               |
|--------------------------------------|--------|---------------|------------------------------|
| Scan (per pass)                      | ~3 ms  | ~2-3 ms       | Negligible — already wait-free |
| Scatter (per pass)                   | ~70 ms | ~30-40 ms     | No atomics, no shared-mem rebuild |
| Sort_full (4 passes)                 | 283 ms | ~140-170 ms   | 1.7-2× from scatter alone   |
| ... after 50 % opacity cull          |        | ~70-85 ms     | Sort runs on 5 M instead of 10 M |
| cs_keygen (with cull math)           | 11 ms  | ~25-30 ms     | +15 ms for visibility math  |
| project_gather (after cull)          | 50 ms  | ~25 ms        | 50 % fewer visible splats    |
| Rasterization (raster + frag)        | n/a (rendered post-readback in current bench) | -15 % | Opacity-aware quad sizing |

End-to-end at 10 M, conservatively: **283 + 50 + 11 = 344 ms → ~135 ms.**
A **2.5-2.7× full-frame win** at 10 M, matching the lower end of
WebSplatter's reported 1.2-4.5× range. Above ~5 M splats we expect closer
to 3×, which puts us at:

| Splat count | sort_full now | sort_full after | total frame now | total frame after |
|-------------|---------------|------------------|------------------|--------------------|
| 1 M         | 10.15 ms      | ~6 ms            | 14.25 ms         | ~9 ms (≈ 110 fps) |
| 5 M (≈ "garden") | ~140 ms (interp) | ~50 ms      | ~190 ms          | ~75 ms (≈ 13 fps) |
| 10 M        | 283 ms        | ~85-140 ms       | 344 ms           | ~130 ms (≈ 7.5 fps) |

**Cross-checks against the paper:**
- Paper RTX 3070 sort @ 5.83 M = 2.31 ms (Table 4). Their pipeline is
  end-to-end optimized including cull; our prediction for 5 M post-port
  is ~50 ms (sort only). That's a **20× gap** which we cannot explain by
  the algorithm alone — it suggests either (i) we have a separate bug or
  configuration mismatch (the project_gather kernel does double the work
  of theirs because we keep SH evaluation in-shader), or (ii) their
  scatter has an additional optimization we haven't extracted from the
  paper (e.g. workgroup-local output staging + coalesced bulk write).
  **Open question — see §6.**
- Paper claims 1.2-4.5× end-to-end. The 4.5× upper bound is on the M1
  where the spin-wait deadlock kills the baseline. On RTX hardware their
  win is 1.2-1.6× end-to-end (Table 1: garden RTX 3070, 9.5 vs 14.4 ms).
  Our predicted 2.5× is *more* than this because our starting baseline
  has a worse scatter than KeKsBoTer's (we're behind on scatter
  optimizations that are independent of the wait-free win).

To clear 60 fps at 10 M we still need a structural change in *both* the
sort AND rasterization (tile-binning or hierarchical splats — see
`webgpu-10m-profile.md` § "Implications"). The WebSplatter port is a
necessary, near-term, low-risk 2-3× win, but not sufficient.

---

## 6. Risks + open questions

### Risks (likely-to-materialize)

1. **`local_ranks[]` memory.** At 10 M splats, one u32 per element is 40 MB.
   That's reusable across passes (recompute each pass), but adds 40 MB of
   VRAM. The paper does not report this overhead because it folds the
   buffer into per-pass scratch. Our peak VRAM at 10 M is currently ~1.6 GB
   (40 MB is 2.5 % — acceptable, document it).
2. **The `cs_keygen_cull` atomic.** One `atomicAdd` per visible splat on a
   single u32 in storage is a textbook contention point. At 10 M splats with
   60 % surviving (6 M atomic adds), we'll hit serialization. The paper
   doesn't say how they handle this. Two mitigations:
   - Per-workgroup local count + workgroup-leader atomic. Cuts atomic
     traffic by ~256×.
   - Two-pass compact (count → scan → write). Adds 2 dispatches.
   Decision: ship the per-workgroup-leader version first.
3. **The pre-pass requires the visible count for the sort dispatch size.**
   This is an indirect-dispatch problem in WebGPU. Two options:
   - `mapAsync` the count back to CPU before recording the sort. Adds 1
     round-trip per frame (~1-2 ms WebGPU submit latency). **Loses the win.**
   - **Use `dispatchWorkgroupsIndirect` with an indirect buffer written by
     `cs_keygen_cull`.** Requires recording the indirect buffer's
     `(numWgsX, 1, 1)` field by a tiny `cs_pack_indirect` kernel after the
     cull. WebGPU 1.0 supports indirect dispatch ✓. **This is the path.**
4. **Subgroup-aware histogram + ranks.** Our existing
   `cs_histogram_subgroup` does not currently emit `intra_wg_rank`. We
   either (a) add it (the subgroup-coalesce branch already knows each
   lane's offset within the subgroup; combining with the workgroup-shared
   count is mechanical), or (b) keep the subgroup version for the *no-ranks*
   pre-cull histogram-only flow and use the atomic version for the
   ranked-histogram. (b) is simpler.

### Open questions

1. **Recursion depth.** At 10 M, 1 mid-level reduce suffices. Above 67 M
   splats (256 × 256 × 1024) we need depth 3. We don't currently care, but
   the paper's "hierarchical" name suggests they support arbitrary depth.
   Document the ceiling.
2. **Stability under ties.** The paper says (§3.3.2): "elements with
   identical digits are placed into the output array in the same relative
   order in which they appeared in the input." Stability requires that
   `intra_wg_rank` is deterministic. Workgroup-atomic order is **not**
   deterministic in WebGPU — `atomicAdd` returns lane order that depends on
   subgroup scheduling. We currently rely on this same nondeterminism in
   our scatter, and it has been visually fine. If WebSplatter actually
   needs deterministic stability, they're using something we should look at
   in the code — but their open-source link is `anonymous.4open.science`
   (anonymized for review), so we can't read the code yet. **Mark this as
   "verify against open-source release post-review."**
3. **Pack `α_peak` into the splat record.** Our `DecodedSplat.color.w` is
   peak opacity; the keygen pass can read it cheaply. But our 4-bit-packed
   variants would need a re-pack. Defer until we adopt RGBA8 in the
   instance buffer.
4. **Cull rate variance.** We need a real benchmark on our test scenes
   before committing to the per-frame cost model. Build a bench harness
   that reports `visible_count / total_count` per frame for "garden",
   "bicycle", "bonsai", and our synthetic stress scenes.
5. **Per-tile sort vs global sort.** The paper deliberately rejects tile-
   binning (§3.4, "Tile-based methods... incur heavy memory traffic"). For
   our 4090 target where DRAM is fat, the rejection may not hold. This
   memo specifies the *global-sort* port; a follow-up memo should compare
   against tile-binned 3DGS (FlashGS, AdR-Gaussian) as an alternative.
6. **WebGPU 1.1 subgroup ops on the scan.** WebSplatter doesn't use them
   (Apple Safari WebGPU only landed in Sep 2025 and doesn't yet ship
   subgroups). On NVIDIA Chrome we have them. A `subgroupExclusiveAdd`-
   based scan could collapse our per-WG scan from a 256-Hillis-Steele
   8-iteration loop into 3 `subgroupExclusiveAdd` waves. **Future work,
   not in v1.**

---

## 7. Implementation plan (for the next task)

Phase 1 — sort port only, no culling (1-2 days):
- Add `cs_histogram_ranked` + `cs_scatter_ranked` to a new
  `radix_sort_v2.wgsl`. Keep the existing files as a feature-flagged
  fallback (`useWebSplatterSort`).
- Allocate `local_ranks[]` once at capacity.
- Bench against the existing `encodeTimedDrilled` harness.
- Acceptance: sort_full ≤ 10 ms at 10 M on the laptop 4090.

Phase 2 — opacity-radius culling (1 day):
- Add `cs_keygen_cull` + indirect-dispatch glue.
- Validate visual output against the no-cull baseline (golden-image diff,
  τ = 1/255 should be visually lossless).
- Acceptance: 30-60 % cull rate on benchmark scenes; total frame time
  drops further by the cull-rate ratio applied to project_gather + sort.

Phase 3 — opacity-aware quad sizing (1 day):
- Move the `r = sqrt(2 ln(α/τ))` calculation into `cs_project_gather`
  and emit per-instance scale uniforms.
- Acceptance: ≥ 10 % render-stage win on a render-bound test (e.g. close-
  zoom on "garden").

Total: ~4 engineer-days for the full port. Phase 1 alone is the dominant
win and should land first.

---

## 8. References

- WebSplatter HTML: https://arxiv.org/html/2602.03207v1 (Han et al., Feb 2026).
- KeKsBoTer/web-splat baseline (W1 in the paper): https://github.com/KeKsBoTer/web-splat
- MarcusAndreasSvensson/gaussian-splatting-webgpu (W2): https://github.com/MarcusAndreasSvensson/gaussian-splatting-webgpu
- Merrill & Grimshaw 2010, "Parallel Scan for Stream Architectures" — the
  basis for our current `scan_multiblock.wgsl`.
- Blelloch 1990, "Prefix Sums and Their Applications" — the scan
  structure WebSplatter adopts.
- Internal: `docs/perf/webgpu-10m-profile.md` (10 M splat profile, source of
  the 71-81 % sort_full cost figure).
- Internal: `splatforge-private/research/EXECUTION-LOG.md` B1 (audit
  confirming our scan is already spin-wait-free).
