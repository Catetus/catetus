# LODGE-style hierarchical LOD — design memo

**Status:** design only. No code shipped. Intended to be implemented on top
of the existing `ComputeDecodePipeline` (`packages/viewer/src/webgpu/index.ts`)
and the manifest-driven chunked streaming already present in
`packages/viewer/src/manifest.ts`. See "Phase breakdown" for the staged plan.

**Goal.** Hit 60 fps @ 10 M-splat scenes in the browser by **rendering only
1–2 M splats per frame** drawn from a precomputed LOD pyramid + spatial
chunking, instead of rasterizing all 10 M every frame. This is the only
remaining big-leverage rendering lever after B4b (opacity-radius cull, real-
scene-KILLed: 0.28–0.88 % cull rate on trained scenes) and B7 (atomic-free
scatter, DRAM-bound: +0.27 fps @ 10 M). Sort scaling does not get us there
on its own — `sort_full` at 10 M is 71–81 % of frame time, and even
sub-linear scatter improvements top out at ~10–15 % gains.

**Primary reference.** Kulhanek et al., *"LODGE: Level-of-Detail Large-Scale
Gaussian Splatting with Efficient Rendering"*, NeurIPS 2025, arXiv
[2505.23158](https://arxiv.org/abs/2505.23158).
Cross-checked against Octree-GS (arXiv 2403.17898) and Hierarchical 3DGS
(arXiv 2406.12080) for the parent-Gaussian-merging math, since LODGE itself
*does not merge* — it smooths-then-prunes.

LODGE's headline numbers from the paper are worth restating before the
design, because they set the budget we are aiming for:

| dataset / scene  | full-rep #G | full-rep PSNR | full-rep time | LODGE #G | LODGE PSNR | LODGE time |
|------------------|-------------|---------------|---------------|----------|------------|------------|
| SmallCity (outdoor, H3DGS dataset) | 2.64 M | 26.62 | 15.17 ms | 877 K | 26.57 | **3.88 ms** |
| H3DGS-SmallCity (table 1)          | —      | —     | —        | —     | 26.57 | **257 fps** (A100) |
| H3DGS-Campus  (table 1)            | —      | —     | —        | —     | 24.75 | **219 fps** (A100) |
| Zip-NeRF Alameda                   | —      | —     | —        | —     | 22.41 | **230 fps** (A100) |
| Zip-NeRF London                    | —      | —     | —        | —     | 26.34 | **253 fps** (A100) |
| Zip-NeRF NYC                       | —      | —     | —        | —     | 27.40 | **280 fps** (A100) |

And — critically for us — table 4 reports **35–43 fps on iPhone 13 Mini,
iPhone 15 Pro, MacBook Air M3, and an HP Chromebook**, via the web-based
GaussianSplats3D renderer (Kellogg). That's the existence proof we need:
LODGE-shaped representations run real-time in a JS/WebGPU viewer on
low-end laptops.

---

## 1. Algorithm summary

LODGE is **four ideas stacked**:

### 1.1 Depth-aware 3D smoothing — the LOD-build primitive

LODGE does **not** merge K children into 1 parent (unlike Hierarchical-3DGS
and Octree-GS). Instead:

1. Start from the trained, fully-densified base set `G(0)`.
2. For each coarser level `l ≥ 1`, **copy all Gaussians from `G(0)`** and
   **convolve each Gaussian with a 3D Mip-Splatting filter** sized to the
   target view distance `d_l`. The smoothed Gaussian becomes (eq. 3 in the
   paper):

   ```
   G̃(x) = √( |Σ| / |Σ + (s·d_l / f) · I| )
         · exp( -½ · (x-μ)ᵀ · (Σ + (s·d_l / f) · I)⁻¹ · (x-μ) )
   ```

   where `s` is a hyperparameter (LODGE uses `s = 0.2` per supp.mat),
   `f` is the focal length, and `d_l` is the LOD's nominal viewing depth.
3. **Prune by importance score** (RadSplat τ): score each Gaussian by its
   max contribution across the training views; drop everything below the
   per-LOD threshold. LODGE prunes three times per LOD, with thresholds
   `0.2γ → 0.6γ → γ` (where γ is the per-scene hyperparameter, ~0.02 on
   most scenes).
4. **Fine-tune** for 1 000 steps per prune (so each LOD costs ~3 000
   training iterations on top of the base 30 000), randomizing `d_l` over
   `U(0.7·d_l, 1.3·d_l)` for robustness to off-trajectory cameras.

Why this works: the 3D smoothing filter enforces a Nyquist-safe size
(`> 2·d_l/f` pixel-equivalent) on every Gaussian at the target depth.
Gaussians that were carrying high-frequency detail get blurred into mush
and lose importance score; the prune then removes them outright. The
fine-tune compensates the residual low-frequency error.

**Coarsening ratio in practice.** LODGE's ablation (table 3) on SmallCity:

| representation | #G loaded | #G visible per frame | PSNR | time |
|----------------|-----------|----------------------|------|------|
| Full           | 2 639 K   | 925 K                | 26.62 | 15.17 ms |
| LOD `d=10`     | 3 465 K   | **324 K**            | 26.45 | 5.61 ms |
| LOD `d=10,28`  | 3 815 K   | **209 K**            | 26.50 | 4.75 ms |
| LOD `d=10,28,47,63` | 4 145 K | 172 K              | 26.50 | 4.07 ms |
| LOD `d=10,28` + clusters + vis. filter + opacity blend (=LODGE) | 877 K | 268 K | **26.57** | **3.88 ms** |

Note the *loaded* count goes UP (each LOD level is a copy of `G(0)` with a
filter; the L1 set isn't strictly smaller than L0). What goes DOWN is
*visible per frame* — from 925 K to 268 K. That's the rendering win.

Visible #G is what dominates `sort_full` (which is O(N log N) in survivors,
but with very cache-hostile scatter beyond ~2-3 M; see
`docs/perf/webgpu-10m-profile.md`). Cutting from 10 M visible to ~1-2 M
visible is exactly the lever we need.

### 1.2 LOD selection — per-Gaussian, distance-banded

The active set at camera center `c` is (eq. 2 in the paper):

```
G̃(c) = ⋃_{l=0}^{L-1} { g_i ∈ G(l) : d_l ≤ ||μ_i - c||₂ < d_{l+1} }
```

Each Gaussian in `G(l)` is "active" for camera positions whose distance to
the Gaussian falls inside `[d_l, d_{l+1})`. So **a single splat is
expressed at exactly one LOD level per frame** — there's no per-Gaussian
multi-level blending inside a chunk.

`d_0 = 0`, `d_L = ∞`. The intermediate thresholds are picked by a greedy
search over training views to minimize *average #visible Gaussians per
16×16 tile* (the per-tile thread-divergence cost; LODGE makes the
near-linear argument in §3 of the paper). For SmallCity this lands at
`d_1 = 10 m, d_2 = 28 m`.

### 1.3 Spatial chunking — K-means over training cameras

To avoid recomputing the active set every frame, LODGE precomputes one
active set per *chunk*:

- Run **K-means on the training camera positions** (not on splat
  positions). Number of clusters from eq. 5:

  ```
  K = (4 / d_1) · max_i ||c_i - c̄||
  ```

  where `c̄` is the centroid of training cameras. For outdoor scenes this
  comes out to roughly **one chunk per 5 m of camera traversal**.

- For each chunk centroid `m_k`, evaluate eq. 2 at `c = m_k`, **with each
  `d_l` offset by `r_k`** (the chunk radius = distance to the next-closest
  chunk centroid). This guarantees the chunk's active set is sufficient for
  any camera inside the chunk's Voronoi cell.

- Additionally per-chunk: **re-run importance pruning** on the active set
  using the training cameras inside the chunk *plus random-orientation
  perturbations* at those camera positions (so chunks are robust to camera
  rotation, not just translation). This is the "Visibility filtering" step
  — the row labeled "+ vis. filter" in table 3 above.

The result is a per-chunk static list of active Gaussian indices, which the
viewer streams in/out as the camera traverses chunks.

### 1.4 Opacity blending across chunk boundaries

Loading only the closest chunk's active set produces visible "popping" at
chunk boundaries (figure 7 in the paper). LODGE fixes this with a
**two-chunk blend**: render the union of the two closest chunks, and
modulate the opacity of splats that are *only in one* of the two sets
(symmetric difference) by a linear ramp (eq. 4):

```
α̂_i = α_i · t
t   = clamp(t̄, 0, 1)
t̄   = ((c - m_o)ᵀ · (m_f - m_o)) / ||m_o - m_f||²
```

`m_f` is the centroid of the chunk that *owns* Gaussian `i`; `m_o` is the
other (closer-or-second-closest) chunk's centroid. The blend uses the
**projection of `c - m_o` onto the line `m_o → m_f`**, not Euclidean
distance — this keeps the blend smooth even when the camera doesn't pass
directly through a chunk centroid.

When the camera crosses the bisector between two chunks (`t̄ = 1` for the
old chunk = 0 for the new), the old chunk's exclusive splats are fully
faded, so the stream can drop them and load the next-closest chunk in the
background without visible artifacts.

Crucially: **only the union of the two chunks needs to be in GPU memory
at any moment.** That's the streaming bound. From LODGE's ablation,
*two* chunks at a typical scale = ~877 K splats total loaded on SmallCity
(vs 2.64 M for the full representation). For our 10 M-splat target
scenes, scaled-up linearly: ~3-4 M loaded, ~1-2 M visible per frame.

### 1.5 Why we picked LODGE over Octree-GS / Hierarchical-3DGS

| approach | merge children → parent? | per-frame compute | viable on WebGPU? |
|----------|--------------------------|-------------------|-------------------|
| Hierarchical-3DGS (Kerbl 2024) | Yes — weighted Gaussian merge into BVH interior nodes; depth+SH propagation | Per-frame BVH cut + interpolation between LOD levels (LERP on each splat) | **No** — per-frame graph-cut is a CPU-side traversal; reported 27-38 fps on A100 but the cut step doesn't parallelize well to compute-shader-only |
| Octree-GS (Ren 2024) | Yes — anchor-based densification | Per-frame MLP eval over anchors to pick LOD | **No** — needs an MLP forward per anchor per frame; LODGE explicitly cites this as ~2× slowdown vs. their approach |
| **LODGE** (Kulhanek 2025) | **No** — smooth + prune + fine-tune (offline) | **None** — chunk lookup is `O(K)` over ~20-100 chunks; active set is precomputed | **Yes** — runtime is plain "render this static list of splats, with a per-splat scalar opacity-multiplier from the blend equation". Maps directly onto our existing pipeline. |

LODGE is *the simplest of the three at runtime* — by design, because
it was built to ship on mobile. That's exactly the constraint shape we
have on WebGPU.

The trade-off LODGE accepts: training cost (3× per-LOD fine-tune × 1 000
iterations × N_levels = ~6 000 extra iters for 2 LOD levels), plus the
offline precomputation of `d_l` and the K-means chunking. That cost lives
in **our offline pipeline (Modal A100, ~$1-2 per scene)**, not in the
viewer.

---

## 2. Integration plan — what stays vs what changes

### 2.1 Stays unchanged

- **`packages/viewer/src/webgpu/radix_sort.wgsl`, `scan_multiblock.wgsl`,
  `histogram_subgroup.wgsl`.** The sort is unchanged. We just pass it a
  smaller `count`. At ~1-2 M survivors the per-frame sort cost from
  `docs/perf/webgpu-10m-profile.md` is **~10-15 ms** (interpolating the
  1 M=10.15 ms and 10 M=283 ms breakdowns), comfortably inside the
  16.6 ms 60-fps budget if the rest of the frame is ~5 ms.

- **`cs_decode` + the canonical 64-byte `DecodedSplat` layout.** Per-chunk
  decode happens once on upload and survives across frames; LOD doesn't
  change this. Each LOD level is a separate glTF chunk with its own
  decode call, but the kernel is unchanged.

- **`cs_keygen` + `cs_project_gather`.** These run on the survivors after
  LOD selection. No math changes — the active-set selection upstream just
  determines which splats they iterate over.

- **`packages/viewer/src/manifest.ts`.** The existing `ChunkDescriptor`
  already carries a `lod: number` field and a `loadPriority`. We've been
  emitting `lod: 0` for every chunk; the LODGE generator will emit `lod ∈
  {0,1,2}` and the streamer will start to honor it.

### 2.2 Changes

- **`packages/viewer/src/webgpu/cs_cull.wgsl`** — repurpose for the
  *boundary-blend opacity ramp*. The opacity-radius cull predicate (B4b)
  gets `KILL`ed as the default but the file is the right place for a new
  kernel `cs_lod_blend` that:
  1. Reads the per-splat `chunk_id` from a new per-splat side buffer.
  2. Reads the camera position and the two active chunk centroids from a
     uniform.
  3. Computes `t̄` per eq. 4.
  4. Writes a modified opacity = `α · t̄` into the splat (or into a
     per-splat instance-side buffer, depending on whether we want the
     unmodified opacity preserved for re-blending next frame).

  This is a single-pass, 1-thread-per-splat compute shader analogous to
  `cs_cull`. It reuses the cull's `survivor count → compact_indices`
  scaffolding: every splat that's in *one of the two active chunks* is a
  "survivor"; every splat that's in *both* keeps `t = 1` and every splat
  in the symmetric difference gets `t < 1`.

  Implementation note: the existing cull's prefix-sum compaction is
  exactly the primitive we need to *gather only the active union into the
  sort input*. The kernel becomes "this splat is active iff its chunk_id
  ∈ {chunk_near, chunk_far}", and the existing scan + compact + project
  flow falls out of `encodeWithCull` virtually unchanged.

- **`packages/viewer/src/webgpu/index.ts` `ComputeDecodePipeline`.** Add:
  1. A new `LodSelector` class that, given the current camera position,
     returns `[chunk_near, chunk_far, t_blend]`. CPU-side, runs once per
     frame, O(K) over ~20-100 chunks (negligible).
  2. A `chunkOffsets[]` storage buffer (one u32 per chunk, prefix-summed
     splat counts) so the cull kernel can map splat-index → chunk-id in
     O(log K) via a binary search, OR — simpler — a per-splat `chunkId:
     u32` side buffer (4 bytes/splat × 4 M = 16 MB, fine).
  3. A new `encodeWithLod(...)` method analogous to `encodeWithCull` that
     dispatches: `cs_lod_blend` → `cs_compact` → `cs_project_cmpct` →
     `radix sort` → `cs_gather`. The internal kernels are reused; only
     the predicate inside what's currently `cs_cull` gets swapped for
     the chunk-membership predicate.

- **`apps/web/scripts/`** — add an offline LOD generator (Phase A below).
  Does not touch the viewer runtime; emits LOD-pyramid PLY/glTF files that
  the existing chunk streamer consumes.

- **`apps/web/public/scenes/<scene>/`** — new on-disk layout:

  ```
  scenes/<scene>/
    manifest.json                          # references all LOD chunks
    lod0/
      chunk_0000.glb  ...  chunk_NNNN.glb  # finest LOD, near-camera
    lod1/
      chunk_0000.glb  ...  chunk_MMMM.glb  # mid LOD
    lod2/
      chunk_0000.glb  ...  chunk_PPPP.glb  # coarsest LOD, far-camera
    chunks.json                            # chunk centroids + radii + LOD-membership
  ```

  Today's manifests live in `benches/scenes/real/manifest.json`; the new
  layout is a superset (each LOD chunk is just a `ChunkDescriptor` with
  `lod: l`).

- **`packages/viewer/src/__tests__/`** — add a parity test:
  `lod_active_set_matches_python.test.ts`. The offline generator emits
  per-chunk active-set lists; the runtime computes them; they must
  match bit-for-bit for a fixed (scene, camera) pair.

### 2.3 What about B4b's `cs_cull` infrastructure?

The opacity-radius predicate is dead in production (real-scene KILL), but
the **compute infrastructure** behind it — `cs_cull → scan_multiblock →
cs_compact → cs_project_cmpct` — is exactly the predicate-and-gather
flow we need for LOD chunk boundaries. We:

- Keep `cs_cull.wgsl` on `main` (already there as of `3886d19`).
- Replace the *predicate body* (the `alive = 1` decision) with the
  chunk-membership test described above.
- Reuse the scan, compact, and project_cmpct kernels verbatim.

That's a ~50-line WGSL change inside `cs_cull` + a ~30-line orchestration
change inside `ComputeDecodePipeline`. The hard part is the offline
precompute, not the runtime.

---

## 3. Phase breakdown

Total estimated effort: **5-7 weeks of focused engineering**. Spread
across one ML engineer (offline) + one rendering engineer (runtime),
the critical path is Phase C.

### Phase A — Offline LOD generator (1-2 weeks)

**Goal.** From a trained 3DGS PLY (e.g. `bonsai.ply` 1.16 M splats,
`bicycle.ply` 3.62 M splats, or any future 10 M scene), emit a **3-level
LOD pyramid as PLYs**.

**Files to create:**

- `apps/web/scripts/lod-generate.py` — Python entry point. Takes
  `--input bonsai.ply --output scenes/bonsai/`.
- `apps/web/scripts/lib/depth_aware_smooth.py` — eq. 3 from LODGE. Given
  a trained 3DGS PLY + a depth `d_l`, return a new PLY where each
  Gaussian's `Σ` has been replaced by `Σ + (s·d_l/f)·I`. Pure NumPy.
  (Inflates only the covariance — positions, colors, SH stay unchanged.)
- `apps/web/scripts/lib/importance_prune.py` — RadSplat-style importance
  score: for each Gaussian, the max over training cameras of its alpha
  contribution. Implemented as a gsplat-1.5.3 rasterizer hook (we already
  use gsplat on the 4090 box per `apps/diff-repack/` patterns). Drop any
  splat below threshold τ.
- `apps/web/scripts/lib/lod_finetune.py` — 1 000 fine-tune iterations on
  the smoothed-then-pruned set, randomizing `d_l ∈ U(0.7d_l, 1.3d_l)` per
  step. Uses gsplat's standard DSSIM+L1 loss. Runs on the **4090 box** at
  ~3 min per LOD level; or on **Modal A100** for $0.50-$1 per scene.

**Predicted output.** For a 10 M-splat scene at `d_1 = 10 m, d_2 = 28 m`:
- `lod0/`: ~10 M splats (the original, optionally re-pruned).
- `lod1/`: ~5-6 M splats (smoothed + pruned — LODGE's table 3 implies
  about 35-45 % reduction on a single level for SmallCity, but with
  visibility-filter the actual *loaded* count after Phase B drops more).
- `lod2/`: ~2-3 M splats.

**Predicted fps win in isolation: 0.** Phase A alone produces files but
doesn't change the viewer. The win lands in Phase C.

**Hardware:** 4090 (Tailscale, single-tenant per
`feedback_serialize_4090_gpu_tasks.md`) OR Modal A100. Fine-tune dominates
wall time (~10-30 min per LOD level on 4090; one full scene = 30-90 min).

**Open question:** for our 1-3 M-splat real scenes (`bonsai`, `bicycle`),
does the LOD pyramid even give a worthwhile reduction? LODGE's
demonstrations all start at 2-7 M splats and end at <1 M. For our
scenes the answer is probably "small win"; the big payoff comes when
we have 10 M-splat scenes from D2 (progressive bitstream) or future
larger captures.

### Phase B — Spatial chunking (1 week)

**Goal.** Given a trained PLY + that scene's training camera poses,
produce a chunk index file.

**Files to create:**

- `apps/web/scripts/lod-chunk.py` — entry point.
- `apps/web/scripts/lib/kmeans_chunks.py` — K-means over training camera
  *positions* (3D world space). Number of clusters from eq. 5
  (`K = (4/d_1) · max_i ||c_i - c̄||`). For our typical small-scene
  captures (~200 training cameras), `K ≈ 10-30`. For 10 M-splat
  city-scale captures, `K ≈ 100-300`.
- `apps/web/scripts/lib/chunk_active_set.py` — per chunk, compute the
  active set via eq. 2 (with `d_l` offset by chunk radius `r_k`). Output:
  `chunks.json` with `[{centroid: [x,y,z], radius: r, active_splat_ids:
  [...], lod_membership: [{splat_id, lod_level}, ...]}, ...]`.
- `apps/web/scripts/lib/visibility_filter.py` — per-chunk importance
  pruning using *only* the training cameras inside the chunk plus 4-8
  random-orientation perturbations per camera. Drops more splats from
  the per-chunk active set; LODGE's table 3 shows this contributes the
  biggest single reduction (#G loaded drops from 3 815 K → 612 K =
  6× compaction).

**Predicted output.** A scene-level `chunks.json` plus per-chunk
glTF/`.mgs` files. Each chunk's binary content is the union of its
LOD-membership splats, repacked as a standard manifest chunk.

**Predicted fps win in isolation: small.** This step doesn't change the
runtime by itself, BUT if the runtime is taught to load only the
two-nearest chunks at LOD 0 (skipping LOD 1/2 for now), we'd see a
**1.5-3× fps lift** because the visible #G drops from `all of LOD 0` to
`~2 chunks of LOD 0`. That's the cheapest interim ship in this whole
plan.

**Hardware:** CPU + 4090 (for the visibility-filter rasterization step).
~10-30 min per scene.

**Open question:** how aggressively should `K` scale? Eq. 5 gives a
roughly-5m chunk radius for outdoor scenes, but for our small indoor
scenes (`bonsai` is ~2 m diameter) eq. 5 gives `K ≈ 1-2` — degenerate.
We'll need a per-scene-class fallback (probably `K = max(8, eq.5)`).

### Phase C — WebGPU runtime LOD selector + chunk loader (1-2 weeks)

**Goal.** Teach the viewer to:
1. Read the multi-LOD manifest.
2. Per frame, compute `[chunk_near, chunk_far, t_blend]` on the CPU.
3. Stream the two active chunks (LOD-aware) into the GPU and decode them.
4. Render only the union.

**Files to create:**

- `packages/viewer/src/lod/selector.ts` — `LodSelector` class. Inputs:
  camera position, list of chunk centroids. Output:
  `{near: chunkId, far: chunkId, t: number}`. O(K) scan; trivial.
- `packages/viewer/src/lod/streamer.ts` — async chunk loader. Tracks
  `{currentlyLoadedChunks: Set<chunkId>}`; on each frame, if
  `{near, far}` ≠ currently loaded, kick off a fetch for the missing
  one. Drops the chunk that's no longer in either slot.
- `packages/viewer/src/lod/chunk_id_buffer.ts` — manages a per-splat
  `chunk_id: u32` storage buffer on the GPU. Populated at chunk-upload
  time (every splat in chunk K gets `chunk_id = K`).

**Files to modify:**

- `packages/viewer/src/webgpu/index.ts` — add `useLod: boolean` to
  `ComputeDecodePipelineInit`. When true, swap `encodeWithCull` →
  `encodeWithLod`. The internal flow is byte-identical to
  `encodeWithCull` except the cull predicate is the chunk-membership
  test.
- `packages/viewer/src/webgpu/cs_cull.wgsl` — add a new kernel
  `cs_lod_blend` (or a uniform flag inside `cs_cull` to swap predicates).
  Predicate: `alive = (chunk_id ∈ {chunk_near, chunk_far})`. The per-splat
  output also writes `inst.color.a = α · t̄` (the blend modulation) so
  the rasterizer never needs to know about the blend.

**Predicted fps win.** This is where it lands.

Working backwards from the LODGE numbers and our `webgpu-10m-profile.md`
baseline (10 M splats: 6.5 fps, `sort_full` = 284 ms, `project_gather` =
50 ms, total ~351 ms):

If LOD selection reduces visible-per-frame from 10 M to ~1 M:
- `sort_full`: 284 ms → ~10 ms (1 M baseline measured)
- `project_gather`: 50 ms → ~3 ms
- `cs_lod_blend + scan + compact`: new overhead ≈ ~15-25 ms over 10 M
  splats (the *full* set has to be evaluated by the blend predicate;
  we can't skip it because we don't know per-splat chunk-membership
  without scanning). This is the same shape as B4b's 23.3 ms cull cost
  at 10 M.
- Total: ~35-45 ms = **22-28 fps**.

If LOD selection reduces visible-per-frame from 10 M to ~2 M *and* the
predicate-evaluation pass is short-circuited (e.g. we keep a CPU-side
list of `splat_id` ranges per chunk and only dispatch the predicate over
the loaded chunks' splats, not the full 10 M):
- `sort_full`: 284 ms → ~22 ms (2× scale from 1 M baseline)
- `project_gather`: 50 ms → ~6 ms
- `cs_lod_blend + scan + compact`: ~2-4 ms (small dispatch over loaded
  union only)
- Total: ~30-32 ms = **30-33 fps**.

To actually hit **60 fps @ 10 M**, we need the visible-per-frame count
to drop to **~700 K-1 M** *and* the predicate pass to be small. That's
the design target — and it's consistent with LODGE's table 3 numbers
(SmallCity: 2.6 M total → 268 K visible = 9.8× reduction).

**Predicted shipping fps for our existing 1-3 M scenes:**
- `bonsai` (1.16 M): already ~67 fps without LOD. LOD might lift to 100 fps
  but it's not the leverage point — `bonsai` doesn't need LOD.
- `bicycle` (3.62 M): currently ~30-35 fps. LOD with `d_1 = 10 m` predicted
  to lift to ~80 fps.
- Future 10 M scene: predicted to lift from 6.5 fps to **45-60 fps**.

**Hardware:** dev laptop with WebGPU (4090 / M3 Max). No GPU training.

**Open question:** what's the actual `sort_full` curve for `N ≤ 2 M`?
The `webgpu-10m-profile.md` only measures 1 M and 10 M. If sort scales
sub-linearly below 2 M (likely — cache-fits cleanly into L2), the wins
are even better than the linear extrapolation above.

### Phase D — Blending across chunk boundaries (1 week)

**Goal.** Ship the eq. 4 opacity blend so chunk transitions don't pop.

Most of this work is *already* covered by Phase C if the per-splat
opacity modulation is implemented in `cs_lod_blend`. Phase D is the
**validation + tuning pass**:

- Implement a test harness in `packages/viewer/bench/lod-popping.bench.ts`
  that flies a camera through a chunk boundary and records per-pixel
  ΔL across consecutive frames. Pre-blend baseline: expected ~10-30 %
  of pixels show >5 % L change at boundary crossing. Post-blend target:
  <2 %.
- Tune the eq. 4 `t̄` formula on real scenes — the paper uses the
  projection-onto-chunk-axis form but reports that on the Zip-NeRF
  indoor dataset the camera frequently lies between *three or more*
  chunks (because chunks fully cover a 3D volume, not a 1D trajectory).
  LODGE says empirically 2-chunk blending was sufficient; we should
  verify on our scenes.
- Add the **async chunk reload** that LODGE describes: when `t̄` crosses
  the bisector, drop the leaving chunk's data and start fetching the
  next-closest chunk in the background. This requires a JS-side
  prefetch heuristic (predicting which chunk the camera will enter
  next), which is straightforward velocity-extrapolation.

**Predicted fps win:** zero (this is a quality-of-rendering fix, not a
speed fix).

**Hardware:** dev laptop only.

**Open question:** memory budget on mobile. Two chunks of LOD 0 +
LOD 1 + LOD 2 unions could be ~3-5 M splats × 64 bytes/splat = ~250 MB
of GPU memory. Likely fits on iPhone 15 Pro (8 GB shared) but probably
NOT on iPhone 13 Mini (4 GB shared). Mobile may need to skip LOD 0 in
the far chunk (load only LOD 1+2 of the second chunk).

---

## 4. Buildability cross-check — what we reuse

| existing primitive | reused as |
|--------------------|-----------|
| `cs_cull.wgsl` predicate + scan + compact + project_cmpct flow | The chunk-membership "alive" predicate plugs into the same flow. ~80% code reuse. |
| `cs_cull.cachedSurvivors` readback machinery | Used identically to pass "active-union count" to the sort dispatch. |
| `RadixSort` (radix_sort.ts, radix_sort.wgsl) | Untouched — just sees a smaller `count`. |
| `cs_decode` + `BYTES_PER_DECODED_SPLAT = 64` canonical layout | Each LOD chunk decodes the same way; we just have ~3× more chunks per scene. |
| `ChunkDescriptor.lod: number` (manifest.ts) | Already in the type. Currently always 0; LOD generator emits 0/1/2. |
| `ChunkDescriptor.loadPriority` | Wire the LodSelector's `[near, far]` pair to set priorities. Existing priority-based fetcher consumes this. |
| `apps/diff-repack/` gsplat 1.5.3 patterns | Reused by `lod_finetune.py` for the fine-tune step. |
| Modal A100 + 4090 patterns in `tasks/scripts/` | Reused for the offline LOD-generation pipeline. |
| `.mgs2` progressive bitstream (`docs/perf/progressive-bitstream-spec.md`) | **Orthogonal but stackable.** A LOD-aware viewer + `.mgs2` chunks means each chunk can also be progressively decoded. Spec-level harmony but no code coupling. |

The single biggest reuse: **`encodeWithCull` is 90 % of `encodeWithLod`**.
The predicate inside `cs_cull` is the only meaningful change.

---

## 5. Open questions — load-bearing for the build

1. **Predicate-pass cost at 10 M.** The cull predicate at 10 M cost 23 ms.
   If `cs_lod_blend` over 10 M splats costs the same, we lose half the
   sort savings. **Mitigation:** the LodSelector can keep a CPU-side
   per-chunk splat-index range and only dispatch the predicate over the
   loaded union (~2-4 M splats max), not the full 10 M. Needs a quick
   prototype to confirm dispatch overhead doesn't dominate.

2. **Memory budget for two LOD-aware chunks on mobile.** Per Phase D.
   Need a real measurement on iPhone 13 Mini (4 GB shared) — possibly we
   ship LOD 1+2 only on low-memory devices.

3. **Does the depth-aware-smooth + prune cycle work on our existing
   trained PLYs without retraining?** LODGE assumes you start from a
   Mip-Splatting-augmented + RadSplat-pruned base. Our `bonsai`/`bicycle`
   PLYs are vanilla Inria 3DGS. The smoothing math is identity-on-Σ if
   the base wasn't trained with the Mip filter, so the LOD set may carry
   aliasing artifacts. **Mitigation:** retrain the base with the 2D Mip
   filter (cheap — 30K iters on a 4090 takes ~30 min for a 1 M scene).

4. **Chunk count for indoor / small scenes.** Eq. 5 degenerates for
   ≤ 2 m scenes. Need a per-scene-class minimum `K` (probably 8).

5. **Predicted real-scene speedup at 10 M is interpolated, not
   measured.** The `webgpu-10m-profile.md` baseline at 10 M is on
   synthetic data; we don't have a trained 10 M-splat scene to measure
   today. Phase A is gated on getting a real 10 M scene (either from
   D2 progressive-bitstream upscale or from a future capture).

6. **K-means chunk centers vs. trajectory-aware chunks.** Outdoor
   linear-trajectory captures (H3DGS SmallCity, our future drone-flight
   captures) chunk cleanly along a 1D line. Indoor free-flight captures
   (Zip-NeRF, our `bonsai`) want a 3D Voronoi tessellation. LODGE uses
   K-means for both; on the indoor case, 3-chunk-blend may be required
   instead of 2-chunk. Empirical test in Phase D.

7. **LOD threshold auto-selection at our scene scale.** LODGE's greedy
   `d_l` search reportedly converges in ~5-10 evaluations of a tile-cost
   histogram over 50 training views. We need to port their cost function
   (it's described qualitatively in §3 of the paper but not given as
   pseudocode). **Mitigation:** start with hardcoded thresholds (`d_1 =
   1 m, d_2 = 5 m` indoor; `d_1 = 10 m, d_2 = 30 m` outdoor); auto-select
   is a Phase A.2 follow-up.

---

## 6. Summary timeline

| phase | wall time | output | shipping fps lift |
|-------|-----------|--------|-------------------|
| A — Offline LOD generator | 1-2 weeks | 3-level LOD PLYs per scene | 0 (build artifact only) |
| B — Spatial chunking + visibility filter | 1 week | Per-chunk active sets + chunks.json | Small if shipped as "load nearest chunk only" interim (1.5-3×) |
| C — Runtime LOD selector + cs_lod_blend | 1-2 weeks | `useLod: true` viewer mode | **Main win** — 10 M @ 6.5 fps → 30-45 fps; full 60 fps if predicate is range-limited |
| D — Boundary blending + async reload | 1 week | Pop-free traversal | 0 fps but ships the quality |

**Total: 5-7 weeks.** Critical path is Phase C. Phases A+B can run in
parallel with Phase C scaffolding (the runtime can ship against
hand-built LOD pyramids during dev). Phases B and C have a clean
deferred-win in between — shipping "load nearest chunk only" without
the blend gives a real fps lift at the cost of visible popping (only OK
for non-interactive demos).

**Confidence.** High for the algorithm (LODGE is published with code at
`https://lodge-gs.github.io/`, mobile-deployed numbers in table 4 are
the existence proof). Medium for our exact fps wins, because Phase A's
LOD-pyramid coarsening ratio depends on the base PLY's training
distribution — we can't predict it precisely without running it. Low
risk on Phase C runtime — it's a small WGSL diff against infrastructure
that already works.

---

## A.1 Phase A.1 BUILT — offline chunker (2026-05-15)

`crates/splatforge-lodge/` + the `splatforge lodge build` CLI subcommand
ship the **offline chunker** half of Phase A. This section documents the
on-disk format the CLI emits, which the Phase A.2 viewer-side manifest
loader will consume.

### A.1.1 What the Phase-A.1 chunker does and does NOT do

The chunker takes a trained 3DGS PLY and emits a `.lodge` directory
containing a `manifest.json` and per-(level, chunk) PLY files. Within
each LOD level, splats are decimated from the original set by
**importance-weighted uniform 3D-grid binning**: per occupied cell, the
splat with the highest `opacity * det(scale)^(2/3)` score survives, all
others are dropped. This is the conservative "smooth-then-prune"
approximation from LODGE §3.1 with the smoothing step skipped (no
Σ inflation, no fine-tune). For Phase A.1 that's deliberate — we ship
the *structure* (manifest schema + chunked layout) so the runtime team
can start integrating, and the smoothing + fine-tune layer is a Phase
A.2 follow-up (needs gsplat on a GPU).

It does NOT yet:
- Apply LODGE eq. 3 depth-aware 3D smoothing per level
- Fine-tune each level for 1 000 iters
- Run K-means over training-camera positions to set the spatial chunk
  partition (we Morton-sort and slice at fixed splat-count chunks
  instead — see §A.1.4)
- Compute LODGE eq. 4 boundary-blend ramps at build time

Those are deferred to Phase A.2 (camera-aware chunk K-means) and the
ML-side fine-tune pass (Modal A100). The manifest schema below has the
fields they will populate — they are emitted today with heuristic
values, which the runtime can use as-is for a first integration.

### A.1.2 On-disk layout

```
<scene>.lodge/
  manifest.json
  level_0/
    chunk_0000.ply
    chunk_0001.ply
    ...
  level_1/
    chunk_0000.ply
    ...
  level_N/
    chunk_0000.ply
```

Per-level chunk PLYs are plain Inria-flavored PLYs written by
`splatforge-ply::write_ply`. Each chunk is independently loadable by
any 3DGS toolchain that consumes PLYs; the LODGE-specific metadata
lives entirely in `manifest.json`.

### A.1.3 `manifest.json` schema (version 1)

```jsonc
{
  // Schema version. Bump on breaking changes. Current = 1.
  "version": 1,

  // Source PLY filename, for provenance only. Readers do not open this.
  "source": "bonsai_iter7000.ply",

  // Splat count of level 0 (the original).
  "original_splat_count": 1157141,

  // Scene-wide AABB: [[min_x, min_y, min_z], [max_x, max_y, max_z]].
  "bbox": [[-1.5, -0.2, -1.4], [1.6, 2.1, 1.5]],

  // Pyramid levels, fine -> coarse. levels[0] is the original.
  "levels": [
    {
      // Level index, 0-based. Level 0 is the original PLY contents.
      "level": 0,

      // Total splats at this level (= sum over chunks).
      "splat_count": 1157141,

      // Splat count relative to level 0. Always 1.0 for level 0.
      "reduction": 1.0,

      // Nominal camera-distance band edge for this level. Phase A.1
      // emits a linear heuristic (level / (L-1)) * scene_diag * 1.5;
      // Phase A.2's training-view greedy search replaces this with the
      // LODGE eq. 5 values. Used by the runtime selector to pick which
      // level a chunk is "active" at.
      "depth_threshold": 0.0,

      // Chunks, in Morton sweep order. Concatenating their splats in
      // this order reproduces the level (set-equal to the source
      // splats at level 0; lossy approximation at coarser levels).
      "chunks": [
        {
          "index": 0,
          "path": "level_0/chunk_0000.ply",
          "splat_count": 96428,

          // Chunk-local AABB.
          "bbox": [[-1.5, -0.2, -1.4], [-0.1, 1.0, 0.2]],

          // Splat-position centroid. Used by the Phase A.2 runtime to
          // pick the two nearest chunks to the camera per frame
          // (LODGE eq. 4 boundary blend).
          "centroid": [-0.78, 0.41, -0.6],

          // Bounding-sphere radius (max distance from centroid to any
          // splat). The runtime expands the per-chunk active set by
          // this distance when the camera is near the chunk edge.
          "radius": 1.32,

          // blake3 hex digest of the chunk PLY bytes. Lets streaming
          // clients detect a stale/swapped chunk without re-fetching
          // the whole manifest.
          "blake3": "ab12...cd34"
        },
        // ...
      ]
    },
    // level_1, level_2, ...
  ]
}
```

All fields are required (no defaults on the read side). The schema is
small enough that a TypeScript type alias on the runtime side is a
1:1 transcription of the JSON.

### A.1.4 Spatial chunking algorithm

Within each LOD level we:

1. Compute a 48-bit 3D Morton code for each surviving splat over the
   *scene-wide* (not per-level) bounding box. Using the scene-wide
   bbox keeps chunk indices comparable across levels — a "north-east"
   chunk at level 0 occupies the same spatial region as a "north-east"
   chunk at level 2, so the runtime can pick level-N's chunk-K to
   replace level-0's chunk-K when the camera recedes.
2. Sort by Morton code, then slice into `ceil(splat_count /
   chunk_target_splats)` evenly-sized contiguous ranges. Default
   `chunk_target_splats = 100 000`.

This is **not** LODGE's K-means-over-camera-positions approach. The
spec calls out that Phase A.2 will re-cluster using training camera
positions once those are available (the on-disk format is unchanged —
the partition just gets recomputed). Phase A.1's Morton slicing gives
roughly cubic spatial groupings that are good enough for an initial
runtime integration and a VRAM-pressure measurement.

### A.1.5 Decimation algorithm — importance-weighted grid argmax

Inside `decimate_to` we iterate the grid resolution by halving /
growing `cells_per_axis` until the survivor count lands within ±15 %
of the target, capped at 12 iterations. The survivor-tracking
"best-so-far" keeps the search stable for lattice-like inputs (where
every grid resolution either over- or under-shoots due to the cubic
cell-count step).

This decimator is deterministic: the per-cell tiebreak is `(higher
importance, lower index) wins`, so two builds of the same PLY produce
byte-identical chunk PLYs (and identical blake3 digests in the
manifest). That stability is important for the round-trip sanity gate
and for caching across redeploys.

### A.1.6 Round-trip sanity

`splatforge lodge unpack -i <dir> --level 0 -o out.ply` reassembles
level 0 by concatenating chunks in manifest order. **The reassembled
PLY contains the same SET of splats as the input PLY** — every splat
is preserved, in possibly different order (the chunker Morton-sorts
within each chunk). Per-splat byte content (position, scale, rotation,
opacity, SH coefficients) is unchanged at level 0. This gives a clean
"binary-equality on the splat set" sanity check; the round-trip CLI
test in `crates/splatforge-cli/tests/cli_smoke.rs::
lodge_unpack_roundtrips_level0_splat_count` enforces it.

For coarser levels (L1+), reassembly is lossy by design — splats are
dropped, not modified. PSNR delta vs. the original is bounded by
LODGE table 3's "+ prune only" row (~0.17 dB drop on SmallCity for a
single coarsening step), with the additional caveat that we skip
LODGE's smoothing + fine-tune so visible aliasing is possible at
unusual camera distances. Phase A.2's fine-tune step recovers that
margin.

### A.1.7 CLI

```bash
# Build a 5-level pyramid (default opts: target_top=100_000,
# chunk_target_splats=100_000, coarsen_ratio=2.0)
splatforge lodge build -i scene.ply -o scene.lodge --levels 5

# Print a summary of an existing pyramid
splatforge lodge info -i scene.lodge

# Reassemble one level back into a flat PLY
splatforge lodge unpack -i scene.lodge --level 0 -o roundtrip.ply
```

Measured on `bonsai_iter7000.ply` (1.16 M splats, 287 MB input PLY,
SH degree 3) running M3-Max release build: **1.9 s wall-time** to
emit a 4-level pyramid (L0 1.16 M, L1 372 k, L2 190 k, L3 98 k) at
430 MB total on disk (= L0 287 MB + coarser-copies overhead).

---

## A.3 Phase A.3 BUILT — per-frame WGSL LOD compute pass + boundary blend (2026-05-15)

`packages/viewer/src/lodge/lod-math.ts`, `lod-pipeline.ts` and the WGSL
kernels `cs_lod_select.wgsl` + `cs_lod_blend.wgsl` ship the runtime
LOD-selection compute pass + LODGE eq. 4 boundary blend on top of the
Phase A.2 loader.

### A.3.1 What Phase A.3 adds

1. **`cs_lod_select.wgsl`** — one thread per chunk (workgroup 64).
   Reads chunk centroid + radius + camera position, evaluates LODGE
   eq. 2 (distance band) + a screen-space-size heuristic to bump
   coarser at projected radius < `ss_size_threshold` px. Emits per-chunk
   `ChunkActivation { level, active, slot, t_blend }` records.

2. **`cs_lod_blend.wgsl`** — per-splat alpha modulation. Reads each
   decoded splat's owning `chunk_id`, looks up its activation record,
   and writes `splat.opacity = active ? alpha * t_blend : 0`. Inactive
   splats fall through the downstream cull predicate (`alpha >= tau`)
   and never reach project / sort.

3. **`LodgeLODPipeline`** — TS orchestrator that wraps a
   `LodgeChunkLoader` with the per-frame LOD decision, the near/far
   chunk picker for eq. 4, and the byte-layout encoders for the WGSL
   uniform / chunk-record / level-record buffers. Exposes
   `prepareFrame(camera, focalY)` (CPU-only, sub-ms per call on a
   100 M-splat scene) and `streamFrame(camera, focalY)` (CPU + I/O,
   ensures the eq. 4 partners are resident in the GPU pipeline).

### A.3.2 LODGE eq. 4 boundary blend

The boundary-blend ramp is the projection of `(camera - m_o)` onto
the line `m_o → m_f`:

```
t = clamp( ((c - m_o) · (m_f - m_o)) / ||m_o - m_f||² , 0, 1 )
```

The "near" chunk's splats get `α' = α · (1 - t)` and the "far" chunk's
splats get `α' = α · t`. When the camera passes the bisector
(`t = 0.5`) the partners swap roles continuously — no popping. Single-
chunk levels degenerate to `t_blend = 1.0` (no fade).

### A.3.3 Buffer layout (kept in sync with WGSL → tested via emulator)

| buffer | bytes/record | shape |
|--------|--------------|-------|
| `ls_chunks[]` (ChunkDesc) | 32 | `centroid(vec4) + level(u32) + chunk_index(u32) + splat_count(u32) + _pad(u32)` |
| `ls_levels[]` (LevelDesc) | 16 | `depth_threshold(f32) + level(u32) + 2 u32 pad` |
| `ls_activation[]` (ChunkActivation) | 16 | `level(u32) + active(u32) + slot(u32) + t_blend(f32)` |
| `LodSelectUniforms` | 128 | 8 vec4: camera, scene_center, depth_thresholds[0..3], depth_thresholds2[4..7], counts(packed), near_centroid, far_centroid, ss_focal(packed) |

`LOD_MAX_LEVELS = 8` (depth thresholds fit two vec4s).

### A.3.4 Test gates

`packages/viewer/src/__tests__/lodge-lod-math.test.ts` (18 tests):
- LODGE eq. 4 ramp behaviour at endpoints, midpoints, off-axis cameras,
  degenerate near==far.
- `selectChunkActivation` decision tree: distance band, SS-size bump,
  slot assignment, chunk-radius slack at the band edge.
- `pickNearFarChunks` semantics on 2-chunk and 1-chunk levels.

`packages/viewer/src/__tests__/lodge-lod-pipeline.test.ts` (10 tests):
- Level selection at scene center vs scene far.
- Record-order semantics (per-(level, chunk) flattening).
- Active-splat-count accounting.
- Near/far chunk picking at off-center cameras.
- Byte-layout encoder constants (`CHUNK_RECORD_BYTES`,
  `LEVEL_RECORD_BYTES`, `ACTIVATION_BYTES`, `LOD_UNIFORMS_BYTES`).
- Activation decode round-trip.
- Streaming integration with a synthetic-PLY mock fetcher: cache-hit
  semantics across consecutive `streamFrame()` calls.

`packages/viewer/src/__tests__/lodge-lod-wgsl-emulation.test.ts`
(4 tests):
- A hand-written WGSL "interpreter" that mirrors `cs_lod_select.wgsl`
  byte-for-byte, fed the encoder output, must agree with the JS
  reference on the same activations. This is the closest we can get
  to gating the GPU kernel without spinning up real WebGPU inside
  vitest.

Total: 160 / 160 viewer tests pass. `pnpm lint` clean.
`cargo test -p splatforge-lodge`: 8 / 8 pass (Phase A.1 untouched).

### A.3.5 4090 bench harness

`packages/viewer/bench/real-scene-lodge.bench.ts` +
`real-scene-lodge.html` drive the LOD pipeline against a real `.lodge`
directory through the existing `ComputeDecodePipeline`. Discovered by
`run-bench-windows.mjs` when any `*.lodge/` directory lives in
`SF_BENCH_PLY_DIR`; merged into `results.json::realSceneLodge`. Per
scene, the bench measures fps at every level that fits in the device's
`maxBufferSize` budget, picks the level the CPU heuristic would
choose, and reports speedup vs L0.

### A.3.6 Out of scope (deferred to Phase B)

- LODGE depth-aware 3D smoothing (eq. 3) — the Phase A.1 decimator
  approximates it with importance-weighted grid argmax; the
  smoothing + per-LOD fine-tune is the ML-side Phase B job (gsplat,
  Modal A100, ~$1/scene).
- K-means over training-camera positions for chunk partition — Phase
  A.1 uses Morton-sorted contiguous slices. Camera-aware re-clustering
  is a re-emit of the on-disk `manifest.json` with the same schema.
- Greedy training-view search for `d_l` thresholds — Phase A.1 emits
  a linear heuristic; Phase B's training-view bench replaces it.
- Async chunk prefetch / velocity-extrapolated streaming — Phase A.3
  loads (near, far) synchronously each frame; the production path
  will overlap fetch with the current frame's render.
