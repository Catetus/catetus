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
