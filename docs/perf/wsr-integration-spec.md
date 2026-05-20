# Weighted Sum Rendering (WSR) Integration Spec

## 1. TL;DR

WSR replaces the non-commutative per-fragment alpha-blend with a commutative depth-weighted sum, making rasterization order-independent and eliminating the radix sort entirely. On bicycle (3.62 M splats), `sort_full` measured at 30.15 ms and accounts for 58 % of GPU time at 19.20 fps (real-scene bench, `real-scene.bench.ts`, RTX 4090 Laptop). Removing the sort reclaims that 30.15 ms and collapses `cs_keygen` (~8 ms/frame at 3.62 M, extrapolated from 11.19 ms at 10 M) plus the gather rearrangement; the remaining per-frame GPU cost becomes projection only (~12–14 ms at 3.62 M). Expected bicycle fps post-WSR: 1000 / (52.2 ms frame − 30.15 ms sort − 8 ms keygen + ~2 ms WSR accumulate overhead) ≈ **52–60 fps**, a **2.7–3.1× improvement** over the current 19.20 fps baseline. At 10 M splats the sort alone is 283.78 ms (71–81 % of frame); WSR converts that to ~0, enabling the structural path to 60 fps that no sort optimization can reach.

## 2. Mechanism

### 2.1 Standard 3DGS alpha-blend (current)

Sequential alpha-blending in back-to-front sorted order:

```
C = Σ_i  c_i · α_i · Π_{j < i} (1 − α_j)
```

The product term `Π_{j<i}(1 − α_j)` is non-commutative — it requires the exact sorted depth order. Radix sort exists solely to provide this order.

### 2.2 WSR rendering equation

Hou et al. (arXiv:2410.18931, ICLR 2025) replace the product with a learnable depth-based scalar weight `w_i`:

```
C = ( c_B · w_B + Σ_i  c_i · α_i · w(d_i) )
    ─────────────────────────────────────────
    ( w_B + Σ_i  α_i · w(d_i) )
```

Where `c_i` is splat color, `α_i` is splat opacity at the fragment, `d_i` is view-space depth, `w(d_i)` is the learned weight, `c_B` / `w_B` are background. Each term is order-independent — no sort needed.

### 2.3 LC-WSR — the chosen variant

Adopt **LC-WSR (Linear Clamped)**:

```
w(d_i) = max(0,  1  −  d_i / σ  −  v_i)
```

`σ` is a per-scene learnable scalar (init: scene depth range). `v_i` is a per-splat learnable bias (init: 0). Cheapest variant (one multiply, one subtract, one max), best quality on Mip-NeRF360, models occlusion via clamp.

For pretrained 3DGS without fine-tuning: `σ = 2 × mean_scene_depth`, `v_i = 0`. Degrades to normalized linear falloff. Expect 1–3 dB PSNR drop on strong-occlusion scenes vs fine-tuned model.

### 2.4 Per-pixel accumulation

Two accumulators per pixel:
- **Numerator:** `N_px = Σ_i  c_i · α_i · w(d_i)` (vec3 RGB)
- **Denominator:** `D_px = Σ_i  α_i · w(d_i)` (scalar f32)

Final: `C = (c_B · w_B + N_px) / (w_B + D_px)`. `w_B ≈ 1e-4` prevents div-by-zero.

## 3. Existing Pipeline Diff

### 3.1 Eliminated kernels

| Kernel | Current cost (bicycle 3.62M) | Decision |
|--------|------------------------------|----------|
| `cs_keygen` | ~8 ms (extrapolated) | Removed — no depth key needed |
| `RadixSort` (`radix_sort.wgsl` + `radix_sort.ts`) | **30.15 ms measured** | Bypassed entirely. Code stays for fallback path. |
| `cs_gather` rearrange | ~3–4 ms | Removed |

Total reclaimed at bicycle 3.62M: ~38–42 ms/frame.

### 3.2 Survived, modified

| Kernel | Change |
|--------|--------|
| `cs_project_gather.wgsl` → `cs_project_wsr` | Remove `g_indices[i]` indirection. Each thread `i` projects `g_splats[i]` directly. Output `(clip_xy, d_i, cov2d, α_i·c_i·w(d_i), α_i·w(d_i))` to a `WSRSplat` storage buffer. `σ` from uniform; `v_i[]` from storage. Workgroup 256, dispatch `ceil(N/256)`. |
| `decode.wgsl` | Unchanged — canonical decoded form is still needed. |

### 3.3 New kernels

| Kernel | File (new) | Responsibility |
|--------|-----------|---------------|
| `cs_wsr_clear` | `packages/viewer/src/webgpu/cs_wsr_clear.wgsl` | Clear numerator + denominator storage textures to 0 at frame start. 16×16 workgroup, dispatch `ceil(W/16)×ceil(H/16)`. |
| `cs_wsr_resolve` | `packages/viewer/src/webgpu/cs_wsr_resolve.wgsl` | Combine `N_px` and `D_px` into final RGBA via `C = (w_B · c_B + N_px) / (w_B + D_px)`. 16×16 workgroup. |

## 4. Per-Pixel State

### 4.1 Storage textures

| Buffer | Format | Size (1920×1080) | Usage |
|--------|--------|------------------|-------|
| `wsr_numerator_tex` | `rgba32float` (RGB sum + unused) | 31.6 MB | `STORAGE_BINDING \| COPY_SRC` |
| `wsr_denominator_tex` | `r32float` | 7.9 MB | `STORAGE_BINDING \| COPY_SRC` |

Total: **39.5 MB at 1920×1080.** Acceptable — existing `instanceBuffer` is 480 MB at 10M.

### 4.2 Initialization

Per-frame clear via dedicated 16×16 compute pass (WebGPU 1.0 lacks a portable storage-texture clear command).

### 4.3 Addressing

Both bound as `var<storage>` 2D textures with `read_write` access in the accumulate pass. Screen-space addressing: `px = vec2u(floor((ndc.xy * 0.5 + 0.5) * viewport))`.

## 5. Two-Pass vs One-Pass

### 5.1 Options

**Option A — Fragment shader rasterization with additive blending:** Keep `vs_splat`/`fs_splat` style rasterization. Replace alpha-blend with `blendOperation: 'add'`, `blendFactor: 'one'` on two render targets (numerator MRT0 rgba32float, denominator MRT1 r32float). GPU's fixed-function ROP performs accumulation natively.

**Option B — Compute scatter pass:** One thread per splat, computes screen footprint from covariance, scatter-adds via atomic operations to storage textures.

### 5.2 Recommendation: Option A

**Choose fragment shader path.** Rationale:

1. **ROP hardware accelerates additive blending** at ~192 GB/s rgba32float on the 4090 — far exceeds compute-shader `atomicAdd` ALU serialization.
2. **B7.1 finding applies directly.** EXECUTION-LOG entry `2026-05-15 B7.1` established scatter at 10M is *DRAM-write-bound, not atomic-bound* on the 4090 (atomic-free B7.1 was +0.27 fps within noise). At per-pixel level, the bandwidth cost is the same in both options but Option A has zero atomic overhead and benefits from GPU quad-coverage culling (threads execute only for covered pixels).
3. **WebGPU 1.0 compatibility.** Storage-buffer atomics are extension-gated on some implementations; additive blending via render targets is unconditional.
4. **Minimal delta.** Existing rasterization already computes 2D Gaussian footprint and clip position.

Fragment path requires:
- Two `GPURenderPassColorAttachments` (numerator rgba32float, denominator r32float), `loadOp: 'clear'`, `storeOp: 'store'`.
- Blend state `{ operation: 'add', srcFactor: 'one', dstFactor: 'one' }` for both targets.
- Follow-up `cs_wsr_resolve` reads both, writes final RGBA.

### 5.3 Why not Option B

Compute scatter would eliminate quad-overhead at sub-pixel footprints, but our cull predicate already removes `r_max < 0.5 px`. Real-scene bicycle splats have median footprint well above 4 px — fragment rasterization is the right unit of work.

## 6. Training-Side Change

### 6.1 New learnable parameters

- `σ` — scene-level positive scalar, init `scene_diameter / 2.0`
- `v_i` — per-splat scalar bias, shape `[N]`, init 0

Without fine-tuning, set `σ = 2 × mean_scene_depth`, `v_i = 0`.

### 6.2 Where in the training loop

**gsplat** (`nerfstudio-project/gsplat`):
- `gsplat/cuda/csrc/backward.cu` and `gsplat/rendering.py`
- Replace `alpha_blend` with WSR accumulation. Backward pass propagates gradients through `w(d_i)`: `∂w/∂σ = d_i / σ²` (when clamp inactive), `∂w/∂v_i = -1` (when clamp inactive, else 0).

**Inria reference** (`graphdeco-inria/gaussian-splatting`):
- `gaussian_renderer/__init__.py:render()`
- CUDA rasterizer (`diff-gaussian-rasterization`) patched to expose `d_i` to fragment accumulation and switch from product transmittance to weighted sum.

**Loss function unchanged** — still `(1 - λ) · L1 + λ · SSIM` over rendered pixels; gradient flows through WSR forward.

**Parameter registration:**
- `σ`: `torch.nn.Parameter` scalar, init `scene_diameter / 2.0`, same Adam lr schedule as opacity.
- `v_i`: attribute column alongside `opacity`/`scales`/`quats`, added to `optimizer.param_groups`.

No new training infrastructure. Only the mathematical form of forward/backward.

## 7. Quality Fallback Path

### 7.1 Known failure mode

Dark foreground objects against bright backgrounds. Sorted alpha-blend correctly blocks background via cumulative transmittance `Π(1 − α_j)`. WSR with finite `w_B` will bleed some background luminance through dark foreground — visible as halo/glow. Risk highest on outdoor (bicycle, garden, bright sky).

### 7.2 Options considered

**(a) Tile-density hybrid:** WSR for tiles with > N_tile splats (sort cost dominates), sorted alpha-blend for sparse tiles (sort negligible, artifacts more visible).

**(b) Duplex-GS proxies (arXiv:2508.03180):** Ellipsoidal proxies sort at cell level, WSR within cells. Better occlusion fidelity but requires retraining.

### 7.3 Recommendation: option (a), deferred to PR4

PR1–PR3 ship WSR globally without hybrid. Measure PSNR / LPIPS vs sorted alpha-blend on bonsai + bicycle. If PSNR drops > 1 dB at 1.16M bonsai or > 2 dB at 3.62M bicycle, activate hybrid in PR4.

Hybrid preferred over Duplex-GS proxies because:
1. Reuses existing `cs_project_gather` + `instanceBuffer` path for sparse tiles.
2. Duplex-GS requires retraining; hybrid works on pretrained scenes.
3. Hybrid's only new infra is a tile-count buffer (byproduct of existing tile binning).

Start `N_tile = 256` splats/tile, widen until hybrid's fps gain disappears.

## 8. Validation Plan

### 8.1 Scenes

| Scene | Splats | Source | Purpose |
|-------|--------|--------|---------|
| Bonsai | 1.16M | Inria Mip-NeRF360 | Indoor baseline |
| Bicycle | 3.62M | Inria Mip-NeRF360 | Outdoor fps target |
| Synthetic 1M | 1M | `buildSyntheticScene(1e6)` | Regression gate |
| Synthetic 10M | 10M | `buildSyntheticScene(1e7)` | Sort-elim headroom |

### 8.2 Metrics

- **fps:** `runRealSceneBench`, 30 iterations × 3 viewpoints (front/orbit45/orbit90), median. Timestamp-query via `encodeTimed` for per-stage breakdown.
- **PSNR:** WSR-rendered vs sorted-alpha-blend reference at identical poses, Y-channel.
- **LPIPS:** AlexNet backbone, vgg variant, via readback path.

All comparisons relative to sorted alpha-blend on same hardware — isolates rendering approximation cost from compression/training artifacts.

### 8.3 Success criteria

| Criterion | Threshold |
|-----------|-----------|
| Bicycle fps (3.62M, no fine-tune) | ≥ 40 fps RTX 4090 Laptop (2× current 19.20) |
| Bicycle fps (3.62M, fine-tuned) | ≥ 50 fps |
| Synthetic 10M fps | ≥ 50 fps (vs ~6.5 fps current) |
| PSNR Δ vs sorted (bonsai pretrained) | ≤ 2 dB |
| PSNR Δ vs sorted (bonsai fine-tuned) | ≤ 0.5 dB |
| LPIPS Δ (bicycle 3.62M) | ≤ 0.05 |

Use `real-scene.bench.ts` (`packages/viewer/bench/real-scene.bench.ts`) with the WSR pipeline substituted for `ComputeDecodePipeline.encode`. Stage timings should show `sortFull ≈ 0` and new `wsrAccumulate` window replacing it.

## 9. Build Sequence

### PR1 — WSR fragment-shader path (feature-flagged, no training) — 3–4 days

New files: `cs_wsr_clear.wgsl`, `cs_wsr_resolve.wgsl`. Modify rasterization shaders to branch on `use_wsr: u32` uniform — additive write to two render targets when WSR, alpha-blend when off. Modify `ComputeDecodePipeline.encode` to skip keygen/sort/gather when `useWSR: true`. No training parameters — heuristic `σ = scene_depth_range`, `v_i = 0`.

**Gate:** `pnpm test` passes; `pnpm build` clean; visual inspection on bonsai shows no NaN/all-black.

### PR2 — Bench validation — 1–2 days

Add `runWSRBench` to `real-scene.bench.ts`. Report `sortFull` (~0), `wsrAccumulate`, `wsrResolve`. Add `computePSNR(ref, test)` utility. Run via existing Tailscale 4090 harness. Record in EXECUTION-LOG B8.2.

**Gate:** Bicycle fps ≥ 40 AND bonsai PSNR Δ ≤ 2 dB, OR documented tradeoff to proceed to PR3.

### PR3 — Training-side σ + v_i — 3–5 days

Extend `manifest.ts` with `wsrSigma: number` and `wsrPerSplatBiases: { uri, byteLength }`. New `wsr_params.ts` loader. `cs_project_wsr` reads σ from uniform and v_i[] from storage. Document the gsplat patch separately in `docs/perf/wsr-training-patch.md`. This PR does NOT run training — that's tracked separately as A10.

**Gate:** Round-trip render with hand-crafted σ, v_i[] (Python-scripted, not trained); PSNR Δ vs PR2 measured.

### PR4 — Tile-density hybrid fallback — 3–4 days

New `cs_tile_density.wgsl` — counts splats per tile from 2D covariance footprint, writes `tile_mode[]` (0=sorted, 1=WSR). Modify rasterization to branch per-tile. Initial `N_tile = 64`.

**Gate:** Bicycle PSNR Δ ≤ 0.5 dB AND fps ≥ 45. Raise N_tile if tile-density kernel overhead drops fps below 40.

### PR5 — Make WSR default — 1 day

Flip default `useWSR: true`. Keep sorted path behind `useLegacySort?: boolean` for regression. Update `webgpu-10m-profile.md` with post-WSR numbers.

**Gate:** All tests pass; bicycle ≥ 50 fps; no visual regressions.

**Time budget:** 11–16 engineer-days (2–3 calendar weeks).

## 10. Comparison with StochasticSplats and Duplex-GS

### StochasticSplats (arXiv:2503.24366, ICCV 2025)

Monte Carlo estimator with per-fragment random sampling. Claims >4× fps at "reasonable quality." Per-fragment sampling needs PRNG (ALU cost) or noise texture (bandwidth). At sample counts giving ≤ 1 dB PSNR Δ, real speedup is closer to 1.5–2× on dense scenes; 4× claim is at lower SPP with visible noise.

WSR is deterministic, zero sampling noise, predictable cost. DRA Round 2 buildability 5/5 ("pure fragment-shader change") vs StochasticSplats' implicit sampling framework.

### Duplex-GS (arXiv:2508.03180, Aug 2025)

Ellipsoidal proxies sort at cell level (~10–100× fewer than splats), WSR within cells. Better outdoor occlusion fidelity but requires retraining all scenes with cell-proxy extension. WSR is pure rendering change on pretrained 3DGS. Buildability gap is the deciding factor: 2 weeks vs 4–6 weeks.

If WSR fails the bicycle PSNR gate, Duplex-GS (or PR4 tile-density hybrid) becomes the fallback.

## 11. Risks and Unknowns

**Risk 1: Bicycle PSNR Δ > 2 dB without fine-tuning.** Halo on bright sky from pretrained model + no v_i optimization. **Kill trigger:** Bicycle PSNR Δ ≥ 2 dB AND tile-density hybrid (N_tile = 64) fails to reduce below 2 dB. Mitigation: restrict WSR to indoor scenes; outdoors stays sorted.

**Risk 2: ROP bandwidth limits gain.** rgba32float + r32float = 20 B/pixel. At 3.62M splats × ~16 px² median footprint × 20 B = ~1.16 GB/frame write. At 192 GB/s ROP that's ~6 ms — still net +24 ms vs 30 ms sort. **Kill trigger:** `wsrAccumulate` > 25 ms on bicycle — net gain < 20%, retreat to tile-density hybrid.

**Risk 3: `w_B` heuristic produces inconsistent backgrounds.** Pretrained models lack scene-specific `σ`/`v_i`; heuristic `w_B` may bleed bright background through dark foreground, or render uncovered pixels black. **Kill trigger:** > 10% of bonsai validation pixels show visible black holes or background bleed — adaptive per-frame `w_B` calibration required before PR5.

## References

- Hou et al., "Sort-free Gaussian Splatting via Weighted Sum Rendering," arXiv:2410.18931 (ICLR 2025). https://arxiv.org/abs/2410.18931
- Liu et al., "Duplex-GS: Proxy-Guided Weighted Blending for Real-Time Order-Independent Gaussian Splatting," arXiv:2508.03180 (Aug 2025). https://arxiv.org/abs/2508.03180
- Kheradmand et al., "StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting," ICCV 2025, arXiv:2503.24366. https://arxiv.org/abs/2503.24366
- Measured baseline: `docs/perf/webgpu-10m-profile.md`
- Real-scene bench: `packages/viewer/bench/real-scene.bench.ts`
- B7.1 atomic-free finding: `catetus-private/research/EXECUTION-LOG.md` entry `2026-05-15 B7.1`
- Round 2 DRA synthesis: `catetus-private/research/EXECUTION-LOG.md` entry `2026-05-15 round2-DRA-synthesis`
