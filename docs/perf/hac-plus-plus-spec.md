# HAC++ Implementation Spec for Catetus

**Track**: A3.2 (post-A3.1 Scaffold-GS checkpoint)
**Status**: Spec authored 2026-05-15. Implementation pending A3.1 BUILT.
**Reference**: Chen et al., "HAC++: Towards 100X Compression of 3D Gaussian Splatting," IEEE TPAMI 47(11):10210–10226, 2025. arXiv:2501.12255. Code: github.com/YihangChen-ee/HAC-plus.
**Predicted gain over our v0.1 hash-grid hyperprior** (`repack_v2_neural.py`): +2-4× at iso-PSNR. End-state ≈100× over vanilla 3DGS.

---

## 0. TL;DR

HAC++ replaces our current "position → hash-grid → MLP → (μ, σ)" entropy model with a four-piece system:

1. **Inter-anchor context** — same hash-grid hyperprior we already have, but applied to *anchor features* (Scaffold-GS style) instead of raw splat attributes.
2. **Intra-anchor context** — channel-wise autoregressive: split each anchor's feature vector into chunks, predict chunk *k* conditioned on the hyperprior **and** decoded chunks 0..k-1.
3. **Adaptive Quantization Module (AQM)** — per-attribute, per-anchor learned quantization step (a continuous scale Q that's also entropy-coded).
4. **Differentiable mask** — Gumbel-style binary masks on (a) offsets within each anchor and (b) implicitly on anchors themselves, trained jointly with an RD-loss `L = D + λ R`.

We already have piece #1's primitives. Pieces #2-#4 are new code but each is a self-contained module that plugs into the existing entropy pipeline.

---

## 1. Algorithm Summary

### 1.1 Setup (Scaffold-GS recap)
A3.1 produces a Scaffold-GS checkpoint with:
- `anchor_xyz: [N_a, 3]` — N_a anchor positions on a coarse voxel grid (~5-10% of full 3DGS count).
- `anchor_feat: [N_a, F]` — per-anchor feature vector, F=32 by default.
- `anchor_offsets: [N_a, K, 3]` — K=10 child-Gaussian offsets per anchor.
- `anchor_scaling: [N_a, 6]` — 3 anchor-extent scales + 3 offset-extent scales.
- Three small MLPs (`opacity_mlp`, `cov_mlp`, `color_mlp`) that decode `(feat, viewdir) → per-child opacity/covariance/SH`.

The bitstream is dominated by `anchor_feat` (~50% of the size), then `anchor_offsets` + `anchor_scaling`, then MLP weights (negligible). HAC++ compresses each of these four anchor-side tensors. The MLPs are sent uncompressed.

### 1.2 Inter-anchor context (hash-grid hyperprior)
For each anchor position `x_i ∈ ℝ³`, query a multi-resolution hash grid `g(x_i) ∈ ℝ^{L·F_g}` (L=12 levels, F_g=2 features/level in the paper; we currently use L=8, F_g=4). Feed `g(x_i)` through a 2-layer MLP → `(μ_inter, σ_inter, Q_inter)` predictions for each anchor attribute.

This is **mathematically identical** to our `repack_v2_neural.py` v0.1, with two differences:
- HAC++ also predicts a per-anchor, per-attribute quantization step `Q` (AQM), not just `(μ, σ)`.
- HAC++ operates on `anchor_feat` (compact, F=32) instead of raw 56-D splat attributes.

### 1.3 Intra-anchor context (channel-wise AR)
Split the F-dim feature vector into C chunks of size F/C (paper uses C=4, so 8 channels per chunk). For chunk index *k*:

```
input_k  = concat( hyperprior_feat, decoded_chunk_0, …, decoded_chunk_{k-1} )
(μ_k, σ_k, Q_k) = MLP_intra_k(input_k)        # one MLP per chunk
ŷ_k             = round(y_k / Q_k) * Q_k       # AQM-quantized
bits_k          = -log2 P_{N(μ_k, σ_k)}(ŷ_k)   # entropy under Gaussian
```

Chunk-0 has no decoded predecessors so its MLP only sees the hyperprior. This is the same channel-wise AR pattern Minnen & Singh used for learned image compression (NeurIPS 2020), transplanted to splat anchors.

**Why it works**: features within a single anchor share semantics (color, geometry hints) and are highly correlated. The hyperprior captures spatial correlation; the channel-wise AR captures *within-anchor* correlation that the hyperprior alone misses. Paper reports +0.7 dB at iso-bitrate from this piece alone.

### 1.4 Adaptive Quantization Module (AQM)
Classic neural compression uses a fixed quantization step Q=1 plus uniform noise during training (Ballé et al. straight-through trick). HAC++ predicts a continuous step `Q_i,j ∈ ℝ_+` per anchor *i*, per attribute *j*:

```
Q_i,j = softplus( Q_base_j + ΔQ_{i,j} )       # ΔQ output by the MLP
ŷ_i,j = round( y_i,j / Q_i,j ) * Q_i,j        # forward (STE on backward)
```

The `Q_base_j` is a learnable scalar (one per attribute group: feat, offsets, scaling). `ΔQ` is a per-anchor refinement predicted from the same MLP that outputs `(μ, σ)`. Q itself is *not* entropy-coded directly — it's reconstructed from `(g(x_i), decoded predecessors)` deterministically, so encoder/decoder agree on Q for free.

**At training time**: replace `round(·)` with `+ U(-Q/2, +Q/2)` (additive uniform noise) so gradients flow. **At inference time**: real rounding + range coding.

### 1.5 Differentiable masks
HAC++ learns two binary masks:

1. **Offset mask** `m_off ∈ {0,1}^{N_a × K}` — drops individual child Gaussians within each anchor.
2. **Anchor mask** `m_anc ∈ {0,1}^{N_a}` — *implicitly derived*: an anchor is killed when **all** its K offsets are masked out (`m_anc_i = OR_k m_off_{i,k}`). This means we don't learn a separate anchor mask; pruning anchors falls out of the offset training.

Mask values are learned via Gumbel-sigmoid straight-through:

```
mask_logit = MLP_mask(anchor_feat)            # [N_a, K]
mask_soft  = sigmoid((mask_logit + gumbel_noise) / τ)
mask_hard  = (mask_soft > 0.5).float()         # STE: use mask_hard fwd, mask_soft bwd
```

τ anneals from 1.0 → 0.1 over training. The mask enters the rate term as `R_total = Σ_i m_anc_i · R_anchor_i` so dropping an anchor zeros its bit cost.

### 1.6 RD Loss
```
L = D(I_render, I_gt) + λ · R(bitstream)
  = (1 - SSIM) + L1   +   λ · Σ_attr Σ_anchor m_i · bits_{i,attr}
```
- D: standard Scaffold-GS distortion (L1 + (1 - DSSIM) with weights 0.8/0.2).
- R: sum of `-log2 P_Gaussian(ŷ | μ, σ)` across all entropy-coded attributes, gated by the offset mask.
- λ values from paper: `{1e-3, 4e-3, 1e-2, 2e-2, 4e-2}` sweep gives the RD curve. Pick λ=4e-3 as our default "balanced" point.

Training: 30k iters total. **0-3k**: Scaffold-GS only (warmup, no entropy model). **3k-10k**: transition — Scaffold-GS densification keeps running, entropy MLPs train with detached `anchor_feat` (no grad through to anchors yet). **10k-30k**: full end-to-end RD loss; densification frozen. Per-iter, **5%** of anchors are randomly sampled for entropy-loss computation (memory cap).

---

## 2. Integration Plan vs Existing Code

### 2.1 What we already have

| Existing primitive | File | Reusable for HAC++? |
|---|---|---|
| `HashGrid` (Instant-NGP style, pure torch) | `apps/diff-repack/posthac_codec.py:52-99` & `posthac_explore.py:43-117` | **Yes, directly.** Becomes the inter-anchor hyperprior backbone. |
| `HyperpriorMLP(n_in, n_attrs)` outputs (μ, log σ) | `posthac_codec.py:102-114` | **Yes, with signature extension.** Add a third head for log-Q (AQM). |
| `train_hyperprior` NLL trainer | `posthac_codec.py:195-227` | **Replace.** New trainer needs the full RD loss + differentiable rendering hook into Scaffold-GS. |
| `encode_residuals` / `decode_residuals` via constriction `QuantizedGaussian` | `posthac_codec.py:287-319` | **Yes, mostly.** Each entropy-coded chunk uses the same loop. AQM means we pass per-symbol Q into the quantizer step instead of step=1. |
| `write_posthac_container` / `read_posthac_container` binary I/O | `posthac_codec.py:351-429` | **Extend.** New container needs: per-chunk MLP weights, AQM `Q_base` table, offset mask bitmap, and Scaffold-GS MLPs (opacity/cov/color). |
| k-NN anchor prediction (IDW) | `splat_delta_explore.py` & `compose_explore.py` | **Indirect.** HAC++ does *not* use k-NN — it uses the hash grid for spatial context. Our k-NN code remains useful as a **fallback inter-anchor entropy** baseline if the hash-grid hyperprior under-performs on indoor scenes. Keep alive for ablations. |
| f32 → uniform 8-bit quantization | `posthac_codec.py:169-180` | **Replace per-attribute.** AQM is the new quantizer. Keep the per-column min/max only as a fallback. |

### 2.2 What stays vs what changes

**Stays unchanged**:
- `HashGrid` class signature and trilinear interpolation logic. Increase `n_levels` from 8 → 12 to match paper defaults; keep `log2_hashmap_size=15` (1.05 MB per table — fits our payload budget) or grow to 17 if quality plateaus.
- `constriction` range coder framing.
- The "snap (μ, σ) to a discrete grid for encoder/decoder determinism" trick at `posthac_codec.py:266-283` — keep this verbatim; HAC++'s extra `Q` head needs the same treatment.
- f32 PLY I/O (`load_ply`, `write_ply`).

**Changes**:
- The input to the hyperprior is `anchor_xyz` (from Scaffold-GS), not all-splat positions. ~5-10× fewer entries.
- The thing being entropy-coded is `[anchor_feat, anchor_offsets, anchor_scaling]`, not raw 56-D splat attributes.
- A new file `apps/diff-repack/hac_plus_plus.py` houses the channel-wise AR MLPs, AQM, and mask logic.
- A new trainer in `apps/scaffold-gs/train_hac.py` wraps Scaffold-GS's existing trainer with the entropy loss and the 0-3k/3k-10k/10k-30k schedule.
- New bitstream container `apps/diff-repack/hacpp_container.py` (extends posthac container).

### 2.3 Scaffold-GS checkpoint coupling

A3.1 lands `bonsai_real_scaffold_30k.pth` containing:
```python
{
  "anchor_xyz":   FloatTensor[N_a, 3],
  "anchor_feat":  FloatTensor[N_a, 32],
  "anchor_offsets": FloatTensor[N_a, 10, 3],
  "anchor_scaling": FloatTensor[N_a, 6],
  "anchor_opacity_mlp": state_dict,
  "anchor_cov_mlp":     state_dict,
  "anchor_color_mlp":   state_dict,
  "voxel_size": float,
  "feature_dim": 32,
  "n_offsets": 10,
}
```

HAC++ loads this checkpoint, freezes the geometry/MLPs initially, and bolts the entropy model on top.

---

## 3. Phase Breakdown

Effort: **3-4 weeks** end-to-end. Predictions are versus the previous phase's output; absolute size is for bonsai_real (small indoor — A3.1's working scene).

### Phase A — Anchor-level entropy model (replace v0.1 hyperprior)
**Goal**: Compress only `anchor_feat` with hash-grid hyperprior, no AR, no AQM, no masks. Establishes the harness.

- **Files to create**:
  - `apps/diff-repack/hac_plus_plus.py` — new file. Classes: `AnchorHyperprior(nn.Module)` (wraps `HashGrid` from `posthac_codec.py` + a `HyperpriorMLP` with extra `n_attrs` for the new F=32 feature dim).
  - `apps/scaffold-gs/train_hac_phase_a.py` — loads A3.1's checkpoint, freezes everything except hyperprior + a copy of `anchor_feat`, runs ~5k iters of NLL on `anchor_feat` while keeping the renderer frozen.
- **Files to modify**: None (Phase A is additive).
- **Predicted size**: bonsai_real Scaffold-GS native ≈ 11 MB → 4-5 MB (≈2.5×, single-stream Gaussian entropy code on a 32-D vector is the well-known starting point).
- **Predicted ΔPSNR vs Scaffold-GS native**: 0.0 dB ± 0.1 (lossless on `anchor_feat` quantization at 8-bit step=1; render loss only from the rounded `anchor_feat`).
- **Hardware**: Modal A100, ~10 min. CPU works too but slower.
- **Wall-time estimate**: 3 days code + 1 day debug determinism vs constriction.
- **Buildability cross-check**: pattern-clone `train_hyperprior()` from `posthac_codec.py:195-227`, change `codes_t` to `anchor_feat`, change `pos_t` to `anchor_xyz_norm` (normalize anchors to [0,1]³ via `normalize_positions` from `posthac_codec.py:183`). The whole class is ~80 lines.

### Phase B — Intra-anchor channel-wise AR
**Goal**: Add channel-wise AR within `anchor_feat`. F=32 split into C=4 chunks of 8 channels each.

- **Files to create**:
  - Add `class IntraAnchorAR(nn.Module)` to `hac_plus_plus.py`. Holds `nn.ModuleList([ChunkMLP(input_dim_k) for k in range(C)])`. Each `ChunkMLP` takes `(hyperprior_feats[N, L·F_g], decoded_chunks_0..k-1[N, k·8])` → `(μ, log σ)` for the next 8 channels. Hidden size 64.
  - `apps/diff-repack/hacpp_codec.py` — encode/decode driver that loops over chunks, range-codes chunk *k*, decodes it, feeds into chunk *k+1*'s context. Reuse `_predict_all` determinism logic verbatim from `posthac_codec.py:234-284`.
- **Files to modify**: `train_hac_phase_a.py` → `train_hac_phase_b.py` adding the per-chunk MLPs to the optimizer, NLL summed across chunks. Encoder/decoder needs the chunk loop.
- **Predicted size vs Phase A**: 4-5 MB → 3.0-3.5 MB (≈1.4× from intra-anchor correlation). Paper reports +0.7 dB at iso-bitrate so we can also frame this as +0.7 dB at fixed 4.0 MB.
- **Predicted ΔPSNR**: still 0.0 (lossless feature quantization at step=1). All gain is on the rate side.
- **Hardware**: Modal A100, ~15 min training.
- **Wall-time estimate**: 4 days. The risk is encoder/decoder determinism across the AR loop — every chunk's quantized values must be bit-identical at decode time. Reuse the "round (μ, σ) to a 1/1024 grid before f32→f64 conversion" technique from `posthac_codec.py:266-272`.
- **Buildability cross-check**: `IntraAnchorAR` is conceptually a stack of MLPs with growing inputs. Loop in Python is fine — N_a is ~100k, C=4 chunks, so 400k forward passes, ~1 sec on GPU. No need for fused cuda.

### Phase C — AQM + differentiable mask
**Goal**: Replace the uniform step=1 quantizer with the AQM per-attribute step Q, and add the offset mask.

- **Files to create**:
  - Add `class AdaptiveQuantizer(nn.Module)` to `hac_plus_plus.py`. Two paths: training (additive uniform noise) and inference (round). Forward: `ŷ = round_or_noise(y / Q) * Q`. Backward: STE.
  - Add `class OffsetMask(nn.Module)` — one MLP `R^{F+3} → R^K` per anchor that emits K mask logits, Gumbel-sigmoid + STE.
  - Add a third head to each `ChunkMLP` outputting `log Q_k` (softplus → Q_k). Update `IntraAnchorAR.forward` to return `(μ, σ, Q)` per chunk.
  - Container format update: store K-bit-per-anchor offset mask as a packed bitmap. `write_hacpp_container()` in `hacpp_codec.py`, extending the format from `posthac_codec.py:347-388`.
- **Files to modify**: `hac_plus_plus.py` (heads, quantizer), `hacpp_codec.py` (Q in the QuantizedGaussian width call), `train_hac_phase_b.py` → `train_hac_phase_c.py`.
- **Predicted size vs Phase B**: 3.0-3.5 MB → **1.5-2.0 MB** (≈1.7×; the mask alone prunes ~30% of offsets on indoor scenes per paper, and AQM saves another ~15-20% on bits per kept attribute).
- **Predicted ΔPSNR vs Phase B**: -0.1 to -0.3 dB (the mask kills marginal Gaussians). Net vs Scaffold-GS native is still ≥ -0.3 dB.
- **Hardware**: Modal A100, ~20 min training.
- **Wall-time estimate**: 5 days. Highest-risk piece: getting the mask Gumbel temperature schedule and τ annealing right. If the mask collapses to all-zero or all-one, training kills the model.
- **Buildability cross-check**: AQM is 30 lines. The offset mask MLP is 20 lines. The annoying part is the encoder needing to skip masked positions in the range-code loop — pattern this after `encode_residuals` in `posthac_codec.py:287-304` but loop over only `m_off[i, k] == 1` entries per anchor.

### Phase D — End-to-end RD optimization
**Goal**: Add Scaffold-GS rendering loss back into the loop. Train the full `L = D(render) + λ R` objective on the 0/3k/10k/30k schedule. This is where the final compression ratio lands.

- **Files to create**: `train_hac_phase_d.py` (final trainer). Reuses Scaffold-GS's `train.py` from upstream; we patch in (a) the entropy MLP optimizer, (b) the `L = D + λR` loss, (c) the 3-stage training schedule.
- **Files to modify**: The Scaffold-GS upstream `train.py` needs a hook to call our `entropy_forward()` on every iteration that's past the 3k warmup. Either fork the file or monkey-patch.
- **Predicted size vs Phase C**: 1.5-2.0 MB → **1.0-1.3 MB** at λ=4e-3 (paper-default "balanced" RD point). End-to-end RD lets the renderer trade some fidelity for bits adaptively. 8.5× vs Phase A, ~10× vs Scaffold-GS native, **~100× vs vanilla 3DGS** on bonsai_real (assuming vanilla = 100 MB, which matches the standard bonsai_real native size).
- **Predicted ΔPSNR vs Scaffold-GS native at λ=4e-3**: -0.0 to +0.2 dB (paper actually reports HAC++ *improves* PSNR on most scenes because the mask drops bad Gaussians that hurt the render). On bonsai_real specifically the indoor scene quality is dominated by Scaffold-GS's anchor geometry, which we're not retraining — so realistic estimate is -0.2 to 0.0 dB.
- **Hardware**: Modal A100, ~2-3 hours. This is the long one. **A4.1 must not be sharing Modal at the same time** (per the discipline notes).
- **Wall-time estimate**: 5-7 days. Bulk of the time is hyperparameter sweep: λ ∈ {1e-3, 4e-3, 1e-2, 2e-2, 4e-2} to produce the RD curve, plus mask temperature schedule, plus learning-rate schedule for the entropy MLPs (paper uses 2e-3 for entropy, 1.6e-4 for Scaffold geometry).
- **Buildability cross-check**: this is a wrapping/glue phase. The novel piece is the 3-stage training schedule: write a `phase_for(iter)` helper that returns one of `{WARMUP, TRANSITION, RD_FULL}` and gates the loss accordingly. ~150 lines on top of Phase C.

---

## 4. File Layout (final state)

```
apps/diff-repack/
  hac_plus_plus.py          # new — AnchorHyperprior, IntraAnchorAR, AQM, OffsetMask
  hacpp_codec.py            # new — encode/decode driver + container I/O
  posthac_codec.py          # existing — HashGrid + HyperpriorMLP reused by hac_plus_plus.py
  posthac_explore.py        # existing — kept as v0.1 baseline reference
  splat_delta_explore.py    # existing — kept; potential fallback inter-anchor model
  compose_explore.py        # existing — kept; entropy ablation harness
apps/scaffold-gs/
  train_hac_phase_a.py      # new — Phase A trainer (NLL only)
  train_hac_phase_b.py      # new — Phase A + intra-anchor AR
  train_hac_phase_c.py      # new — Phase B + AQM + mask
  train_hac_phase_d.py      # new — full RD with rendering loss
docs/perf/
  hac-plus-plus-spec.md     # this file
```

The four trainer files are roughly v1, v2, v3, v4 of the same script; we keep all four committed so an ablation re-run is one CLI flag away.

---

## 5. Open Questions (need experiment to resolve)

1. **Hash-grid size on indoor scenes**: HAC++ paper tunes `log2_hashmap_size` ∈ {15, 17, 19}; default is 17 (524k entries × 12 levels × 2 features = ~50 MB raw, → ~6 MB after their hash-grid binarization). For our bonsai_real (one scene), 15 may suffice. Open Q: does our current 8-level / log2=15 grid from `posthac_codec.py` undertrain on Scaffold-GS anchors? Run Phase A at both 15 and 17 and pick.

2. **Quantization of the hash grid itself**: HAC++ further binarizes hash-grid entries via straight-through. Our v0.1 codec leaves the grid as f32. At 100× target this becomes significant: 12 levels × 32k × 2 × 4 B = 3 MB uncompressed grid is now bigger than the entropy-coded anchors. Phase A baseline must include hash-grid f16 storage at minimum; Phase D needs straight-through 1-bit grid quantization to actually hit 100×.

3. **Channel-chunk size C**: paper uses C=4 (chunks of F/C = 8). Is C=8 (chunks of 4) better for our F=32? More chunks = more AR signal = better rate, but C MLPs forward/back at training. Phase B should ablate C ∈ {2, 4, 8}.

4. **Mask schedule sensitivity**: τ-annealing schedule is reported by paper as "τ: 1.0 → 0.1 linearly over 10k-30k". Indoor scenes have less anchor redundancy than outdoor (bonsai_real ≈ 100k anchors vs garden ≈ 400k). The mask may collapse on indoor — keep an eye on `m_anc.mean()` per iteration; if it falls below 0.5 we're killing too many anchors.

5. **5%-sample-per-iter for entropy loss**: paper's memory-cap trick. On bonsai_real with 100k anchors, 5% = 5000 anchors per iteration. We can probably afford 100% on A100 80GB. Determine whether full-sampling vs 5% makes RD curve different.

6. **Determinism across PyTorch versions**: our v0.1 already snaps MLP outputs to a 1/1024 grid for cross-implementation bit-exactness. AQM adds a *learned* per-attribute Q step into this critical path. Confirm Q itself is also snapped (paper doesn't explicitly say, but it's required for the range coder to not desync between encoder/decoder).

7. **Compatibility with our existing PostHAC container**: we have a deployed Rust decoder for the v0.1 container (`catetus-posthac` crate). HAC++'s bitstream is a *strict superset* of v0.1 if we version-bump. Or do we ship HAC++ as a parallel format with a different magic? Recommend the latter — HAC++ is sufficiently more complex that a forked decoder is cleaner than versioning. Magic `HCPP` (`0x50504348`).

---

## 6. Acceptance Criteria

A3.2 ships when:

1. `apps/scaffold-gs/train_hac_phase_d.py --scene bonsai_real --lambda 4e-3` produces a bitstream `bonsai_real.hacpp` of size ≤ 1.5 MB.
2. The bitstream decodes losslessly via `python -m apps.diff_repack.hacpp_codec decode` to a PLY that re-renders within -0.3 dB PSNR of the Scaffold-GS native checkpoint at the validation cameras.
3. RD curve at λ ∈ {1e-3, 4e-3, 1e-2, 2e-2, 4e-2} plotted in `docs/perf/hac-plus-plus-rd-curve.md`.
4. Determinism check: encoder/decoder round-trip on three machines (Modal A100, M3 laptop, 4090 box) produces bit-identical PLY output.

---

## 7. Dependencies

- `torch ≥ 2.1` (already in posthac harness).
- `constriction ≥ 0.4` (already in posthac harness — provides `QuantizedGaussian` range coder).
- `plyfile` (already in posthac harness).
- Scaffold-GS upstream code: cloned during A3.1; checkpoint already loaded with `torch.load`.
- **No new heavy deps**: deliberately avoiding `tinycudann` — pure-torch HashGrid from `posthac_codec.py` is enough at our scale (N_a ≈ 100k, batch sizes ≤ 16k).

---

## 8. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Encoder/decoder desync in AR loop (Phase B) | Medium | Reuse the 1/1024-grid-snap trick from v0.1; add a roundtrip CI test that encodes then decodes 1000 anchors and asserts bit-equality. |
| Mask collapses to all-zero (Phase C) | Medium | Warmup mask with a regularizer `α · (m.mean() - 0.7)^2` for the first 5k post-transition iters to anchor it near 0.7. |
| Phase D blows Modal hour budget | High | Run a 3k-iter "smoke" at λ=4e-3 first; only kick off the full λ-sweep after the smoke compresses cleanly. |
| AQM gradient instability (Q → ∞ or Q → 0) | Low | softplus the log-Q output and clamp Q ∈ [0.01, 100]; the paper does this too. |
| Indoor scenes have too few anchors for the hash-grid hyperprior to be useful (under-trains) | Low | Confirmed-mitigated by Phase A ablation between `log2_hashmap_size` ∈ {15, 17} — if 15 wins, we have headroom and indoor is fine. |

---

## 9. Sources

- arXiv:2501.12255 — HAC++: Towards 100X Compression of 3D Gaussian Splatting
- arXiv:2403.14530 — HAC (predecessor, ECCV 2024)
- github.com/YihangChen-ee/HAC-plus — reference PyTorch implementation
- Minnen & Singh 2020, "Channel-wise Autoregressive Entropy Models for Learned Image Compression" — the channel-wise AR pattern HAC++ borrows
- Internal: `apps/diff-repack/posthac_codec.py` (v0.1 hash-grid hyperprior, the literal starting point)
- Internal: `apps/diff-repack/splat_delta_explore.py` (k-NN anchor predictor; useful as ablation baseline, not on the main HAC++ path)
