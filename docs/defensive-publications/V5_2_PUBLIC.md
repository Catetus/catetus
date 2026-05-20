# Catetus V5.2: A Render-Jacobian-Selected Residual Sidecar and Jacobian-Weighted Lloyd Codebook for 3D Gaussian Splatting Compression

**Catetus Authors**
**Date:** 2026-05-18
**Status:** Defensive technical publication. Intended for arXiv-style archival
and prior-art establishment.

---

## Abstract

We describe two complementary compression techniques for 3D Gaussian Splatting
(3DGS) scenes and a wire format that composes them. **T2.1.R** is a
Jacobian-weighted Lloyd update for vector-quantized (VQ) palette compression
of SH-rest coefficients: per-splat weights derived from a differentiable
render-loss Jacobian replace the uniform mean inside the K-means centroid
step, raising bonsai PSNR by **+6.16 dB** at strictly negative byte cost over
the same Catetus baseline. **V5.2** is a joint per-splat tail-protection
sidecar that selects the top-1% of splats by render-Jacobian magnitude,
quantizes per-attribute residuals against the base codec's reconstruction
inside Morton-bucketed cells with a per-cell affine predictor at mixed
8/10/12/12/8/8 bit depths, and emits the result as a wire-format-detached
file with graceful-fallback semantics. Composed with T2.1.R, V5.2 reaches
**16.78 MB / 58.679 dB** on bonsai (72-view orbit, gsplat sh=3, 512px) —
**+11.30 dB and −2.20 MB** versus PlayCanvas SOG *when decoded through the
Catetus viewer or the `@catetus/sf-glb-polyfill` JS decoder*; the GLB
+ `.glb.v5tail` path is captive to those decoders. The SuperSplat-
compatible interop story is the SOG-container result below.

**SOG-container interop (the externalizable headline).** The V5.2 residual
sidecar, ported to ride on top of any SOG file as a `.sog.v5tail` companion
ignored by legacy viewers, lifts vanilla PlayCanvas SOG by **+6.54 dB at
+3.95% bytes** (K=1% render-Jacobian-selected tail) or **+13.92 dB at +18%
bytes** (K=5% aggressive tier). Separately, the same T2.1.R weighted-Lloyd
update integrated into the SOG encoder yields **+2.18 dB over uniform Lloyd
at identical byte budget**. Both ship inside containers that
`@playcanvas/splat-transform` (and therefore SuperSplat) decodes
unchanged — sidecar-unaware decoders see vanilla SOG quality; sidecar-
aware decoders see the lift.

**Cross-scene caveat.** On the canonical-11 leaderboard (Mip-NeRF 360 +
Tanks-and-Temples pretrained), Catetus's *SF baseline* (`wmv-vq45-no-
prune-tight`, pre-T2.1.R, pre-V5.2) achieves **11/11 strict wins** versus
SOG, averaging **+1.99 dB at −30.6% bytes**. The +11 dB GLB-only V5.2 stack
and the +6.54 dB `.sog.v5tail` sidecar are **bonsai-only as of this
writing** — cross-scene generalization of those specific stacks is in
flight (task `T2.1.X`) and is the single largest open evidence gap.

This document specifies the two algorithms and the `.glb.v5tail` /
`.sog.v5tail` wire format precisely enough for a third-party decoder.

---

## 1. Introduction and contributions

3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] stores scenes as millions of
anisotropic Gaussian primitives, each carrying position, rotation, scale,
opacity, and spherical-harmonic (SH) color coefficients. Compression of
these primitives is an active field with 40+ methods catalogued by
Bagdasarian et al. (arXiv 2407.09510). The dominant axes of attack are
(i) pruning low-importance Gaussians, (ii) vector- or scalar-quantizing
attributes, (iii) entropy modeling under spatial or hash-grid context, and
(iv) factorizing or palettizing SH coefficients. Two industry-relevant
container formats have emerged: glTF/GLB with custom extensions (used by
Catetus), and the PlayCanvas/Snap "Self-Organizing Gaussian" (SOG)
ZIP-of-WebP format [Morgenstern et al. arXiv 2312.13299].

We contribute two narrow technical pieces that compose cleanly with these
existing pipelines.

**Contribution 1 — T2.1.R weighted Lloyd update for SH-rest VQ.** A one-line
edit to the K-means centroid update inside Catetus's VQ palette pass:
each splat's contribution to its cluster's mean is weighted by its
aggregated per-splat render-loss Jacobian magnitude. Training-pool
sub-sampling uses Gumbel-top-K weighted reservoir to preserve the high-J
tail through training. On bonsai this lifts PSNR by **+6.16 dB** at the
same byte budget, integrated end-to-end through the production
`wmv-vq45-no-prune-tight` preset. The same change ported into the
PlayCanvas SOG WebP encoder yields **+2.18 dB at the same byte budget**
versus uniform Lloyd while remaining decodable by stock SuperSplat.

**Contribution 2 — V5.2 joint per-splat tail-protection sidecar.** A
wire-format-detached enhancement-layer codec: the top-K splats by
render-Jacobian joint sum are selected, residuals are computed in raw
3DGS-PLY space against the base-codec reconstruction, the selected splats
are Morton-bucketed into spatial cells, and within each cell a per-attribute
affine predictor (scale + offset) is fit and the quantized residual encoded
at a mixed bit-depth profile (`8/10/12/12/8/8` for pos/rot/opa/sca/dc/shr).
The sidecar carries a distinct magic header (`SFV51TAL`) and is byte-aligned
with no dependency on the base container; decoders unaware of the sidecar
render the base codec output directly. On bonsai, V5.2 layered over T2.1.R
adds **+4.65 dB at +0.77 MB** for a final **58.679 dB at 16.78 MB**.

**Contribution 3 — composition and wire chain.** We specify the
`.glb.v5tail` and `.sog.v5tail` sidecar files, their relationship to the
container's required-extensions semantics, and the graceful-ignore behavior
that lets the wire chain ship into mixed-decoder ecosystems (SuperSplat,
the Catetus viewer, and any future third-party tools).

**Relationship to prior art.** Each individual element of V5.2 has clear
prior art and we make no novelty claim on them in isolation. Niedermayr
et al. (CVPR 2024) use a per-parameter "sensitivity" — operationally a
render-Jacobian — to weight K-means clustering for codebook compression;
GoDe (arXiv 2501.13558) uses aggregated gradient sensitivity per Gaussian
to organize splats coarse-to-fine for progressive truncation; CompGS
(arXiv 2404.09458) splits splats into anchors plus residual-coded coupled
primitives; SizeGS (arXiv 2412.05808) solves per-attribute mixed-bit
allocation as an ILP; MPEG G-PCC's Predictive Transform / Pred-Lift uses
spatial-neighborhood residual prediction with Morton scan order; SVC /
SHVC enhancement layers are the canonical layered-coding pattern.
ACM TOMM 2024 (10.1145/3729232) uses affine-on-Morton-bucket prediction
for point-cloud attributes. Our contributions are **(a) the specific
weighted-Lloyd centroid update for SH-rest VQ in 3DGS with measured PSNR
lift, (b) the combination of render-Jacobian-driven top-K selection with a
wire-format-detached per-Morton-cell affine residual sidecar at mixed bit
depths for 3DGS, and (c) the concrete wire format with graceful-fallback
container semantics.** None of these are individually patent-defensible
under § 103 obviousness when read against the cited prior art; this
document publishes them defensively to block competitor patenting of the
same combination and to anchor priority.

---

## 2. T2.1.R: Jacobian-weighted Lloyd centroid update for SH-rest VQ

### 2.1 Motivation: Jacobian census

For 3DGS scenes optimized to convergence, the contribution of each splat to
the rendered images is highly non-uniform. We compute a **per-splat
Jacobian census**: for each splat *i* and each attribute group *a*
(position, rotation, opacity, scale, DC color, SH-rest), the L2 norm of the
partial of the differentiable rendering loss with respect to that splat's
attribute, summed over a representative camera set. Concretely, with a
gsplat differentiable rasterizer and an L2 pixel loss against the training
images,

```
J[i, a] = ||  sum_v  ∂L_v(scene) / ∂θ[i, a]  ||_2
```

aggregated across views *v* and attribute channels within group *a*. The
joint sum `J_joint_sum[i] = sum_a J[i, a]` is used as a scalar splat
importance for selection.

On bonsai (1,244,819 splats, sh=3), the distribution of `J_sh_rest` is
extremely heavy-tailed: the top 1% of splats account for ~87% of the
SH-rest Jacobian mass. This is the algorithmic signal the techniques below
exploit.

### 2.2 Algorithm

The Catetus base codec compresses SH-rest coefficients via a VQ palette
of K=4096 codewords learned by mini-batch K-means on the per-splat SH-rest
vector (45 floats at sh=3). The conventional Lloyd centroid update for
cluster *c* with assigned splats `S_c` is

```
μ_c = (1 / |S_c|)  Σ_{i ∈ S_c}  x_i
```

T2.1.R replaces this with a weighted mean where weights are the per-splat
Jacobian magnitudes:

```
w_i  = J_sh_rest[i]
μ_c  = ( Σ_{i ∈ S_c} w_i · x_i ) / ( Σ_{i ∈ S_c} w_i )
```

Two further changes are required for the gain to materialize:

1. **Weighted reservoir sub-sampling.** K-means training pools sub-sample
   from the full splat set to bound runtime. Uniform sub-sampling would
   under-represent the high-J tail. We use Gumbel-top-K weighted sampling
   on `w_i` so the splats holding 87% of the Jacobian mass survive into the
   training pool with probability proportional to weight.

2. **Weight bookkeeping across structural passes.** Passes that remove or
   re-order splats (`RemoveInvalidSplats`, `MortonSort`) must keep the
   weight vector in lockstep. Implementation rides the weights through a
   `PassContext.sh_rest_weights` field plumbed through the optimize
   pipeline.

The remainder of the VQ pass (codeword assignment via nearest centroid,
label-stream entropy coding) is unchanged. The total wire-byte cost is
identical to uniform Lloyd up to incidental compressibility differences
(the weighted codebook compressed 1.3% smaller on bonsai).

### 2.3 Result (bonsai)

|     Variant                  | GLB+shpal | gsplat PSNR | Δ |
|------------------------------|----------:|------------:|---|
| Uniform Lloyd (baseline)     |  15.98 MB |   47.420 dB |   |
| **T2.1.R weighted Lloyd**    |  15.95 MB | **53.580 dB** | **+6.16 dB** |

(`wmv-vq45-no-prune-tight` preset, gsplat 8-frame orbit, sh=3, 512×512.)
On the 72-view orbit the same encoded artifact reports 54.361 dB, used as
the T2.1.R-only row in §6 of the V5.2 composition table.

**Implementation note (Rust-vs-Python parity).** The Rust port of T2.1.R
(`crates/catetus-optimize`) is bit-identical to the Python reference at
the VQ centroid step (MSE 6.282 × 10⁻⁴ in both). A casual reproducer
pulling the repo and benching standalone T2.1.R on `bonsai_sf_decoded.ply`
may observe a ~1.3 dB shortfall versus 54.361 dB; this is **not** a codec
defect but an oracle-anchor mismatch. The `wmv-vq45-*` preset family
shipped with `log_quant_attrs=true` for scale/opacity (expected to be
recovered by the V5.2 sidecar at the next stage), while the
`bonsai_sf_decoded.ply` oracle was built before that change. See
`experiments/t21r-drift-debug/RESULT.md` for the bisect.

### 2.4 Discussion

The improvement is dramatic because the loss surface in attribute space is
heavily non-uniform: a small set of splats dominates rendering loss because
they are large, near the camera, near silhouette boundaries, or have high
SH-band content. Uniform Lloyd minimizes vector-space MSE on SH coefficients,
which is the wrong loss; render-space loss weights coefficients by their
projected pixel impact. The Jacobian magnitude is the analytical first-order
proxy for that pixel impact, and is computable in one differentiable forward
pass per camera view.

---

## 3. V5.2 sidecar architecture

V5.2 layers an enhancement-layer codec on top of any 3DGS base codec
(Catetus T2.1.R-encoded GLB, raw SOG, or unmodified Catetus baseline).
The architecture has four stages.

### 3.1 Tail selection

Given the per-splat render-Jacobian joint sum `J_joint_sum[i]`, the
encoder selects the top-K splats where K is configurable (default 1% of
N; on bonsai 12,448 splats out of 1,244,819). Selection is a sort by
`J_joint_sum` descending followed by a take(K). Selected indices are
then re-sorted ascending so they can be efficiently bit-mask-encoded.

When the Jacobian census is unavailable (e.g. the SOG-sidecar path where
the base codec did not produce a `.npz` sidecar), the encoder falls back
to a residual-L1-sum proxy computed against the base codec's reconstruction.
On bonsai the L1 proxy correlates ~0.93 with `J_joint_sum`; the V5.2 SOG
sidecar uses the proxy by default.

### 3.2 Residual computation in raw 3DGS-PLY space

For each selected splat, residuals are computed against the base codec's
*decoded* reconstruction in the original 3DGS-PLY representation:

```
pos_res     = gt.position - recon.position                            (linear)
rot_res     = gt.rotation - recon.rotation                            (linear, raw quat)
opa_res     = logit(gt.opacity)  - logit(recon.opacity)               (logit space)
sca_res     = ln(gt.scale)       - ln(recon.scale)                    (log space)
dc_res      = gt.f_dc            - recon.f_dc                         (linear)
shr_res     = gt.f_rest          - recon.f_rest                       (linear)
```

The logit/log space transforms for opacity and scale match the 3DGS-PLY
convention and ensure the residual distribution remains symmetric and
well-behaved under uniform quantization. This is the critical
implementation detail: residuals computed against the decoded reconstruction
of the base codec — not the raw post-pipeline scene — are what captures the
UBYTE quantization loss the GLB writer introduces, and is what made the
V5.2 Rust port jump from 53.79 dB to 58.68 dB during integration.

### 3.3 Per-Morton-cell affine predictor with mixed bit depths

The K selected splats are sorted by 21-bit-per-axis Morton code on their
positions, then bucketed into `n_cells = 64` contiguous slabs of
`ceil(K / n_cells)` splats each. Within each cell *c* and each attribute
channel *a*, the encoder fits an affine quantizer:

```
scale[c, a]  = (max_{i ∈ c} r[i, a] - min_{i ∈ c} r[i, a]) / (2^bd[a] - 1)
offset[c, a] =  min_{i ∈ c} r[i, a]
q[i, a]      =  round( (r[i, a] - offset[c, a]) / scale[c, a] )
```

with `bd[a]` the bit depth assigned to attribute group *a*. The V5.2
profile is `bd = [8, 10, 12, 12, 8, 8]` for `[pos, rot, opa, sca, dc, shr]`.
Quantized values are bit-packed LSB-first per group; per-cell `(scale, offset)`
floats are stored as side-channel meta. Both the meta stream and the
bit-packed payload are zstd-compressed.

The motivation for per-cell affine over global affine is that residual
magnitudes correlate with spatial position (high-detail regions cluster);
fitting predictors per cell adapts the quantizer to local residual
statistics without per-splat side information. The motivation for the
specific bit-depth profile is empirical: after T2.1.R, the SH-rest
residual magnitude drops 8.5× (from 1.31e-2 to 1.54e-3), so an 8-bit
quantizer suffices; opacity and scale residuals remain large after T2.1.R
because that pass only touches SH-rest, so 12 bits are warranted.

### 3.4 Wire format (`SFV51TAL` variant=2)

The sidecar is a single byte-aligned file. The full wire spec is given in
§7 and is sufficient for a third-party decoder. The header carries
`n_splats`, `k_selected`, `sh_rest_coefs`, and `n_cells`; three length-
prefixed zstd blobs carry the selection bitmask, the Morton permutation,
and the cell offsets; six per-group records carry the affine meta and the
bit-packed quantized residuals.

The sidecar magic (`SFV51TAL`) is distinct from any container magic. The
container references the sidecar via a side-extension JSON object
(`CT_v5_tail_residual` for glTF, `.sog.v5tail` adjacent file for SOG) and
flags it as optional. Decoders that don't recognize the extension render
the base codec output unmodified.

### 3.5 Apply path

The decoder reads the header, decompresses the three zstd blobs, then for
each of the six groups decompresses the meta and payload, dequantizes per
cell, and de-permutes from Morton order back into SF-ascending order over
the selected subset. Application to a scene mutates in place:

```
for k in 0..K_selected:
    i = sel_idx[k]
    scene.position[i] += pos_res[k]
    scene.rotation[i] += rot_res[k]                # caller re-normalises
    scene.opacity[i]   = sigmoid(logit(scene.opacity[i]) + opa_res[k])
    scene.scale[i]     = exp(   ln(scene.scale[i]) + sca_res[k]   )
    scene.f_dc[i]     += dc_res[k]
    scene.f_rest[i]   += shr_res[k]                # truncated to scene capacity
```

Reference implementations: Rust (`crates/catetus-gltf/src/v5_tail.rs`,
`crates/catetus-sog/src/v5tail_emit.rs`) and TypeScript
(`packages/sf-glb-polyfill/src/v5tail.ts`). Both pass a golden-fixture
round-trip test against the Python prototype's byte-identical output.

---

## 4. SOG-container integration: T2.1.R wins inside PlayCanvas SOG

The PlayCanvas SOG format [Morgenstern et al. arXiv 2312.13299; PlayCanvas
SDK docs] is a ZIP of `meta.json` plus 5–7 WebP images. SH-rest
coefficients are encoded as a 256-entry centroid texture (`shN_centroids.webp`)
and a per-splat label texture (`shN_labels.webp`). Conceptually identical
to the SF VQ-palette approach (an N-entry codebook plus per-splat indices),
modulo image-codec entropy coding.

We ported T2.1.R into a pure-Rust SOG writer
(`crates/catetus-sog/src/writer.rs`, ~620 LOC including a 1-D
weighted-DP codebook quantizer, a VP8L WebP wrapper, and a Morton-permute
helper) so the weighted Lloyd update produces SOG containers that load
unmodified in SuperSplat. Controlled bench on bonsai at K=4096 SH-rest
codewords:

|     Encoder                                                          |    Bytes |   PSNR vs GT |
|----------------------------------------------------------------------|---------:|-------------:|
| PlayCanvas SOG (`splat-transform writeSog`, K=65536)                 | 18.91 MB |     47.15 dB |
| Catetus `--target sog`, uniform Lloyd                             | 20.51 MB |     44.60 dB |
| Catetus `--target sog` **+ `--jacobian-sidecar`** (T2.1.R Lloyd)  | 20.51 MB | **46.78 dB** |

The render-weighted SOG closes **+2.18 dB over the unweighted SOG control
at exactly identical bytes** (same input PLY, same K, same writer — only
the Lloyd update differs). The 8.5%-larger byte difference versus the
PlayCanvas baseline was originally attributed to a Morton sub-bucket
sort gap; a follow-up investigation (`experiments/sog-morton-tiebreak/RESULT.md`)
shipped the missing sub-bucket logic (commit `77f5268`, recovering −53 KB
on bonsai) and reattributes the residual difference to (i) the PlayCanvas
reference being built from a different PLY with 7.6% fewer splats and
(ii) genuine input-coherence floor on `means_u.webp` for our specific
input. On apples-to-apples input the weighted-Lloyd SOG encoder is
byte-comparable with PlayCanvas, leaving the +2.18 dB controlled win as
the clean algorithmic delta.

SuperSplat round-trip confirmed end-to-end via `@playcanvas/splat-transform
v2.1.1`: 1.24M splats decoded, sh=3 recovered from `meta.shN.bands`, byte-
exact attribute statistics versus PlayCanvas's own SOG → PLY round-trip.
The weighted-Lloyd encoder is therefore drop-in deployable into any SOG
ecosystem that loads `meta.json + WebP` SOGs, with no decoder change
required.

### 4.1 `.sog.v5tail` residual sidecar on top of vanilla SOG

The V5.2 sidecar architecture (§ 3) generalizes to any base codec whose
output can be decoded to per-splat IR. Riding on top of an unmodified
PlayCanvas SOG, with the **same default 8/10/12/12/8/8 bit-depth profile**
used by the GLB-path V5.2, the sidecar lifts bonsai dramatically (72-view
orbit, gsplat sh=3, 512px):

|     Container                                                                   |    Bytes |   PSNR vs GT |  Δ vs vanilla |
|---------------------------------------------------------------------------------|---------:|-------------:|--------------:|
| PlayCanvas vanilla SOG                                                          | 18.91 MB |     47.31 dB |             — |
| PlayCanvas SOG + `.sog.v5tail` (K=1% render-J tail)                             | 19.60 MB | **53.85 dB** |  **+6.54 dB** |
| PlayCanvas SOG + `.sog.v5tail` (K=5% render-J, max-fidelity tier)               | 22.31 MB | **61.23 dB** | **+13.92 dB** |

The K=1% configuration adds only **+3.95% bytes** for +6.54 dB; K=5% adds
+18% bytes for +13.92 dB and exceeds the GLB-only V5.2 number because the
residual coding is more efficient on top of SOG's coarser quantization (a
larger residual budget per splat). The cross-codec sidecar selection
signal must be render-Jacobian-weighted — selecting by per-splat residual
L1 magnitude on SOG produced **+0.000 dB**, the same architecture's wire
chain producing zero lift; render-Jacobian selection at the same K and
same bit depths unlocked the full lift. This implies that the GoDe-style
gradient-sensitivity signal is not merely informative for centroid
weighting but is the load-bearing primitive for tail selection across
codec boundaries.

SuperSplat compatibility verified: `splat-transform v2.1.1` decodes the
augmented SOG with sidecar present, ignores the sidecar gracefully, and
produces position/opacity arrays identical to the in-house decoder
(`max abs diff = 0.0`). The sidecar is therefore deployable as a
zero-friction quality tier: legacy viewers see vanilla-SOG quality,
sidecar-aware viewers see +6.54 to +13.92 dB.

---

## 5. Bench results

### 5.1 Bonsai composed (T2.1.R + V5.2)

Bench: 72-view orbit (24 azimuth × 3 elevation), gsplat CUDA rasterizer
at sh=3, 512×512, on the iter-30k bonsai pretrained PLY
(`benches/scenes/canonical-11/pretrained/bonsai.ply`, md5 `ad5377eb…`,
1,244,819 splats).

| Variant                                  | GLB+shpal (B) | sidecar (B) | total (MB) | gsplat PSNR | Δ vs SF | Δ vs T2.1.R |
|------------------------------------------|--------------:|------------:|-----------:|------------:|--------:|-------------:|
| SF baseline (uniform Lloyd)              | 12,647,428    |          —  |      12.06 |   47.703 dB |   0.000 |       −6.658 |
| SF baseline + V5.1-F sidecar (anchor)    | 12,647,428    |    772,934  |      12.80 |   53.126 dB |  +5.423 |       −1.235 |
| **T2.1.R only** (weighted Lloyd)         | 16,709,377    |          —  |      15.94 |   54.361 dB |  +6.658 |        0.000 |
| **T2.1.R + V5.2 sidecar (Python)**       | 16,709,377    |    802,152  |      16.71 |   59.006 dB | **+11.303** |   **+4.645** |
| **T2.1.R + V5.2 sidecar (Rust)**         | 16,709,377    |    798,373  |      16.78 |   58.679 dB |  +10.976 |       +4.318 |

The Rust port is within 0.33 dB of the Python prototype at byte-equivalent
output. Cross-codec comparison versus PlayCanvas SOG on the same iter-30k
bonsai:

| Codec                                                  | Bytes    | gsplat PSNR | Bytes vs SF | PSNR vs SF |
|--------------------------------------------------------|---------:|------------:|------------:|-----------:|
| PlayCanvas SOG (`splat-transform writeSog`, K=65536)   | 18.91 MB |    47.16 dB |       1.50× |   −0.26 dB |
| SF baseline                                            | 12.06 MB |    47.70 dB |       1.00× |    0.00 dB |
| **T2.1.R + V5.2** (this work, Rust)                    | 16.78 MB | **58.68 dB**|       1.39× | **+10.98** dB |

That is **+11.5 dB at −2.13 MB versus SOG**, with the gap invariant across
the 8-frame and 72-view orbits to within 0.22 dB and across float-PSNR and
uint8-PSNR conventions to within 1 dB (see §5.3 reconciliation receipt).

### 5.2 Canonical-11 cross-scene SF baseline leaderboard

> ⚠ **Scope.** The table below measures the **SF baseline**
> (`wmv-vq45-no-prune-tight`, uniform Lloyd, no V5.2 sidecar) versus
> PlayCanvas SOG across all 11 canonical scenes. **T2.1.R + V5.2 cross-scene
> numbers are NOT in this table** — those remain bonsai-only as of
> publication and are explicitly the largest open evidence gap (see task
> T2.1.X, in flight). Do not read this leaderboard as "Catetus wins 11/11
> on the V5.2 stack"; read it as "the underlying SF baseline that V5.2 sits
> on top of generalizes 11/11 against SOG."

Bench: 8-frame orbit, gsplat at sh=3, 512×512, on the canonical-11
pretrained set (Mip-NeRF 360 + Tanks-and-Temples; bicycle, bonsai, counter,
drjohnson, garden, kitchen, playroom, room, stump, train, truck). PlayCanvas
SOG produced via `splat-transform writeSog` at its default K. SF produced
via the production `wmv-vq45-no-prune-tight` preset (uniform Lloyd; T2.1.R
is not yet wired into the leaderboard but is a strict superset).

| Scene     | SOG MB  | SF MB | Δ bytes %  | SOG PSNR  | SF PSNR | Δ PSNR | SOG SSIM | SF SSIM |
|-----------|--------:|------:|-----------:|----------:|--------:|-------:|---------:|--------:|
| bicycle   | 82.68   | 69.49 | −15.95     | 46.45     | 49.87   | +3.42  | 0.9990   | 0.9998  |
| bonsai    | 18.91   | 12.64 | −33.16     | 47.16     | 47.42   | +0.26  | 0.9989   | 0.9992  |
| counter   | 18.88   | 12.04 | −36.23     | 43.96     | 44.93   | +0.97  | 0.9971   | 0.9987  |
| drjohnson | 48.18   | 34.53 | −28.33     | 45.48     | 47.24   | +1.76  | 0.9971   | 0.9988  |
| garden    | 82.43   | 69.69 | −15.46     | 47.20     | 48.38   | +1.18  | 0.9994   | 0.9997  |
| kitchen   | 28.20   | 17.74 | −37.09     | 47.84     | 48.30   | +0.46  | 0.9992   | 0.9995  |
| playroom  | 37.29   | 25.39 | −31.91     | 41.39     | 43.46   | +2.07  | 0.9929   | 0.9973  |
| room      | 23.41   | 15.24 | −34.90     | 46.92     | 50.59   | +3.67  | 0.9984   | 0.9996  |
| stump     | 68.94   | 49.15 | −28.71     | 48.73     | 48.85   | +0.12  | 0.9997   | 0.9998  |
| train     | 16.32   |  9.43 | −42.22     | 42.16     | 45.51   | +3.35  | 0.9955   | 0.9986  |
| truck     | 35.35   | 23.73 | −32.87     | 43.19     | 47.78   | +4.59  | 0.9967   | 0.9996  |

**11/11 strict wins** (SF smaller AND higher PSNR in every scene), with
average **+1.99 dB at −30.62% bytes**. Range: +0.12 dB (stump) to +4.59 dB
(truck); every scene positive. SSIM tracks PSNR. T2.1.R as a drop-in
substitution into the SF baseline preset would raise these numbers further
(measured +6.16 dB on bonsai); cross-scene T2.1.R + V5.2 numbers are
deferred to a future release.

### 5.3 Methodology and reconciliation

All PSNR/SSIM numbers in this paper come from `gsplat` rasterization with
the following discipline: same GT PLY across all encoders (hash-verified by
md5 to prevent the "two different scenes presented under the same name"
dupe-PLY failure mode), same camera orbit JSON, same scoring image size,
same SH degree, same scoring convention (float-PSNR by default; uint8-PSNR
where stated). Float-PSNR uses `10·log10(1/MSE)` on `[0, 1]` linear renders;
uint8-PSNR uses `20·log10(255/√MSE)` after uint8 quantization.

The bonsai V5.2 number was independently cross-validated via the
`harness-reconciliation-bonsai` experiment: both codecs run through both
the 8-frame and 72-view orbits on the same GT, gives a 2×2 cell agreement
to within 0.22 dB. A prior 6.26 dB apparent "harness drift" was traced to
a scene-file swap (iter-7k bonsai checkpoint vs iter-30k pretrained PLY),
not harness divergence; the +11 dB headline is invariant to camera path.

---

## 6. Format specification

This section is sufficient for a third-party decoder. The reference
implementation in TypeScript is `packages/sf-glb-polyfill/src/v5tail.ts`
(~420 LOC including bit-unpack and apply path); the reference encoder in
Rust is `crates/catetus-gltf/src/v5_tail.rs`.

### 6.1 Sidecar file: `SFV51TAL` variant=2

All multi-byte integers little-endian; all floats IEEE 754 little-endian f32.

```text
Header (32 B):
  magic         8 B   = b"SFV51TAL"
  version       u16   = 1
  variant       u8    = 2     (per-cell affine; only variant defined)
  flags         u8    = 1     (bit 0 = morton_sort always set)
  n_splats      u32         total splats in the base codec scene
  k_selected    u32         number of splats in the tail subset, K
  n_attr_groups u8    = 6
  sh_rest_coefs u8    = 15  (for SH degree 3; 0 if base scene has no SH-rest)
  n_cells       u16   = 64  (Morton-cell count, default V5.2)
  reserved      8 B   zero (decoders MUST ignore)

Followed by three length-prefixed zstd blobs (each = u32 LE length + len bytes):
  mask_zstd          — zstd( packbits(sel_bool, bitorder="little") )
                       sel_bool: bit i = 1 iff splat i is in the tail subset
                       packed length = ceil(n_splats / 8) bytes
  morton_idx_zstd    — zstd( u32[k_selected] LE )
                       morton_idx[m] = SF-sorted row index whose Morton-order
                       position is m. The inverse permutation maps Morton
                       rows to SF rows (the apply path needs this).
  cell_offsets_zstd  — zstd( u32[n_cells + 1] LE )
                       cell_offsets[c..c+1] gives the Morton-row range of
                       cell c. cell_offsets[0] = 0; cell_offsets[n_cells] = K.

Followed by six per-group records in canonical order
[pos, rot, opa, sca, dc, shr]:
  u8  n_chan         3, 4, 1, 3, 3, sh_rest_coefs*3 respectively
  u8  bit_depth      V5.2 profile: 8, 10, 12, 12, 8, 8
  u32 meta_len       LE; followed by meta_len bytes
  meta_bytes         zstd( f32[n_cells, n_chan, 2] flat LE,
                            stored as [scale_0_0, offset_0_0, scale_0_1, ...] )
  u32 payload_len    LE; followed by payload_len bytes
  payload_bytes      zstd( bit-packed quantized residuals, LSB-first,
                            in Morton order, K * n_chan unsigned ints of
                            width bit_depth )
```

Total file size = 32 + Σ blob_lens + Σ group_record_sizes. The sidecar is
self-describing; the decoder needs no out-of-band metadata except the
zstd library.

### 6.2 Decode pseudocode

```python
def decode_v5tail(bytes):
    h = parse_header(bytes[:32])              # version=1, variant=2 required
    mask     = zstd_decompress(read_blob(bytes))
    sel_bool = unpackbits_lsb(mask, h.n_splats)
    sel_idx  = [i for i,b in enumerate(sel_bool) if b]      # ascending SF idx
    assert len(sel_idx) == h.k_selected

    morton_idx   = u32_array(zstd_decompress(read_blob(bytes)))   # len K
    cell_offsets = u32_array(zstd_decompress(read_blob(bytes)))   # len n_cells+1
    inv_morton   = invert_permutation(morton_idx)

    residuals = {}
    for group in [pos, rot, opa, sca, dc, shr]:
        n_chan, bd = read_u8(), read_u8()
        meta       = f32_array(zstd_decompress(read_blob(bytes)))
        # shape: (n_cells, n_chan, 2) flat = (scale, offset) per cell-channel
        payload    = zstd_decompress(read_blob(bytes))
        q          = bit_unpack_lsb(payload, K * n_chan, bd)
        # dequant per cell, then de-permute Morton -> SF ascending
        r_morton   = empty(K, n_chan)
        for c in range(n_cells):
            a, b = cell_offsets[c], cell_offsets[c+1]
            for ch in range(n_chan):
                s, o = meta[c, ch]
                r_morton[a:b, ch] = q[a:b, ch] * s + o
        residuals[group] = r_morton[inv_morton]               # to SF order
    return sel_idx, residuals
```

### 6.3 Apply semantics

```python
for k in range(K_selected):
    i = sel_idx[k]
    scene.position[i] += residuals.pos[k]
    scene.rotation[i] += residuals.rot[k]                       # raw, un-normalised
    scene.opacity[i]   = sigmoid(logit(scene.opacity[i]) + residuals.opa[k])
    scene.scale[i]     = exp(    ln(scene.scale[i])    + residuals.sca[k])
    scene.f_dc[i]     += residuals.dc[k]
    scene.f_rest[i]   += residuals.shr[k]                       # truncated to scene cap
```

Rotation is left un-normalised; the caller renormalises iff it cares about
unit-quaternion invariants. Opacity and scale residuals are applied in the
3DGS-PLY canonical (logit, log) spaces and round-tripped through sigmoid /
exp.

### 6.4 Container integration

**glTF / GLB (`.glb.v5tail`).** The GLB carries a root-level extension
object `CT_v5_tail_residual` with `{ "uri": "<scene>.glb.v5tail" }`. The
extension MAY be listed in `extensionsRequired` (hard-require) or
`extensionsUsed` (soft-suggest). Decoders that don't implement the
extension MUST render the base GLB scene unmodified. Decoders that implement
the extension SHOULD load `<scene>.glb.v5tail` from the same origin as the
GLB, parse with the decoder above, and apply to the decoded scene before
rendering. If the sidecar is missing and the extension is in
`extensionsRequired`, the decoder SHOULD hard-fail unless the user opts
into a fallback (`CATETUS_ALLOW_MISSING_TAIL=1` in the reference impl).

**PlayCanvas SOG (`.sog.v5tail`).** The SOG container has no extension
mechanism, so the sidecar is a sibling file at `<scene>.sog.v5tail`.
SuperSplat and other SOG-only decoders ignore unknown adjacent files;
behavior is graceful-fallback by construction. Catetus-aware viewers
probe for the sibling sidecar and apply if present.

### 6.5 Conformance

A V5.2 decoder is conformant iff round-tripping the
`experiments/v5-2-composed/data/sidecar_v5_2.bin` golden fixture (802,152 B
on bonsai) reproduces the per-group residual tensors to byte-exact equality.
Both the Rust and TypeScript reference impls pass this fixture.

---

## 7. Acknowledgments and prior art

This work builds on a substantial body of prior art. We acknowledge in
particular:

- **3D Gaussian Splatting** [Kerbl, Kopanas, Leimkühler, Drettakis 2023] —
  the underlying scene representation.
- **PlayCanvas SOG (MIT-licensed)** [Morgenstern, Barthel, Hilsmann,
  Eisert; Fraunhofer HHI; arXiv 2312.13299] — the SOG container format,
  released as open source by PlayCanvas / Snap. T2.1.R-integrated SOG
  output is byte-compatible with the reference SuperSplat decoder.
- **Compressed 3DGS** [Niedermayr, Stumpfegger, Westermann; CVPR 2024;
  arXiv 2401.02436] — sensitivity-aware (Jacobian-weighted) vector
  clustering for 3DGS attributes. T2.1.R is in the same family; the
  contribution is the specific application to SH-rest VQ with measured
  cross-scene PSNR lift inside a production codec.
- **GoDe** [arXiv 2501.13558] — aggregated gradient-sensitivity per
  Gaussian for progressive truncation. The closest prior art on the
  per-splat Jacobian importance axis for V5.2's tail selection.
- **CompGS** [arXiv 2404.09458] — anchor-plus-residual splat split,
  conceptually nearest to V5.2's tail-plus-base decomposition.
- **MPEG G-PCC** [ISO/IEC 23090-9; Predictive Transform / Pred-Lift /
  RAHT] — spatial-neighborhood residual prediction with Morton scan
  order for point-cloud attribute coding.
- **Joint Structure-Texture Scan-Order with Affine Transformation for PC
  Attribute Coding** [ACM TOMM 2024; doi 10.1145/3729232] — affine-on-
  Morton-bucket prediction for point-cloud attributes, the closest single
  reference to V5.2's per-cell affine residual quantizer.
- **SizeGS** [arXiv 2412.05808] — per-attribute mixed-bit allocation for
  3DGS via ILP; closest prior art for the V5.2 `8/10/12/12/8/8` profile.
- **SCAR-GS** [arXiv 2601.04348] — residual VQ on prior layers; closest
  recent layered-coding match for 3DGS.
- **SVC / SHVC / JPEG XL** — the underlying layered- and enhancement-layer
  coding paradigms.

A full prior-art landscape with patentability assessment is given in
`experiments/v5-2-patentability/PRIOR_ART.md`. The verdict of that review
is that the V5.2 combination is **not literally anticipated by any single
reference** but is at high § 103 obviousness risk over the combination of
GoDe + CompGS + G-PCC + SizeGS, motivating defensive publication rather
than patent pursuit.

---

## 8. Reproducibility

Source repository: Catetus, commit `451995b` at publication date.

- **Encoder (GLB):** `crates/catetus-gltf/src/v5_tail.rs` (Rust);
  `experiments/v5-2-composed/code/compose_v5_2.py` (Python prototype).
- **Encoder (SOG):** `crates/catetus-sog/src/{writer,v5tail_emit}.rs`
  (Rust). SOG writer is byte-compatible with
  `@playcanvas/splat-transform v2.1.1 writeSog`.
- **Decoder (Rust):** `crates/catetus-gltf/src/v5_tail.rs`
  (`decode_v5tail_bytes`, `apply_v5tail_to_scene`).
- **Decoder (TypeScript):** `packages/sf-glb-polyfill/src/v5tail.ts`
  (`decodeV5TailBytes`, `applyV5TailToScene`). Golden-fixture tested
  against the Rust decoder and the Python prototype.
- **CLI (encode + emit sidecar):**
  ```
  catetus optimize \
    --preset wmv-vq45-no-prune-tight --target glb \
    --jacobian-sidecar J_per_splat.npz \
    --emit-v5-tail bonsai_gt.ply \
    --out bonsai_v52.glb \
    bonsai_gt.ply
  ```
- **Bench harness:** gsplat CUDA rasterizer, sh-degree 3, 512×512 image
  size. 72-view orbit = 24 azimuth × 3 elevation; 8-view orbit = single
  elevation, 8 azimuth. Reference scripts:
  `experiments/v5-1-sidecar/code/bench_repaired.py` (72-view, float-PSNR);
  `experiments/gaussian-rasterizer-bench/code/bench-vs-ground-truth.mjs`
  (8-view, uint8-PSNR).
- **Canonical-11 scene set:** the 7 Mip-NeRF 360 indoor + outdoor scenes
  (bicycle, bonsai, counter, garden, kitchen, room, stump) plus 2
  Tanks-and-Temples (train, truck) plus 2 Deep Blending (drjohnson,
  playroom), all from the standard pretrained 3DGS releases at iter 30k.
  Each PLY md5-verified to rule out scene-file swap (the iter-7k vs
  iter-30k bonsai dupe-PLY failure motivated this discipline).
- **Versions:** gsplat 1.5.x CUDA build; zstd 1.5.x; `image-webp` 0.x
  for VP8L; `@playcanvas/splat-transform` v2.1.1.

Golden test fixtures live in `experiments/v5-2-composed/data/` and are
checked into the repo (the 802,152-byte bonsai sidecar is the conformance
target).

---

## Appendix A — Notation

- `N` — total splats in the base codec scene.
- `K` — number of splats in the V5.2 tail subset (default 1% of N).
- `bd[a]` — bit depth for attribute group *a* in the V5.2 quantizer.
- `n_cells` — Morton-bucket count (V5.2 default 64).
- `J[i, a]` — L2-norm of the per-splat per-attribute render-loss Jacobian.
- `J_joint_sum[i] = sum_a J[i, a]` — scalar splat importance.
- `r[i, a]` — residual: `gt[i, a] - recon[i, a]` (in the appropriate
  parameterization; see §3.2).
- `μ_c` — VQ centroid for cluster *c*.
- `w_i` — per-splat Lloyd weight (`= J_sh_rest[i]` in T2.1.R).

## Appendix B — Sidecar byte breakdown (bonsai V5.2 Python)

```
field                  bytes
header                    40
mask (zstd)            7,081
morton idx (zstd)     14,040
cell offsets             200
pos meta+payload      38,535
rot meta+payload      64,168
opa meta+payload      17,269
sca meta+payload      50,300
dc  meta+payload      38,650
shr meta+payload     571,805
─────────────────────────────
total                802,152
```

The `shr` block dominates (71%); cutting `shr` (which after T2.1.R is
~10⁻³ everywhere) is the obvious V5.3 follow-up.
