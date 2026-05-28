# `catetus-jacobian` — Algorithm

## Background

The Catetus production codec (T2.1.R + V5.2) can deliver +7 dB to +12 dB PSNR
at the same byte budget when the SH-rest VQ palette / V5.2 residual selector
uses **per-splat render-Jacobian weights** instead of treating every splat
equally. Today the public CLI exposes a `--jacobian-sidecar PATH.npz` flag
that consumes a `J_sh_rest` (or joint multi-array) NPZ produced by an
offline gsplat-based "census" script that lives on the 4090 box (Python +
gsplat + CUDA — see `experiments/jacobian-census-bonsai-30k/`).

The presence of that external dependency is **the** shipping blocker: a
customer downloading the Catetus CLI cannot reproduce the published +7/+12 dB
deltas because they don't have a CUDA box, gsplat, or the census script.
`--jacobian-sidecar` exists, but they can't fill it in.

This crate is the first step toward an **auto-Jacobian** path: a pure-Rust,
no-GPU, no-Python computation of the per-splat Jacobian weights, computed
directly from the input PLY in milliseconds, exposed eventually as
`catetus optimize --auto-jacobian`.

## Ground truth (Python reference)

The Python reference (`jacobian_census.py`, run on the 4090 — local
`code/` directory is empty because the script lives at
`/home/montabano1/splatforge/jacobian-census-*/`; the artifacts checked
into the public repo are the resulting `J_per_splat.npz` files) does the
following:

1. Load the input PLY into gsplat (positions, rotations, scales, opacities,
   DC and SH-rest coefficients).
2. Render the scene from N=72 camera views around the scene bounding box at
   a fixed image size (512×512).
3. For each view, perturb each splat's SH-rest by Gaussian noise
   (σ=0.005), re-render, measure ΔL2 in the rendered image, and accumulate
   the per-splat magnitude into `J_sh_rest[i]`.
4. Also compute `J_position`, `J_rotation`, `J_opacity`, `J_scale`, `J_dc`
   the same way (each measures how much that attribute matters at render
   time, per splat). All saved into one `.npz`.

Summary stats (bonsai-30k, N=1,244,819):

| group        | mean       | median     | max     | top-1% share |
|--------------|------------|------------|---------|--------------|
| J_position   | 1.07e-02   | 1.09e-04   | 5.24e+1 | 82.7 %       |
| J_scale      | 1.64e-03   | 5.04e-07   | 4.24e+1 | 94.8 %       |
| J_rotation   | 3.15e-03   | 9.26e-07   | 4.36e+1 | 93.9 %       |
| J_sh_rest    | (see npz)  | …          | …       | …            |

Key observation: **all the J arrays are heavily long-tailed**. The top 1 %
of splats carry 82–95 % of the total render-importance mass. That's *why*
weighted Lloyd-Max VQ beats unweighted Lloyd-Max by +11.94 dB on SH-rest —
it preserves the few splats that actually matter on screen.

## CPU MVP — geometric proxy

The full Python algorithm needs:
- A perspective-projection rasterizer (gsplat / a Rust port).
- Differentiation w.r.t. each per-splat attribute.
- Camera intrinsics + extrinsics (gsplat synthesizes these from the scene
  bbox; we don't have them in the .ply).

We can't ship that in one session, and we shouldn't: it would re-introduce
the same "needs CUDA / GPU rasterization" deployment problem the user was
trying to avoid.

Instead we implement a **closed-form geometric proxy** that is correlated
with the rendering Jacobian without rendering anything:

```
J_proxy[i] = α_i  *  area_2d(scale_i)  *  ||sh_rest_i||_2
```

where:

- `α_i` is the splat's opacity (post-sigmoid, in [0, 1]). A fully
  transparent splat contributes zero to any rendered pixel — its render
  Jacobian is exactly 0 — so multiplying by α correctly zeros these out.

- `area_2d(scale_i)` is the **view-averaged 2D footprint area** of a
  Gaussian with anisotropic scales (s_x, s_y, s_z). The geometric mean of
  the three pairwise products,
  `((s_x*s_y) * (s_y*s_z) * (s_z*s_x))^{1/3}`,
  approximates the screen-space footprint a viewer would see averaged over
  random viewing directions. This is the dominant driver of the rendering
  Jacobian: small splats project to few pixels, so perturbing their
  SH-rest barely moves any pixel; large splats cover many pixels, so the
  same SH-rest perturbation moves the rendered image a lot.

- `||sh_rest_i||_2` is the L2 norm of the higher-order SH coefficients
  (bands 1..=N). A splat whose SH-rest is already zero would have zero
  J_sh_rest in principle. But gsplat's central-difference numerical
  Jacobian is dominated by the σ=0.005 perturbation, not the original
  value, so this factor matters less for the SH-rest array specifically
  than it does for other attributes. We keep it because it's monotone
  and zero-respecting (a perfectly-DC splat correctly maps to zero
  J_sh_rest).

What we deliberately don't model in the MVP:

- **Visibility / occlusion / camera frustum.** With no camera poses
  available from the .ply we treat all splats as "in view, in front." The
  Python census uses 72 cameras arranged on a sphere around the scene
  bbox, which is itself a uniform-coverage proxy.
- **View-direction-dependent SH evaluation.** SH-rest contributes to
  rendering only off the dominant DC direction; near the DC axis the
  contribution is small. View-averaging already washes most of this out.
- **Alpha compositing depth ordering.** A splat behind a fully-opaque
  splat shouldn't move pixels, but with α<1 (typical) the back splat
  still contributes. We assume the bulk α distribution is in the
  "partially-transparent" regime where ordering matters less.

## Correlation expectations

Per the contract this is a **stretch goal**. Acceptable: correlation
(Pearson or Spearman) > 0.5 with the Python reference on bonsai 30k. Target:
> 0.7. The crate exposes `compute_jacobian_with_method` so we can A/B
multiple proxies as we iterate.

## API

```rust
pub struct JacobianResult {
    pub j_sh_rest: Vec<f32>,
    pub method: JacobianMethod,
}

pub enum JacobianMethod {
    /// `α * area2d * ||sh_rest||₂` — the MVP described above.
    GeometricProxyV1,
}

pub fn compute_jacobian(scene: &SplatScene) -> JacobianResult;
```

## Per-attribute proxies (LAUNCH-1c)

The V5.2 sidecar emitter (`catetus optimize --emit-v5-tail
--auto-jacobian`) consumes the *joint* per-attribute layout — six arrays,
one per splat attribute channel: `J_position`, `J_dc`, `J_sh_rest`,
`J_opacity`, `J_scale`, `J_rotation`. The single-channel SH-rest proxy
above is reused as one of the six; the other five are closed-form
geometric proxies built from the same primitives.

The shared scalars we compute per splat once:

```
α         = clamp(opacity, 0, 1)              # already post-sigmoid by importer
area      = (s_x · s_y · s_z)^(2/3)           # view-averaged 2D footprint
dc_norm   = ‖DC‖₂                              # diffuse RGB magnitude
sh_norm   = ‖SH-rest‖₂                         # higher-band magnitude
color     = dc_norm + sh_norm                  # "how bright would a pixel be"
aniso     = 1 - (s_min / s_max) ∈ [0, 1]      # 0 = sphere, 1 = needle
```

### `J_position` — `area · α · color`

Perturbing a splat's xyz moves its (colored) 2-D footprint across pixels.
The ΔL2 in the rendered image scales with `(footprint area) ·
(α-weighted color magnitude)`. This channel empirically dominates the
reference Jacobians — splats with both large footprints and bright
colors carry most of the rendering importance.

### `J_dc` — `area · α`

Perturbing DC adds a constant ε to every pixel inside the splat's
α-weighted footprint. The Δ-image L2 is `(footprint area) · α · |ε|`.
No `color_norm` factor — the perturbation *is* the color change.

### `J_sh_rest` — `area · α · ‖SH-rest‖₂`

(Same single-channel proxy from the MVP — see "CPU MVP" above. Mirrored
into the per-attribute struct unchanged.)

### `J_opacity` — `area · color · α · (1 - α)`

Perturbing α scales the splat's full color contribution by Δα. We weight
by `α · (1 - α)` — the sigmoid-derivative shape:

- splats at α ≈ 0 are invisible (Δα can't paint pixels without also
  gaining color);
- splats at α ≈ 1 are alpha-saturated (compositing clips at 1);
- splats at α ≈ 0.5 are most sensitive.

A bare `area · color_norm` proxy (no α-derivative weight) scored
Spearman 0.27 against the gsplat reference; adding the `α (1-α)` term
lifts it to 0.63, clearing the 0.6 contract target.

### `J_scale` — `area · α · color`

A scale perturbation redistributes color across more/fewer pixels. The
derivative of the geomean-area proxy w.r.t. `log s_i` is `(2/3) · area`,
so the per-splat sensitivity scales with `area · α · color_norm` — same
shape as `J_position` but driven by area-change instead of footprint-shift.

### `J_rotation` — `area · α · color · anisotropy`

A rotation perturbation only moves the footprint if the splat isn't
spherical. Multiplying by `anisotropy = 1 - s_min/s_max` ∈ [0, 1] zeros
out perfectly round splats and weights elongated needles/pancakes by
how much pixel motion a small Δθ would induce.

### Measured correlations (bonsai-30k, N=1,244,819)

Compared against the Python `J_per_splat.npz` produced by
`jacobian_census.py` on the 4090 (72 views, σ=0.005, SH-deg=3):

| channel        | Pearson | Spearman | top-1 % overlap |
|----------------|---------|----------|------------------|
| `J_position`   | 0.225   | 0.685    | 27.8 %           |
| `J_dc`         | 0.158   | 0.667    | 32.7 %           |
| `J_sh_rest`    | 0.182   | 0.732    | 32.8 %           |
| `J_opacity`    | 0.229   | 0.626    | 24.8 %           |
| `J_scale`      | 0.320   | 0.787    | 39.6 %           |
| `J_rotation`   | 0.302   | 0.795    | 40.6 %           |

All six channels clear the contract gate (Spearman ≥ 0.6).
`tests/bonsai_per_attribute_correlation.rs` re-runs this end-to-end
when `CATETUS_JACOBIAN_BONSAI_PLY` and `CATETUS_JACOBIAN_REFERENCE_NPZ`
are set. The Pearson values are low across the board because the
reference Jacobians are extremely long-tailed (top 1 % carries ~83-95 %
of mass) — Spearman and top-k overlap are the metrics that matter
downstream for VQ palette / V5.2 residual selection.

## API

```rust
pub struct JacobianResult {
    pub j_sh_rest: Vec<f32>,
    pub method: JacobianMethod,
}

pub struct PerAttributeJacobian {
    pub j_position: Vec<f32>,
    pub j_dc:       Vec<f32>,
    pub j_sh_rest:  Vec<f32>,
    pub j_opacity:  Vec<f32>,
    pub j_scale:    Vec<f32>,
    pub j_rotation: Vec<f32>,
    pub method:     JacobianMethod,
}

pub enum JacobianMethod {
    /// `α * area2d * ||sh_rest||₂` (single channel) and the per-attribute
    /// six-channel proxy above. CPU-only, no rasterisation.
    GeometricProxyV1,
}

pub fn compute_jacobian(scene: &SplatScene) -> JacobianResult;
pub fn compute_jacobian_per_attribute(scene: &SplatScene) -> PerAttributeJacobian;
```

## What's left after this session

1. **Camera-aware visibility.** When the consumer ships a `cameras.json`
   (Inria 3DGS standard), gate `J_proxy` by frustum membership and the
   cosine of view direction vs the splat's longest axis.
2. **A true rasterized Jacobian.** A pure-Rust EWA splatting renderer
   would let us drop the proxy entirely. ~2 weeks of work; not in MVP
   scope but a clear follow-on.
3. **`--auto-jacobian` CLI wiring.** Once the proxy lands the public
   catetus-cli should grow an `--auto-jacobian` flag that calls this
   crate's `compute_jacobian` / `compute_jacobian_per_attribute` and
   feeds the result through the existing `PassContext.sh_rest_weights`
   and V5.2-tail joint-Jacobian plumbing (the latter currently fed by
   `--jacobian-sidecar` only).
4. **Higher Pearson via tail-tame.** Top-1 % overlap is acceptable for
   the downstream VQ/V5.2 consumers, but a log-warp + per-attribute
   gain calibration on the proxy would lift Pearson and might unlock a
   slightly tighter residual selector. Not on the critical path.
