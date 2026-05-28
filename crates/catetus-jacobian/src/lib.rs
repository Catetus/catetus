//! `catetus-jacobian` — per-splat rendering-Jacobian proxies.
//!
//! See `ALGORITHM.md` for the full derivation. TL;DR: the Python reference
//! (`jacobian_census.py`, runs on a CUDA box via gsplat) renders the scene
//! from 72 views, perturbs each splat's per-attribute parameters by Gaussian
//! noise, and measures the resulting per-splat L2 change in the rendered
//! image. The output `J_per_splat.npz` has six arrays — one per attribute
//! channel: `J_position`, `J_dc`, `J_sh_rest`, `J_opacity`, `J_scale`,
//! `J_rotation`.
//!
//! This crate approximates all six with closed-form CPU proxies built from
//! the same primitives: a view-averaged 2D footprint area, opacity, DC
//! magnitude, SH-rest magnitude, and a scale anisotropy factor. None of
//! them require rendering, cameras, or GPU.
//!
//! ```text
//! J_xyz     [i] ≈ area_2d(s) * α * (‖dc‖ + ‖sh_rest‖)
//! J_dc      [i] ≈ area_2d(s) * α
//! J_sh_rest [i] ≈ area_2d(s) * α * ‖sh_rest‖
//! J_opacity [i] ≈ area_2d(s) * (‖dc‖ + ‖sh_rest‖)
//! J_scale   [i] ≈ area_2d(s) * α * (‖dc‖ + ‖sh_rest‖)         (∂area/∂log s)
//! J_rotation[i] ≈ area_2d(s) * α * (‖dc‖ + ‖sh_rest‖) * anisotropy
//! ```
//!
//! The proxies correlate with the Python ground truth strongly enough to
//! drive the same VQPaletteShRest weighted-Lloyd algorithm (and the V5.2
//! sidecar emitter, which consumes the *joint* per-attribute layout) that
//! yield the +7..+12 dB headlines on the canonical-11 bench.
//!
//! Public crate (Apache-2.0). Builds on the public `catetus-ply` /
//! `catetus-core` IR and powers the public `catetus optimize
//! --auto-jacobian` CLI flag — letting customers reproduce the T2.1.R
//! (+6.24 dB over SuperSplat) and V5.2 fidelity tiers without needing
//! CUDA or gsplat.

use catetus_core::ir::{Color, Splat, SplatScene};

/// Which proxy variant to use. The crate is built for A/B experimentation
/// as we iterate toward a renderer-grade per-splat Jacobian, so each new
/// proxy lands as a new enum variant and the public entry point dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JacobianMethod {
    /// `α * area_2d(scale) * ||sh_rest||₂` — the MVP described in
    /// `ALGORITHM.md`. Pure-CPU, no rasterization, no GPU.
    #[default]
    GeometricProxyV1,
}

/// One per-splat Jacobian estimate, parallel to `SplatScene::splats`.
#[derive(Debug, Clone)]
pub struct JacobianResult {
    /// `j_sh_rest[i]` ≥ 0, parallel to `scene.splats`.
    pub j_sh_rest: Vec<f32>,
    /// Which proxy produced the values (for logging / debug).
    pub method: JacobianMethod,
}

/// Per-attribute per-splat Jacobian proxies. Every field is `Vec<f32>` of
/// length `scene.splats.len()`, and every value is ≥ 0.
///
/// Mirrors the six arrays in `J_per_splat.npz` produced by
/// `jacobian_census.py` (see `ALGORITHM.md`):
///
/// - `j_position` — perturbing this splat's `xyz`.
/// - `j_dc` — perturbing this splat's diffuse colour.
/// - `j_sh_rest` — perturbing this splat's SH bands ≥ 1.
/// - `j_opacity` — perturbing this splat's α.
/// - `j_scale` — perturbing this splat's `log scale` (per axis).
/// - `j_rotation` — perturbing this splat's rotation quaternion.
#[derive(Debug, Clone)]
pub struct PerAttributeJacobian {
    pub j_position: Vec<f32>,
    pub j_dc: Vec<f32>,
    pub j_sh_rest: Vec<f32>,
    pub j_opacity: Vec<f32>,
    pub j_scale: Vec<f32>,
    pub j_rotation: Vec<f32>,
    pub method: JacobianMethod,
}

/// Compute the per-splat SH-rest rendering-Jacobian proxy.
///
/// Uses [`JacobianMethod::GeometricProxyV1`]. Pure-CPU, parallelized over
/// splats with rayon. Output length matches `scene.splats.len()`.
pub fn compute_jacobian(scene: &SplatScene) -> JacobianResult {
    compute_jacobian_with_method(scene, JacobianMethod::default())
}

/// Like [`compute_jacobian`] but with an explicit method selector. Exposed
/// for A/B testing as new proxies land.
pub fn compute_jacobian_with_method(scene: &SplatScene, method: JacobianMethod) -> JacobianResult {
    let j_sh_rest = match method {
        JacobianMethod::GeometricProxyV1 => geometric_proxy_v1(&scene.splats),
    };
    JacobianResult { j_sh_rest, method }
}

/// Compute the per-attribute per-splat rendering-Jacobian proxy.
///
/// Returns one `Vec<f32>` per attribute channel, all parallel to
/// `scene.splats`. This is the entry point the V5.2 sidecar emitter
/// (`catetus optimize --emit-v5-tail --auto-jacobian`) calls — the V5.2
/// residual selector consumes the *joint* multi-array layout, not just
/// SH-rest.
///
/// Pure-CPU, parallelised over splats. Each channel uses a closed-form
/// proxy built from `(α, area_2d, ‖dc‖, ‖sh_rest‖, anisotropy)`. See
/// `ALGORITHM.md` for the per-channel derivation.
pub fn compute_jacobian_per_attribute(scene: &SplatScene) -> PerAttributeJacobian {
    compute_jacobian_per_attribute_with_method(scene, JacobianMethod::default())
}

/// Like [`compute_jacobian_per_attribute`] but with an explicit method
/// selector. Today there's only one variant; the indirection is here so
/// future proxies (e.g. with camera-aware visibility) can A/B without
/// changing the public signature.
pub fn compute_jacobian_per_attribute_with_method(
    scene: &SplatScene,
    method: JacobianMethod,
) -> PerAttributeJacobian {
    match method {
        JacobianMethod::GeometricProxyV1 => geometric_per_attribute_v1(&scene.splats),
    }
}

/// `α * area_2d(scale) * ||sh_rest||₂` proxy. See `ALGORITHM.md`.
///
/// Parallelized over splats. Order-preserving (output index = input index).
fn geometric_proxy_v1(splats: &[Splat]) -> Vec<f32> {
    use rayon::prelude::*;
    splats
        .par_iter()
        .map(|s| {
            let alpha = s.opacity.clamp(0.0, 1.0);
            let area = area_2d_view_avg(s.scale);
            let sh_norm = sh_rest_l2_norm(&s.color);
            alpha * area * sh_norm
        })
        .collect()
}

/// Per-attribute six-channel proxy. See per-channel derivation in
/// `ALGORITHM.md`. Parallelised over splats.
fn geometric_per_attribute_v1(splats: &[Splat]) -> PerAttributeJacobian {
    use rayon::prelude::*;

    // Single parallel pass — we want one read per splat, six writes.
    let rows: Vec<[f32; 6]> = splats
        .par_iter()
        .map(|s| {
            let alpha = s.opacity.clamp(0.0, 1.0);
            let area = area_2d_view_avg(s.scale);
            let dc_norm = dc_l2_norm(&s.color);
            let sh_norm = sh_rest_l2_norm(&s.color);
            // "Total radiometric magnitude" — how brightly this splat would
            // colour a pixel if α=1. DC dominates; SH-rest is a refinement.
            let color_norm = dc_norm + sh_norm;
            let aniso = scale_anisotropy(s.scale);

            // J_position: shifting xyz moves a coloured footprint of α-mass
            //   across pixels. Proportional to (footprint area) * α * colour
            //   magnitude. Empirically dominates the reference Jacobians.
            let j_pos = area * alpha * color_norm;

            // J_dc: perturbing DC adds a constant to every pixel inside the
            //   footprint, weighted by α. No colour-magnitude factor — the
            //   perturbation IS the colour change.
            let j_dc = area * alpha;

            // J_sh_rest: same as existing geometric_proxy_v1.
            let j_sh = area * alpha * sh_norm;

            // J_opacity: shifting α scales the splat's (DC+SH-rest)
            //   colour contribution by Δα. We weight by `α · (1 - α)`:
            //   - splats at α ≈ 0 barely paint anything, so an opacity
            //     perturbation can't be visible (no down-stream pixels
            //     to redistribute);
            //   - splats at α ≈ 1 are alpha-saturated and clipping at 1
            //     similarly absorbs the perturbation;
            //   - splats at α ≈ 0.5 are most sensitive.
            //   This matches the sigmoid-derivative shape and empirically
            //   correlates with the gsplat reference much better than a
            //   bare `area * color_norm` proxy.
            let j_op = area * color_norm * alpha * (1.0 - alpha);

            // J_scale: ∂(area)/∂(log s_i) ≈ (2/3)·area for the geomean
            //   proxy. The colour weight matters because a scale change
            //   redistributes colour across more/fewer pixels.
            let j_sc = area * alpha * color_norm;

            // J_rotation: a rotation perturbation only moves the footprint
            //   if the footprint isn't a circle. Multiply by anisotropy in
            //   [0, 1]: 0 means spherical (rotation invisible), 1 means
            //   maximally elongated (small rotations move pixels by ~Δ·L).
            let j_rot = area * alpha * color_norm * aniso;

            [j_pos, j_dc, j_sh, j_op, j_sc, j_rot]
        })
        .collect();

    // Transpose into six channel-vectors in one allocation each.
    let n = rows.len();
    let mut j_position = Vec::with_capacity(n);
    let mut j_dc = Vec::with_capacity(n);
    let mut j_sh_rest = Vec::with_capacity(n);
    let mut j_opacity = Vec::with_capacity(n);
    let mut j_scale = Vec::with_capacity(n);
    let mut j_rotation = Vec::with_capacity(n);
    for r in rows {
        j_position.push(r[0]);
        j_dc.push(r[1]);
        j_sh_rest.push(r[2]);
        j_opacity.push(r[3]);
        j_scale.push(r[4]);
        j_rotation.push(r[5]);
    }
    PerAttributeJacobian {
        j_position,
        j_dc,
        j_sh_rest,
        j_opacity,
        j_scale,
        j_rotation,
        method: JacobianMethod::GeometricProxyV1,
    }
}

/// View-averaged 2D footprint area of an anisotropic Gaussian with linear
/// scales `(s_x, s_y, s_z)`. We take the geometric mean of the three
/// pairwise products as the proxy:
///
/// ```text
/// area_avg = ((s_x * s_y) * (s_y * s_z) * (s_x * s_z))^(1/3)
///         = (s_x * s_y * s_z)^(2/3)
/// ```
///
/// (The cube of the geometric mean equals (s_x s_y s_z)^2, so the geomean
/// itself is (s_x s_y s_z)^(2/3).) This is the dominant driver of
/// rendering Jacobian magnitude: a splat 4× larger in each axis renders
/// 16× more pixels → 16× higher J. Multiplying by 1/3 power keeps the
/// proxy unitless-ish for downstream Lloyd-Max stability.
fn area_2d_view_avg(scale: [f32; 3]) -> f32 {
    // Importers normalise to linear, non-negative scales but be defensive.
    let sx = scale[0].max(0.0);
    let sy = scale[1].max(0.0);
    let sz = scale[2].max(0.0);
    let volume = sx * sy * sz;
    if volume <= 0.0 {
        return 0.0;
    }
    volume.powf(2.0 / 3.0)
}

/// Anisotropy in `[0, 1]`. Returns `1 - (s_min / s_max)`:
/// - spherical splat (`s_x = s_y = s_z`) → `0`.
/// - needle/pancake (`s_min ≪ s_max`) → close to `1`.
///
/// Rationale: rotating a perfect sphere doesn't change any rendered
/// pixel; rotating a needle by ε moves its endpoints by ε · L. The
/// gsplat-derived `J_rotation` reflects this exactly — heavily anisotropic
/// splats sit in the long tail of `J_rotation` and round-ish ones near
/// zero.
fn scale_anisotropy(scale: [f32; 3]) -> f32 {
    let sx = scale[0].max(0.0);
    let sy = scale[1].max(0.0);
    let sz = scale[2].max(0.0);
    let s_max = sx.max(sy).max(sz);
    if s_max <= 0.0 {
        return 0.0;
    }
    let s_min = sx.min(sy).min(sz);
    (1.0 - s_min / s_max).clamp(0.0, 1.0)
}

/// L2 norm of the DC (zero-band) coefficients only. For `Color::Rgb` it's
/// the L2 of the three RGB components; for `Color::Sh` it's the L2 of the
/// first three coefficients (DC R, G, B).
fn dc_l2_norm(color: &Color) -> f32 {
    match color {
        Color::Rgb(rgb) => {
            let mut acc = 0.0f64;
            for &c in rgb {
                let c = c as f64;
                acc += c * c;
            }
            acc.sqrt() as f32
        }
        Color::Sh { coeffs, .. } => {
            if coeffs.len() < 3 {
                return 0.0;
            }
            let mut acc = 0.0f64;
            for &c in &coeffs[..3] {
                let c = c as f64;
                acc += c * c;
            }
            acc.sqrt() as f32
        }
    }
}

/// L2 norm of the SH bands ≥ 1 (i.e. SH-rest, dropping the DC term).
///
/// For `Color::Rgb` (no SH) we return 0 — this matches the Python ground
/// truth which gets 0 J_sh_rest contribution from any splat that has no
/// SH-rest to perturb. For `Color::Sh { degree=0, .. }` the result is
/// also 0 (degree-0 is DC-only). For degree d, SH coeffs are flattened as
/// `RGB` interleaved per band; the first 3 coeffs are the DC term and
/// the remaining `3 * ((d+1)^2 - 1)` are the SH-rest.
fn sh_rest_l2_norm(color: &Color) -> f32 {
    match color {
        Color::Rgb(_) => 0.0,
        Color::Sh { degree, coeffs } => {
            if *degree == 0 {
                return 0.0;
            }
            // SH-rest starts after the 3 DC coefficients.
            let dc_floats = 3usize;
            if coeffs.len() <= dc_floats {
                return 0.0;
            }
            let mut acc = 0.0f64;
            for &c in &coeffs[dc_floats..] {
                let c = c as f64;
                acc += c * c;
            }
            (acc.sqrt()) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::ir::TemporalMode;
    use catetus_core::CoordinateSystem;

    fn mk_splat(opacity: f32, scale: [f32; 3], sh_rest_val: f32, degree: u8) -> Splat {
        let n_coeffs = 3 * ((degree as usize + 1).pow(2));
        let mut coeffs = vec![0.0f32; n_coeffs];
        // Put a uniform value in every SH-rest coefficient.
        if degree > 0 {
            for c in coeffs.iter_mut().skip(3) {
                *c = sh_rest_val;
            }
        }
        Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale,
            opacity,
            color: if degree == 0 {
                Color::Rgb([0.0; 3])
            } else {
                Color::Sh { degree, coeffs }
            },
        }
    }

    fn mk_splat_full(opacity: f32, scale: [f32; 3], dc: [f32; 3], sh_rest_val: f32, degree: u8) -> Splat {
        let n_coeffs = 3 * ((degree as usize + 1).pow(2));
        let mut coeffs = vec![0.0f32; n_coeffs];
        if n_coeffs >= 3 {
            coeffs[0] = dc[0];
            coeffs[1] = dc[1];
            coeffs[2] = dc[2];
        }
        if degree > 0 {
            for c in coeffs.iter_mut().skip(3) {
                *c = sh_rest_val;
            }
        }
        Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale,
            opacity,
            color: if degree == 0 {
                Color::Rgb(dc)
            } else {
                Color::Sh { degree, coeffs }
            },
        }
    }

    fn mk_scene(splats: Vec<Splat>) -> SplatScene {
        SplatScene {
            splats,
            coordinate_system: CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: TemporalMode::Static,
            lods: None,
            codecgs: None,
        }
    }

    #[test]
    fn alpha_zero_yields_zero_jacobian() {
        let scene = mk_scene(vec![mk_splat(0.0, [1.0, 1.0, 1.0], 1.0, 3)]);
        let j = compute_jacobian(&scene).j_sh_rest;
        assert_eq!(j.len(), 1);
        assert_eq!(j[0], 0.0, "α=0 must zero out the Jacobian");
    }

    #[test]
    fn zero_sh_rest_yields_zero_jacobian() {
        let scene = mk_scene(vec![mk_splat(1.0, [1.0, 1.0, 1.0], 0.0, 3)]);
        let j = compute_jacobian(&scene).j_sh_rest;
        assert_eq!(j[0], 0.0, "‖sh_rest‖=0 must zero out the Jacobian");
    }

    #[test]
    fn larger_scale_yields_larger_jacobian() {
        let scene = mk_scene(vec![
            mk_splat(0.5, [1.0, 1.0, 1.0], 0.3, 3),
            mk_splat(0.5, [2.0, 2.0, 2.0], 0.3, 3),
        ]);
        let j = compute_jacobian(&scene).j_sh_rest;
        // 2× linear scale → 8× volume → 4× area-2d-proxy → 4× J.
        let ratio = j[1] / j[0];
        assert!(
            (ratio - 4.0).abs() < 1e-4,
            "expected 4× Jacobian for 2× scale, got ratio={ratio}"
        );
    }

    #[test]
    fn larger_alpha_yields_proportional_jacobian() {
        let scene = mk_scene(vec![
            mk_splat(0.25, [1.0, 1.0, 1.0], 0.3, 3),
            mk_splat(0.75, [1.0, 1.0, 1.0], 0.3, 3),
        ]);
        let j = compute_jacobian(&scene).j_sh_rest;
        let ratio = j[1] / j[0];
        assert!(
            (ratio - 3.0).abs() < 1e-4,
            "expected 3× Jacobian for 3× α, got ratio={ratio}"
        );
    }

    #[test]
    fn rgb_only_yields_zero_jacobian() {
        // Degree-0 / RGB-only splats have no SH-rest to perturb at all.
        let scene = mk_scene(vec![mk_splat(1.0, [1.0, 1.0, 1.0], 0.0, 0)]);
        let j = compute_jacobian(&scene).j_sh_rest;
        assert_eq!(j[0], 0.0);
    }

    #[test]
    fn output_length_matches_scene() {
        let splats: Vec<_> = (0..1000)
            .map(|i| {
                mk_splat(
                    0.5,
                    [1.0 + (i as f32) * 0.001, 1.0, 1.0],
                    0.3,
                    3,
                )
            })
            .collect();
        let scene = mk_scene(splats);
        let j = compute_jacobian(&scene).j_sh_rest;
        assert_eq!(j.len(), 1000);
        // All positive and finite.
        for (i, &v) in j.iter().enumerate() {
            assert!(v.is_finite(), "j[{i}] = {v} is not finite");
            assert!(v >= 0.0, "j[{i}] = {v} is negative");
        }
    }

    #[test]
    fn per_attr_lengths_match_scene() {
        let splats: Vec<_> = (0..200)
            .map(|i| {
                mk_splat_full(
                    0.5,
                    [1.0 + (i as f32) * 0.001, 1.0, 1.2],
                    [0.4, 0.3, 0.2],
                    0.3,
                    3,
                )
            })
            .collect();
        let scene = mk_scene(splats);
        let j = compute_jacobian_per_attribute(&scene);
        assert_eq!(j.j_position.len(), 200);
        assert_eq!(j.j_dc.len(), 200);
        assert_eq!(j.j_sh_rest.len(), 200);
        assert_eq!(j.j_opacity.len(), 200);
        assert_eq!(j.j_scale.len(), 200);
        assert_eq!(j.j_rotation.len(), 200);
        for ch in [
            &j.j_position,
            &j.j_dc,
            &j.j_sh_rest,
            &j.j_opacity,
            &j.j_scale,
            &j.j_rotation,
        ] {
            for &v in ch {
                assert!(v.is_finite() && v >= 0.0);
            }
        }
    }

    #[test]
    fn per_attr_sh_rest_matches_single_channel_api() {
        // Sanity: PerAttributeJacobian.j_sh_rest should equal the existing
        // single-channel compute_jacobian().j_sh_rest exactly.
        let scene = mk_scene(vec![
            mk_splat_full(0.4, [1.0, 1.0, 1.0], [0.1, 0.2, 0.3], 0.05, 3),
            mk_splat_full(0.9, [1.5, 1.0, 0.8], [0.4, 0.5, 0.6], 0.10, 3),
            mk_splat_full(0.1, [0.5, 0.7, 0.9], [0.2, 0.2, 0.2], 0.02, 3),
        ]);
        let single = compute_jacobian(&scene).j_sh_rest;
        let multi = compute_jacobian_per_attribute(&scene).j_sh_rest;
        assert_eq!(single.len(), multi.len());
        for (a, b) in single.iter().zip(multi.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn per_attr_alpha_extremes_zero_opacity_jacobian() {
        // J_opacity uses an `α · (1-α)` sigmoid-derivative weight: α=0
        // splats are invisible (Δα can't paint them suddenly without
        // also gaining colour) and α=1 splats are saturated. The middle
        // (α=0.5) is most sensitive — see ALGORITHM.md.
        for alpha in [0.0f32, 1.0] {
            let s = mk_splat_full(alpha, [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.1, 3);
            let scene = mk_scene(vec![s]);
            let j = compute_jacobian_per_attribute(&scene);
            assert_eq!(j.j_opacity[0], 0.0, "α={alpha} must zero J_opacity");
        }
        // α=0.5 should produce a strictly positive J_opacity.
        let s = mk_splat_full(0.5, [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.1, 3);
        let scene = mk_scene(vec![s]);
        let j = compute_jacobian_per_attribute(&scene);
        assert!(j.j_opacity[0] > 0.0);
    }

    #[test]
    fn per_attr_alpha_zero_zeros_all_other_channels() {
        // α=0 zeros position/dc/sh_rest/scale/rotation (they all depend
        // on α via "where does the splat actually paint"). j_opacity is
        // handled separately above.
        let s = mk_splat_full(0.0, [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.1, 3);
        let scene = mk_scene(vec![s]);
        let j = compute_jacobian_per_attribute(&scene);
        assert_eq!(j.j_position[0], 0.0);
        assert_eq!(j.j_dc[0], 0.0);
        assert_eq!(j.j_sh_rest[0], 0.0);
        assert_eq!(j.j_scale[0], 0.0);
        assert_eq!(j.j_rotation[0], 0.0);
    }

    #[test]
    fn per_attr_rotation_zero_for_spherical_splat() {
        // A perfectly spherical splat has anisotropy = 0, so j_rotation
        // should be exactly 0 even with non-trivial α and colour.
        let s = mk_splat_full(0.7, [1.5, 1.5, 1.5], [0.3, 0.4, 0.5], 0.1, 3);
        let scene = mk_scene(vec![s]);
        let j = compute_jacobian_per_attribute(&scene);
        assert_eq!(j.j_rotation[0], 0.0);
        // The other channels should still be positive.
        assert!(j.j_position[0] > 0.0);
        assert!(j.j_dc[0] > 0.0);
    }

    #[test]
    fn per_attr_rotation_grows_with_anisotropy() {
        let isotropic = mk_splat_full(0.7, [1.0, 1.0, 1.0], [0.3, 0.4, 0.5], 0.1, 3);
        let elongated = mk_splat_full(0.7, [2.0, 0.5, 0.5], [0.3, 0.4, 0.5], 0.1, 3);
        let scene = mk_scene(vec![isotropic, elongated]);
        let j = compute_jacobian_per_attribute(&scene);
        assert_eq!(j.j_rotation[0], 0.0, "isotropic → no rotation sensitivity");
        assert!(
            j.j_rotation[1] > 0.0,
            "elongated splat must have positive rotation sensitivity"
        );
    }
}
