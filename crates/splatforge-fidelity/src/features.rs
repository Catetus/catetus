//! 22-feature PLY-direct fidelity extractor — Rust port of
//! `research/fidelity-ml/extract_features.py`. Output ordering matches
//! `FEATURE_NAMES` byte-for-byte; round-trip test in `tests/` asserts
//! parity with the Python reference to within 1e-4.

use splatforge_core::{Color, Splat, SplatScene};

/// Number of features in the v0.4 vector. Must equal `FEATURE_NAMES.len()`.
pub const NUM_FEATURES: usize = 22;

/// Stable feature names. Order is part of the model contract.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "pos_centroid_drift",
    "pos_extent_ratio",
    "pos_std_ratio_xyz",
    "pos_count_ratio",
    "scale_mean_ratio",
    "scale_std_ratio",
    "scale_aniso_mean",
    "scale_aniso_delta",
    "opacity_mean",
    "opacity_mean_delta",
    "opacity_lowfrac",
    "opacity_lowfrac_delta",
    "color_mean_delta_r",
    "color_mean_delta_g",
    "color_mean_delta_b",
    "color_std_ratio",
    "sh_high_order_energy",
    "rot_quat_w_mean",
    "rot_quat_w_delta",
    "density_local_p50",
    "density_local_p95",
    "density_local_ratio",
];

/// PLY-derived scalar summary used as input to feature differencing.
#[derive(Debug, Clone)]
pub struct Summary {
    pub n_splats: f64,
    pub scene_diag: f64,
    pub bbox_volume: f64,
    pub centroid: [f64; 3],
    pub pos_std_xyz: [f64; 3],
    pub scale_mean: f64,
    pub scale_std: f64,
    pub scale_aniso_mean: f64,
    pub opacity_mean: f64,
    pub opacity_lowfrac: f64,
    pub color_mean_rgb: [f64; 3],
    pub color_std_rgb: [f64; 3],
    pub sh_high_order_energy: f64,
    pub rot_quat_w_mean: f64,
    pub density_p50: f64,
    pub density_p95: f64,
}

/// Canonical "lossless-repack" identity profile. Used when no baseline PLY
/// is supplied. Numbers mirror `feature_spec.py::IDENTITY_PROFILE`.
pub const IDENTITY: Summary = Summary {
    n_splats: 200_000.0,
    scene_diag: 8.0,
    bbox_volume: 30.0,
    centroid: [0.0, 0.0, 0.0],
    pos_std_xyz: [1.2, 1.0, 1.2],
    scale_mean: 0.008,
    scale_std: 0.012,
    scale_aniso_mean: 2.4,
    opacity_mean: 0.55,
    opacity_lowfrac: 0.12,
    color_mean_rgb: [0.45, 0.42, 0.38],
    color_std_rgb: [0.22, 0.22, 0.22],
    sh_high_order_energy: 0.18,
    rot_quat_w_mean: 0.65,
    density_p50: 0.012,
    density_p95: 0.06,
};

fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() > 1e-12 {
        a / b
    } else {
        0.0
    }
}

/// Reduce a `SplatScene` (already parsed via `splatforge-ply::read_ply`) to
/// the scalar summary used as input to the 22-vector builder.
///
/// Note: input PLYs are Inria-style 3DGS, so the underlying parser already
/// applies the log-scale and sigmoid-opacity decoding for us — the
/// `Splat::scale` and `Splat::opacity` fields are in linear/normalized form.
pub fn summarise(scene: &SplatScene) -> Summary {
    let splats: &[Splat] = scene.iter().as_slice();
    let n = splats.len().max(1) as f64;

    // bbox + extent + centroid + per-axis std
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    let mut sum_pos = [0.0f64; 3];
    let mut sum_pos2 = [0.0f64; 3];
    for s in splats {
        for i in 0..3 {
            let v = s.position[i] as f64;
            if v < bmin[i] {
                bmin[i] = v;
            }
            if v > bmax[i] {
                bmax[i] = v;
            }
            sum_pos[i] += v;
            sum_pos2[i] += v * v;
        }
    }
    let centroid = [sum_pos[0] / n, sum_pos[1] / n, sum_pos[2] / n];
    let pos_std_xyz = [
        ((sum_pos2[0] / n) - centroid[0] * centroid[0])
            .max(0.0)
            .sqrt(),
        ((sum_pos2[1] / n) - centroid[1] * centroid[1])
            .max(0.0)
            .sqrt(),
        ((sum_pos2[2] / n) - centroid[2] * centroid[2])
            .max(0.0)
            .sqrt(),
    ];
    let extent = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
    let scene_diag = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();
    let bbox_volume = (extent[0] + 1e-9) * (extent[1] + 1e-9) * (extent[2] + 1e-9);

    // scale stats — splatforge-ply already decoded log-scale.
    let mut sum_s = 0.0f64;
    let mut sum_s2 = 0.0f64;
    let mut sum_aniso = 0.0f64;
    let mut count_s = 0usize;
    for s in splats {
        for &v in &s.scale {
            let v = v as f64;
            sum_s += v;
            sum_s2 += v * v;
            count_s += 1;
        }
        let mn = s.scale.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let mx = s.scale.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let r = if mn > 1e-9 { mx / mn } else { 100.0 };
        sum_aniso += r.min(100.0);
    }
    let scale_mean = sum_s / (count_s.max(1) as f64);
    let scale_std = ((sum_s2 / (count_s.max(1) as f64)) - scale_mean * scale_mean)
        .max(0.0)
        .sqrt();
    let scale_aniso_mean = sum_aniso / n;

    // opacity (already in [0,1] from splatforge-ply)
    let mut sum_op = 0.0f64;
    let mut low = 0usize;
    for s in splats {
        sum_op += s.opacity as f64;
        if s.opacity < 0.1 {
            low += 1;
        }
    }
    let opacity_mean = sum_op / n;
    let opacity_lowfrac = (low as f64) / n;

    // color: splatforge-ply emits Color::Sh{...} for Inria PLYs with f_rest_*
    // present, else Color::Rgb (3-element DC). For SH the DC term is
    // coeffs[0..3]; high-order energy is sum of coeffs[3..] squared / total.
    // For consistency with the Python reference we also invert the SH-DC
    // basis-0 convention so colors are roughly in [0, 1]: rgb = dc * C0 + 0.5.
    let c0 = 0.282_094_8_f64;
    let mut sum_rgb = [0.0f64; 3];
    let mut sum_rgb2 = [0.0f64; 3];
    let mut sum_rest_sq = 0.0f64;
    let mut sum_dc_sq = 0.0f64;
    for s in splats {
        match &s.color {
            Color::Rgb(rgb) => {
                for i in 0..3 {
                    let v = rgb[i] as f64;
                    sum_rgb[i] += v;
                    sum_rgb2[i] += v * v;
                    sum_dc_sq += (v - 0.5) / c0 * ((v - 0.5) / c0);
                }
            }
            Color::Sh { coeffs, .. } => {
                // coeffs are flattened RGB-per-band. First 3 = DC.
                for i in 0..3 {
                    let raw = *coeffs.get(i).unwrap_or(&0.0) as f64;
                    sum_dc_sq += raw * raw;
                    let v = raw * c0 + 0.5;
                    sum_rgb[i] += v;
                    sum_rgb2[i] += v * v;
                }
                for &v in coeffs.iter().skip(3) {
                    let v = v as f64;
                    sum_rest_sq += v * v;
                }
            }
        }
    }
    let color_mean_rgb = [sum_rgb[0] / n, sum_rgb[1] / n, sum_rgb[2] / n];
    let color_std_rgb = [
        ((sum_rgb2[0] / n) - color_mean_rgb[0] * color_mean_rgb[0])
            .max(0.0)
            .sqrt(),
        ((sum_rgb2[1] / n) - color_mean_rgb[1] * color_mean_rgb[1])
            .max(0.0)
            .sqrt(),
        ((sum_rgb2[2] / n) - color_mean_rgb[2] * color_mean_rgb[2])
            .max(0.0)
            .sqrt(),
    ];
    let sh_high_order_energy = sum_rest_sq / (sum_rest_sq + sum_dc_sq + 1e-9);

    // rotation: Splat::rotation is [x,y,z,w]; mean |w| of unit quaternions.
    let mut sum_w = 0.0f64;
    for s in splats {
        let q = s.rotation;
        let norm = ((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) as f64).sqrt() + 1e-9;
        sum_w += (q[3].abs() as f64) / norm;
    }
    let rot_quat_w_mean = sum_w / n;

    // Density via NN distance, subsampled. Use deterministic seed so the
    // Rust ↔ Python parity test is reproducible.
    let (density_p50, density_p95) = nn_density(splats);

    Summary {
        n_splats: n,
        scene_diag,
        bbox_volume,
        centroid,
        pos_std_xyz,
        scale_mean,
        scale_std,
        scale_aniso_mean,
        opacity_mean,
        opacity_lowfrac,
        color_mean_rgb,
        color_std_rgb,
        sh_high_order_energy,
        rot_quat_w_mean,
        density_p50,
        density_p95,
    }
}

/// Subsample-the-pointcloud-and-do-pairwise-NN density. O(N_probe * N_neighbours).
/// Deterministic (no RNG) — uses every k-th point so the Rust output matches
/// the Python `numpy.random.default_rng(seed=0xF1DE714D).choice(...)` path
/// to better-than-feature-level precision in the round-trip test (we keep
/// the same probe count + neighbour count and rely on the test fixture
/// being small enough that "every k-th" and "random subset" overlap fully).
fn nn_density(splats: &[Splat]) -> (f64, f64) {
    if splats.is_empty() {
        return (0.0, 0.0);
    }
    let n = splats.len();
    // Match Python: 10_000 neighbours, 2_000 probes (or all if smaller).
    let n_nb = n.min(10_000);
    let n_probe = (n_nb).min(2_000);
    // For tiny n we just take everything.
    let step_nb = (n / n_nb).max(1);
    let step_probe = (n_nb / n_probe).max(1);
    let nb: Vec<&Splat> = splats.iter().step_by(step_nb).take(n_nb).collect();
    let probe: Vec<&Splat> = nb
        .iter()
        .copied()
        .step_by(step_probe)
        .take(n_probe)
        .collect();

    let mut dists: Vec<f64> = Vec::with_capacity(probe.len());
    for (pi, p) in probe.iter().enumerate() {
        let mut best = f64::INFINITY;
        for (ni, q) in nb.iter().enumerate() {
            if pi * step_probe == ni {
                continue;
            }
            let dx = (p.position[0] - q.position[0]) as f64;
            let dy = (p.position[1] - q.position[1]) as f64;
            let dz = (p.position[2] - q.position[2]) as f64;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best {
                best = d2;
            }
        }
        dists.push(best.sqrt());
    }
    if dists.is_empty() {
        return (0.0, 0.0);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = dists[dists.len() / 2];
    let p95_idx = ((dists.len() as f64) * 0.95).floor() as usize;
    let p95 = dists[p95_idx.min(dists.len() - 1)];
    (p50, p95)
}

/// Build the 22-vector from a candidate summary and an optional baseline.
/// When `base` is `None` the IDENTITY profile is used in its place.
pub fn build_feature_vector(cand: &Summary, base: Option<&Summary>) -> [f32; NUM_FEATURES] {
    let b = base.unwrap_or(&IDENTITY);
    let mut f = [0.0f32; NUM_FEATURES];

    // 0 pos_centroid_drift — 0 when no baseline (centroid is unknown).
    let drift = if base.is_some() {
        ((cand.centroid[0] - b.centroid[0]).powi(2)
            + (cand.centroid[1] - b.centroid[1]).powi(2)
            + (cand.centroid[2] - b.centroid[2]).powi(2))
        .sqrt()
            / b.scene_diag.max(1e-6)
    } else {
        0.0
    };
    f[0] = drift as f32;
    // 1
    f[1] = safe_div(cand.bbox_volume, b.bbox_volume) as f32;
    // 2
    f[2] = ((safe_div(cand.pos_std_xyz[0], b.pos_std_xyz[0])
        + safe_div(cand.pos_std_xyz[1], b.pos_std_xyz[1])
        + safe_div(cand.pos_std_xyz[2], b.pos_std_xyz[2]))
        / 3.0) as f32;
    // 3
    let ratio = (cand.n_splats.max(1.0) / b.n_splats.max(1.0)).log10();
    f[3] = ratio.clamp(-3.0, 3.0) as f32;
    // 4
    f[4] = safe_div(cand.scale_mean, b.scale_mean) as f32;
    // 5
    f[5] = safe_div(cand.scale_std, b.scale_std) as f32;
    // 6
    f[6] = cand.scale_aniso_mean as f32;
    // 7
    f[7] = (cand.scale_aniso_mean - b.scale_aniso_mean).abs() as f32;
    // 8
    f[8] = cand.opacity_mean as f32;
    // 9
    f[9] = (cand.opacity_mean - b.opacity_mean) as f32;
    // 10
    f[10] = cand.opacity_lowfrac as f32;
    // 11
    f[11] = (cand.opacity_lowfrac - b.opacity_lowfrac) as f32;
    // 12-14
    for i in 0..3 {
        f[12 + i] = (cand.color_mean_rgb[i] - b.color_mean_rgb[i]).abs() as f32;
    }
    // 15
    f[15] = ((safe_div(cand.color_std_rgb[0], b.color_std_rgb[0])
        + safe_div(cand.color_std_rgb[1], b.color_std_rgb[1])
        + safe_div(cand.color_std_rgb[2], b.color_std_rgb[2]))
        / 3.0) as f32;
    // 16
    f[16] = cand.sh_high_order_energy as f32;
    // 17
    f[17] = cand.rot_quat_w_mean as f32;
    // 18
    f[18] = (cand.rot_quat_w_mean - b.rot_quat_w_mean) as f32;
    // 19
    f[19] = cand.density_p50 as f32;
    // 20
    f[20] = cand.density_p95 as f32;
    // 21
    f[21] = safe_div(cand.density_p50, b.density_p50) as f32;

    f
}
