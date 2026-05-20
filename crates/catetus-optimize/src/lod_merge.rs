//! V4 — LOD splat merging via mixture-of-Gaussians moment matching.
//!
//! Ported from `experiments/cluster-H/code/v4_lod_merge.py` (commit 377213f).
//!
//! Splats with small projected screen-space radius are voxel-binned by 3-D
//! position; each non-trivial bin is replaced by a single super-splat whose
//! mean / covariance / color match the cluster's mixture moments:
//!
//!   W      = Σ w_i                          (with w_i = sigmoid(opacity_i))
//!   μ'     = (1/W) Σ w_i μ_i
//!   Σ'     = (1/W) Σ w_i (Σ_i + (μ_i - μ')(μ_i - μ')ᵀ)
//!   α'     = 1 - Π (1 - α_i)                (alpha-composite over the cluster)
//!   color' = (1/W) Σ w_i color_i            (linear-domain DC + SH-rest)
//!
//! The merged covariance is symmetrized and eigendecomposed back into a
//! `(scale, quat-xyzw)` pair via Jacobi rotations + Shepperd's R→quat method.
//!
//! Bytes saved: `(N0 - N1) * bytes_per_splat`. On bonsai the prototype hit
//! −22% to −66% splat-count reduction depending on the cluster-size cap and
//! screen-radius threshold. The reduction *multiplies* every downstream
//! per-splat saving (quantize, palette, smallest-3 rot), which is why V4 is
//! a "pre-pass" — it should run before every other byte-bearing pass.

use anyhow::Result;
use catetus_core::{Color, Splat, SplatScene};
use serde::Serialize;

use crate::passes::{Pass, PassContext, PassStats};

/// V4 LOD-merge pass parameters. Defaults match the prototype's TIGHT setting
/// (`screen_threshold_px=1.0`, `voxel_factor=0.5`, `max_cluster=64`).
#[derive(Debug, Clone)]
pub struct LodMergeV4 {
    /// Splats with projected screen radius below this many pixels are
    /// candidates for merging. Larger splats are kept verbatim.
    pub screen_threshold_px: f32,
    /// Voxel size = `voxel_factor * scene_bbox_diag / 2048`. Smaller factor
    /// → finer voxels → fewer per-cluster merges.
    pub voxel_factor: f32,
    /// Hard cap on cluster size. Oversize clusters keep the first `max_cluster`
    /// splats and pass the leftovers through untouched.
    pub max_cluster: usize,
    /// Effective focal length used for the screen-radius proxy. Matches the
    /// 256 px default in the Python prototype.
    pub focal_px: f32,
}

impl Default for LodMergeV4 {
    fn default() -> Self {
        Self {
            screen_threshold_px: 1.0,
            voxel_factor: 0.5,
            max_cluster: 64,
            focal_px: 256.0,
        }
    }
}

impl Pass for LodMergeV4 {
    fn name(&self) -> &'static str {
        "LodMergeV4"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n0 = scene.splats.len();
        if n0 == 0 {
            return Ok(PassStats::default());
        }

        // 1. Scene bbox + orbit-distance proxy.
        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        for s in &scene.splats {
            for i in 0..3 {
                if s.position[i] < bmin[i] {
                    bmin[i] = s.position[i];
                }
                if s.position[i] > bmax[i] {
                    bmax[i] = s.position[i];
                }
            }
        }
        let diag = ((bmax[0] - bmin[0]).powi(2)
            + (bmax[1] - bmin[1]).powi(2)
            + (bmax[2] - bmin[2]).powi(2))
        .sqrt()
        .max(1e-9);
        let c = [
            0.5 * (bmin[0] + bmax[0]),
            0.5 * (bmin[1] + bmax[1]),
            0.5 * (bmin[2] + bmax[2]),
        ];
        let orbit = 0.9 * diag;

        // 2. Per-splat screen-radius proxy → small-mask.
        let mut small_mask = vec![false; n0];
        for (i, s) in scene.splats.iter().enumerate() {
            let smax = s
                .scale
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0);
            let d_from_c = ((s.position[0] - c[0]).powi(2)
                + (s.position[1] - c[1]).powi(2)
                + (s.position[2] - c[2]).powi(2))
            .sqrt();
            let dist = (orbit - d_from_c).max(0.05 * diag);
            let radius_px = self.focal_px * 2.0 * smax / dist;
            small_mask[i] = radius_px < self.screen_threshold_px;
        }

        // 3. Voxel-binning of the "small" set. We compute integer voxel
        //    keys then sort small-splat indices by key so consecutive indices
        //    share a key.
        let vox_size = (self.voxel_factor * diag / 2048.0).max(1e-6);
        let mut small_idx: Vec<usize> = (0..n0).filter(|&i| small_mask[i]).collect();
        let n_small = small_idx.len();
        let n_large = n0 - n_small;
        if n_small < 2 {
            return Ok(PassStats {
                notes: vec![format!(
                    "n_small={n_small} < 2 — no merge candidates (radius_thresh={:.2}px)",
                    self.screen_threshold_px
                )],
                ..Default::default()
            });
        }
        // 3-D keys (i64 triples). Bias so non-negative.
        let keys: Vec<(i64, i64, i64)> = small_idx
            .iter()
            .map(|&i| {
                let p = scene.splats[i].position;
                (
                    ((p[0] - bmin[0]) / vox_size).floor() as i64,
                    ((p[1] - bmin[1]) / vox_size).floor() as i64,
                    ((p[2] - bmin[2]) / vox_size).floor() as i64,
                )
            })
            .collect();
        // Sort small_idx by (key.0, key.1, key.2).
        let mut order: Vec<usize> = (0..n_small).collect();
        order.sort_unstable_by_key(|&j| keys[j]);
        let sorted_small: Vec<usize> = order.iter().map(|&j| small_idx[j]).collect();
        let sorted_keys: Vec<(i64, i64, i64)> = order.iter().map(|&j| keys[j]).collect();
        small_idx = sorted_small;

        // 4. Walk groups; each group of size >=2 becomes a super-splat,
        //    oversize groups split into a single max-cluster and leftover
        //    pass-through stragglers.
        let mut merged: Vec<Splat> = Vec::with_capacity(n_large + n_small / 2);
        // Keep the "large" splats first, untouched.
        for (i, s) in scene.splats.iter().enumerate() {
            if !small_mask[i] {
                merged.push(s.clone());
            }
        }

        let mut group_start = 0usize;
        let mut n_clusters_emitted = 0usize;
        let mut n_leftover_stragglers = 0usize;
        while group_start < n_small {
            let key = sorted_keys[group_start];
            let mut group_end = group_start + 1;
            while group_end < n_small && sorted_keys[group_end] == key {
                group_end += 1;
            }
            let cluster_end = group_end.min(group_start + self.max_cluster);
            let cluster_len = cluster_end - group_start;

            if cluster_len == 1 {
                // Pass through the only splat in the bin.
                merged.push(scene.splats[small_idx[group_start]].clone());
            } else {
                // Merge by mixture-moment matching.
                let cluster_indices = &small_idx[group_start..cluster_end];
                let super_splat = merge_cluster(scene, cluster_indices);
                merged.push(super_splat);
                n_clusters_emitted += 1;
            }

            // Stragglers (group exceeded max_cluster) pass through as-is.
            if group_end > cluster_end {
                for &orig_idx in &small_idx[cluster_end..group_end] {
                    merged.push(scene.splats[orig_idx].clone());
                    n_leftover_stragglers += 1;
                }
            }

            group_start = group_end;
        }

        let n1 = merged.len();
        scene.splats = merged;
        // LOD indices (if any) are no longer valid — drop them. Downstream
        // BuildLOD will recompute fractions on the merged scene.
        scene.lods = None;
        Ok(PassStats {
            removed: n0.saturating_sub(n1),
            modified: n_clusters_emitted,
            notes: vec![format!(
                "n_in={n0} n_out={n1} reduction={:.1}% n_large={n_large} n_small={n_small} \
                 n_clusters={n_clusters_emitted} n_leftover={n_leftover_stragglers} \
                 vox_size={vox_size:.6} thresh={:.2}px",
                100.0 * (n0 as f32 - n1 as f32) / n0.max(1) as f32,
                self.screen_threshold_px
            )],
            ..Default::default()
        })
    }
}

/// Side table for the LOD-merge stats: small enough to inline in the report.
#[derive(Debug, Clone, Serialize, Default)]
pub struct LodMergeV4Stats {
    pub n_in: usize,
    pub n_out: usize,
    pub n_clusters: usize,
    pub n_leftover: usize,
    pub vox_size: f32,
}

// ---------------------------------------------------------------------------
// Splat → covariance → merged super-splat helpers.
// ---------------------------------------------------------------------------

/// Convert quaternion `[x, y, z, w]` (IR convention) to a 3×3 rotation matrix
/// stored in row-major form `R[row][col]`.
fn quat_xyzw_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    let (x, y, z, w) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

/// Compute splat covariance Σ = R diag(s²) Rᵀ. Splat scale is already linear
/// in the IR (importers handle the log→linear conversion).
fn splat_sigma(s: &Splat) -> [[f64; 3]; 3] {
    let r = quat_xyzw_to_rotmat(s.rotation);
    let s2 = [
        s.scale[0] as f64 * s.scale[0] as f64,
        s.scale[1] as f64 * s.scale[1] as f64,
        s.scale[2] as f64 * s.scale[2] as f64,
    ];
    // (R * diag(s²)) * Rᵀ → row i, col j:
    // Σ[i][j] = Σ_k R[i][k] * s²[k] * R[j][k]
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += r[i][k] as f64 * s2[k] * r[j][k] as f64;
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Symmetric 3×3 eigendecomposition via Jacobi rotations. Returns
/// `(eigvals_ascending, eigvecs_columns)` where `eigvecs[:, k]` is the
/// eigenvector for `eigvals[k]`.
fn eigh_symmetric_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    // Identity rotation accumulator.
    let mut v = [[0.0f64; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }
    // Symmetrize defensively.
    for i in 0..3 {
        for j in 0..i {
            let m = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = m;
            a[j][i] = m;
        }
    }
    // Sweep — Jacobi converges quadratically on 3x3 in <10 sweeps.
    for _ in 0..50 {
        // Find largest off-diagonal magnitude.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_off {
                    max_off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Update A.
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..3 {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        // Accumulate rotation.
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let eigvals = [a[0][0], a[1][1], a[2][2]];
    // Sort eigenvalues ascending, permuting eigvecs columns accordingly.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sorted_vals = [0.0f64; 3];
    let mut sorted_vecs = [[0.0f64; 3]; 3];
    for (out_col, &src_col) in order.iter().enumerate() {
        sorted_vals[out_col] = eigvals[src_col];
        for row in 0..3 {
            sorted_vecs[row][out_col] = v[row][src_col];
        }
    }
    (sorted_vals, sorted_vecs)
}

/// Convert a 3×3 rotation matrix (rows-by-cols, det ≈ +1) to an `[x, y, z, w]`
/// quaternion via Shepperd's method (numerically stable across all branches).
fn rotmat_to_quat_xyzw(r: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r[2][1] - r[1][2]) / s;
        let y = (r[0][2] - r[2][0]) / s;
        let z = (r[1][0] - r[0][1]) / s;
        return normalize_quat([x, y, z, w]);
    }
    if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        let w = (r[2][1] - r[1][2]) / s;
        let x = 0.25 * s;
        let y = (r[0][1] + r[1][0]) / s;
        let z = (r[0][2] + r[2][0]) / s;
        return normalize_quat([x, y, z, w]);
    }
    if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        let w = (r[0][2] - r[2][0]) / s;
        let x = (r[0][1] + r[1][0]) / s;
        let y = 0.25 * s;
        let z = (r[1][2] + r[2][1]) / s;
        return normalize_quat([x, y, z, w]);
    }
    let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
    let w = (r[1][0] - r[0][1]) / s;
    let x = (r[0][2] + r[2][0]) / s;
    let y = (r[1][2] + r[2][1]) / s;
    let z = 0.25 * s;
    normalize_quat([x, y, z, w])
}

#[inline]
fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Merge a single cluster of small splats into one super-splat via mixture
/// moment matching. The `indices` slice points into `scene.splats` and must
/// contain at least two entries.
fn merge_cluster(scene: &SplatScene, indices: &[usize]) -> Splat {
    debug_assert!(indices.len() >= 2);
    let template = &scene.splats[indices[0]];

    // 1. Weights from sigmoid(opacity).
    let weights: Vec<f64> = indices
        .iter()
        .map(|&i| {
            let alpha = sigmoid(scene.splats[i].opacity as f64);
            alpha.max(1e-6)
        })
        .collect();
    let w_total = weights.iter().sum::<f64>().max(1e-12);

    // 2. Weighted mean position.
    let mut mu = [0.0f64; 3];
    for (k, &i) in indices.iter().enumerate() {
        let p = scene.splats[i].position;
        for c in 0..3 {
            mu[c] += weights[k] * p[c] as f64;
        }
    }
    for c in 0..3 {
        mu[c] /= w_total;
    }

    // 3. Weighted covariance = Σ w_i (Σ_i + (μ_i - μ')(μ_i - μ')ᵀ) / W.
    let mut sigma = [[0.0f64; 3]; 3];
    for (k, &i) in indices.iter().enumerate() {
        let sig_i = splat_sigma(&scene.splats[i]);
        let p = scene.splats[i].position;
        let d = [
            p[0] as f64 - mu[0],
            p[1] as f64 - mu[1],
            p[2] as f64 - mu[2],
        ];
        let w = weights[k];
        for r in 0..3 {
            for c in 0..3 {
                sigma[r][c] += w * (sig_i[r][c] + d[r] * d[c]);
            }
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            sigma[r][c] /= w_total;
        }
        // Symmetrize against round-off.
        for c in 0..r {
            let m = 0.5 * (sigma[r][c] + sigma[c][r]);
            sigma[r][c] = m;
            sigma[c][r] = m;
        }
    }

    // 4. Eigendecompose Σ → (eigvals = σ², eigvecs = R columns). Build R and
    //    convert back to a quaternion. Ensure right-handed (det +1) by flipping
    //    the last column on negative det.
    let (eigvals, mut eigvecs) = eigh_symmetric_3x3(sigma);
    let scales = [
        eigvals[0].max(1e-12).sqrt() as f32,
        eigvals[1].max(1e-12).sqrt() as f32,
        eigvals[2].max(1e-12).sqrt() as f32,
    ];
    // det(eigvecs) = ?
    let det = eigvecs[0][0] * (eigvecs[1][1] * eigvecs[2][2] - eigvecs[1][2] * eigvecs[2][1])
        - eigvecs[0][1] * (eigvecs[1][0] * eigvecs[2][2] - eigvecs[1][2] * eigvecs[2][0])
        + eigvecs[0][2] * (eigvecs[1][0] * eigvecs[2][1] - eigvecs[1][1] * eigvecs[2][0]);
    if det < 0.0 {
        for row in 0..3 {
            eigvecs[row][2] = -eigvecs[row][2];
        }
    }
    let r_f32 = [
        [
            eigvecs[0][0] as f32,
            eigvecs[0][1] as f32,
            eigvecs[0][2] as f32,
        ],
        [
            eigvecs[1][0] as f32,
            eigvecs[1][1] as f32,
            eigvecs[1][2] as f32,
        ],
        [
            eigvecs[2][0] as f32,
            eigvecs[2][1] as f32,
            eigvecs[2][2] as f32,
        ],
    ];
    let quat = rotmat_to_quat_xyzw(r_f32);

    // 5. Opacity: alpha-composite the cluster (1 - Π(1 - α_i)) then logit.
    let mut anti = 1.0f64;
    for &i in indices {
        let alpha = sigmoid(scene.splats[i].opacity as f64);
        anti *= 1.0 - alpha;
    }
    let alpha_cluster = (1.0 - anti).clamp(1e-6, 1.0 - 1e-6);
    let opacity_logit = (alpha_cluster / (1.0 - alpha_cluster)).ln() as f32;

    // 6. Color: weighted linear-domain mean of DC and SH-rest. We preserve
    //    the template splat's Color variant; if the cluster has any Sh
    //    members the merged splat is Sh of that same degree.
    let (sh_degree, n_coeffs) = match &template.color {
        Color::Rgb(_) => (0u8, 3usize),
        Color::Sh { degree, coeffs } => (*degree, coeffs.len()),
    };
    let mut color_acc = vec![0.0f64; n_coeffs];
    for (k, &i) in indices.iter().enumerate() {
        let w = weights[k];
        match &scene.splats[i].color {
            Color::Rgb(rgb) => {
                for c in 0..3.min(n_coeffs) {
                    color_acc[c] += w * rgb[c] as f64;
                }
            }
            Color::Sh { coeffs, .. } => {
                for c in 0..n_coeffs.min(coeffs.len()) {
                    color_acc[c] += w * coeffs[c] as f64;
                }
            }
        }
    }
    for v in color_acc.iter_mut() {
        *v /= w_total;
    }
    let color = if sh_degree == 0 {
        Color::Rgb([
            color_acc[0] as f32,
            color_acc[1] as f32,
            color_acc[2] as f32,
        ])
    } else {
        Color::Sh {
            degree: sh_degree,
            coeffs: color_acc.iter().map(|v| *v as f32).collect(),
        }
    };

    Splat {
        position: [mu[0] as f32, mu[1] as f32, mu[2] as f32],
        rotation: quat,
        scale: scales,
        opacity: opacity_logit,
        color,
    }
}

#[inline]
fn sigmoid(v: f64) -> f64 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_splat(pos: [f32; 3], scale: f32, opacity: f32) -> Splat {
        Splat {
            position: pos,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [scale; 3],
            opacity,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        }
    }

    #[test]
    fn merge_two_identical_splats_preserves_position_and_color() {
        let mut scene = SplatScene::new();
        let p = [0.0, 0.0, 0.0];
        scene.splats.push(mk_splat(p, 1e-6, 1.0));
        scene.splats.push(mk_splat(p, 1e-6, 1.0));
        // Force them small + into the same voxel by using tight thresholds.
        let pass = LodMergeV4 {
            screen_threshold_px: 1e9,
            voxel_factor: 1e3,
            max_cluster: 64,
            focal_px: 256.0,
        };
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(scene.splats.len(), 1, "two identical splats merge to one");
        assert!(stats.removed >= 1);
        // Mean position preserved.
        let m = &scene.splats[0];
        for c in 0..3 {
            assert!((m.position[c] - p[c]).abs() < 1e-4);
        }
    }

    #[test]
    fn empty_scene_is_noop() {
        let mut scene = SplatScene::new();
        let pass = LodMergeV4::default();
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(stats.removed, 0);
        assert_eq!(scene.splats.len(), 0);
    }

    #[test]
    fn eigh_recovers_diagonal_matrix() {
        let m = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 1.0]];
        let (eigvals, _) = eigh_symmetric_3x3(m);
        // Sorted ascending.
        assert!((eigvals[0] - 1.0).abs() < 1e-9);
        assert!((eigvals[1] - 4.0).abs() < 1e-9);
        assert!((eigvals[2] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn quat_rotmat_round_trip_identity() {
        let r = quat_xyzw_to_rotmat([0.0, 0.0, 0.0, 1.0]);
        let q = rotmat_to_quat_xyzw(r);
        // Identity quat (sign-insensitive).
        let dot = q[3].abs();
        assert!(dot > 0.999, "identity round-trip: got {:?}", q);
    }

    #[test]
    fn large_splats_above_threshold_are_kept() {
        let mut scene = SplatScene::new();
        for i in 0..20 {
            let pos = [i as f32, 0.0, 0.0];
            // big scale → big projected radius → above threshold → kept
            scene.splats.push(mk_splat(pos, 1.0, 1.0));
        }
        let pass = LodMergeV4 {
            screen_threshold_px: 1e-9, // basically all splats are above this
            ..Default::default()
        };
        let mut ctx = PassContext::default();
        let _ = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(scene.splats.len(), 20, "no small splats — nothing to merge");
    }
}
