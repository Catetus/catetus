//! Individual optimization passes. Each pass is deterministic given a
//! `PassContext` seed and the input scene.

use anyhow::Result;
use serde::Serialize;
use splatforge_core::{Color, LodLevel, SplatScene};

/// Per-pass statistics returned by `Pass::run`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PassStats {
    /// Number of splats removed.
    pub removed: usize,
    /// Number of splats modified in place.
    pub modified: usize,
    /// Synthetic duration in milliseconds (not wall clock — runtime injects).
    pub duration_ms: u64,
    /// Optional notes (e.g. for stub passes).
    pub notes: Vec<String>,
}

/// Mutable context carried through a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct PassContext {
    /// Deterministic seed for any pass that needs pseudo-randomness.
    pub seed: u64,
}

/// Pass trait: every optimization pass implements this.
pub trait Pass {
    /// Stable identifier for the pass (used in reports).
    fn name(&self) -> &'static str;

    /// Mutate the scene in place.
    fn run(&self, scene: &mut SplatScene, ctx: &mut PassContext) -> Result<PassStats>;
}

/// Drop splats whose position, rotation, scale, opacity, or DC color contains a
/// NaN or Inf value.
#[derive(Debug, Default, Clone)]
pub struct RemoveInvalidSplats;

impl Pass for RemoveInvalidSplats {
    fn name(&self) -> &'static str {
        "RemoveInvalidSplats"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let before = scene.splats.len();
        scene.splats.retain(|s| {
            s.position.iter().all(|v| v.is_finite())
                && s.rotation.iter().all(|v| v.is_finite())
                && s.scale.iter().all(|v| v.is_finite())
                && s.opacity.is_finite()
                && match &s.color {
                    Color::Rgb(c) => c.iter().all(|v| v.is_finite()),
                    Color::Sh { coeffs, .. } => coeffs.iter().all(|v| v.is_finite()),
                }
        });
        Ok(PassStats {
            removed: before - scene.splats.len(),
            ..Default::default()
        })
    }
}

/// Drop splats whose opacity is below a threshold.
#[derive(Debug, Clone)]
pub struct OpacityPrune {
    /// Opacity threshold; splats `<= threshold` are removed.
    pub threshold: f32,
}

impl Default for OpacityPrune {
    fn default() -> Self {
        Self { threshold: 0.01 }
    }
}

impl Pass for OpacityPrune {
    fn name(&self) -> &'static str {
        "OpacityPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let before = scene.splats.len();
        let t = self.threshold;
        scene.splats.retain(|s| s.opacity > t);
        Ok(PassStats {
            removed: before - scene.splats.len(),
            ..Default::default()
        })
    }
}

/// Drop splats farther than `dist_sigma * σ` from the scene centroid.
#[derive(Debug, Clone)]
pub struct FloaterPrune {
    /// Reserved (k-nearest neighbor count) — currently unused in stub.
    pub k_neighbors: usize,
    /// Sigma multiplier; defaults to 5.
    pub dist_sigma: f32,
}

impl Default for FloaterPrune {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            dist_sigma: 5.0,
        }
    }
}

impl Pass for FloaterPrune {
    fn name(&self) -> &'static str {
        "FloaterPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.len() < 4 {
            return Ok(PassStats::default());
        }
        let n = scene.splats.len() as f64;
        let mut centroid = [0.0f64; 3];
        for s in &scene.splats {
            for (i, c) in centroid.iter_mut().enumerate() {
                *c += s.position[i] as f64;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n;
        }
        let mut sum_sq = 0.0f64;
        for s in &scene.splats {
            let d2 = (0..3)
                .map(|i| {
                    let d = s.position[i] as f64 - centroid[i];
                    d * d
                })
                .sum::<f64>();
            sum_sq += d2;
        }
        let sigma = (sum_sq / n).sqrt();
        let max = (self.dist_sigma as f64) * sigma;
        let max_sq = max * max;

        let before = scene.splats.len();
        scene.splats.retain(|s| {
            let d2 = (0..3)
                .map(|i| {
                    let d = s.position[i] as f64 - centroid[i];
                    d * d
                })
                .sum::<f64>();
            d2 <= max_sq
        });
        Ok(PassStats {
            removed: before - scene.splats.len(),
            ..Default::default()
        })
    }
}

fn quantize_f32(v: f32, bits: u8, min: f32, max: f32) -> f32 {
    let steps = ((1u32 << bits.min(31)) - 1) as f32;
    let span = (max - min).max(1e-9);
    let t = ((v - min) / span).clamp(0.0, 1.0);
    let q = (t * steps).round() / steps;
    min + q * span
}

/// In-place position quantization (preserves layout).
#[derive(Debug, Clone)]
pub struct QuantizePosition {
    /// Bits per axis component.
    pub bits: u8,
}

impl Default for QuantizePosition {
    fn default() -> Self {
        Self { bits: 16 }
    }
}

impl Pass for QuantizePosition {
    fn name(&self) -> &'static str {
        "QuantizePosition"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.is_empty() {
            return Ok(PassStats::default());
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for s in &scene.splats {
            for i in 0..3 {
                if s.position[i] < mn[i] {
                    mn[i] = s.position[i];
                }
                if s.position[i] > mx[i] {
                    mx[i] = s.position[i];
                }
            }
        }
        let mut modified = 0usize;
        for s in &mut scene.splats {
            for i in 0..3 {
                let q = quantize_f32(s.position[i], self.bits, mn[i], mx[i]);
                if (q - s.position[i]).abs() > f32::EPSILON {
                    modified += 1;
                }
                s.position[i] = q;
            }
        }
        Ok(PassStats {
            modified,
            ..Default::default()
        })
    }
}

/// In-place scale quantization.
#[derive(Debug, Clone)]
pub struct QuantizeScale {
    /// Bits per axis component.
    pub bits: u8,
}

impl Default for QuantizeScale {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

impl Pass for QuantizeScale {
    fn name(&self) -> &'static str {
        "QuantizeScale"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.is_empty() {
            return Ok(PassStats::default());
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for s in &scene.splats {
            for i in 0..3 {
                if s.scale[i] < mn[i] {
                    mn[i] = s.scale[i];
                }
                if s.scale[i] > mx[i] {
                    mx[i] = s.scale[i];
                }
            }
        }
        let mut modified = 0usize;
        for s in &mut scene.splats {
            for i in 0..3 {
                let q = quantize_f32(s.scale[i], self.bits, mn[i], mx[i]);
                if (q - s.scale[i]).abs() > f32::EPSILON {
                    modified += 1;
                }
                s.scale[i] = q;
            }
        }
        Ok(PassStats {
            modified,
            ..Default::default()
        })
    }
}

/// In-place rotation quantization (per-component on the unit quaternion).
#[derive(Debug, Clone)]
pub struct QuantizeRotation {
    /// Bits per component.
    pub bits: u8,
}

impl Default for QuantizeRotation {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

impl Pass for QuantizeRotation {
    fn name(&self) -> &'static str {
        "QuantizeRotation"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let mut modified = 0usize;
        for s in &mut scene.splats {
            for i in 0..4 {
                let q = quantize_f32(s.rotation[i], self.bits, -1.0, 1.0);
                if (q - s.rotation[i]).abs() > f32::EPSILON {
                    modified += 1;
                }
                s.rotation[i] = q;
            }
            // re-normalize after quantization
            let n = (s.rotation.iter().map(|v| v * v).sum::<f32>()).sqrt();
            if n > 0.0 {
                for v in &mut s.rotation {
                    *v /= n;
                }
            }
        }
        Ok(PassStats {
            modified,
            ..Default::default()
        })
    }
}

/// Truncate SH coefficients to the requested degree.
#[derive(Debug, Clone, Default)]
pub struct ReduceSHDegree {
    /// Target SH degree (0 collapses to plain RGB).
    pub target_degree: u8,
}

impl Pass for ReduceSHDegree {
    fn name(&self) -> &'static str {
        "ReduceSHDegree"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let mut modified = 0usize;
        for s in &mut scene.splats {
            let new_color = match &s.color {
                Color::Rgb(_) => continue,
                Color::Sh { degree, coeffs } => {
                    if *degree <= self.target_degree {
                        continue;
                    }
                    modified += 1;
                    if self.target_degree == 0 {
                        let dc = [
                            coeffs.first().copied().unwrap_or(0.0),
                            coeffs.get(1).copied().unwrap_or(0.0),
                            coeffs.get(2).copied().unwrap_or(0.0),
                        ];
                        Color::Rgb(dc)
                    } else {
                        let bands = (self.target_degree as usize + 1).pow(2);
                        let needed = 3 * bands;
                        let truncated = coeffs.iter().copied().take(needed).collect();
                        Color::Sh {
                            degree: self.target_degree,
                            coeffs: truncated,
                        }
                    }
                }
            };
            s.color = new_color;
        }
        Ok(PassStats {
            modified,
            ..Default::default()
        })
    }
}

/// Sort splats by 30-bit interleaved Morton code of normalized positions.
#[derive(Debug, Default, Clone)]
pub struct MortonSort;

fn morton_code(p: [f32; 3], mn: [f32; 3], mx: [f32; 3]) -> u64 {
    let mut out = 0u64;
    for axis in 0..3 {
        let span = (mx[axis] - mn[axis]).max(1e-9);
        let t = ((p[axis] - mn[axis]) / span).clamp(0.0, 1.0);
        let q = (t * 1023.0).round() as u32; // 10 bits per axis
        for bit in 0..10 {
            let b = (q >> bit) & 1;
            out |= (b as u64) << (3 * bit + axis);
        }
    }
    out
}

impl Pass for MortonSort {
    fn name(&self) -> &'static str {
        "MortonSort"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.is_empty() {
            return Ok(PassStats::default());
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for s in &scene.splats {
            for i in 0..3 {
                if s.position[i] < mn[i] {
                    mn[i] = s.position[i];
                }
                if s.position[i] > mx[i] {
                    mx[i] = s.position[i];
                }
            }
        }
        let mut indexed: Vec<(u64, usize)> = scene
            .splats
            .iter()
            .enumerate()
            .map(|(i, s)| (morton_code(s.position, mn, mx), i))
            .collect();
        indexed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let order: Vec<usize> = indexed.iter().map(|x| x.1).collect();
        let original = std::mem::take(&mut scene.splats);
        scene.splats = order.iter().map(|&i| original[i].clone()).collect();
        Ok(PassStats {
            modified: scene.splats.len(),
            ..Default::default()
        })
    }
}

/// Build subsampled LOD index lists from the (presumably morton-sorted) scene.
///
/// For each `f` in `levels`, an `LodLevel { fraction: f, indices }` is appended
/// to `scene.lods`. `indices` references the main `scene.splats` array; the
/// first level (LOD0) covering the full scene is always inserted.
#[derive(Debug, Default, Clone)]
pub struct BuildLOD {
    /// LOD splat fractions in (0, 1]. e.g. `[0.5, 0.25]` produces two
    /// additional levels in addition to LOD0.
    pub levels: Vec<f32>,
}

impl Pass for BuildLOD {
    fn name(&self) -> &'static str {
        "BuildLOD"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        let mut notes = Vec::new();
        let mut levels: Vec<LodLevel> = Vec::with_capacity(self.levels.len() + 1);
        // LOD0: full scene.
        let lod0_indices: Vec<u32> = (0..n as u32).collect();
        notes.push(format!("LOD0 count={n}"));
        levels.push(LodLevel {
            fraction: 1.0,
            indices: lod0_indices,
        });
        for (i, frac) in self.levels.iter().copied().enumerate() {
            // Skip invalid fractions; record a note.
            if !(frac > 0.0 && frac <= 1.0) || n == 0 {
                notes.push(format!("LOD{} skipped (fraction={frac})", i + 1));
                continue;
            }
            // Target count rounded; "every 1/f-th splat".
            let stride = (1.0_f32 / frac).max(1.0).round() as usize;
            let mut indices: Vec<u32> = Vec::with_capacity(n / stride + 1);
            let mut k = 0usize;
            while k < n {
                indices.push(k as u32);
                k += stride;
            }
            notes.push(format!(
                "LOD{} fraction={frac} stride={stride} count={}",
                i + 1,
                indices.len()
            ));
            levels.push(LodLevel {
                fraction: frac,
                indices,
            });
        }
        scene.lods = Some(levels);
        Ok(PassStats {
            notes,
            ..Default::default()
        })
    }
}

/// Experimental object-aware pruning.
///
/// Heuristic: compute a coarse k-NN density per splat (`k = 8`). Splats whose
/// density rank falls in the lowest decile **and** whose opacity is below
/// `0.1` are pruned, unless they carry a semantic label listed in
/// `protect_labels`.
#[derive(Debug, Default, Clone)]
pub struct ObjectAwarePruneExperimental {
    /// Semantic labels exempt from pruning.
    pub protect_labels: Vec<String>,
}

fn squared_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

impl Pass for ObjectAwarePruneExperimental {
    fn name(&self) -> &'static str {
        "ObjectAwarePruneExperimental"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        if n < 10 {
            return Ok(PassStats {
                notes: vec!["ObjectAwarePrune: scene too small, skipped".to_string()],
                ..Default::default()
            });
        }
        let k = 8usize.min(n - 1);

        // Density score = sum of squared distances to k nearest neighbors.
        // Lower score = denser; higher score = sparser. We rank by score.
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, si) in scene.splats.iter().enumerate() {
            let mut dists: Vec<f32> = scene
                .splats
                .iter()
                .enumerate()
                .filter_map(|(j, sj)| {
                    if i == j {
                        None
                    } else {
                        Some(squared_dist(si.position, sj.position))
                    }
                })
                .collect();
            // Partial sort: select k smallest deterministically.
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let sum: f32 = dists.iter().take(k).copied().sum();
            scores.push((i, sum));
        }

        // Determine the density threshold: lowest-decile means SPARSEST 10%
        // (largest scores). Compute rank from descending sort.
        let mut by_score = scores.clone();
        by_score.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let decile = n.div_ceil(10);
        let sparsest_set: std::collections::HashSet<usize> =
            by_score.iter().take(decile).map(|(i, _)| *i).collect();
        let threshold_score = by_score
            .get(decile.saturating_sub(1))
            .map(|(_, s)| *s)
            .unwrap_or(f32::INFINITY);

        let labels = scene.semantic_labels.clone();
        let protect = &self.protect_labels;

        let before = scene.splats.len();
        // Use enumerate-retain with shared index. Because `Vec::retain` doesn't
        // pass an index, build a keep mask first.
        let keep: Vec<bool> = (0..n)
            .map(|i| {
                let is_sparse = sparsest_set.contains(&i);
                let low_opacity = scene.splats[i].opacity < 0.1;
                let protected = labels
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .map(|lbl| protect.iter().any(|p| p == &lbl.0))
                    .unwrap_or(false);
                let prune = is_sparse && low_opacity && !protected;
                !prune
            })
            .collect();

        // Rebuild splats + parallel labels.
        let mut new_splats = Vec::with_capacity(n);
        let mut new_labels: Option<Vec<_>> = labels.as_ref().map(|_| Vec::with_capacity(n));
        for (i, splat) in scene.splats.iter().enumerate() {
            if keep[i] {
                new_splats.push(splat.clone());
                if let (Some(src), Some(dst)) = (labels.as_ref(), new_labels.as_mut()) {
                    if let Some(l) = src.get(i) {
                        dst.push(l.clone());
                    }
                }
            }
        }
        scene.splats = new_splats;
        scene.semantic_labels = new_labels;
        let removed = before - scene.splats.len();

        Ok(PassStats {
            removed,
            notes: vec![format!(
                "density_threshold_score={threshold_score:.6}, decile_size={decile}"
            )],
            ..Default::default()
        })
    }
}
