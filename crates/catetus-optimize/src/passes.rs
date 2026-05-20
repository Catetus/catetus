//! Individual optimization passes. Each pass is deterministic given a
//! `PassContext` seed and the input scene.

use anyhow::Result;
use catetus_core::{Color, LodLevel, SplatScene};
use serde::Serialize;
use std::sync::Mutex;

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
    /// Optional per-splat SH-rest rendering Jacobian. When present, length
    /// MUST equal `scene.splats.len()` and is kept in lock-step by passes
    /// that reorder or filter splats (currently `RemoveInvalidSplats` and
    /// `MortonSort`). `VQPaletteShRest` consumes this as the per-splat
    /// weight `w_i` in the weighted-Lloyd centroid update — the render-space
    /// codec from `experiments/render-space-lloyd-max/RESULT.md` (V1: +11.94
    /// dB SH-rest-only at the same byte budget vs the unweighted baseline).
    /// When `None`, `VQPaletteShRest` falls back to the original unweighted
    /// arithmetic mean (no behaviour change).
    pub sh_rest_weights: Option<Vec<f32>>,
    /// Optional per-splat "origin index": the row this splat had in the
    /// original input file BEFORE any pipeline pass ran. The CLI seeds it
    /// to `0..N` for V5.2 sidecar runs; `RemoveInvalidSplats` and
    /// `MortonSort` keep it in lockstep (drop / permute exactly the same
    /// entries as `scene.splats`). Used by the V5.2 joint-tail sidecar
    /// encoder to map post-pipeline splat indices back to the GT PLY rows
    /// so it can subtract GT - recon residuals. When `None`, no tracking
    /// is done — structural passes short-circuit.
    pub splat_origin_idx: Option<Vec<u32>>,
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
    fn run(&self, scene: &mut SplatScene, ctx: &mut PassContext) -> Result<PassStats> {
        let before = scene.splats.len();
        // Compute keep-mask so we can drop the corresponding entries from any
        // PassContext-carried per-splat side-channel (e.g. SH-rest Jacobian
        // weights used by VQPaletteShRest in render-space-Lloyd-Max mode).
        let keep: Vec<bool> = scene
            .splats
            .iter()
            .map(|s| {
                s.position.iter().all(|v| v.is_finite())
                    && s.rotation.iter().all(|v| v.is_finite())
                    && s.scale.iter().all(|v| v.is_finite())
                    && s.opacity.is_finite()
                    && match &s.color {
                        Color::Rgb(c) => c.iter().all(|v| v.is_finite()),
                        Color::Sh { coeffs, .. } => coeffs.iter().all(|v| v.is_finite()),
                    }
            })
            .collect();
        let mut idx = 0;
        scene.splats.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
        if let Some(w) = ctx.sh_rest_weights.as_mut() {
            if w.len() == keep.len() {
                let mut i = 0;
                w.retain(|_| {
                    let k = keep[i];
                    i += 1;
                    k
                });
            }
        }
        if let Some(o) = ctx.splat_origin_idx.as_mut() {
            if o.len() == keep.len() {
                let mut i = 0;
                o.retain(|_| {
                    let k = keep[i];
                    i += 1;
                    k
                });
            }
        }
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

/// Rate-distortion prune: keep the top `target_ratio` fraction of splats
/// ranked by a cheap distortion proxy, dropping the rest.
///
/// The proxy approximates each splat's pixel-domain contribution:
///
///   distortion = opacity * max(scale)^2 * (1 + mean(SH_rest^2))
///
/// where `opacity` is linear `[0,1]`, `scale` is linear world units (already
/// `exp`'d from the INRIA log-space on import), and `SH_rest` is the
/// degree-1+ band energy (zero for `Color::Rgb`, full coeffs for
/// `Color::Sh`). The proxy is intentionally cheap: O(n) and pass-local, so
/// the pipeline cost is dominated by the prune-sort, not the metric.
///
/// Bytes saved per splat is roughly constant under the downstream
/// `web-mobile` quantization (~22 bytes/splat: pos15 + scale8 + rot8 +
/// SH-degree-0), so distortion alone is a sufficient ordering for an
/// equal-byte comparison vs `OpacityPrune` at the same keep-rate.
///
/// `target_ratio` is the *fraction of splats to keep* (0.0..=1.0). 1.0 is
/// a no-op; 0.0 empties the scene. Explicit keep-rate (rather than a
/// slope) makes the preset configuration straightforward.
#[derive(Debug, Clone)]
pub struct RDPrune {
    /// Fraction of splats to keep in `[0, 1]`. Clamped on entry.
    pub target_ratio: f32,
}

impl Default for RDPrune {
    fn default() -> Self {
        // 0.7 is the sweet spot from the w2-rd-prune sweep on bonsai +
        // bicycle: ~30% smaller GLB at the same fidelity-score band as
        // the OpacityPrune-only `web-mobile` baseline. See
        // experiments/w2-rd-prune/RESULT.md.
        Self { target_ratio: 0.7 }
    }
}

impl Pass for RDPrune {
    fn name(&self) -> &'static str {
        "RDPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        let ratio = self.target_ratio.clamp(0.0, 1.0);
        let keep = (ratio * n as f32).round() as usize;
        if keep >= n {
            // No-op at ratio == 1.0; faster than running the rank pass.
            return Ok(PassStats {
                notes: vec![format!("ratio={ratio:.3} no_op")],
                ..Default::default()
            });
        }
        if keep == 0 {
            let removed = n;
            scene.splats.clear();
            return Ok(PassStats {
                removed,
                notes: vec![format!("ratio={ratio:.3} cleared")],
                ..Default::default()
            });
        }

        // Per-splat distortion proxy. f32 is fine: the dynamic range is
        // ~9 orders of magnitude on bonsai (2e-9..3e-1) which fits inside
        // f32 mantissa without losing rank-order stability.
        let dist: Vec<f32> = scene
            .splats
            .iter()
            .map(|s| {
                let alpha = s.opacity;
                // Largest linear axis. Clamp to a small floor so a
                // degenerate zero-scale splat doesn't tie with everyone
                // else at distortion=0.
                let smax = s
                    .scale
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(1e-6);
                let sh_pow = match &s.color {
                    Color::Rgb(_) => 1.0_f32,
                    Color::Sh { coeffs, degree } => {
                        // DC term occupies coeffs[0..3] (RGB interleaved
                        // per band). The "rest" is the higher bands.
                        let total = (*degree as usize + 1).pow(2);
                        if total <= 1 || coeffs.len() < 3 * total {
                            1.0
                        } else {
                            let rest = &coeffs[3..];
                            let n_rest = rest.len().max(1) as f32;
                            let e: f32 = rest.iter().map(|v| v * v).sum::<f32>() / n_rest;
                            1.0 + e
                        }
                    }
                };
                alpha * smax * smax * sh_pow
            })
            .collect();

        // Find the keep-th largest distortion via `select_nth_unstable`.
        // O(n) average; nothing to gain from sorting the full slice
        // because we only need the threshold value.
        let mut dist_copy = dist.clone();
        let nth = n - keep; // we want the `keep` largest, so the cutoff is the (n-keep)-th smallest.
        dist_copy.select_nth_unstable_by(nth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = dist_copy[nth];

        // Retain in place. Splats at exactly `threshold` are kept only
        // until we've satisfied the keep count — this guarantees the
        // post-prune length is exactly `keep` even when many splats tie
        // (common for zero-SH RGB scenes where many alphas are equal).
        let strictly_above = dist.iter().filter(|&&d| d > threshold).count();
        let mut tie_budget = keep.saturating_sub(strictly_above);
        let mut idx: usize = 0;
        let mut kept_count: usize = 0;
        scene.splats.retain(|_| {
            let d = dist[idx];
            idx += 1;
            if d > threshold {
                kept_count += 1;
                true
            } else if d == threshold && tie_budget > 0 {
                tie_budget -= 1;
                kept_count += 1;
                true
            } else {
                false
            }
        });
        let removed = n - kept_count;
        Ok(PassStats {
            removed,
            notes: vec![format!(
                "ratio={ratio:.3} kept={kept_count} threshold={threshold:.4e}"
            )],
            ..Default::default()
        })
    }
}

/// Drop "needle" splats whose maximum/minimum scale ratio exceeds `max_ratio`.
///
/// Inria 3DGS training routinely emits extremely anisotropic gaussians where
/// one eigenvalue dwarfs the other two (ratio > 100). When neighbouring detail
/// splats are later quantized or pruned, these needles become visible spikes
/// across the rendered scene. Dropping them at the optimizer stage eliminates
/// the artifact without needing per-renderer mitigations.
///
/// Note on units: the in-memory `SplatScene.scale` is **linear** — importers
/// (e.g. the Inria PLY reader) convert the log-space scales emitted by 3DGS
/// training into linear space before populating the IR. So the ratio is simply
/// `max(scale) / min(scale)` with no `exp()` involved.
#[derive(Debug, Clone)]
pub struct AspectRatioPrune {
    /// Splats with `max(scale)/min(scale) > max_ratio` are dropped. Default 8.0.
    pub max_ratio: f32,
}

impl Default for AspectRatioPrune {
    fn default() -> Self {
        Self { max_ratio: 8.0 }
    }
}

impl Pass for AspectRatioPrune {
    fn name(&self) -> &'static str {
        "AspectRatioPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let max_ratio = self.max_ratio.max(1.0);
        let before = scene.splats.len();
        // First pass: compute ratios + a keep mask so we can report median/max
        // of the *dropped* set without an extra clone of the splat vec.
        let mut dropped_ratios: Vec<f32> = Vec::new();
        let keep: Vec<bool> = scene
            .splats
            .iter()
            .map(|s| {
                // Absolute values: Inria converts log-space to linear via
                // `exp()`, so values should already be positive, but a few
                // importers/test fixtures pass raw scales through. Guard
                // against zero/negative without panicking.
                let sx = s.scale[0].abs();
                let sy = s.scale[1].abs();
                let sz = s.scale[2].abs();
                let smin = sx.min(sy).min(sz);
                let smax = sx.max(sy).max(sz);
                if !smin.is_finite() || !smax.is_finite() || smin <= 0.0 {
                    // Degenerate scale → drop (consistent with RemoveInvalidSplats
                    // intent; an axis of zero extent is unrenderable anyway).
                    dropped_ratios.push(f32::INFINITY);
                    return false;
                }
                let ratio = smax / smin;
                if ratio > max_ratio {
                    dropped_ratios.push(ratio);
                    false
                } else {
                    true
                }
            })
            .collect();

        let mut idx = 0usize;
        scene.splats.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
        let removed = before - scene.splats.len();

        // Stats: median + max of dropped ratios. Use partial_cmp; INFINITY
        // values (degenerate splats) naturally sort to the top.
        let mut notes = Vec::new();
        if !dropped_ratios.is_empty() {
            let mut sorted = dropped_ratios.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            let max = *sorted.last().unwrap();
            notes.push(format!(
                "max_ratio={:.2} dropped={} median_dropped_ratio={:.2} max_dropped_ratio={:.2}",
                max_ratio, removed, median, max
            ));
        } else {
            notes.push(format!(
                "max_ratio={:.2} dropped=0 (no needle splats found)",
                max_ratio
            ));
        }

        Ok(PassStats {
            removed,
            notes,
            ..Default::default()
        })
    }
}

/// Drop splats farther than `dist_sigma * σ` from the scene centroid.
#[derive(Debug, Clone)]
pub struct FloaterPrune {
    /// k-nearest-neighbor count used for the per-splat isolation metric.
    pub k_neighbors: usize,
    /// Multiplier on the median k-NN distance — a splat is a floater iff its
    /// k-NN distance exceeds `dist_sigma * median(knn_dist)`. This is the
    /// "radius outlier removal" semantics used by Open3D / PCL. Lower is more
    /// aggressive; 3.0 catches the sparse-densification halo around real
    /// captures while sparing corner/boundary splats of dense subjects.
    pub dist_sigma: f32,
}

impl Default for FloaterPrune {
    fn default() -> Self {
        // k=8 is the standard PointCleanNet / Open3D choice. 3.0× median is
        // a strong floater filter for capture-derived 3DGS scenes — bonsai
        // interior k-NN ≈ 0.02 units, halo k-NN ≈ 0.5–2 units → 25–100× the
        // median, comfortably above 3× threshold.
        Self {
            k_neighbors: 8,
            dist_sigma: 3.0,
        }
    }
}

impl Pass for FloaterPrune {
    fn name(&self) -> &'static str {
        "FloaterPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        let k = self.k_neighbors.max(1);
        // Need at least k+1 splats for any non-trivial k-NN. Below that the
        // pass is a no-op (rather than guessing what "outlier" means on a
        // handful of points).
        if n <= k {
            return Ok(PassStats::default());
        }

        // Build positions array once. f64 because the centroid-relative
        // distance accumulation underflows in f32 at typical scene scales
        // (bonsai_real bbox ~16 units; 16² · 270 000 splats overflows the
        // mantissa).
        let pos: Vec<[f64; 3]> = scene
            .splats
            .iter()
            .map(|s| {
                [
                    s.position[0] as f64,
                    s.position[1] as f64,
                    s.position[2] as f64,
                ]
            })
            .collect();

        // Bbox + cell-size derivation.
        //
        // Strategy: derive `cell_size` from the *mean inter-splat spacing*
        // rather than a fixed cells-per-axis count. The previous
        // implementation clamped `cells_per_axis` to 96, which on a 3.6M-splat
        // bicycle (concentrated in <5% of the AABB volume — the bike body
        // surrounded by a near-empty sky-dome) packed ~10k splats per dense
        // cell. Each per-splat k-NN scan then visited ~270k candidates,
        // pushing the pass into O(N²)-ish wall time (>5 min on bicycle).
        //
        // Using `cell_size = c * (V/N)^(1/3)` keeps the *expected* cell
        // occupancy uniform regardless of scene scale. With `c = 2.0` a
        // 27-cell neighborhood expects ~(2·c)³ ≈ 64 splats — enough for a
        // stable k=8 estimate without quadratic blow-up. A sparse HashMap
        // grid then handles the long-tail empty cells without allocating a
        // 512³-cell dense array.
        //
        // We further bound the worst-case per-query work via `MAX_CANDIDATES`
        // below — if a single neighborhood produces more candidates than
        // that (degenerate co-located splats), we sub-sample deterministically
        // by stepping through the list, which preserves the median k-NN
        // estimate to within a few percent.
        let (bmin, bmax) = bbox_f64(&pos);
        let extent = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        let bbox_diag = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2])
            .sqrt()
            .max(1e-9);
        let longest = extent[0].max(extent[1]).max(extent[2]).max(1e-9);
        // Volume of the AABB. Degenerate axes (planar scenes) get clamped to
        // `longest` so `cell_size` stays sane on flat point sets.
        let vol = extent[0].max(1e-6 * longest)
            * extent[1].max(1e-6 * longest)
            * extent[2].max(1e-6 * longest);
        let mean_spacing = (vol / n as f64).cbrt().max(1e-12);
        // c=2.0: with uniform distribution a (2·c)³ neighborhood has ~64
        // splats. Clamped so we never go finer than 1/512 of the longest
        // axis (avoids HashMap explosion on absurd scenes) and never coarser
        // than 1/4 (preserves spatial culling on tiny inputs).
        let cell_size = (2.0 * mean_spacing).clamp(longest / 512.0, longest / 4.0);

        // Sparse spatial hash: only allocates cells that contain splats, so
        // we can afford a fine grid (effective cells_per_axis up to 512)
        // without paying the dense Vec<Vec<u32>> memory cost.
        let cell_of = |p: [f64; 3]| -> (i32, i32, i32) {
            (
                ((p[0] - bmin[0]) / cell_size).floor() as i32,
                ((p[1] - bmin[1]) / cell_size).floor() as i32,
                ((p[2] - bmin[2]) / cell_size).floor() as i32,
            )
        };
        let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>> =
            std::collections::HashMap::with_capacity(n / 8 + 1);
        for (i, p) in pos.iter().enumerate() {
            grid.entry(cell_of(*p)).or_default().push(i as u32);
        }
        let cells_count = grid.len();
        // FxHash-quality deterministic ordering isn't required; HashMap's
        // default RandomState is fine because the only consumer is
        // `get(&key)` lookups — no iteration order leaks into output.

        // Per-splat k-NN scan, parallelised across splats. Each query is
        // independent (grid is shared read-only), so this is an embarrassingly
        // parallel rayon `par_iter`.
        //
        // Worst-case bound per query: we cap candidates examined at
        // `MAX_CANDIDATES`. This guards against the degenerate "100k
        // co-located splats in one cell" case without changing the median
        // k-NN estimate (any sub-sample of >>k splats is representative for
        // the k-th order statistic).
        const MAX_CANDIDATES: usize = 4096;
        const MAX_RADIUS: i32 = 3;
        use rayon::prelude::*;
        let knn_d2: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let p = pos[i];
                let (cx, cy, cz) = cell_of(p);
                let mut heap = BoundedMaxHeap::with_capacity(k + 1);
                let mut examined: usize = 0;
                let mut radius: i32 = 0;
                // Expand outward in ring shells, starting with the *central*
                // cell (ring=0, the cell the query splat lives in). On each
                // bump of `radius`, the inner `ring == radius` check skips
                // cells we already visited in prior shells. Continue until
                // the heap holds k+1 entries *or* we've scanned out to
                // MAX_RADIUS — whichever comes first.
                while heap.len() < k + 1 && radius <= MAX_RADIUS {
                    for dx in -radius..=radius {
                        for dy in -radius..=radius {
                            for dz in -radius..=radius {
                                let ring = dx.abs().max(dy.abs()).max(dz.abs());
                                if ring != radius {
                                    continue;
                                }
                                let key = (cx + dx, cy + dy, cz + dz);
                                let Some(bucket) = grid.get(&key) else {
                                    continue;
                                };
                                // Deterministic stride sub-sampling for
                                // pathologically-dense cells. Picks every
                                // (len/cap)-th entry so we still cover the
                                // full cell uniformly.
                                let len = bucket.len();
                                let stride =
                                    ((len + MAX_CANDIDATES - 1) / MAX_CANDIDATES.max(1)).max(1);
                                let mut idx = 0usize;
                                while idx < len {
                                    let j = bucket[idx] as usize;
                                    let q = pos[j];
                                    let dx0 = q[0] - p[0];
                                    let dy0 = q[1] - p[1];
                                    let dz0 = q[2] - p[2];
                                    let d2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
                                    heap.push(d2, k + 1);
                                    examined += 1;
                                    if examined >= MAX_CANDIDATES {
                                        break;
                                    }
                                    idx += stride;
                                }
                                if examined >= MAX_CANDIDATES {
                                    break;
                                }
                            }
                            if examined >= MAX_CANDIDATES {
                                break;
                            }
                        }
                        if examined >= MAX_CANDIDATES {
                            break;
                        }
                    }
                    radius += 1;
                }
                heap.top().unwrap_or(f64::INFINITY)
            })
            .collect();

        // Radius-outlier semantics: threshold = dist_sigma × median(knn_d).
        // Multiplicative is the right shape for capture-derived 3DGS, where
        // floater k-NN distances are 10–100× the dense-subject median rather
        // than a Gaussian-tail above it. Boundary splats of dense regions
        // sit at ~sqrt(3)≈1.73× median, well below the 3.0× default.
        // The MAD-augmented additive form (median + k·MAD) over-prunes such
        // boundaries because MAD vanishes inside the dense subject.
        let knn_d: Vec<f64> = knn_d2.par_iter().map(|&d2| d2.sqrt()).collect();
        let median = median_inplace(&mut knn_d.clone());
        // We still surface MAD in the notes so an operator can sanity-check
        // the distribution shape without rerunning the pipeline.
        let mut abs_dev: Vec<f64> = knn_d.iter().map(|&d| (d - median).abs()).collect();
        let mad = median_inplace(&mut abs_dev);
        // Floor at a fraction of bbox_diag so a hypothetical perfectly-aligned
        // grid (median≈0 because of floating-point noise) doesn't end up
        // pruning everything above a vanishingly small threshold.
        let floor = bbox_diag * 1e-4;
        let threshold = (self.dist_sigma as f64 * median).max(median + floor);

        let before = scene.splats.len();
        // Two-pass retain so the `knn_d` slice indexes line up. The closure
        // captures `idx` and steps it for every original splat.
        let mut idx: usize = 0;
        scene.splats.retain(|_| {
            let keep = knn_d[idx] <= threshold;
            idx += 1;
            keep
        });
        let removed = before - scene.splats.len();
        Ok(PassStats {
            removed,
            notes: vec![format!(
                "knn_median={:.4} mad={:.4} threshold={:.4} cells={} cell_size={:.5}",
                median, mad, threshold, cells_count, cell_size
            )],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod floater_prune_perf_tests {
    use super::*;
    use catetus_core::{Color, Splat, SplatScene};
    use std::time::Instant;

    /// Build a 100k-splat synthetic "scene": a dense cluster (95k) of points
    /// uniformly inside a unit cube plus a sparse halo of 5k floaters spread
    /// across a 100×-larger AABB. This reproduces the bicycle / inria
    /// shape (concentrated subject + sky-dome floaters) at small enough
    /// scale to run in CI.
    fn make_synthetic_scene(seed: u64) -> SplatScene {
        // Cheap LCG so the test is deterministic without pulling in `rand`.
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut next = || -> f32 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545F4914F6CDD1D);
            ((v >> 32) as u32 as f32) / (u32::MAX as f32)
        };
        let mut scene = SplatScene::new();
        let mk_splat = |p: [f32; 3]| Splat {
            position: p,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.01, 0.01, 0.01],
            opacity: 1.0,
            color: Color::Sh {
                degree: 0,
                coeffs: vec![0.5, 0.5, 0.5],
            },
        };
        // 95k dense splats in [0,1]^3.
        for _ in 0..95_000 {
            let x = next();
            let y = next();
            let z = next();
            scene.splats.push(mk_splat([x, y, z]));
        }
        // 5k halo splats uniformly in [-50, 51]^3 — well outside the dense
        // cluster, so any sane k-NN outlier detector should flag them.
        for _ in 0..5_000 {
            let p = [
                next() * 101.0 - 50.0,
                next() * 101.0 - 50.0,
                next() * 101.0 - 50.0,
            ];
            scene.splats.push(mk_splat(p));
        }
        scene
    }

    /// O(N²) brute-force reference k-NN for cross-checking the spatial-hash
    /// implementation. Only used in tests, on a sub-sample of splats.
    fn brute_force_knn_d(pos: &[[f64; 3]], k: usize, indices: &[usize]) -> Vec<f64> {
        indices
            .iter()
            .map(|&i| {
                let p = pos[i];
                let mut dists: Vec<f64> = pos
                    .iter()
                    .map(|q| {
                        let dx = q[0] - p[0];
                        let dy = q[1] - p[1];
                        let dz = q[2] - p[2];
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    })
                    .collect();
                let m = (k + 1).min(dists.len() - 1);
                dists.select_nth_unstable_by(m, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                dists[m]
            })
            .collect()
    }

    #[test]
    fn runs_under_5s_on_100k_splats() {
        let mut scene = make_synthetic_scene(0xCAFEBABE);
        assert_eq!(scene.splats.len(), 100_000);
        let pass = FloaterPrune::default();
        let mut ctx = PassContext::default();
        let t0 = Instant::now();
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        let dt = t0.elapsed();
        eprintln!(
            "FloaterPrune on 100k synthetic: {:?}, removed={}, kept={}, notes={:?}",
            dt,
            stats.removed,
            scene.splats.len(),
            stats.notes
        );
        assert!(
            dt.as_secs_f64() < 5.0,
            "FloaterPrune took {:?} on 100k splats — should be <5s",
            dt
        );
        // The 5k halo splats are an order of magnitude isolated; with
        // dist_sigma=3 we expect to catch the vast majority of them. The
        // dense 95k cluster's boundary splats sit at ~sqrt(3)·median so
        // they survive. Sanity-check the overall removal rate.
        assert!(
            stats.removed >= 4_000 && stats.removed <= 10_000,
            "expected to drop most of the 5k halo splats; removed={}",
            stats.removed
        );
    }

    #[test]
    fn jaccard_matches_brute_force() {
        // 10k-splat scene (5% as many as the perf test) so the O(N²)
        // brute-force reference is fast enough for CI. The behaviour of the
        // spatial-hash vs brute-force k-NN is independent of N, so a 10k
        // scene is sufficient to validate correctness.
        let mut scene = SplatScene::new();
        let mut state = 0xDEADBEEFu64 | 1;
        let mut next = || -> f32 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32 as f32) / (u32::MAX as f32)
        };
        let mk_splat = |p: [f32; 3]| Splat {
            position: p,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.01; 3],
            opacity: 1.0,
            color: Color::Sh {
                degree: 0,
                coeffs: vec![0.5, 0.5, 0.5],
            },
        };
        for _ in 0..9_500 {
            scene.splats.push(mk_splat([next(), next(), next()]));
        }
        for _ in 0..500 {
            scene.splats.push(mk_splat([
                next() * 101.0 - 50.0,
                next() * 101.0 - 50.0,
                next() * 101.0 - 50.0,
            ]));
        }

        // Snapshot positions before the spatial-hash pass mutates the scene.
        let pos: Vec<[f64; 3]> = scene
            .splats
            .iter()
            .map(|s| {
                [
                    s.position[0] as f64,
                    s.position[1] as f64,
                    s.position[2] as f64,
                ]
            })
            .collect();
        let n = pos.len();

        // Run the production pass on a clone; record which original indices
        // it drops by snapshotting positions before and after.
        let mut scene_a = scene.clone();
        let pass = FloaterPrune::default();
        let mut ctx = PassContext::default();
        pass.run(&mut scene_a, &mut ctx).expect("run");
        let kept_a: std::collections::HashSet<(u64, u64, u64)> = scene_a
            .splats
            .iter()
            .map(|s| {
                (
                    s.position[0].to_bits() as u64,
                    s.position[1].to_bits() as u64,
                    s.position[2].to_bits() as u64,
                )
            })
            .collect();
        let dropped_a: std::collections::HashSet<usize> = (0..n)
            .filter(|&i| {
                let p = scene.splats[i].position;
                !kept_a.contains(&(
                    p[0].to_bits() as u64,
                    p[1].to_bits() as u64,
                    p[2].to_bits() as u64,
                ))
            })
            .collect();

        // Brute-force k-NN over the *same* indices to recover a reference
        // dropped-set under the same multiplicative-median threshold.
        let k = pass.k_neighbors.max(1);
        let knn_d_b = brute_force_knn_d(&pos, k, &(0..n).collect::<Vec<_>>());
        let mut sorted = knn_d_b.clone();
        let mid = sorted.len() / 2;
        sorted.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let median_b = sorted[mid];
        let threshold_b = pass.dist_sigma as f64 * median_b;
        let dropped_b: std::collections::HashSet<usize> =
            (0..n).filter(|&i| knn_d_b[i] > threshold_b).collect();

        // Jaccard(dropped_a, dropped_b) ≥ 0.95: the spatial-hash sampling
        // can disagree with brute-force on a handful of borderline splats
        // sitting right at the multiplicative-median threshold, but the
        // vast majority of the floater set must match.
        let inter = dropped_a.intersection(&dropped_b).count();
        let union = dropped_a.union(&dropped_b).count();
        let jaccard = if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        };
        eprintln!(
            "dropped_hash={} dropped_brute={} inter={} jaccard={:.4}",
            dropped_a.len(),
            dropped_b.len(),
            inter,
            jaccard
        );
        assert!(
            jaccard >= 0.95,
            "Jaccard(hash, brute) = {:.4} < 0.95",
            jaccard
        );
    }
}

// -- internals -------------------------------------------------------------

fn bbox_f64(pos: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut bmin = [f64::INFINITY; 3];
    let mut bmax = [f64::NEG_INFINITY; 3];
    for p in pos {
        for i in 0..3 {
            if p[i] < bmin[i] {
                bmin[i] = p[i];
            }
            if p[i] > bmax[i] {
                bmax[i] = p[i];
            }
        }
    }
    (bmin, bmax)
}

/// In-place median via `select_nth_unstable_by`. f64 NaNs are sorted to the
/// end deterministically, then the median is the slice midpoint.
fn median_inplace(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mid = xs.len() / 2;
    xs.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    xs[mid]
}

/// Tiny binary max-heap of f64 distances, used to extract the k-th smallest
/// without sorting. Manual rather than `BinaryHeap<NotNan<f64>>` to avoid the
/// `ordered-float` dep and the per-push NaN check.
struct BoundedMaxHeap {
    buf: Vec<f64>,
}

impl BoundedMaxHeap {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }
    fn len(&self) -> usize {
        self.buf.len()
    }
    fn top(&self) -> Option<f64> {
        self.buf.first().copied()
    }
    /// Push `v`, evicting the current max if the heap exceeds `cap`. NaN
    /// inputs are silently dropped (only finite distances are meaningful for
    /// a k-NN estimate).
    fn push(&mut self, v: f64, cap: usize) {
        if !v.is_finite() {
            return;
        }
        if self.buf.len() < cap {
            self.buf.push(v);
            self.sift_up(self.buf.len() - 1);
        } else if let Some(&top) = self.buf.first() {
            if v < top {
                self.buf[0] = v;
                self.sift_down(0);
            }
        }
    }
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.buf[i] > self.buf[parent] {
                self.buf.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, mut i: usize) {
        let n = self.buf.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut largest = i;
            if l < n && self.buf[l] > self.buf[largest] {
                largest = l;
            }
            if r < n && self.buf[r] > self.buf[largest] {
                largest = r;
            }
            if largest == i {
                break;
            }
            self.buf.swap(i, largest);
            i = largest;
        }
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
    /// When true, quantize in **log space** rather than linear. Scales follow a
    /// heavy-tailed log-normal distribution (3DGS optimizer parameterizes them
    /// via `exp(s)`), so uniform quantization in linear space crushes the
    /// long tail — small splats (the majority of the population) collapse to
    /// the same low bin and dequantize back to ~zero. The PLY writer's
    /// `ln(scale)` then maps those zeros to `ln(EPSILON) ≈ -15.94`, which
    /// matches what SuperSplat would render: heavy overdraw of large
    /// splats hiding the missing fine detail.
    ///
    /// In log space, a step of `span/2^bits` is multiplicative — at 12 bits
    /// over a typical bonsai log-range of [-19, 1] that's an 0.5%
    /// relative-error grid that preserves both the tiny and the large
    /// splats faithfully.
    ///
    /// Defaults to `false` for backwards compatibility. Set to `true` in
    /// presets whose downstream GLB writer also stores log-space scale
    /// (see `WriteOpts::log_quant_attrs`).
    pub log_space: bool,
}

impl Default for QuantizeScale {
    fn default() -> Self {
        Self {
            bits: 8,
            log_space: false,
        }
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
        // Helper: map IR-space linear scale into the working space for
        // quantization, and back.
        let fwd = |v: f32| -> f32 {
            if self.log_space {
                v.max(f32::MIN_POSITIVE).ln()
            } else {
                v
            }
        };
        let inv = |t: f32| -> f32 {
            if self.log_space {
                t.exp()
            } else {
                t
            }
        };
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for s in &scene.splats {
            for i in 0..3 {
                let t = fwd(s.scale[i]);
                if t < mn[i] {
                    mn[i] = t;
                }
                if t > mx[i] {
                    mx[i] = t;
                }
            }
        }
        let mut modified = 0usize;
        for s in &mut scene.splats {
            for i in 0..3 {
                let t_in = fwd(s.scale[i]);
                let t_q = quantize_f32(t_in, self.bits, mn[i], mx[i]);
                let q = inv(t_q);
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

pub(crate) fn morton_code(p: [f32; 3], mn: [f32; 3], mx: [f32; 3]) -> u64 {
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
    fn run(&self, scene: &mut SplatScene, ctx: &mut PassContext) -> Result<PassStats> {
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
        // Reorder the SH-rest Jacobian weights in lockstep so they stay
        // aligned with `scene.splats` (consumed downstream by
        // `VQPaletteShRest` in render-space-Lloyd-Max mode).
        if let Some(w) = ctx.sh_rest_weights.as_mut() {
            if w.len() == order.len() {
                let permuted: Vec<f32> = order.iter().map(|&i| w[i]).collect();
                *w = permuted;
            }
        }
        // Same lockstep for splat_origin_idx (used by the V5.2 sidecar
        // encoder to map post-pipeline splats back to GT PLY rows).
        if let Some(o) = ctx.splat_origin_idx.as_mut() {
            if o.len() == order.len() {
                let permuted: Vec<u32> = order.iter().map(|&i| o[i]).collect();
                *o = permuted;
            }
        }
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

// =========================================================================
// CodecGSLite — H.264 PLAS-sorted parameter-image codec pass
// =========================================================================
//
// CodecGS-Lite encodes the scene as 7 H.264 video streams (xyz-hi, xyz-lo,
// f_dc, f_rest, opacity, scale, rot) over a PLAS-sorted 2D grid of splats.
// The proven Python reference is in `experiments/w2-codecgs/code/encode.py`;
// the corresponding result table is `experiments/w2-codecgs/RESULT.md`:
//
//   bonsai (1.16M splats):  splat-transform .sog 17.4 MB / fid 0.8422
//                           CodecGS-Lite h264/30 8.19 MB / fid 0.8501
//                           => 47% of SOG bytes at +0.008 fidelity.
//
//   bicycle (4.0M splats):  splat-transform .sog 48.5 MB / fid 0.8375
//                           CodecGS-Lite h264/30 24.96 MB / fid 0.8428
//                           => 51% of SOG bytes at +0.005 fidelity.
//
// v0.1 implementation shells out to the Python prototype + ffmpeg + the GLB
// packer (`pack_codecgs_glb.py`). A pure-Rust PLAS-lite + ffmpeg wiring is
// tracked as a follow-up — the prototype already takes ~10s for bonsai.
//
// The pass does not mutate `scene.splats`. It writes the encoded `.glb` to
// `out_glb_path` as a side effect and returns the byte count in the stats
// `notes`. The CLI is expected to invoke it with the user-chosen output
// path; library callers can wire it up directly.

/// Output codec for `CodecGSLite`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecGSKind {
    /// H.264 baseline (avc1.42E01F) — universal WebCodecs decode.
    H264,
    /// H.265 / HEVC — better RD curve, narrower decode coverage.
    H265,
    /// AV1 (SVT-AV1) — best RD curve, even narrower decode coverage.
    Av1,
}

impl CodecGSKind {
    fn name(self) -> &'static str {
        match self {
            CodecGSKind::H264 => "h264",
            CodecGSKind::H265 => "h265",
            CodecGSKind::Av1 => "av1",
        }
    }
}

/// PLAS-sorted parameter-image codec pass. See module-level docs above for
/// the wire format and measured savings.
///
/// This pass shells to Python + ffmpeg for v0.1. It REQUIRES:
///   - `python3` on PATH with the prototype's dependencies (numpy, plyfile)
///   - `ffmpeg` on PATH with `libx264` (and `libx265` / `libsvtav1` if you
///     opt into those codecs)
///   - The encoder source tree at `experiments/w2-codecgs/code/`
///   - The GLB packer at `experiments/w3-codecgs-integrate/code/pack_codecgs_glb.py`
///
/// Use `CodecGSLite::is_supported()` to probe at runtime before adding it to
/// a pipeline.
#[derive(Debug, Clone)]
pub struct CodecGSLite {
    /// Input PLY path. The pass re-loads the PLY because the encoder needs
    /// the full f_rest channel which the IR currently strips at SPZ-write
    /// time on some paths.
    pub input_ply: std::path::PathBuf,
    /// Output `.glb` path with the `SF_codecgs_lite` extension.
    pub out_glb_path: std::path::PathBuf,
    /// Repo root (used to locate the prototype Python scripts).
    pub repo_root: std::path::PathBuf,
    /// Codec selection.
    pub codec: CodecGSKind,
    /// Uniform CRF (per-channel-QP recipe still to come — see RESULT.md).
    pub crf: u8,
    /// Sort strategy: "auto" (real PLAS if available, else lite), "plas",
    /// "lite".
    pub sort: String,
}

impl CodecGSLite {
    /// Sane default: H.264 / CRF=30 / plas-lite.
    pub fn h264_crf30(
        input_ply: std::path::PathBuf,
        out_glb_path: std::path::PathBuf,
        repo_root: std::path::PathBuf,
    ) -> Self {
        Self {
            input_ply,
            out_glb_path,
            repo_root,
            codec: CodecGSKind::H264,
            crf: 30,
            sort: "auto".to_string(),
        }
    }

    /// Returns Ok(()) if `python3` and `ffmpeg` are on PATH and the prototype
    /// scripts are present. Use this from CLI before scheduling the pass so
    /// users see a clean error message instead of an opaque subprocess fail.
    pub fn is_supported(repo_root: &std::path::Path) -> std::result::Result<(), String> {
        for bin in ["python3", "ffmpeg"] {
            let ok = std::process::Command::new(bin)
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                return Err(format!("`{bin}` not found on PATH"));
            }
        }
        for rel in [
            "experiments/w2-codecgs/code/encode.py",
            "experiments/w3-codecgs-integrate/code/pack_codecgs_glb.py",
        ] {
            if !repo_root.join(rel).exists() {
                return Err(format!("missing prototype script: {rel}"));
            }
        }
        Ok(())
    }
}

impl Pass for CodecGSLite {
    fn name(&self) -> &'static str {
        "CodecGSLite"
    }
    fn run(&self, _scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        use std::process::Command;
        CodecGSLite::is_supported(&self.repo_root)
            .map_err(|e| anyhow::anyhow!("CodecGSLite preconditions: {e}"))?;

        // Bundle directory: <out_glb_path>.bundle/
        let bundle_dir = self.out_glb_path.with_extension("bundle");
        std::fs::create_dir_all(&bundle_dir)?;

        // 1. Encode per-channel mp4s with the Python prototype.
        let encode_py = self.repo_root.join("experiments/w2-codecgs/code/encode.py");
        let status = Command::new("python3")
            .arg(&encode_py)
            .arg("--in")
            .arg(&self.input_ply)
            .arg("--out-dir")
            .arg(&bundle_dir)
            .arg("--codec")
            .arg(self.codec.name())
            .arg("--crf")
            .arg(self.crf.to_string())
            .arg("--sort")
            .arg(&self.sort)
            .status()?;
        if !status.success() {
            anyhow::bail!("encode.py exit status: {status}");
        }

        // 2. Pack the bundle into a single .glb with SF_codecgs_lite.
        let pack_py = self
            .repo_root
            .join("experiments/w3-codecgs-integrate/code/pack_codecgs_glb.py");
        let status = Command::new("python3")
            .arg(&pack_py)
            .arg("--bundle-dir")
            .arg(&bundle_dir)
            .arg("--out")
            .arg(&self.out_glb_path)
            .status()?;
        if !status.success() {
            anyhow::bail!("pack_codecgs_glb.py exit status: {status}");
        }

        let bytes_out = std::fs::metadata(&self.out_glb_path)?.len();
        Ok(PassStats {
            modified: 0,
            removed: 0,
            duration_ms: 0,
            notes: vec![
                format!("CodecGSLite: codec={} crf={}", self.codec.name(), self.crf),
                format!("CodecGSLite: out={} bytes", bytes_out),
                format!("CodecGSLite: bundle dir kept at {}", bundle_dir.display()),
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// QuantizeSHRest — per-channel symmetric int quantization of SH coefficients
// at degrees 1..=3. The pass round-trips dequantized values into the scene so
// downstream passes / writers still see f32; it also parks a side-table in a
// thread-local register so the GLB writer can emit BYTE/SHORT accessors using
// the per-channel ranges (which the deferred reader needs to reconstruct).
//
// Per SOG / mkkellogg parity, this brings sh-rest from 180 b/s (fp32 × 45) to
// 45 b/s (b=8) or 33.75 b/s (b=6). Per-channel ranges add 180 B per scene,
// amortized to ~0 over the splat count.
// ---------------------------------------------------------------------------

/// Per-channel SH-rest quantization side table. One entry per SH coefficient
/// scalar across all degrees 1..=3 (3+5+7 = 15 coefs × 3 channels = 45).
/// The on-wire integer is `q = clamp(round(v / range_per_ch * scale), -scale, scale)`
/// where `scale = 2^(bits-1) - 1`. Dequant is the symmetric inverse.
#[derive(Debug, Clone, Serialize)]
pub struct ShRestQuantTable {
    /// Quantization width in bits (signed; 2..=16).
    pub bits: u8,
    /// Per-channel range (length 45, one per non-DC scalar in the standard
    /// `[3 + coef_idx*3 + ch]` packing).
    pub ranges: Vec<f32>,
    /// Robust-percentile used to derive `ranges` (for diagnostics).
    pub range_percentile: f32,
}

/// In-process register for the most recent `QuantizeSHRest` side table. The
/// writer drains it just before encoding the GLB. Mirrors the
/// `LAST_DELTA_STREAM` pattern in `splat_delta.rs`.
static LAST_SH_REST_QUANT_TABLE: Mutex<Option<ShRestQuantTable>> = Mutex::new(None);

/// Drain the last `QuantizeSHRest` side table. Returns `None` if no
/// `QuantizeSHRest` pass has run since the previous drain.
pub fn take_last_sh_rest_quant_table() -> Option<ShRestQuantTable> {
    LAST_SH_REST_QUANT_TABLE
        .lock()
        .ok()
        .and_then(|mut g| g.take())
}

/// Reset the side-table register without consuming it. Used by tests.
#[doc(hidden)]
pub fn clear_last_sh_rest_quant_table() {
    if let Ok(mut g) = LAST_SH_REST_QUANT_TABLE.lock() {
        *g = None;
    }
}

/// `QuantizeSHRest` pass: symmetric int quantization of all SH coefficients
/// at degrees 1..=3. The DC term (`coeffs[0..3]`) is untouched.
#[derive(Debug, Clone)]
pub struct QuantizeSHRest {
    /// Signed-int width. 8 ⇒ BYTE accessor (-127..=127), 9..=16 ⇒ SHORT.
    pub bits: u8,
    /// Robust percentile (0..100) of |v| per channel for range estimation.
    /// 99.5 matches SplatDelta and the SOG sh-rest range table.
    pub range_percentile: f32,
}

impl Default for QuantizeSHRest {
    fn default() -> Self {
        Self {
            bits: 8,
            range_percentile: 99.5,
        }
    }
}

impl Pass for QuantizeSHRest {
    fn name(&self) -> &'static str {
        "QuantizeSHRest"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        // 45 scalars: 3 coefs degree-1, 5 degree-2, 7 degree-3, × 3 channels.
        const N_SCALARS: usize = 45;
        let bits = self.bits.clamp(2, 16);
        let levels = ((1u32 << (bits as u32 - 1)) - 1) as f32;

        // 1) Per-channel ranges from the configured percentile of |v|. Only
        //    splats with `Color::Sh` and a coeffs vector of the expected
        //    length contribute. A splat with Rgb-only color is ignored.
        let mut ranges = vec![1e-9f32; N_SCALARS];
        let mut have_any = false;
        for ch in 0..N_SCALARS {
            let mut vs: Vec<f32> = scene
                .splats
                .iter()
                .filter_map(|s| match &s.color {
                    Color::Sh { coeffs, .. } if coeffs.len() >= 3 + N_SCALARS => {
                        Some(coeffs[3 + ch].abs())
                    }
                    _ => None,
                })
                .collect();
            if vs.is_empty() {
                continue;
            }
            have_any = true;
            vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let pct = self.range_percentile.clamp(0.0, 100.0);
            let idx = ((pct / 100.0) * vs.len() as f32) as usize;
            let idx = idx.min(vs.len() - 1);
            ranges[ch] = vs[idx].max(1e-9);
        }

        // 2) Round-trip quantize → dequantize in place. Downstream sees the
        //    reconstructed f32 values (so e.g. the GLB writer's BYTE path
        //    will quantize the same numbers again with the same ranges and
        //    produce bit-identical output).
        let mut modified = 0usize;
        if have_any {
            for s in scene.splats.iter_mut() {
                if let Color::Sh { coeffs, .. } = &mut s.color {
                    if coeffs.len() < 3 + N_SCALARS {
                        continue;
                    }
                    for ch in 0..N_SCALARS {
                        let r = ranges[ch];
                        let v = coeffs[3 + ch];
                        let t = (v / r).clamp(-1.0, 1.0);
                        let q = (t * levels).round();
                        let dq = (q / levels).max(-1.0) * r;
                        coeffs[3 + ch] = dq;
                    }
                    modified += 1;
                }
            }
        }

        // 3) Park the side table for the writer.
        let table = ShRestQuantTable {
            bits,
            ranges,
            range_percentile: self.range_percentile,
        };
        if let Ok(mut g) = LAST_SH_REST_QUANT_TABLE.lock() {
            *g = Some(table);
        }

        Ok(PassStats {
            modified,
            notes: vec![format!(
                "bits={bits} pct={:.2} channels={N_SCALARS}",
                self.range_percentile
            )],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod sh_rest_quant_tests {
    use super::*;
    use catetus_core::{Color, Splat};

    fn sh_splat(values: [f32; 45]) -> Splat {
        let mut coeffs = vec![0.0f32, 0.0, 0.0]; // DC = 0
        coeffs.extend_from_slice(&values);
        Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            opacity: 1.0,
            color: Color::Sh { degree: 3, coeffs },
        }
    }

    #[test]
    fn quantize_sh_rest_round_trip_within_step() {
        clear_last_sh_rest_quant_table();
        // 8 splats with synthetic patterns of known magnitude per channel.
        let mut scene = SplatScene::new();
        let mut originals = Vec::new();
        for s_idx in 0..8 {
            let mut vals = [0.0f32; 45];
            for ch in 0..45 {
                // Spread values in [-0.5..0.5] per channel.
                let phase = (s_idx as f32 * 0.13 + ch as f32 * 0.07).sin();
                vals[ch] = 0.5 * phase * (((ch % 5) + 1) as f32) / 5.0;
            }
            originals.push(vals);
            scene.splats.push(sh_splat(vals));
        }

        let bits = 8;
        let pass = QuantizeSHRest {
            bits,
            range_percentile: 99.5,
        };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).expect("run");

        let table = take_last_sh_rest_quant_table().expect("side table parked");
        assert_eq!(table.bits, bits);
        assert_eq!(table.ranges.len(), 45);

        let levels = ((1u32 << (bits as u32 - 1)) - 1) as f32;
        for (i, splat) in scene.splats.iter().enumerate() {
            if let Color::Sh { coeffs, .. } = &splat.color {
                for ch in 0..45 {
                    let orig = originals[i][ch];
                    let recon = coeffs[3 + ch];
                    let r = table.ranges[ch].max(1e-9);
                    // |error| <= r / levels (one step of the dequant grid).
                    let step = r / levels;
                    let err = (recon - orig).abs();
                    assert!(
                        err <= step * 1.5 + 1e-6,
                        "splat {i} ch {ch}: err={err} > step={step} (orig={orig} recon={recon} r={r})"
                    );
                }
            } else {
                panic!("expected sh");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizeRotationPacked — per-component min/max int quantization of quaternion
// components. Parks a side table so the GLB writer can emit ROTATION as a
// normalized UBYTE (bits<=8) or USHORT (bits>8) VEC4 accessor with min/max,
// dropping 16 B/splat (FP32 VEC4) down to 4 B/splat (bits=8) or 8 B/splat
// (bits=12 packed as SHORT). On bonsai sh=3 this drops ROTATION from ~14.6 MB
// (Morton-zstd of FP32) to ~3-5 MB after the same lossless wrap.
//
// `QuantizeRotation` already quantizes the *values* (snaps to a bits-bit grid
// per component in [-1, 1]) but does not change the wire format. This pass
// runs the same forward quantization AND parks the side table for the writer
// so the bits actually reach disk.
// ---------------------------------------------------------------------------

/// Per-component rotation quantization side table. The on-wire integer is
/// `q = quantize_u8/u16(v, mins[c], maxs[c])` for c=0..3; dequant is the
/// glTF normalized-integer inverse.
#[derive(Debug, Clone, Serialize)]
pub struct RotationQuantTable {
    /// Quantization width in bits. 2..=8 ⇒ UBYTE accessor, 9..=16 ⇒ USHORT.
    pub bits: u8,
    /// Per-component min across the scene (length 4).
    pub mins: [f32; 4],
    /// Per-component max across the scene (length 4).
    pub maxs: [f32; 4],
}

static LAST_ROTATION_QUANT_TABLE: Mutex<Option<RotationQuantTable>> = Mutex::new(None);

/// Drain the last `QuantizeRotationPacked` side table.
pub fn take_last_rotation_quant_table() -> Option<RotationQuantTable> {
    LAST_ROTATION_QUANT_TABLE
        .lock()
        .ok()
        .and_then(|mut g| g.take())
}

#[doc(hidden)]
pub fn clear_last_rotation_quant_table() {
    if let Ok(mut g) = LAST_ROTATION_QUANT_TABLE.lock() {
        *g = None;
    }
}

/// `QuantizeRotationPacked` pass: per-component min/max quantization of
/// quaternion components. Round-trips dequantized values back into the scene
/// so downstream passes see f32; parks the side table so the GLB writer can
/// emit a UBYTE / USHORT normalized VEC4 accessor with min/max.
#[derive(Debug, Clone)]
pub struct QuantizeRotationPacked {
    /// Bits per component. 2..=8 ⇒ UBYTE, 9..=16 ⇒ USHORT.
    pub bits: u8,
}

impl Default for QuantizeRotationPacked {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

impl Pass for QuantizeRotationPacked {
    fn name(&self) -> &'static str {
        "QuantizeRotationPacked"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.is_empty() {
            // Still park a default side-table so the writer doesn't fall back
            // to FP32 just because the scene is empty.
            let table = RotationQuantTable {
                bits: self.bits.clamp(2, 16),
                mins: [-1.0; 4],
                maxs: [1.0; 4],
            };
            if let Ok(mut g) = LAST_ROTATION_QUANT_TABLE.lock() {
                *g = Some(table);
            }
            return Ok(PassStats::default());
        }
        let bits = self.bits.clamp(2, 16);
        // 1) Per-component min/max across the scene.
        let mut mins = [f32::INFINITY; 4];
        let mut maxs = [f32::NEG_INFINITY; 4];
        for s in &scene.splats {
            for i in 0..4 {
                let v = s.rotation[i];
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
        }
        // Guard degenerate ranges (all-equal component → mins == maxs).
        for i in 0..4 {
            if !mins[i].is_finite() {
                mins[i] = -1.0;
            }
            if !maxs[i].is_finite() {
                maxs[i] = 1.0;
            }
            if maxs[i] - mins[i] < 1e-12 {
                // Pad around the constant so dequant is well-defined.
                mins[i] -= 0.5;
                maxs[i] += 0.5;
            }
        }
        // 2) Round-trip quantize → dequantize in place (no renormalization;
        //    the writer's BYTE accessor reconstruction matches this map
        //    exactly, so the scene sees the same numbers the renderer will).
        let levels = ((1u32 << bits.min(31)) - 1) as f32;
        let mut modified = 0usize;
        for s in &mut scene.splats {
            for i in 0..4 {
                let span = (maxs[i] - mins[i]).max(1e-12);
                let t = ((s.rotation[i] - mins[i]) / span).clamp(0.0, 1.0);
                let q = (t * levels).round() / levels;
                let dq = mins[i] + q * span;
                if (dq - s.rotation[i]).abs() > f32::EPSILON {
                    modified += 1;
                }
                s.rotation[i] = dq;
            }
        }
        // 3) Park the side table for the writer.
        let table = RotationQuantTable { bits, mins, maxs };
        if let Ok(mut g) = LAST_ROTATION_QUANT_TABLE.lock() {
            *g = Some(table);
        }
        Ok(PassStats {
            modified,
            notes: vec![format!(
                "bits={bits} mins={mins:?} maxs={maxs:?}",
                bits = bits,
                mins = mins,
                maxs = maxs
            )],
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// QuantizeDCPacked — per-channel min/max int quantization of the SH DC term
// (coeffs[0..3] under `Color::Sh`, or `Color::Rgb`). Parks a side table so the
// GLB writer can emit `SH_DEGREE_0_COEF_0` as a normalized UBYTE / USHORT VEC3
// accessor with min/max, dropping 12 B/splat (FP32 VEC3) down to 3 B/splat
// (bits=8). On bonsai sh=3 this drops DC from ~10.2 MB (Morton-zstd of FP32)
// to ~2-3 MB.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct DcQuantTable {
    /// Quantization width in bits. 2..=8 ⇒ UBYTE accessor, 9..=16 ⇒ USHORT.
    pub bits: u8,
    /// Per-channel min across the scene (length 3).
    pub mins: [f32; 3],
    /// Per-channel max across the scene (length 3).
    pub maxs: [f32; 3],
}

static LAST_DC_QUANT_TABLE: Mutex<Option<DcQuantTable>> = Mutex::new(None);

pub fn take_last_dc_quant_table() -> Option<DcQuantTable> {
    LAST_DC_QUANT_TABLE.lock().ok().and_then(|mut g| g.take())
}

#[doc(hidden)]
pub fn clear_last_dc_quant_table() {
    if let Ok(mut g) = LAST_DC_QUANT_TABLE.lock() {
        *g = None;
    }
}

/// `QuantizeDCPacked` pass: per-channel min/max quantization of DC. Round-trips
/// dequantized values back into the scene so downstream passes see f32.
#[derive(Debug, Clone)]
pub struct QuantizeDCPacked {
    pub bits: u8,
}

impl Default for QuantizeDCPacked {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

impl Pass for QuantizeDCPacked {
    fn name(&self) -> &'static str {
        "QuantizeDCPacked"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        if scene.splats.is_empty() {
            let table = DcQuantTable {
                bits: self.bits.clamp(2, 16),
                mins: [0.0; 3],
                maxs: [1.0; 3],
            };
            if let Ok(mut g) = LAST_DC_QUANT_TABLE.lock() {
                *g = Some(table);
            }
            return Ok(PassStats::default());
        }
        let bits = self.bits.clamp(2, 16);
        let mut mins = [f32::INFINITY; 3];
        let mut maxs = [f32::NEG_INFINITY; 3];
        let dc_of = |s: &catetus_core::Splat| -> [f32; 3] {
            match &s.color {
                Color::Rgb(c) => *c,
                Color::Sh { coeffs, .. } => [
                    coeffs.first().copied().unwrap_or(0.0),
                    coeffs.get(1).copied().unwrap_or(0.0),
                    coeffs.get(2).copied().unwrap_or(0.0),
                ],
            }
        };
        for s in &scene.splats {
            let dc = dc_of(s);
            for i in 0..3 {
                if dc[i] < mins[i] {
                    mins[i] = dc[i];
                }
                if dc[i] > maxs[i] {
                    maxs[i] = dc[i];
                }
            }
        }
        for i in 0..3 {
            if !mins[i].is_finite() {
                mins[i] = 0.0;
            }
            if !maxs[i].is_finite() {
                maxs[i] = 1.0;
            }
            if maxs[i] - mins[i] < 1e-12 {
                mins[i] -= 0.5;
                maxs[i] += 0.5;
            }
        }
        let levels = ((1u32 << bits.min(31)) - 1) as f32;
        let mut modified = 0usize;
        for s in &mut scene.splats {
            let mut dc = dc_of(s);
            for i in 0..3 {
                let span = (maxs[i] - mins[i]).max(1e-12);
                let t = ((dc[i] - mins[i]) / span).clamp(0.0, 1.0);
                let q = (t * levels).round() / levels;
                dc[i] = mins[i] + q * span;
            }
            match &mut s.color {
                Color::Rgb(c) => {
                    if dc != *c {
                        modified += 1;
                    }
                    *c = dc;
                }
                Color::Sh { coeffs, .. } => {
                    if coeffs.len() >= 3
                        && (coeffs[0] != dc[0] || coeffs[1] != dc[1] || coeffs[2] != dc[2])
                    {
                        modified += 1;
                    }
                    if coeffs.len() < 3 {
                        coeffs.resize(3, 0.0);
                    }
                    coeffs[0] = dc[0];
                    coeffs[1] = dc[1];
                    coeffs[2] = dc[2];
                }
            }
        }
        let table = DcQuantTable { bits, mins, maxs };
        if let Ok(mut g) = LAST_DC_QUANT_TABLE.lock() {
            *g = Some(table);
        }
        Ok(PassStats {
            modified,
            notes: vec![format!("bits={bits} mins={mins:?} maxs={maxs:?}")],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod packed_quant_tests {
    use super::*;
    use catetus_core::{Color, Splat};

    fn make_splat(rot: [f32; 4], dc: [f32; 3]) -> Splat {
        let mut coeffs = vec![dc[0], dc[1], dc[2]];
        coeffs.resize(48, 0.0);
        Splat {
            position: [0.0; 3],
            rotation: rot,
            scale: [1.0; 3],
            opacity: 1.0,
            color: Color::Sh { degree: 3, coeffs },
        }
    }

    #[test]
    fn rotation_packed_round_trip_within_step() {
        clear_last_rotation_quant_table();
        let mut scene = SplatScene::new();
        let originals: Vec<[f32; 4]> = (0..16)
            .map(|i| {
                let t = i as f32 * 0.13;
                let q = [t.sin(), t.cos(), (t * 1.7).sin(), (t * 0.6).cos()];
                let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-9);
                [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
            })
            .collect();
        for r in &originals {
            scene.splats.push(make_splat(*r, [0.5; 3]));
        }
        let pass = QuantizeRotationPacked { bits: 8 };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).expect("run");
        let table = take_last_rotation_quant_table().expect("side table parked");
        assert_eq!(table.bits, 8);
        let levels = ((1u32 << 8) - 1) as f32;
        for (i, s) in scene.splats.iter().enumerate() {
            for c in 0..4 {
                let span = (table.maxs[c] - table.mins[c]).max(1e-12);
                let step = span / levels;
                let err = (s.rotation[c] - originals[i][c]).abs();
                assert!(
                    err <= step * 1.5 + 1e-6,
                    "splat {i} c {c}: err={err} > step={step}"
                );
            }
        }
    }

    #[test]
    fn dc_packed_round_trip_within_step() {
        clear_last_dc_quant_table();
        let mut scene = SplatScene::new();
        let originals: Vec<[f32; 3]> = (0..16)
            .map(|i| {
                let t = i as f32 * 0.07;
                [t.sin() * 2.0, t.cos() * 0.5, (t * 1.3).sin() * 1.5]
            })
            .collect();
        for dc in &originals {
            scene.splats.push(make_splat([0.0, 0.0, 0.0, 1.0], *dc));
        }
        let pass = QuantizeDCPacked { bits: 8 };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).expect("run");
        let table = take_last_dc_quant_table().expect("side table parked");
        assert_eq!(table.bits, 8);
        let levels = ((1u32 << 8) - 1) as f32;
        for (i, s) in scene.splats.iter().enumerate() {
            if let Color::Sh { coeffs, .. } = &s.color {
                for c in 0..3 {
                    let span = (table.maxs[c] - table.mins[c]).max(1e-12);
                    let step = span / levels;
                    let err = (coeffs[c] - originals[i][c]).abs();
                    assert!(
                        err <= step * 1.5 + 1e-6,
                        "splat {i} c {c}: err={err} > step={step}"
                    );
                }
            } else {
                panic!("expected SH");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizeRotationSmallest3 — SOG's "smallest-3" quaternion codec. For each
// unit quaternion, drop the largest-magnitude component (recoverable from
// unit-length), sign-flip so the dropped component is non-negative, and store
// the remaining 3 components quantized to `component_bits` bits each. Add a
// 2-bit tag for which component was dropped. Total = 3*component_bits + 2 bits.
//
// At `component_bits = 10` the 32 bits pack into a single u32 (per-splat,
// 4 B). Compared to `QuantizeRotationPacked { bits = 8 }` (also 4 B per splat,
// 4 components * 8 bits), smallest-3 with 10 bits gives ~4x finer per-
// component resolution because the three kept components live in
// [-1/sqrt(2), +1/sqrt(2)] — the dropped component being the largest
// guarantees this.
//
// This pass parks a side table so the GLB writer can emit ROTATION as a
// SCALAR UNSIGNED_INT accessor (componentType 5125) with an
// `SF_quat_smallest3` extension marker carrying `{ component_bits }`. The
// pass round-trips dequantized values into the scene so downstream sees
// exactly what the renderer will reconstruct.
// ---------------------------------------------------------------------------

/// Side table emitted by `QuantizeRotationSmallest3`. Read by the GLB writer.
#[derive(Debug, Clone, Serialize)]
pub struct RotationSmallest3Table {
    /// Bits per stored component (3 components are stored, 1 dropped). Valid
    /// range is 6..=10. At 10 bits the 3*10 + 2-bit-tag layout fits exactly
    /// in a 32-bit word; below 10 the layout is left-padded inside the u32.
    pub component_bits: u8,
}

static LAST_ROTATION_SMALLEST3_TABLE: Mutex<Option<RotationSmallest3Table>> = Mutex::new(None);

/// Drain the last `QuantizeRotationSmallest3` side table.
pub fn take_last_rotation_smallest3_table() -> Option<RotationSmallest3Table> {
    LAST_ROTATION_SMALLEST3_TABLE
        .lock()
        .ok()
        .and_then(|mut g| g.take())
}

#[doc(hidden)]
pub fn clear_last_rotation_smallest3_table() {
    if let Ok(mut g) = LAST_ROTATION_SMALLEST3_TABLE.lock() {
        *g = None;
    }
}

/// `QuantizeRotationSmallest3` pass: SOG-style 3-of-4 quaternion encoding.
#[derive(Debug, Clone)]
pub struct QuantizeRotationSmallest3 {
    /// Bits per stored component; valid 6..=10. At 10 bits the full layout
    /// fits in a single u32 (3*10 + 2-bit tag = 32).
    pub component_bits: u8,
}

impl Default for QuantizeRotationSmallest3 {
    fn default() -> Self {
        Self { component_bits: 10 }
    }
}

/// Quantize a value in `[-1/sqrt(2), +1/sqrt(2)]` to an unsigned `bits`-bit
/// integer using SOG's mapping: `q = round((v / sqrt(2) + 0.5) * levels)`,
/// where `levels = (1 << bits) - 1`. Decoder inverse:
/// `v = (q / levels - 0.5) * sqrt(2)`. Exactly matches the PlayCanvas
/// `unpackQuat` decoder at `tasks/scripts/splat-transform-bench/.../index.mjs`
/// (search `unpackQuat` / `decodeRotationInto`).
#[inline]
fn quantize_smallest3_component(v: f32, bits: u8) -> u32 {
    let levels = ((1u32 << bits) - 1) as f32;
    let sqrt2 = std::f32::consts::SQRT_2;
    let t = (v / sqrt2 + 0.5).clamp(0.0, 1.0);
    (t * levels).round() as u32
}

#[inline]
fn dequantize_smallest3_component(q: u32, bits: u8) -> f32 {
    let levels = ((1u32 << bits) - 1) as f32;
    let sqrt2 = std::f32::consts::SQRT_2;
    (q as f32 / levels) * sqrt2 - sqrt2 * 0.5
}

/// Encode a (possibly non-unit) quaternion `[x, y, z, w]` into the smallest-3
/// integer triple plus the 2-bit max-component tag. Returns
/// `(q0, q1, q2, tag)` where `tag` is the original index of the dropped
/// (largest-magnitude) component. The encoder normalizes first, then
/// sign-flips so the dropped component is non-negative (the decoder
/// reconstructs it via sqrt(1 - sum_others^2), always non-negative).
fn encode_smallest3(rot: [f32; 4], bits: u8) -> (u32, u32, u32, u8) {
    let n = (rot.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-12);
    let q = [rot[0] / n, rot[1] / n, rot[2] / n, rot[3] / n];
    let mut tag: u8 = 0;
    let mut best = q[0].abs();
    for i in 1..4 {
        let a = q[i].abs();
        if a > best {
            best = a;
            tag = i as u8;
        }
    }
    let s = if q[tag as usize] < 0.0 { -1.0 } else { 1.0 };
    let qs = [q[0] * s, q[1] * s, q[2] * s, q[3] * s];
    let mut comps = [0u32; 3];
    let mut k = 0usize;
    for i in 0..4 {
        if i as u8 == tag {
            continue;
        }
        comps[k] = quantize_smallest3_component(qs[i], bits);
        k += 1;
    }
    (comps[0], comps[1], comps[2], tag)
}

/// Decode the smallest-3 triple + tag back into a unit quaternion `[x, y, z, w]`.
fn decode_smallest3(q0: u32, q1: u32, q2: u32, tag: u8, bits: u8) -> [f32; 4] {
    let a = dequantize_smallest3_component(q0, bits);
    let b = dequantize_smallest3_component(q1, bits);
    let c = dequantize_smallest3_component(q2, bits);
    let mut out = [0f32; 4];
    let mut k = 0usize;
    let stored = [a, b, c];
    for i in 0..4 {
        if i as u8 == tag {
            continue;
        }
        out[i] = stored[k];
        k += 1;
    }
    let s2 = stored[0] * stored[0] + stored[1] * stored[1] + stored[2] * stored[2];
    out[tag as usize] = (1.0 - s2.min(1.0)).max(0.0).sqrt();
    out
}

/// Pack `(q0, q1, q2, tag)` into a single u32 using `component_bits` per
/// component plus a 2-bit tag in the high bits — exactly the SOG V2 layout.
#[inline]
pub fn pack_smallest3_u32(q0: u32, q1: u32, q2: u32, tag: u8, bits: u8) -> u32 {
    let mask = (1u32 << bits) - 1;
    (q0 & mask) | ((q1 & mask) << bits) | ((q2 & mask) << (2 * bits)) | (((tag as u32) & 3) << 30)
}

/// Inverse of `pack_smallest3_u32` — extract `(q0, q1, q2, tag)` from a u32.
#[inline]
pub fn unpack_smallest3_u32(packed: u32, bits: u8) -> (u32, u32, u32, u8) {
    let mask = (1u32 << bits) - 1;
    let q0 = packed & mask;
    let q1 = (packed >> bits) & mask;
    let q2 = (packed >> (2 * bits)) & mask;
    let tag = ((packed >> 30) & 3) as u8;
    (q0, q1, q2, tag)
}

impl Pass for QuantizeRotationSmallest3 {
    fn name(&self) -> &'static str {
        "QuantizeRotationSmallest3"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let bits = self.component_bits.clamp(6, 10);
        let table = RotationSmallest3Table {
            component_bits: bits,
        };
        if let Ok(mut g) = LAST_ROTATION_SMALLEST3_TABLE.lock() {
            *g = Some(table);
        }
        if scene.splats.is_empty() {
            return Ok(PassStats::default());
        }
        let mut modified = 0usize;
        for s in &mut scene.splats {
            let original = s.rotation;
            let (q0, q1, q2, tag) = encode_smallest3(original, bits);
            let decoded = decode_smallest3(q0, q1, q2, tag, bits);
            for c in 0..4 {
                if (decoded[c] - original[c]).abs() > f32::EPSILON {
                    modified += 1;
                    break;
                }
            }
            s.rotation = decoded;
        }
        Ok(PassStats {
            modified,
            notes: vec![format!(
                "component_bits={bits} layout=u32 (3*{bits}+2-bit tag = {} bits)",
                3 * bits + 2
            )],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod smallest3_tests {
    use super::*;
    use catetus_core::{Color, Splat};

    fn make_splat(rot: [f32; 4]) -> Splat {
        let mut coeffs = vec![0.0_f32; 48];
        for v in coeffs.iter_mut().take(3) {
            *v = 0.5;
        }
        Splat {
            position: [0.0; 3],
            rotation: rot,
            scale: [1.0; 3],
            opacity: 1.0,
            color: Color::Sh { degree: 3, coeffs },
        }
    }

    #[test]
    fn smallest3_round_trip_close_to_input() {
        clear_last_rotation_smallest3_table();
        let mut scene = SplatScene::new();
        let originals: Vec<[f32; 4]> = (0..32)
            .map(|i| {
                let t = i as f32 * 0.17;
                let q = [t.sin(), t.cos(), (t * 1.7).sin(), (t * 0.6).cos()];
                let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-9);
                [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
            })
            .collect();
        for r in &originals {
            scene.splats.push(make_splat(*r));
        }
        let pass = QuantizeRotationSmallest3 { component_bits: 10 };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).expect("run");
        let table = take_last_rotation_smallest3_table().expect("side table parked");
        assert_eq!(table.component_bits, 10);
        let tol = 5e-3;
        for (i, s) in scene.splats.iter().enumerate() {
            // Account for global sign ambiguity (q and -q encode the same rotation).
            let dot: f32 = (0..4).map(|c| s.rotation[c] * originals[i][c]).sum();
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            for c in 0..4 {
                let err = (s.rotation[c] - sign * originals[i][c]).abs();
                assert!(err < tol, "splat {i} c {c}: err={err} > tol={tol}");
            }
            let n2: f32 = (0..4).map(|c| s.rotation[c] * s.rotation[c]).sum();
            assert!((n2 - 1.0).abs() < 1e-3, "splat {i}: |q|^2 = {n2}");
        }
    }

    #[test]
    fn smallest3_pack_unpack_round_trip() {
        for bits in 6u8..=10 {
            let mask = (1u32 << bits) - 1;
            let q0 = 1234 % (mask + 1);
            let q1 = 5678 % (mask + 1);
            let q2 = 9012 % (mask + 1);
            for tag in 0u8..4 {
                let packed = pack_smallest3_u32(q0, q1, q2, tag, bits);
                let (a, b, c, t) = unpack_smallest3_u32(packed, bits);
                assert_eq!((a, b, c, t), (q0, q1, q2, tag), "bits={bits} tag={tag}");
            }
        }
    }

    #[test]
    fn smallest3_decoder_matches_sog_reference() {
        // Cross-check against the SOG V2 decoder
        // (`tasks/scripts/splat-transform-bench/.../index.mjs` → `decodeRotationInto`):
        //   d0 = (v & 1023) / 1023
        //   qx = d0 * sqrt(2) - 1/sqrt(2)
        let bits = 10u8;
        let levels = ((1u32 << bits) - 1) as f32;
        let q = [0.05f32, -0.10, 0.20, 0.97];
        let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let qn = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
        let (q0, q1, q2, tag) = encode_smallest3(qn, bits);
        let sqrt2 = std::f32::consts::SQRT_2;
        let d0 = q0 as f32 / levels;
        let d1 = q1 as f32 / levels;
        let d2 = q2 as f32 / levels;
        let a = d0 * sqrt2 - sqrt2 * 0.5;
        let b = d1 * sqrt2 - sqrt2 * 0.5;
        let c = d2 * sqrt2 - sqrt2 * 0.5;
        let dropped = (1.0 - (a * a + b * b + c * c).min(1.0)).max(0.0).sqrt();
        let mut out = [0f32; 4];
        let mut k = 0usize;
        let stored = [a, b, c];
        for i in 0..4 {
            if i as u8 == tag {
                continue;
            }
            out[i] = stored[k];
            k += 1;
        }
        out[tag as usize] = dropped;
        let ours = decode_smallest3(q0, q1, q2, tag, bits);
        for i in 0..4 {
            assert!(
                (out[i] - ours[i]).abs() < 1e-6,
                "JS-vs-Rust mismatch at i={i}: js={} rs={}",
                out[i],
                ours[i]
            );
        }
    }
}

#[cfg(test)]
mod codecgs_lite_tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        // We don't have a stable repo root in unit tests; cover just the
        // identifier so consumers can match on it in pipeline reports.
        let p = CodecGSLite {
            input_ply: std::path::PathBuf::from("/dev/null"),
            out_glb_path: std::path::PathBuf::from("/dev/null"),
            repo_root: std::path::PathBuf::from("/dev/null"),
            codec: CodecGSKind::H264,
            crf: 30,
            sort: "auto".to_string(),
        };
        assert_eq!(p.name(), "CodecGSLite");
    }

    #[test]
    fn is_supported_returns_err_when_scripts_missing() {
        // `/dev/null` is not a repo; expect a clean error rather than a
        // panic.
        let r = CodecGSLite::is_supported(std::path::Path::new("/dev/null"));
        assert!(r.is_err());
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

/// Merge near-duplicate splats: position-neighbors within `voxel_size_world`
/// whose normalized attribute distance is ≤ `max_attr_distance` are replaced
/// with a single representative splat.
///
/// Pareto-optimal on `benches/scenes/real/bonsai_iter7000.ply` (1.157M splats):
///
/// | voxel | attr | count ratio | preset bytes ratio | MLP fidelity |
/// |-------|------|-------------|---------------------|--------------|
/// | 0.10  | 0.4  | 1.64×       | 1.61×               | 0.7720       |
/// | 0.10  | 0.6  | 2.79×       | 2.73×               | 0.7034       |
/// | 0.10  | 0.8  | 4.49×       | 4.37×               | 0.6336       |
///
/// (baseline self-MLP = 0.8414; v=0.10 / a=0.6 is the chosen size-min default.)
///
/// Algorithm (mirrors `experiments/bundle/code/bundle_proto.py`):
/// 1. Uniform-grid spatial hash with cell = voxel_size_world.
/// 2. Two-pass scan: for each splat, examine ±1 cell; merge with splat j
///    iff `||pos_i - pos_j|| <= voxel_size_world` AND
///    Euclidean distance over z-normalized
///    `(scale_log, color_dc, opacity_logit)` ≤ `max_attr_distance`.
/// 3. Union-find groups → one representative splat each:
///    - position    = opacity-weighted centroid
///    - scale       = max(group_extent/2, dominant_scale) per axis
///    - rotation    = dominant splat's quaternion (renormalized)
///    - color       = opacity-weighted mean (DC or full SH)
///    - opacity     = 1 − ∏(1 − αᵢ)   (Beer–Lambert composition)
#[derive(Debug, Clone)]
pub struct BundleNeighbors {
    /// Position-neighbor radius in scene world units. Two splats are merge
    /// candidates iff their position distance ≤ this value.
    pub voxel_size_world: f32,
    /// Attribute-distance cap. Distance is Euclidean over the z-normalized
    /// 7-D feature `(scale_log[0..3], dc[0..3], opacity_logit)`. Typical
    /// usable range: 0.2 (very strict, ~10 % compression) → 0.8 (aggressive,
    /// 3–4× compression at noticeable fidelity loss).
    pub max_attr_distance: f32,
}

impl Default for BundleNeighbors {
    fn default() -> Self {
        Self {
            voxel_size_world: 0.1,
            max_attr_distance: 0.5,
        }
    }
}

impl Pass for BundleNeighbors {
    fn name(&self) -> &'static str {
        "BundleNeighbors"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        if n < 2 || self.voxel_size_world <= 0.0 || self.max_attr_distance <= 0.0 {
            return Ok(PassStats::default());
        }

        // ---- 1. Collect positions + 7-D attribute feature.
        let mut pos = vec![[0.0f32; 3]; n];
        let mut feat = vec![[0.0f32; 7]; n];
        for (i, s) in scene.splats.iter().enumerate() {
            pos[i] = s.position;
            // Log-space scale, clamped to avoid -inf for zero scales.
            feat[i][0] = s.scale[0].max(1e-12).ln();
            feat[i][1] = s.scale[1].max(1e-12).ln();
            feat[i][2] = s.scale[2].max(1e-12).ln();
            let dc = match &s.color {
                Color::Rgb(c) => *c,
                Color::Sh { coeffs, .. } => {
                    let mut d = [0.0f32; 3];
                    for k in 0..3 {
                        d[k] = coeffs.get(k).copied().unwrap_or(0.0);
                    }
                    d
                }
            };
            feat[i][3] = dc[0];
            feat[i][4] = dc[1];
            feat[i][5] = dc[2];
            // Opacity logit. Linear in [0,1] is the IR convention; clamp away
            // from the asymptotes so we don't blow up sigma below.
            let p = s.opacity.clamp(1e-6, 1.0 - 1e-6);
            feat[i][6] = (p / (1.0 - p)).ln();
        }

        // ---- 2. Z-normalize feature columns (use f64 accumulation; bonsai's
        // 1M-row sums underflow in f32 for the variance step).
        let mut mu = [0.0f64; 7];
        for v in &feat {
            for k in 0..7 {
                mu[k] += v[k] as f64;
            }
        }
        for k in 0..7 {
            mu[k] /= n as f64;
        }
        let mut var = [0.0f64; 7];
        for v in &feat {
            for k in 0..7 {
                let d = v[k] as f64 - mu[k];
                var[k] += d * d;
            }
        }
        let mut sigma = [0.0f32; 7];
        for k in 0..7 {
            sigma[k] = ((var[k] / (n as f64)).sqrt() as f32).max(1e-6);
        }
        let mut feat_n = vec![[0.0f32; 7]; n];
        for i in 0..n {
            for k in 0..7 {
                feat_n[i][k] = (feat[i][k] - mu[k] as f32) / sigma[k];
            }
        }

        // ---- 3. Build sparse uniform-grid spatial hash, cell = voxel_size_world.
        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        for p in &pos {
            for k in 0..3 {
                if p[k] < bmin[k] {
                    bmin[k] = p[k];
                }
                if p[k] > bmax[k] {
                    bmax[k] = p[k];
                }
            }
        }
        let cell = self.voxel_size_world.max(1e-9);
        let cell_of = |p: [f32; 3]| -> (i64, i64, i64) {
            (
                ((p[0] - bmin[0]) / cell) as i64,
                ((p[1] - bmin[1]) / cell) as i64,
                ((p[2] - bmin[2]) / cell) as i64,
            )
        };
        let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> =
            std::collections::HashMap::new();
        for (i, p) in pos.iter().enumerate() {
            grid.entry(cell_of(*p)).or_default().push(i as u32);
        }

        // ---- 4. Union-find over near-duplicate pairs.
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn uf_find(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        let voxel2 = (self.voxel_size_world as f64) * (self.voxel_size_world as f64);
        let attr2 = (self.max_attr_distance as f64) * (self.max_attr_distance as f64);

        for i in 0..n {
            let p = pos[i];
            let (cx, cy, cz) = cell_of(p);
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let key = (cx + dx, cy + dy, cz + dz);
                        let Some(bucket) = grid.get(&key) else {
                            continue;
                        };
                        for &j_u in bucket {
                            let j = j_u as usize;
                            if j <= i {
                                continue;
                            }
                            // Position test.
                            let q = pos[j];
                            let dpx = (q[0] - p[0]) as f64;
                            let dpy = (q[1] - p[1]) as f64;
                            let dpz = (q[2] - p[2]) as f64;
                            let d2 = dpx * dpx + dpy * dpy + dpz * dpz;
                            if d2 > voxel2 {
                                continue;
                            }
                            // Attribute test.
                            let fi = &feat_n[i];
                            let fj = &feat_n[j];
                            let mut a2 = 0.0f64;
                            for k in 0..7 {
                                let d = (fj[k] - fi[k]) as f64;
                                a2 += d * d;
                                if a2 > attr2 {
                                    break;
                                }
                            }
                            if a2 <= attr2 {
                                let ra = uf_find(&mut parent, i as u32);
                                let rb = uf_find(&mut parent, j_u);
                                if ra != rb {
                                    if ra < rb {
                                        parent[rb as usize] = ra;
                                    } else {
                                        parent[ra as usize] = rb;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- 5. Group members by root.
        let mut roots = vec![0u32; n];
        for i in 0..n {
            roots[i] = uf_find(&mut parent, i as u32);
        }
        let mut comp_id: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut label = vec![0u32; n];
        for i in 0..n {
            let next = comp_id.len() as u32;
            let id = *comp_id.entry(roots[i]).or_insert(next);
            label[i] = id;
        }
        let n_comp = comp_id.len();
        let mut members: Vec<Vec<u32>> = (0..n_comp).map(|_| Vec::new()).collect();
        for i in 0..n {
            members[label[i] as usize].push(i as u32);
        }

        // ---- 6. Build merged splat vec.
        let original = std::mem::take(&mut scene.splats);
        let mut new_splats: Vec<catetus_core::Splat> = Vec::with_capacity(n_comp);
        let mut n_merged_groups = 0usize;
        let mut n_splats_merged = 0usize;
        for grp in &members {
            if grp.len() == 1 {
                new_splats.push(original[grp[0] as usize].clone());
                continue;
            }
            n_merged_groups += 1;
            n_splats_merged += grp.len();

            let mut total_w = 0.0f64;
            let mut dominant: usize = grp[0] as usize;
            let mut dom_w = -1.0f64;
            let mut sum_log1m = 0.0f64;
            for &gi in grp {
                let s = &original[gi as usize];
                let w = s.opacity.clamp(0.0, 1.0 - 1e-7) as f64;
                total_w += w;
                if w > dom_w {
                    dom_w = w;
                    dominant = gi as usize;
                }
                sum_log1m += (1.0 - w).ln();
            }
            let use_count_weights = total_w <= 1e-9;
            let denom = if use_count_weights {
                grp.len() as f64
            } else {
                total_w
            };
            let weight_of = |s: &catetus_core::Splat| -> f64 {
                if use_count_weights {
                    1.0
                } else {
                    s.opacity.clamp(0.0, 1.0 - 1e-7) as f64
                }
            };

            // Weighted-mean position + bbox extent.
            let mut px = 0.0f64;
            let mut py = 0.0f64;
            let mut pz = 0.0f64;
            let mut emin = [f32::INFINITY; 3];
            let mut emax = [f32::NEG_INFINITY; 3];
            for &gi in grp {
                let s = &original[gi as usize];
                let w = weight_of(s);
                px += s.position[0] as f64 * w;
                py += s.position[1] as f64 * w;
                pz += s.position[2] as f64 * w;
                for k in 0..3 {
                    if s.position[k] < emin[k] {
                        emin[k] = s.position[k];
                    }
                    if s.position[k] > emax[k] {
                        emax[k] = s.position[k];
                    }
                }
            }
            let new_pos = [
                (px / denom) as f32,
                (py / denom) as f32,
                (pz / denom) as f32,
            ];

            // Weighted-mean color (preserve Rgb / Sh shape from dominant splat).
            let dom_color = original[dominant].color.clone();
            let new_color = match &dom_color {
                Color::Rgb(_) => {
                    let mut r = 0.0f64;
                    let mut g = 0.0f64;
                    let mut b = 0.0f64;
                    for &gi in grp {
                        let s = &original[gi as usize];
                        let w = weight_of(s);
                        if let Color::Rgb(c) = s.color {
                            r += c[0] as f64 * w;
                            g += c[1] as f64 * w;
                            b += c[2] as f64 * w;
                        }
                    }
                    Color::Rgb([(r / denom) as f32, (g / denom) as f32, (b / denom) as f32])
                }
                Color::Sh { degree, coeffs } => {
                    let cn = coeffs.len();
                    let mut sum = vec![0.0f64; cn];
                    for &gi in grp {
                        let s = &original[gi as usize];
                        let w = weight_of(s);
                        if let Color::Sh { coeffs: cc, .. } = &s.color {
                            for k in 0..cn.min(cc.len()) {
                                sum[k] += cc[k] as f64 * w;
                            }
                        }
                    }
                    Color::Sh {
                        degree: *degree,
                        coeffs: sum.iter().map(|v| (*v / denom) as f32).collect(),
                    }
                }
            };

            // Scale: cover the group extent OR keep the dominant size.
            let dom_scale = original[dominant].scale;
            let new_scale = [
                ((emax[0] - emin[0]) * 0.5).max(dom_scale[0]),
                ((emax[1] - emin[1]) * 0.5).max(dom_scale[1]),
                ((emax[2] - emin[2]) * 0.5).max(dom_scale[2]),
            ];

            // Beer–Lambert opacity composition.
            let alpha = (1.0 - sum_log1m.exp()).clamp(0.0, 1.0) as f32;

            // Rotation: dominant + renormalize.
            let mut rot = original[dominant].rotation;
            let nrm =
                (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3]).sqrt();
            if nrm > 1e-12 {
                for k in 0..4 {
                    rot[k] /= nrm;
                }
            }

            new_splats.push(catetus_core::Splat {
                position: new_pos,
                rotation: rot,
                scale: new_scale,
                opacity: alpha,
                color: new_color,
            });
        }
        scene.splats = new_splats;

        Ok(PassStats {
            removed: n - scene.splats.len(),
            notes: vec![format!(
                "components={} merged_groups={} splats_merged={} voxel={} attr={}",
                n_comp,
                n_merged_groups,
                n_splats_merged,
                self.voxel_size_world,
                self.max_attr_distance
            )],
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// SHDCTQuantize
// ---------------------------------------------------------------------------

/// Per-frequency bit-allocation tables.
///
/// SH coefficients in `Color::Sh { coeffs }` are stored RGB-interleaved per
/// band: `coeffs[3*k + c]` is band-k channel-c (c ∈ {0=R, 1=G, 2=B}).
/// `n = (degree+1)^2` is the band count.
///
/// This pass applies a 1D DCT-II of length `n` to each per-channel band-vector
/// (R-only, G-only, B-only), quantizes each frequency bin with the given bit
/// budget, then inverse-DCTs back so the downstream pipeline sees plain SH
/// coefficients. The size win shows up at the writer when those quantized
/// coefficients zstd / brotli much better than the raw float distribution.
///
/// Pareto from `experiments/sh-dct/` (bonsai_iter7000, deg-3):
/// - `uniform(7)` → SH payload zstd ~40.8 MB (vs truncate_deg1 46.6 MB),
///   ΔE94 mean 0.61 (vs truncate_deg1 1.89). 12.4% smaller, 3.1× lower ΔE94.
/// - `uniform(8)` → 48.8 MB, ΔE94 0.30 (the lowest-distortion option).
#[derive(Debug, Clone)]
pub struct SHDCTQuantize {
    /// Bits per frequency bin. Use `0` to drop a bin (it round-trips to zero).
    /// If shorter than the SH band count, missing entries are treated as 0.
    pub bit_alloc: Vec<u8>,
}

impl SHDCTQuantize {
    /// Uniform N-bit quantization across every DCT frequency (the Pareto
    /// winner from Wave-1 experiments — per-frequency tables underperformed).
    pub fn uniform(bits: u8) -> Self {
        Self {
            bit_alloc: vec![bits; 16],
        }
    }
}

fn sh_dct_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    let pi = std::f32::consts::PI;
    let inv_sqrt_n = (1.0_f32 / n as f32).sqrt();
    let two_over_n_sqrt = (2.0_f32 / n as f32).sqrt();
    for k in 0..n {
        let alpha = if k == 0 { inv_sqrt_n } else { two_over_n_sqrt };
        for nn in 0..n {
            m[k * n + nn] =
                alpha * ((pi * (2.0 * nn as f32 + 1.0) * k as f32) / (2.0 * n as f32)).cos();
        }
    }
    m
}

fn sh_mat_apply(b: &[f32], n: usize, x: &mut [f32]) {
    let mut y = vec![0.0f32; n];
    for k in 0..n {
        let mut acc = 0.0f32;
        for nn in 0..n {
            acc += b[k * n + nn] * x[nn];
        }
        y[k] = acc;
    }
    x.copy_from_slice(&y);
}

fn sh_mat_apply_t(b: &[f32], n: usize, x: &mut [f32]) {
    let mut y = vec![0.0f32; n];
    for k in 0..n {
        let mut acc = 0.0f32;
        for nn in 0..n {
            acc += b[nn * n + k] * x[nn];
        }
        y[k] = acc;
    }
    x.copy_from_slice(&y);
}

impl Pass for SHDCTQuantize {
    fn name(&self) -> &'static str {
        "SHDCTQuantize"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        // Find the SH band count from the first SH splat.
        let n = scene
            .splats
            .iter()
            .find_map(|s| match &s.color {
                Color::Sh { coeffs, .. } => Some(coeffs.len() / 3),
                _ => None,
            })
            .unwrap_or(0);
        if n <= 1 {
            // Nothing to DCT — n=1 is just the DC term, no frequency axis.
            return Ok(PassStats::default());
        }

        let b = sh_dct_matrix(n);

        // Pass 1: forward DCT per splat per channel; track per-(c, k) max.
        // We use per-frequency scales (shared across the corpus, computed
        // online) so that quantization stays within representable range.
        let mut y_all = Vec::<Vec<f32>>::with_capacity(scene.splats.len());
        let mut max_abs = vec![0.0f32; 3 * n];

        for s in &scene.splats {
            let Color::Sh { coeffs, .. } = &s.color else {
                y_all.push(Vec::new());
                continue;
            };
            if coeffs.len() != 3 * n {
                // Mixed-degree scene (unusual). Skip non-conforming splats.
                y_all.push(Vec::new());
                continue;
            }
            let mut buf = vec![0.0f32; 3 * n];
            for c in 0..3 {
                // Gather the per-channel band vector at stride 3.
                for k in 0..n {
                    buf[c * n + k] = coeffs[3 * k + c];
                }
                let slice = &mut buf[c * n..(c + 1) * n];
                sh_mat_apply(&b, n, slice);
                for (k, v) in slice.iter().enumerate() {
                    let a = v.abs();
                    if a > max_abs[c * n + k] {
                        max_abs[c * n + k] = a;
                    }
                }
            }
            y_all.push(buf);
        }

        // Pass 2: per-(c, k) scale from the corpus max + bit budget.
        let mut scale = vec![0.0f32; 3 * n];
        for c in 0..3 {
            for k in 0..n {
                let bits = self.bit_alloc.get(k).copied().unwrap_or(0);
                if bits == 0 || max_abs[c * n + k] == 0.0 {
                    continue;
                }
                let qmax = ((1u32 << (bits.saturating_sub(1))) - 1).max(1) as f32;
                scale[c * n + k] = max_abs[c * n + k] / qmax;
            }
        }

        // Pass 3: quant → dequant → inverse DCT → write back.
        let mut modified = 0usize;
        for (s, mut buf) in scene.splats.iter_mut().zip(y_all.into_iter()) {
            if buf.is_empty() {
                continue;
            }
            for c in 0..3 {
                for k in 0..n {
                    let bits = self.bit_alloc.get(k).copied().unwrap_or(0);
                    if bits == 0 {
                        buf[c * n + k] = 0.0;
                        continue;
                    }
                    let sc = scale[c * n + k];
                    if sc == 0.0 {
                        buf[c * n + k] = 0.0;
                        continue;
                    }
                    let qmax = ((1u32 << (bits.saturating_sub(1))) - 1).max(1) as i32;
                    let q = (buf[c * n + k] / sc).round() as i32;
                    let q = q.clamp(-qmax, qmax);
                    buf[c * n + k] = q as f32 * sc;
                }
                let slice = &mut buf[c * n..(c + 1) * n];
                sh_mat_apply_t(&b, n, slice);
            }
            if let Color::Sh { coeffs, .. } = &mut s.color {
                if coeffs.len() == 3 * n {
                    for c in 0..3 {
                        for k in 0..n {
                            coeffs[3 * k + c] = buf[c * n + k];
                        }
                    }
                    modified += 1;
                }
            }
        }

        Ok(PassStats {
            modified,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod sh_dct_tests {
    use super::*;
    use catetus_core::{Color, Splat, SplatScene};

    fn make_sh_splat(degree: u8, coeffs: Vec<f32>) -> Splat {
        Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1; 3],
            opacity: 0.9,
            color: Color::Sh { degree, coeffs },
        }
    }

    #[test]
    fn sh_dct_round_trip_is_close_at_uniform_8() {
        // Synthesize a 4-band (degree-1) SH splat with non-trivial coeffs.
        let coeffs: Vec<f32> = (0..(3 * 4)).map(|i| (i as f32) * 0.05).collect();
        let original = coeffs.clone();
        let mut scene = SplatScene::new();
        scene.splats.push(make_sh_splat(1, coeffs));
        let pass = SHDCTQuantize::uniform(8);
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).unwrap();
        let Color::Sh {
            coeffs: roundtripped,
            ..
        } = &scene.splats[0].color
        else {
            panic!("expected SH");
        };
        // 8-bit quant on a single-splat corpus should round-trip near-exactly.
        let max_err = original
            .iter()
            .zip(roundtripped.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "8-bit DCT-quant error {} too large",
            max_err
        );
    }

    #[test]
    fn sh_dct_zero_bits_zeroes_the_coeffs() {
        let coeffs: Vec<f32> = (0..(3 * 4)).map(|i| (i as f32) * 0.05).collect();
        let mut scene = SplatScene::new();
        scene.splats.push(make_sh_splat(1, coeffs));
        let pass = SHDCTQuantize {
            bit_alloc: vec![0; 16],
        };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).unwrap();
        let Color::Sh {
            coeffs: roundtripped,
            ..
        } = &scene.splats[0].color
        else {
            panic!();
        };
        assert!(roundtripped.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn sh_dct_is_noop_on_rgb_color() {
        let mut scene = SplatScene::new();
        scene.splats.push(Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1; 3],
            opacity: 0.5,
            color: Color::Rgb([0.2, 0.4, 0.6]),
        });
        let pass = SHDCTQuantize::uniform(7);
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(stats.modified, 0);
    }
}

#[cfg(test)]
mod bundle_tests {
    use super::*;
    use catetus_core::{Color, Splat, SplatScene};

    fn make_splat(pos: [f32; 3], color: [f32; 3], opacity: f32, scale: [f32; 3]) -> Splat {
        Splat {
            position: pos,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale,
            opacity,
            color: Color::Rgb(color),
        }
    }

    #[test]
    fn bundle_merges_near_duplicates() {
        // Three near-identical splats clustered tightly; two outliers far away
        // and with very different colors. Bundle should produce 3 components:
        // {0,1,2} -> 1 merged splat, {3} singleton, {4} singleton.
        let mut scene = SplatScene::new();
        for _ in 0..3 {
            scene.splats.push(make_splat(
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                0.5,
                [0.01, 0.01, 0.01],
            ));
        }
        // perturb positions slightly within the voxel
        scene.splats[1].position = [0.01, 0.0, 0.0];
        scene.splats[2].position = [0.0, 0.01, 0.0];
        scene.splats.push(make_splat(
            [10.0, 10.0, 10.0],
            [1.0, 0.0, 0.0],
            0.9,
            [0.01, 0.01, 0.01],
        ));
        scene.splats.push(make_splat(
            [-10.0, -10.0, -10.0],
            [0.0, 1.0, 0.0],
            0.2,
            [0.01, 0.01, 0.01],
        ));

        let pass = BundleNeighbors {
            voxel_size_world: 0.1,
            max_attr_distance: 0.5,
        };
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(scene.splats.len(), 3, "should collapse to 3 splats");
        assert_eq!(stats.removed, 2);
    }

    #[test]
    fn bundle_beer_lambert_opacity() {
        // Two splats with α=0.5 should merge to α = 1 - 0.25 = 0.75.
        let mut scene = SplatScene::new();
        scene.splats.push(make_splat(
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            0.5,
            [0.01, 0.01, 0.01],
        ));
        scene.splats.push(make_splat(
            [0.001, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            0.5,
            [0.01, 0.01, 0.01],
        ));
        let pass = BundleNeighbors {
            voxel_size_world: 0.1,
            max_attr_distance: 0.5,
        };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(scene.splats.len(), 1);
        let alpha = scene.splats[0].opacity;
        assert!(
            (alpha - 0.75).abs() < 1e-5,
            "Beer-Lambert composition: got {alpha}, expected 0.75"
        );
    }

    #[test]
    fn bundle_skips_when_attr_diverges() {
        // Two splats at the same position but very different colors should
        // NOT merge.
        let mut scene = SplatScene::new();
        scene.splats.push(make_splat(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            [0.01, 0.01, 0.01],
        ));
        scene.splats.push(make_splat(
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            0.5,
            [0.01, 0.01, 0.01],
        ));
        let pass = BundleNeighbors {
            voxel_size_world: 0.1,
            max_attr_distance: 0.1, // strict
        };
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(scene.splats.len(), 2);
        assert_eq!(stats.removed, 0);
    }

    #[test]
    fn bundle_noop_on_empty_or_tiny_scene() {
        let mut empty = SplatScene::new();
        let pass = BundleNeighbors::default();
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut empty, &mut ctx).expect("run");
        assert_eq!(stats.removed, 0);

        let mut single = SplatScene::new();
        single.splats.push(make_splat(
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            0.5,
            [0.01, 0.01, 0.01],
        ));
        let stats = pass.run(&mut single, &mut ctx).expect("run");
        assert_eq!(stats.removed, 0);
    }
}

#[cfg(test)]
mod rd_prune_tests {
    use super::*;
    use catetus_core::{Color, Splat, SplatScene};

    fn rgb_splat(opacity: f32, scale_max: f32) -> Splat {
        Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [scale_max * 0.1, scale_max * 0.1, scale_max],
            opacity,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        }
    }

    #[test]
    fn rd_prune_keeps_high_distortion_splats() {
        // Five splats: distortion ~ opacity * smax^2. Construct so the
        // ranking is unambiguous and the top-2 are the last two we push.
        let mut scene = SplatScene::new();
        scene.splats.push(rgb_splat(0.01, 0.01)); // d = 1e-2 * 1e-4 = 1e-6
        scene.splats.push(rgb_splat(0.05, 0.02)); // d = 5e-2 * 4e-4 = 2e-5
        scene.splats.push(rgb_splat(0.10, 0.05)); // d = 1e-1 * 2.5e-3 = 2.5e-4
        scene.splats.push(rgb_splat(0.50, 0.10)); // d = 5e-1 * 1e-2 = 5e-3
        scene.splats.push(rgb_splat(0.90, 0.20)); // d = 9e-1 * 4e-2 = 3.6e-2

        let pass = RDPrune { target_ratio: 0.4 }; // keep 2 of 5
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(stats.removed, 3);
        assert_eq!(scene.splats.len(), 2);
        // The two surviving splats are the original last two (largest
        // distortion). Their opacities are 0.5 and 0.9 in some order.
        let mut ops: Vec<f32> = scene.splats.iter().map(|s| s.opacity).collect();
        ops.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ops[0] - 0.5).abs() < 1e-6);
        assert!((ops[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rd_prune_ratio_one_is_noop() {
        let mut scene = SplatScene::new();
        for i in 0..10 {
            scene.splats.push(rgb_splat(0.1 * (i as f32 + 1.0), 0.05));
        }
        let before = scene.splats.len();
        let pass = RDPrune { target_ratio: 1.0 };
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(stats.removed, 0);
        assert_eq!(scene.splats.len(), before);
    }

    #[test]
    fn rd_prune_handles_ties_exactly() {
        // 10 identical splats — every distortion is equal. The pass must
        // still produce exactly `round(ratio * n)` survivors via the
        // tie-break budget.
        let mut scene = SplatScene::new();
        for _ in 0..10 {
            scene.splats.push(rgb_splat(0.5, 0.1));
        }
        let pass = RDPrune { target_ratio: 0.3 }; // keep 3
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(scene.splats.len(), 3);
        assert_eq!(stats.removed, 7);
    }

    #[test]
    fn rd_prune_respects_sh_higher_band_energy() {
        // Two splats with identical opacity + scale but different SH
        // higher-band energies. The one with non-zero rest must rank
        // strictly above the other, so at ratio=0.5 we keep the SH-rich.
        let mut scene = SplatScene::new();
        let flat = Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1, 0.1, 0.1],
            opacity: 0.5,
            color: Color::Sh {
                degree: 1,
                // 3 * (1+1)^2 = 12 floats; rest is zero.
                coeffs: {
                    let mut v = vec![0.0; 12];
                    v[0] = 0.4;
                    v[1] = 0.4;
                    v[2] = 0.4;
                    v
                },
            },
        };
        let rich = Splat {
            color: Color::Sh {
                degree: 1,
                coeffs: {
                    let mut v = vec![0.5; 12]; // every band non-trivial
                    v[0] = 0.4;
                    v[1] = 0.4;
                    v[2] = 0.4;
                    v
                },
            },
            ..flat.clone()
        };
        scene.splats.push(flat);
        scene.splats.push(rich);
        let pass = RDPrune { target_ratio: 0.5 };
        let mut ctx = PassContext::default();
        pass.run(&mut scene, &mut ctx).unwrap();
        assert_eq!(scene.splats.len(), 1);
        // The survivor should be the SH-rich splat (nonzero rest).
        if let Color::Sh { coeffs, .. } = &scene.splats[0].color {
            // rest = coeffs[3..]; should be 0.5s
            assert!(coeffs[3] > 0.4);
        } else {
            panic!("expected SH color");
        }
    }
}
