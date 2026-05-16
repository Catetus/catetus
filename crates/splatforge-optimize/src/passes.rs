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

        // Bbox + cell-size derivation. Target ~64 splats per cell so the
        // 27-cell neighborhood scan averages ~1700 candidates — enough for
        // a stable k=8 estimate without quadratic blow-up.
        let (bmin, bmax) = bbox_f64(&pos);
        let extent = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        let bbox_diag = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2])
            .sqrt()
            .max(1e-9);
        // cells_per_axis = round(cbrt(n / target_per_cell)); clamped so we
        // don't degenerate into 1×1×1 (no spatial culling) or 1024³ (cache
        // shred). 16–96 is fine for 10k–10M splats.
        let target_per_cell = 64.0_f64;
        let cells_per_axis = ((n as f64 / target_per_cell).cbrt().round() as i64).clamp(8, 96);
        let cells = cells_per_axis as usize;
        // Cell size from the longest axis so we never get sub-cell splats
        // straddling a boundary in two dimensions.
        let longest = extent[0].max(extent[1]).max(extent[2]).max(1e-9);
        let cell_size = longest / cells_per_axis as f64;

        // Spatial hash: dense Vec<Vec<u32>> indexed by (cx, cy, cz). Dense
        // beats hashmap at this density — 16³ = 4096 cells, all touched.
        let total_cells = cells.checked_mul(cells).and_then(|v| v.checked_mul(cells));
        let Some(total_cells) = total_cells else {
            // Numerical overflow: fall back to the cheap centroid-stdev path
            // (better than panicking on a malformed input).
            return Ok(PassStats {
                notes: vec!["grid_overflow_fallback".into()],
                ..Default::default()
            });
        };
        let cell_idx = |p: [f64; 3]| -> [i64; 3] {
            [
                (((p[0] - bmin[0]) / cell_size) as i64).clamp(0, cells_per_axis - 1),
                (((p[1] - bmin[1]) / cell_size) as i64).clamp(0, cells_per_axis - 1),
                (((p[2] - bmin[2]) / cell_size) as i64).clamp(0, cells_per_axis - 1),
            ]
        };
        let flat = |cx: i64, cy: i64, cz: i64| -> usize {
            (cx as usize * cells + cy as usize) * cells + cz as usize
        };

        let mut grid: Vec<Vec<u32>> = vec![Vec::new(); total_cells];
        for (i, p) in pos.iter().enumerate() {
            let [cx, cy, cz] = cell_idx(*p);
            grid[flat(cx, cy, cz)].push(i as u32);
        }

        // Per-splat k-NN distance (using k+1 because the splat finds itself
        // in its own cell with distance 0). On the 27-cell scan we keep a
        // tiny max-heap of size k+1 to extract the k-th smallest distance
        // without a full sort. The heap is reused across iterations.
        let knn_d2: Vec<f64> = (0..n)
            .map(|i| {
                let p = pos[i];
                let [cx, cy, cz] = cell_idx(p);
                let mut heap = BoundedMaxHeap::with_capacity(k + 1);
                // Initial neighborhood = ±1 cell. If the cell is sparse and
                // we don't fill k+1 entries, expand to ±2 then ±3. Capped
                // at ±3 to keep worst-case O(343 · cell-occupancy) finite.
                let mut radius = 1i64;
                while heap.len() < k + 1 && radius <= 3 {
                    for dx in -radius..=radius {
                        let ncx = cx + dx;
                        if !(0..cells_per_axis).contains(&ncx) {
                            continue;
                        }
                        for dy in -radius..=radius {
                            let ncy = cy + dy;
                            if !(0..cells_per_axis).contains(&ncy) {
                                continue;
                            }
                            for dz in -radius..=radius {
                                let ncz = cz + dz;
                                if !(0..cells_per_axis).contains(&ncz) {
                                    continue;
                                }
                                // Skip cells already scanned at the previous
                                // radius to avoid redoing work.
                                let ring = dx.abs().max(dy.abs()).max(dz.abs());
                                if ring < radius {
                                    continue;
                                }
                                let cell_splats = &grid[flat(ncx, ncy, ncz)];
                                for &j in cell_splats {
                                    let q = pos[j as usize];
                                    let d2 = (q[0] - p[0]) * (q[0] - p[0])
                                        + (q[1] - p[1]) * (q[1] - p[1])
                                        + (q[2] - p[2]) * (q[2] - p[2]);
                                    heap.push(d2, k + 1);
                                }
                            }
                        }
                    }
                    radius += 1;
                }
                // k+1-th smallest = heap top once heap is full; if still
                // under-full (very isolated splat near the bbox corner with
                // no neighbors within 3 cells), use the largest seen, which
                // is already huge and will get pruned — correct behavior.
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
        let knn_d: Vec<f64> = knn_d2.iter().map(|&d2| d2.sqrt()).collect();
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
                "knn_median={:.4} mad={:.4} threshold={:.4} cells={}",
                median, mad, threshold, cells
            )],
            ..Default::default()
        })
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
    xs.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
                        // Convert SH DC coefficient to linear RGB in [0,1].
                        // Inria 3DGS stores f_dc_* as raw SH degree-0 coefficients;
                        // the standard conversion (matching the renderer's
                        // base.ts and lodge/ply.ts paths) is
                        //   color_rgb = SH_C0 * f_dc + 0.5
                        // where SH_C0 = 0.28209479177387814. Without this
                        // bake-down, downstream renderers that consume
                        // Color::Rgb directly see raw (often-negative) SH
                        // values and produce a near-black image.
                        const SH_C0: f32 = 0.282_094_79_f32;
                        let raw0 = coeffs.first().copied().unwrap_or(0.0);
                        let raw1 = coeffs.get(1).copied().unwrap_or(0.0);
                        let raw2 = coeffs.get(2).copied().unwrap_or(0.0);
                        let dc = [
                            (SH_C0 * raw0 + 0.5).clamp(0.0, 1.0),
                            (SH_C0 * raw1 + 0.5).clamp(0.0, 1.0),
                            (SH_C0 * raw2 + 0.5).clamp(0.0, 1.0),
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

/// `BackgroundOverdrawPrune`: drop the largest, faintest splats whose
/// screen-coverage cost dominates fillrate without contributing useful
/// detail.
///
/// Background motivation
/// ---------------------
/// Inria 3DGS often leaves a ring of large, low-opacity gaussians around the
/// real subject — typically representing far walls, sky, or low-frequency
/// ambient color. These splats are NOT picked up by `AspectRatioPrune` (they
/// tend to be isotropic, ratio << 10) and they survive `OpacityPrune` at
/// hero-friendly thresholds because their opacity is in the 0.05–0.5 band
/// where alpha still matters at edges. In a WebGL2/WebGPU rasteriser they
/// individually cover hundreds of thousands of pixels and absolutely murder
/// fillrate when the camera is framed tight on the subject (per the
/// hero-rebuild post-mortem of bonsai @ 250k splats).
///
/// What this pass does
/// -------------------
/// For each splat compute a `coverage_cost = max_scale^2 * opacity` proxy
/// (units: world-area × alpha; a stable proxy for fragments shaded per draw
/// at any fixed viewport). Then drop the `top_fraction` of splats by
/// `coverage_cost` whose opacity is below `opacity_keep_above`. Splats with
/// opacity ≥ `opacity_keep_above` are protected regardless of cost — they
/// are bright/saturated highlights and almost always part of the subject.
///
/// Defaults are tuned to drop the bonsai background ring at hero framing:
/// `top_fraction = 0.05` (drop top 5% by cost) and
/// `opacity_keep_above = 0.5` (protect anything saturated).
#[derive(Debug, Clone)]
pub struct BackgroundOverdrawPrune {
    /// Fraction of splats considered for removal (those with the highest
    /// `max_scale^2 * opacity`). 0.05 = top 5%.
    pub top_fraction: f32,
    /// Opacity floor that grants immunity. Splats with `opacity >= this`
    /// are kept even if their coverage cost is in the top fraction.
    pub opacity_keep_above: f32,
}

impl Default for BackgroundOverdrawPrune {
    fn default() -> Self {
        Self {
            top_fraction: 0.05,
            opacity_keep_above: 0.5,
        }
    }
}

impl Pass for BackgroundOverdrawPrune {
    fn name(&self) -> &'static str {
        "BackgroundOverdrawPrune"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let before = scene.splats.len();
        if before == 0 {
            return Ok(PassStats::default());
        }
        let top_fraction = self.top_fraction.clamp(0.0, 1.0);
        let opacity_floor = self.opacity_keep_above.max(0.0);

        // Compute coverage proxy per splat. Guard against negative/NaN.
        let costs: Vec<f32> = scene
            .splats
            .iter()
            .map(|s| {
                let sx = s.scale[0].abs();
                let sy = s.scale[1].abs();
                let sz = s.scale[2].abs();
                let smax = sx.max(sy).max(sz);
                let op = s.opacity.clamp(0.0, 1.0);
                let cost = smax * smax * op;
                if cost.is_finite() {
                    cost
                } else {
                    0.0
                }
            })
            .collect();

        // Threshold = the (1 - top_fraction) quantile of the cost distribution.
        // Splats above this threshold are *candidates* for removal.
        let mut sorted = costs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut_idx = ((1.0 - top_fraction) * sorted.len() as f32).floor() as usize;
        let cut_idx = cut_idx.min(sorted.len().saturating_sub(1));
        let cost_threshold = sorted[cut_idx];

        // Build a keep mask: candidates that are also faint get dropped.
        let keep: Vec<bool> = scene
            .splats
            .iter()
            .zip(costs.iter())
            .map(|(s, &cost)| {
                let is_candidate = cost > cost_threshold;
                let is_faint = s.opacity < opacity_floor;
                !(is_candidate && is_faint)
            })
            .collect();

        let mut idx = 0usize;
        let mut dropped_costs: Vec<f32> = Vec::new();
        scene.splats.retain(|s| {
            let k = keep[idx];
            if !k {
                let sx = s.scale[0].abs();
                let sy = s.scale[1].abs();
                let sz = s.scale[2].abs();
                let smax = sx.max(sy).max(sz);
                let op = s.opacity.clamp(0.0, 1.0);
                dropped_costs.push(smax * smax * op);
            }
            idx += 1;
            k
        });
        let removed = before - scene.splats.len();

        let mut notes = Vec::new();
        if !dropped_costs.is_empty() {
            let mut s = dropped_costs.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = s[s.len() / 2];
            let max = *s.last().unwrap();
            notes.push(format!(
                "top_fraction={:.3} opacity_keep_above={:.3} dropped={} cost_threshold={:.4} median_dropped_cost={:.4} max_dropped_cost={:.4}",
                top_fraction, opacity_floor, removed, cost_threshold, median, max
            ));
        } else {
            notes.push(format!(
                "top_fraction={:.3} opacity_keep_above={:.3} dropped=0 (no overdraw splats matched)",
                top_fraction, opacity_floor
            ));
        }

        Ok(PassStats {
            removed,
            notes,
            ..Default::default()
        })
    }
}

/// `SubjectCrop`: drop splats far from the dense subject cluster.
///
/// Why
/// ---
/// Inria-3DGS scenes from MipNeRF360 (and similar) include large radial
/// background floaters that pad the bounding box by 5-10× the subject's
/// actual footprint. At hero framing this kills composition: the viewer
/// frames to the *bbox*, so the subject ends up as a tiny dot at center.
///
/// What
/// ----
/// For each axis compute the median and MAD (median absolute deviation),
/// then drop splats whose position on any axis falls outside
/// `[median - k*MAD, median + k*MAD]`. MAD is robust to the long-tailed
/// floater distribution — a regular stdev would be dragged by the very
/// outliers we want to remove.
///
/// Defaults
/// --------
/// `k_mad = 4.0` — empirically keeps the bonsai canopy + pot + ground
/// plane while dropping the distant ceiling / floor floaters.
#[derive(Debug, Clone)]
pub struct SubjectCrop {
    /// Number of median absolute deviations from the per-axis median
    /// beyond which splats are dropped. Lower = tighter crop.
    pub k_mad: f32,
}

impl Default for SubjectCrop {
    fn default() -> Self {
        Self { k_mad: 4.0 }
    }
}

impl Pass for SubjectCrop {
    fn name(&self) -> &'static str {
        "SubjectCrop"
    }
    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let before = scene.splats.len();
        if before == 0 {
            return Ok(PassStats::default());
        }
        let k_mad = self.k_mad.max(0.1);

        // Per-axis median + MAD.
        let mut bounds = [(0.0f32, 0.0f32, 0.0f32); 3]; // (lo, hi, median) per axis
        for (axis, slot) in bounds.iter_mut().enumerate() {
            let mut v: Vec<f32> = scene
                .splats
                .iter()
                .map(|s| s.position[axis])
                .filter(|x| x.is_finite())
                .collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = v[v.len() / 2];
            let mut dev: Vec<f32> = v.iter().map(|x| (x - median).abs()).collect();
            dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = dev[dev.len() / 2].max(1e-6);
            let half_window = k_mad * mad;
            *slot = (median - half_window, median + half_window, median);
        }

        scene.splats.retain(|s| {
            (0..3).all(|axis| {
                let (lo, hi, _) = bounds[axis];
                let p = s.position[axis];
                p.is_finite() && p >= lo && p <= hi
            })
        });
        let removed = before - scene.splats.len();
        let notes = vec![format!(
            "k_mad={:.2} dropped={} crop_x=[{:.3},{:.3}] crop_y=[{:.3},{:.3}] crop_z=[{:.3},{:.3}]",
            k_mad,
            removed,
            bounds[0].0,
            bounds[0].1,
            bounds[1].0,
            bounds[1].1,
            bounds[2].0,
            bounds[2].1,
        )];
        Ok(PassStats {
            removed,
            notes,
            ..Default::default()
        })
    }
}
