//! `SplatDelta` — anchor-stride Morton-order residual codec pass.
//!
//! Hypothesis (validated on bonsai / bicycle, see
//! `experiments/w2-splatdelta/RESULT.md`): spatially-sorted Gaussians have
//! heavy local redundancy. Pick every (1/α)-th splat in Morton order as a
//! full-precision "anchor"; code the rest as quantized residuals against the
//! IDW-weighted average of their k nearest already-decoded anchors.
//!
//! Per the Pareto sweep, the operating point that beats `splat-transform .sog`
//! 1.41-1.57× at parity fidelity is `anchor_stride=64, k_neighbors=2,
//! residual_bits=6, range_percentile=99.5`. Those are the defaults.
//!
//! Integration shape:
//!   - The pass mutates `scene.splats` in place to their *reconstructed*
//!     values (anchors stay full-precision, non-anchors become
//!     pred + dequant(quant(target - pred))). Downstream passes / writers
//!     see a valid `SplatScene` and can run unchanged.
//!   - The pass also serializes the side-channel (anchors raw f32, neighbour
//!     indices u16/u32, residuals int8/int16, per-channel ranges f32) into a
//!     zstd-compressed blob and stows it in a thread-local register. The CLI
//!     drains the register after `pipe.run` and writes a sidecar `.splatdelta`
//!     file next to the .glb / .gltf output, plus a small extension stub on
//!     the glTF asset (`SF_gaussian_splatting_delta`) that points at the
//!     sidecar URI. Bonsai validation in `INTEGRATION_RESULT.md` reports
//!     sidecar bytes against the `.sog` baseline (matching what the prototype
//!     measured); the .glb wrapper around it is currently the full-precision
//!     scene, which a smarter writer can later drop.
//!
//! Determinism: KNN ties on identical positions are broken by anchor index
//! (lowest wins). The Morton key tie-break already orders splats by original
//! index. zstd output is deterministic at a fixed level.

use anyhow::Result;
use catetus_core::{Color, SplatScene};
use serde::Serialize;
use std::sync::Mutex;

use crate::passes::{morton_code, Pass, PassContext, PassStats};

/// 14 attributes the codec Δ-codes per splat (matches the production tuple
/// shape and the Python prototype's `DELTA_ATTRS`).
const DELTA_ATTRS: usize = 14;

/// `SplatDelta` pass: anchor-stride residual codec.
#[derive(Debug, Clone)]
pub struct SplatDelta {
    /// Anchor density: 1 splat in every `anchor_stride` (in Morton order) is
    /// kept full-precision. Empirical Pareto winner on bonsai + bicycle: 64.
    pub anchor_stride: u32,
    /// k nearest anchors for the IDW prediction. Winner: 2 (k=4 trades ~10%
    /// less residual RMSE for ~25-30% more index bytes — not worth it).
    pub k_neighbors: u32,
    /// Residual quantization bits per channel. Winner: 6 (b=7 / b=8 give
    /// identical fidelity for +17-33% bytes).
    pub residual_bits: u8,
    /// Robust percentile (0..100) for per-channel residual range estimation.
    /// Winner: 99.5 (matches the prototype).
    pub range_percentile: f32,
}

impl Default for SplatDelta {
    fn default() -> Self {
        Self {
            anchor_stride: 64,
            k_neighbors: 2,
            residual_bits: 6,
            range_percentile: 99.5,
        }
    }
}

/// Per-pass summary statistics (in addition to the standard `PassStats.notes`
/// emitted by the pass). Useful for downstream readers that don't want to
/// parse the notes strings.
#[derive(Debug, Clone, Serialize)]
pub struct SplatDeltaStats {
    /// Total splats post-pass (always == scene.splats.len()).
    pub n: usize,
    /// Anchor count (full-precision).
    pub n_anchor: usize,
    /// Non-anchor count (delta-coded).
    pub n_nonanchor: usize,
    /// RMSE between the reconstructed and original 14-attr vectors over
    /// non-anchor splats (sanity bound).
    pub rmse_recon: f32,
    /// Bytes of the zstd-compressed sidecar payload (anchors + indices +
    /// residuals + ranges + header).
    pub sidecar_bytes: usize,
    /// Bytes of the raw (uncompressed) sidecar payload (for the
    /// post-mortem; the compressed value is what matters on the wire).
    pub raw_sidecar_bytes: usize,
}

/// In-process register where the last `SplatDelta` run parks its sidecar
/// blob. The CLI calls [`take_last_delta_stream`] after `pipe.run` to drain
/// the register and write the bytes as a `<out>.splatdelta` file.
///
/// We use a Mutex<Option<...>> rather than thread-local because a CLI run is
/// single-threaded across the pipeline anyway and the global is one less
/// plumbing change through `Pass::run`'s signature. Drainage is intentional
/// so a second pipeline invocation in the same process doesn't pick up a
/// stale blob from the first.
static LAST_DELTA_STREAM: Mutex<Option<DeltaStreamBlob>> = Mutex::new(None);

/// Captured side-channel payload from the most recent `SplatDelta::run`.
#[derive(Debug, Clone)]
pub struct DeltaStreamBlob {
    /// Compressed payload (zstd level 19). Ready to write as-is.
    pub compressed: Vec<u8>,
    /// Uncompressed payload length, for diagnostics.
    pub raw_len: usize,
    /// Configuration used to produce the blob.
    pub config: SplatDelta,
    /// Per-pass stats (mirrors what the pass emits via notes).
    pub stats: SplatDeltaStats,
}

/// Drain the most recent `SplatDelta` sidecar blob (if any). After draining
/// the register is empty until the next pass run.
pub fn take_last_delta_stream() -> Option<DeltaStreamBlob> {
    LAST_DELTA_STREAM.lock().ok().and_then(|mut g| g.take())
}

impl SplatDelta {
    fn levels(&self) -> f32 {
        ((1u32 << (self.residual_bits.clamp(2, 16) as u32 - 1)) - 1) as f32
    }
}

impl Pass for SplatDelta {
    fn name(&self) -> &'static str {
        "SplatDelta"
    }

    fn run(&self, scene: &mut SplatScene, _ctx: &mut PassContext) -> Result<PassStats> {
        let n = scene.splats.len();
        if n == 0 {
            return Ok(PassStats::default());
        }

        let stride = (self.anchor_stride.max(2)) as usize;
        let k = (self.k_neighbors.max(1)) as usize;

        // 1. Morton-sort indices. We reuse the canonical morton_code from
        //    passes.rs so the anchor-stride pattern matches the MortonSort
        //    pass byte-for-byte (this is the contract the prototype proved
        //    on against the canonical Rust morton key).
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
        let morton_order: Vec<usize> = indexed.iter().map(|x| x.1).collect();
        let inv_order: Vec<usize> = {
            let mut iv = vec![0usize; n];
            for (rank, &orig) in morton_order.iter().enumerate() {
                iv[orig] = rank;
            }
            iv
        };

        // 2. Anchor / non-anchor split in Morton order.
        let mut anchor_mask = vec![false; n];
        let mut i = 0usize;
        while i < n {
            anchor_mask[i] = true;
            i += stride;
        }
        anchor_mask[n - 1] = true;
        let anchor_idx_morton: Vec<usize> = (0..n).filter(|&i| anchor_mask[i]).collect();
        let nonanchor_idx_morton: Vec<usize> = (0..n).filter(|&i| !anchor_mask[i]).collect();
        let n_anchor = anchor_idx_morton.len();
        let n_nonanchor = nonanchor_idx_morton.len();

        // 3. Build a sorted (positions, attrs) view in Morton order so we
        //    can index directly by Morton rank. Cheaper than chasing indices
        //    through scene.splats on every k-NN hit.
        let mut pos_sorted: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut attrs_sorted: Vec<[f32; DELTA_ATTRS]> = Vec::with_capacity(n);
        for &orig in &morton_order {
            pos_sorted.push(scene.splats[orig].position);
            attrs_sorted.push(extract_attrs(&scene.splats[orig]));
        }

        // 4. k-NN per non-anchor against anchor positions.
        //    Uniform grid is O(n_nonanchor · k · cell_occupancy) and easily
        //    handles 1M splats / 18k anchors. Brute-force would be ~20B ops.
        let anchor_positions: Vec<[f32; 3]> =
            anchor_idx_morton.iter().map(|&mi| pos_sorted[mi]).collect();
        let grid = UniformGrid::build(&anchor_positions);

        // 5. Per-non-anchor: find k nearest anchors, predict + residual.
        let mut residuals: Vec<[f32; DELTA_ATTRS]> = Vec::with_capacity(n_nonanchor);
        let mut nn_idx: Vec<u32> = Vec::with_capacity(n_nonanchor * k);
        for &mi in &nonanchor_idx_morton {
            let q = pos_sorted[mi];
            let nns = grid.knn(q, k);
            // IDW weights.
            let mut weights: [f32; 8] = [0.0; 8];
            let mut wsum = 0.0f32;
            let kk = nns.len();
            for (i, &(d2, _ai)) in nns.iter().enumerate().take(8) {
                let w = 1.0 / (d2.sqrt() + 1e-6);
                weights[i] = w;
                wsum += w;
            }
            if wsum < 1e-9 {
                wsum = 1.0;
            }
            let mut pred = [0.0f32; DELTA_ATTRS];
            for (i, &(_d2, ai_local)) in nns.iter().enumerate() {
                let w = weights[i] / wsum;
                let a_morton = anchor_idx_morton[ai_local as usize];
                let av = &attrs_sorted[a_morton];
                for c in 0..DELTA_ATTRS {
                    pred[c] += w * av[c];
                }
                nn_idx.push(ai_local);
            }
            // Pad nn_idx if grid returned fewer than k (degenerate: e.g.
            // anchor count < k). Repeating the last index keeps the buffer
            // aligned at exactly k per non-anchor.
            for _ in kk..k {
                let pad = nns.last().map(|x| x.1).unwrap_or(0);
                nn_idx.push(pad);
            }
            let target = &attrs_sorted[mi];
            let mut r = [0.0f32; DELTA_ATTRS];
            for c in 0..DELTA_ATTRS {
                r[c] = target[c] - pred[c];
            }
            residuals.push(r);
        }

        // 6. Per-channel range from the requested percentile of |residual|.
        let mut ranges = [1.0f32; DELTA_ATTRS];
        if !residuals.is_empty() {
            for c in 0..DELTA_ATTRS {
                let mut vs: Vec<f32> = residuals.iter().map(|r| r[c].abs()).collect();
                vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let pct = self.range_percentile.clamp(0.0, 100.0);
                let idx = ((pct / 100.0) * vs.len() as f32) as usize;
                let idx = idx.min(vs.len() - 1);
                ranges[c] = vs[idx].max(1e-9);
            }
        }

        // 7. Quantize residuals symmetrically per-channel.
        let levels = self.levels();
        let mut residuals_q: Vec<[i16; DELTA_ATTRS]> = Vec::with_capacity(residuals.len());
        for r in &residuals {
            let mut rq = [0i16; DELTA_ATTRS];
            for c in 0..DELTA_ATTRS {
                let v = (r[c] / ranges[c]).clamp(-1.0, 1.0);
                rq[c] = (v * levels).round() as i16;
            }
            residuals_q.push(rq);
        }

        // 8. Rebuild non-anchor splats from anchors + dequantized residuals.
        //    Anchors stay byte-identical to their input. Recompute the same
        //    IDW pred from the (already-decoded) anchor attrs (note: anchors
        //    are unchanged, so this is equivalent to the encode-side pred).
        let mut rmse_sq_sum: f64 = 0.0;
        let mut rmse_n: usize = 0;
        for (qi, &mi) in nonanchor_idx_morton.iter().enumerate() {
            let q = pos_sorted[mi];
            let nns = grid.knn(q, k);
            let mut wsum = 0.0f32;
            let mut weights: [f32; 8] = [0.0; 8];
            for (i, &(d2, _)) in nns.iter().enumerate().take(8) {
                let w = 1.0 / (d2.sqrt() + 1e-6);
                weights[i] = w;
                wsum += w;
            }
            if wsum < 1e-9 {
                wsum = 1.0;
            }
            let mut pred = [0.0f32; DELTA_ATTRS];
            for (i, &(_d2, ai_local)) in nns.iter().enumerate() {
                let w = weights[i] / wsum;
                let a_morton = anchor_idx_morton[ai_local as usize];
                let av = &attrs_sorted[a_morton];
                for c in 0..DELTA_ATTRS {
                    pred[c] += w * av[c];
                }
            }
            let rq = residuals_q[qi];
            let mut new_attrs = [0.0f32; DELTA_ATTRS];
            for c in 0..DELTA_ATTRS {
                new_attrs[c] = pred[c] + (rq[c] as f32 / levels) * ranges[c];
            }
            // Accumulate RMSE against the original target.
            let target = &attrs_sorted[mi];
            for c in 0..DELTA_ATTRS {
                let d = new_attrs[c] - target[c];
                rmse_sq_sum += (d * d) as f64;
            }
            rmse_n += DELTA_ATTRS;
            // Write into the scene at the *original* index for this Morton
            // rank (inv_order maps original -> morton rank; we want morton ->
            // original, which is morton_order[mi]).
            let orig = morton_order[mi];
            assign_attrs(&mut scene.splats[orig], &new_attrs);
        }
        let rmse_recon = if rmse_n > 0 {
            ((rmse_sq_sum / rmse_n as f64).sqrt()) as f32
        } else {
            0.0
        };

        // 9. Renormalize quaternions on the whole scene (anchors were already
        //    unit; non-anchors may have drifted off-unit after IDW + quant).
        for s in &mut scene.splats {
            let qnorm: f32 = s.rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
            if qnorm > 1e-9 {
                for v in &mut s.rotation {
                    *v /= qnorm;
                }
            }
        }

        // 10. Serialize the side-channel.
        let (raw_bytes, compressed_bytes) = serialize_sidechannel(
            self,
            &anchor_idx_morton,
            &attrs_sorted,
            &nn_idx,
            &residuals_q,
            &ranges,
            n,
        )?;

        let stats = SplatDeltaStats {
            n,
            n_anchor,
            n_nonanchor,
            rmse_recon,
            sidecar_bytes: compressed_bytes.len(),
            raw_sidecar_bytes: raw_bytes,
        };

        // 11. Park the sidecar for the CLI to drain.
        if let Ok(mut g) = LAST_DELTA_STREAM.lock() {
            *g = Some(DeltaStreamBlob {
                compressed: compressed_bytes.clone(),
                raw_len: raw_bytes,
                config: self.clone(),
                stats: stats.clone(),
            });
        }

        // Suppress unused warning when caller doesn't drain it.
        let _ = inv_order;

        let notes = vec![format!(
            "SplatDelta: stride={} k={} bits={} pct={:.1} n_anchor={} n_nonanchor={} rmse={:.6} sidecar={}B (raw={}B)",
            self.anchor_stride,
            self.k_neighbors,
            self.residual_bits,
            self.range_percentile,
            n_anchor,
            n_nonanchor,
            rmse_recon,
            compressed_bytes.len(),
            raw_bytes,
        )];
        Ok(PassStats {
            modified: n_nonanchor,
            notes,
            ..Default::default()
        })
    }
}

// ---------- helpers ----------

/// Pack a `Splat` into the 14-attr delta vector. SH-typed colors are
/// collapsed to their first 3 DC scalars (the prototype operated on the
/// `f_dc_*` slots only; degree-≥1 SH bands are not Δ-coded). A scene that
/// reaches `SplatDelta` is expected to have run `ReduceSHDegree { target=0 }`
/// (the `web-mobile-delta` preset does this implicitly through the standard
/// `web-mobile` prune chain — see `presets.rs`); the fallback here keeps the
/// pass safe on raw SH inputs.
fn extract_attrs(s: &catetus_core::Splat) -> [f32; DELTA_ATTRS] {
    let (c0, c1, c2) = match &s.color {
        Color::Rgb(c) => (c[0], c[1], c[2]),
        Color::Sh { coeffs, .. } => (
            coeffs.first().copied().unwrap_or(0.0),
            coeffs.get(1).copied().unwrap_or(0.0),
            coeffs.get(2).copied().unwrap_or(0.0),
        ),
    };
    [
        s.position[0],
        s.position[1],
        s.position[2],
        s.scale[0],
        s.scale[1],
        s.scale[2],
        s.rotation[0],
        s.rotation[1],
        s.rotation[2],
        s.rotation[3],
        s.opacity,
        c0,
        c1,
        c2,
    ]
}

fn assign_attrs(s: &mut catetus_core::Splat, a: &[f32; DELTA_ATTRS]) {
    s.position = [a[0], a[1], a[2]];
    s.scale = [a[3], a[4], a[5]];
    s.rotation = [a[6], a[7], a[8], a[9]];
    s.opacity = a[10];
    match &mut s.color {
        Color::Rgb(c) => {
            c[0] = a[11];
            c[1] = a[12];
            c[2] = a[13];
        }
        Color::Sh { coeffs, .. } => {
            if !coeffs.is_empty() {
                coeffs[0] = a[11];
            }
            if coeffs.len() >= 2 {
                coeffs[1] = a[12];
            }
            if coeffs.len() >= 3 {
                coeffs[2] = a[13];
            }
        }
    }
}

/// Uniform spatial grid for cheap k-NN over a static anchor cloud.
///
/// Cell sizing target: ~32 anchors / cell so a ±1 ring covers ~864 candidates
/// — comfortably above k=2..8 with room for empty boundary cells.
struct UniformGrid {
    positions: Vec<[f32; 3]>,
    bmin: [f32; 3],
    cell_size: f32,
    cells_per_axis: i32,
    grid: Vec<Vec<u32>>,
}

impl UniformGrid {
    fn build(positions: &[[f32; 3]]) -> Self {
        let n = positions.len();
        if n == 0 {
            return Self {
                positions: Vec::new(),
                bmin: [0.0; 3],
                cell_size: 1.0,
                cells_per_axis: 1,
                grid: vec![Vec::new()],
            };
        }
        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        for p in positions {
            for i in 0..3 {
                if p[i] < bmin[i] {
                    bmin[i] = p[i];
                }
                if p[i] > bmax[i] {
                    bmax[i] = p[i];
                }
            }
        }
        let extent = [
            (bmax[0] - bmin[0]).max(1e-6),
            (bmax[1] - bmin[1]).max(1e-6),
            (bmax[2] - bmin[2]).max(1e-6),
        ];
        let longest = extent[0].max(extent[1]).max(extent[2]);
        let target_per_cell = 32.0_f32;
        let cells_per_axis = ((n as f32 / target_per_cell).cbrt().round() as i32).clamp(2, 128);
        let cell_size = longest / cells_per_axis as f32;
        let cap = cells_per_axis as usize;
        let total = cap * cap * cap;
        let mut grid: Vec<Vec<u32>> = vec![Vec::new(); total];
        let cap_i = cells_per_axis;
        let cell_idx = |p: [f32; 3]| -> [i32; 3] {
            [
                (((p[0] - bmin[0]) / cell_size) as i32).clamp(0, cap_i - 1),
                (((p[1] - bmin[1]) / cell_size) as i32).clamp(0, cap_i - 1),
                (((p[2] - bmin[2]) / cell_size) as i32).clamp(0, cap_i - 1),
            ]
        };
        for (i, p) in positions.iter().enumerate() {
            let [cx, cy, cz] = cell_idx(*p);
            let flat = (cx as usize * cap + cy as usize) * cap + cz as usize;
            grid[flat].push(i as u32);
        }
        Self {
            positions: positions.to_vec(),
            bmin,
            cell_size,
            cells_per_axis: cap_i,
            grid,
        }
    }

    fn flat(&self, cx: i32, cy: i32, cz: i32) -> usize {
        let cap = self.cells_per_axis as usize;
        (cx as usize * cap + cy as usize) * cap + cz as usize
    }

    fn cell_of(&self, p: [f32; 3]) -> [i32; 3] {
        let cap = self.cells_per_axis;
        [
            (((p[0] - self.bmin[0]) / self.cell_size) as i32).clamp(0, cap - 1),
            (((p[1] - self.bmin[1]) / self.cell_size) as i32).clamp(0, cap - 1),
            (((p[2] - self.bmin[2]) / self.cell_size) as i32).clamp(0, cap - 1),
        ]
    }

    /// Return `(d2, index)` for up to k nearest anchors. Index is into the
    /// original `positions` slice the grid was built with.
    fn knn(&self, q: [f32; 3], k: usize) -> Vec<(f32, u32)> {
        if self.positions.is_empty() {
            return Vec::new();
        }
        let [cx, cy, cz] = self.cell_of(q);
        let cap = self.cells_per_axis;
        // Scan rings ±radius until we have at least k candidates AND we've
        // walked one extra ring (so we don't miss a closer point in the next
        // shell). Cap at radius=8 for sanity.
        let mut heap = BoundedMaxHeapKV::with_capacity(k);
        let mut radius = 0i32;
        let mut last_good_radius = -1i32;
        while radius <= 8 {
            let mut any = false;
            for dx in -radius..=radius {
                let ncx = cx + dx;
                if !(0..cap).contains(&ncx) {
                    continue;
                }
                for dy in -radius..=radius {
                    let ncy = cy + dy;
                    if !(0..cap).contains(&ncy) {
                        continue;
                    }
                    for dz in -radius..=radius {
                        let ncz = cz + dz;
                        if !(0..cap).contains(&ncz) {
                            continue;
                        }
                        let ring = dx.abs().max(dy.abs()).max(dz.abs());
                        if ring != radius {
                            continue;
                        }
                        let cell = &self.grid[self.flat(ncx, ncy, ncz)];
                        for &ai in cell {
                            let p = self.positions[ai as usize];
                            let d2 = (p[0] - q[0]) * (p[0] - q[0])
                                + (p[1] - q[1]) * (p[1] - q[1])
                                + (p[2] - q[2]) * (p[2] - q[2]);
                            heap.push(d2, ai, k);
                            any = true;
                        }
                    }
                }
            }
            if heap.len() >= k && last_good_radius >= 0 {
                break;
            }
            if heap.len() >= k && last_good_radius < 0 {
                last_good_radius = radius;
            }
            // Stop early if we've scanned the whole grid and got nothing.
            if !any && radius > 0 && heap.is_empty() && 2 * radius > cap {
                break;
            }
            radius += 1;
        }
        let mut out = heap.into_sorted();
        if out.len() > k {
            out.truncate(k);
        }
        out
    }
}

/// Bounded max-heap of `(distance², anchor_index)`. We pop the largest when
/// over-capacity so we end with the k smallest.
struct BoundedMaxHeapKV {
    buf: Vec<(f32, u32)>,
}

impl BoundedMaxHeapKV {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap + 1),
        }
    }
    fn len(&self) -> usize {
        self.buf.len()
    }
    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    fn push(&mut self, d2: f32, idx: u32, cap: usize) {
        if !d2.is_finite() {
            return;
        }
        if self.buf.len() < cap {
            self.buf.push((d2, idx));
            self.sift_up(self.buf.len() - 1);
        } else if let Some(&top) = self.buf.first() {
            if d2 < top.0 {
                self.buf[0] = (d2, idx);
                self.sift_down(0);
            }
        }
    }
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.buf[i].0 > self.buf[parent].0 {
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
            if l < n && self.buf[l].0 > self.buf[largest].0 {
                largest = l;
            }
            if r < n && self.buf[r].0 > self.buf[largest].0 {
                largest = r;
            }
            if largest == i {
                break;
            }
            self.buf.swap(i, largest);
            i = largest;
        }
    }
    fn into_sorted(mut self) -> Vec<(f32, u32)> {
        self.buf.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        self.buf
    }
}

/// Serialize the sidechannel payload and zstd-compress it.
///
/// Wire layout (little-endian):
///   magic      : 4 bytes  "SFDL"
///   version    : u32      1
///   n_total    : u32
///   n_anchor   : u32
///   n_nonanchor: u32
///   k_neighbors: u8
///   residual_bits: u8
///   idx_bits   : u8     (16 if n_anchor < 65536, else 32)
///   reserved   : u8
///   anchor_stride: u32
///   ranges     : f32 × 14
///   anchor_indices : (u32 or u16) × n_anchor  (indices into the original
///                    morton-sorted splat array; the decoder pairs with the
///                    Morton order it has)
///   anchor_attrs   : f32 × 14 × n_anchor
///   nn_idx         : (u16 or u32) × k × n_nonanchor (indices into the
///                    anchor_attrs array)
///   residuals      : i8 × 14 × n_nonanchor  (when residual_bits <= 8)
///                    OR i16 × 14 × n_nonanchor (when 8 < bits <= 16)
fn serialize_sidechannel(
    cfg: &SplatDelta,
    anchor_idx_morton: &[usize],
    attrs_morton: &[[f32; DELTA_ATTRS]],
    nn_idx: &[u32],
    residuals_q: &[[i16; DELTA_ATTRS]],
    ranges: &[f32; DELTA_ATTRS],
    n_total: usize,
) -> Result<(usize, Vec<u8>)> {
    let n_anchor = anchor_idx_morton.len();
    let n_nonanchor = residuals_q.len();
    let use_u32_idx = n_anchor >= (1 << 16);

    let mut raw: Vec<u8> = Vec::with_capacity(
        4 + 4
            + 4 * 3
            + 4
            + 4
            + 4 * DELTA_ATTRS
            + n_anchor * if use_u32_idx { 4 } else { 2 }
            + n_anchor * DELTA_ATTRS * 4
            + n_nonanchor * cfg.k_neighbors as usize * if use_u32_idx { 4 } else { 2 }
            + n_nonanchor * DELTA_ATTRS * if cfg.residual_bits <= 8 { 1 } else { 2 },
    );
    raw.extend_from_slice(b"SFDL");
    raw.extend_from_slice(&1u32.to_le_bytes());
    raw.extend_from_slice(&(n_total as u32).to_le_bytes());
    raw.extend_from_slice(&(n_anchor as u32).to_le_bytes());
    raw.extend_from_slice(&(n_nonanchor as u32).to_le_bytes());
    raw.push(cfg.k_neighbors.min(255) as u8);
    raw.push(cfg.residual_bits);
    raw.push(if use_u32_idx { 32 } else { 16 });
    raw.push(0);
    raw.extend_from_slice(&cfg.anchor_stride.to_le_bytes());
    for r in ranges {
        raw.extend_from_slice(&r.to_le_bytes());
    }
    if use_u32_idx {
        for &mi in anchor_idx_morton {
            raw.extend_from_slice(&(mi as u32).to_le_bytes());
        }
    } else {
        for &mi in anchor_idx_morton {
            raw.extend_from_slice(&(mi as u16).to_le_bytes());
        }
    }
    for &mi in anchor_idx_morton {
        let a = &attrs_morton[mi];
        for v in a {
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }
    if use_u32_idx {
        for &x in nn_idx {
            raw.extend_from_slice(&x.to_le_bytes());
        }
    } else {
        for &x in nn_idx {
            raw.extend_from_slice(&(x as u16).to_le_bytes());
        }
    }
    if cfg.residual_bits <= 8 {
        for rq in residuals_q {
            for &v in rq {
                raw.push(v as i8 as u8);
            }
        }
    } else {
        for rq in residuals_q {
            for &v in rq {
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let raw_len = raw.len();
    // zstd level 19 matches what the prototype used. Pure-Rust crate via the
    // `zstd` workspace dep.
    let compressed = zstd::stream::encode_all(std::io::Cursor::new(&raw), 19)?;
    Ok((raw_len, compressed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::{Color, Splat, SplatScene};

    fn make_synth_scene(n: usize, seed: u64) -> SplatScene {
        // Deterministic xorshift-based pseudo-random splats with spatial
        // clustering (a real photogrammetry scene has heavy local
        // redundancy — for the test we just want a non-degenerate
        // distribution that the IDW prediction is meaningful on).
        let mut x = seed.wrapping_mul(0xdeadbeef).wrapping_add(1);
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let mut scene = SplatScene::new();
        for i in 0..n {
            // Mix cluster + jitter so neighbors share most attrs.
            let cluster = i / 16;
            let cx = (cluster as f32 * 0.137) % 4.0 - 2.0;
            let cy = ((cluster as f32 * 0.211) % 4.0) - 2.0;
            let cz = ((cluster as f32 * 0.299) % 4.0) - 2.0;
            let jx = rng() * 0.05;
            let jy = rng() * 0.05;
            let jz = rng() * 0.05;
            scene.splats.push(Splat {
                position: [cx + jx, cy + jy, cz + jz],
                rotation: {
                    // Random unit quaternion (xyzw).
                    let a = rng();
                    let b = rng();
                    let c = rng();
                    let d = rng();
                    let n = (a * a + b * b + c * c + d * d).sqrt().max(1e-9);
                    [a / n, b / n, c / n, d / n]
                },
                scale: [
                    0.05 + 0.02 * rng().abs(),
                    0.05 + 0.02 * rng().abs(),
                    0.05 + 0.02 * rng().abs(),
                ],
                opacity: 0.5 + 0.5 * rng().abs(),
                color: Color::Rgb([0.5 + 0.4 * rng(), 0.5 + 0.4 * rng(), 0.5 + 0.4 * rng()]),
            });
        }
        scene
    }

    #[test]
    fn splat_delta_default_runs_on_1k_synth() {
        let mut a = make_synth_scene(1024, 42);
        let mut b = make_synth_scene(1024, 42);
        let mut ctx = PassContext::default();
        let stats_a = SplatDelta::default().run(&mut a, &mut ctx).unwrap();
        let blob_a = take_last_delta_stream().expect("delta blob");
        let stats_b = SplatDelta::default().run(&mut b, &mut ctx).unwrap();
        let blob_b = take_last_delta_stream().expect("delta blob");

        // Determinism: identical input -> identical output scene + identical
        // compressed sidecar bytes.
        assert_eq!(a.splats.len(), b.splats.len());
        for (sa, sb) in a.splats.iter().zip(b.splats.iter()) {
            assert_eq!(sa.position, sb.position);
            assert_eq!(sa.rotation, sb.rotation);
            assert_eq!(sa.scale, sb.scale);
            assert_eq!(sa.opacity, sb.opacity);
        }
        assert_eq!(blob_a.compressed, blob_b.compressed);
        // Notes match too.
        assert_eq!(stats_a.notes, stats_b.notes);
    }

    #[test]
    fn splat_delta_anchor_count_matches_stride() {
        let n = 1024usize;
        let stride = 64u32;
        let mut scene = make_synth_scene(n, 7);
        let mut ctx = PassContext::default();
        let pass = SplatDelta {
            anchor_stride: stride,
            ..SplatDelta::default()
        };
        pass.run(&mut scene, &mut ctx).unwrap();
        let blob = take_last_delta_stream().unwrap();
        let expected_min = n.div_ceil(stride as usize);
        // The pass always force-anchors the last index, so n_anchor may be
        // expected_min or expected_min + 1 depending on whether n-1 is on a
        // stride boundary.
        assert!(
            blob.stats.n_anchor == expected_min || blob.stats.n_anchor == expected_min + 1,
            "expected n_anchor in {{{},{}}}, got {}",
            expected_min,
            expected_min + 1,
            blob.stats.n_anchor
        );
        assert_eq!(blob.stats.n, n);
        assert_eq!(blob.stats.n_anchor + blob.stats.n_nonanchor, n);
    }

    #[test]
    fn splat_delta_reconstruction_rmse_bounded() {
        // On a smooth synthetic distribution with cluster-local jitter
        // <= 0.05, the 14-attr RMSE should stay well under 1.0. Loose bound
        // because IDW + 6-bit quant on opacity / quaternion components can
        // push errors up to ~0.2 even when positions are tight.
        let mut scene = make_synth_scene(2048, 11);
        let originals: Vec<_> = scene.splats.iter().map(extract_attrs).collect();
        let mut ctx = PassContext::default();
        SplatDelta::default().run(&mut scene, &mut ctx).unwrap();
        let mut sq_sum = 0.0f64;
        let mut n = 0usize;
        for (s, orig) in scene.splats.iter().zip(originals.iter()) {
            let recon = extract_attrs(s);
            for c in 0..DELTA_ATTRS {
                let d = recon[c] - orig[c];
                sq_sum += (d * d) as f64;
            }
            n += DELTA_ATTRS;
        }
        let rmse = (sq_sum / n as f64).sqrt();
        assert!(
            rmse < 0.5,
            "reconstruction RMSE {rmse} above sanity bound 0.5"
        );
    }
}
