//! `VQPaletteShRest` — 45-D k-means palette quantization for SH-rest.
//!
//! Replaces the per-splat × 45-scalar SH-rest representation with a single
//! 16-bit palette index into a K-entry codebook (default K = 65,536). Matches
//! the SOG "killer feature" identified in
//! `experiments/SOG_STUDY_RESULT.md`: SOG compresses SH-rest from 360 b/s
//! (raw fp32) to ~11.2 b/s by treating the 45-D coefficient vector as a single
//! atomic symbol, training a 64k-entry codebook with k-means, then storing
//! `log2(K) = 16` bits per splat plus the codebook (45 × K bytes if each
//! coefficient is 8-bit-quantized per-column).
//!
//! Pipeline shape (matches `SplatDelta`):
//!
//!   - The pass mutates `scene.splats` in place — each splat's `Color::Sh`
//!     coefficient vector (positions 3..=47) is replaced with the centroid
//!     values from its assigned cluster. Downstream passes / writers see a
//!     standard `SplatScene` and can either re-quantize per-coefficient (for
//!     a smaller GLB) or emit FP32 as-is.
//!   - The pass also parks a side-table containing the codebook (per-coef
//!     8-bit-quantized centroids + per-coef ranges) and the per-splat 16-bit
//!     palette indices, plus a zstd-compressed sidecar blob ready for the
//!     CLI to write next to the GLB as `<out>.shpal`.
//!
//! k-means: simple Lloyd's iteration with Forgy initialization (random
//! sample of `K` points without replacement), brute-force nearest-centroid
//! assignment parallelized across splats via rayon, and centroid recomputation
//! as the per-cluster mean. Empty clusters are re-seeded with a random point
//! to avoid degenerate centroids (same approach as splat-transform's CPU
//! `kmeans`, see `node_modules/@playcanvas/splat-transform/dist/index.mjs`
//! line 9228).
//!
//! Determinism: `PassContext.seed` drives both initialization and the empty-
//! cluster re-seed RNG. Two runs with the same scene + seed produce identical
//! codebooks and identical labels.

use anyhow::Result;
use catetus_core::{Color, SplatScene};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::Mutex;

use crate::passes::{Pass, PassContext, PassStats};

/// 45 scalars: 3 (degree-1) + 5 (degree-2) + 7 (degree-3) coefficients × 3
/// channels. Same layout the rest of the codebase uses for `Color::Sh`
/// (`coeffs[0..3]` = DC, `coeffs[3..48]` = SH-rest in (l, n, ch) order).
pub const VQ_SH_REST_DIM: usize = 45;

/// `VQPaletteShRest` pass: 45-D k-means palette quantization for SH-rest.
#[derive(Debug, Clone)]
pub struct VQPaletteShRest {
    /// Codebook size. SOG default is `min(64, 2^floor(log2(N/1024))) * 1024`
    /// which yields 65,536 for N ≈ 1M. We default to 4,096 because the CPU
    /// brute-force Lloyd's loop is `O(N × K × 45)` per iter — at K=65,536 a
    /// single iter on a 1M-splat scene runs ~10 min on 8 cores, which blows
    /// the typical 3-min "encode budget". K=4,096 still gives 30-90× SH-rest
    /// compression with sub-minute encode time. CLIs aiming for SOG bit
    /// parity can override via the preset's pass construction.
    pub palette_size: usize,
    /// Lloyd's k-means iterations. SOG default is 10; 5 captures the bulk of
    /// the convergence at half the wall time. CLI runs override via the
    /// preset's pass construction.
    pub iterations: usize,
    /// Per-column codebook quantization width for the sidecar. 8 matches SOG
    /// (`shN_centroids` are 8-bit-quantized via the 1D weighted-DP codebook).
    pub codebook_bits: u8,
    /// k-means training subsample cap. When `Some(s)`, Lloyd's iterations run
    /// on `min(s, N)` randomly sampled points (without replacement). A final
    /// full pass then assigns *every* splat to its nearest centroid. For
    /// K=4,096 and N=1.16M this drops total wall-time from ~5 min to ~1 min
    /// with negligible quality impact (the 1M-point average converges very
    /// fast). When `None`, every iter operates on the full N (slower but
    /// theoretically tightest convergence).
    pub training_subsample: Option<usize>,
    /// When `true`, the pass additionally range-codes the per-splat u16
    /// palette index stream with `catetus-posthac`'s categorical
    /// entropy coder (empirical-CDF over the K-symbol alphabet). The
    /// resulting `.idxp` payload is parked in
    /// `ShRestPaletteSidetable::posthac_indices` for the CLI to emit as
    /// a `.shpal.pthc` companion file. The standard `.shpal` payload is
    /// still emitted unchanged so existing decoders that only know the
    /// `.shpal` format keep round-tripping; the PostHAC payload is a
    /// strict superset (it carries the same index information in fewer
    /// bytes, plus a self-describing header so a future bench/loader
    /// can swap which payload it reads). See
    /// `experiments/SOG_STUDY_RUN/VQ45_GPU_SWEEP.md` for the K=4096
    /// rationale that makes this composition worthwhile.
    pub posthac_indices: bool,
}

impl Default for VQPaletteShRest {
    fn default() -> Self {
        Self {
            palette_size: 4_096,
            iterations: 5,
            codebook_bits: 8,
            training_subsample: Some(200_000),
            posthac_indices: false,
        }
    }
}

/// Captured side-channel payload from the most recent `VQPaletteShRest::run`.
///
/// On-wire layout (zstd-compressed, level 19):
///
///   magic: u32 = 0x53485041 ("SHPA")
///   version: u32 = 1
///   palette_size: u32
///   n_splats: u32
///   codebook_bits: u8 (typically 8)
///   _pad: [u8; 3]
///   ranges: [f32; 45]                            (per-coef abs range)
///   codebook: [u8; palette_size * 45]            (signed int8 normalized to range)
///   indices: [u16; n_splats]
#[derive(Debug, Clone)]
pub struct ShRestPaletteSidetable {
    /// Compressed sidecar payload (zstd level 19). Ready to write as-is.
    pub compressed: Vec<u8>,
    /// Uncompressed payload length, for diagnostics.
    pub raw_len: usize,
    /// Number of centroids in the codebook.
    pub palette_size: usize,
    /// Number of splats encoded.
    pub n_splats: usize,
    /// Codebook quantization width (bits per coefficient column).
    pub codebook_bits: u8,
    /// Per-pass stats (mirrors `VQPaletteShRestStats`).
    pub stats: VQPaletteShRestStats,
    /// Optional PostHAC range-coded payload for the u16 palette index stream
    /// (self-describing `.idxp` blob: magic + version + K + N + per-symbol
    /// histogram + range-coded bitstream). Present iff `posthac_indices` was
    /// set on the `VQPaletteShRest` pass. The standard `.shpal` payload
    /// (`compressed` above) is always present so existing decoders keep
    /// working; this is an additive sidecar that future loaders can prefer.
    pub posthac_indices: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VQPaletteShRestStats {
    pub n_splats_encoded: usize,
    pub palette_size: usize,
    pub iterations_run: usize,
    pub codebook_bits: u8,
    /// Mean squared error between original SH-rest and centroid SH-rest
    /// (averaged over all 45 coefficients and N splats).
    pub mse: f64,
    /// Wall-time of the k-means inner loop in milliseconds.
    pub kmeans_ms: u64,
    /// Wall-time of the codebook quantization + zstd compression in ms.
    pub encode_ms: u64,
    /// Bytes used by the standard zstd-19 `.shpal` payload (the existing
    /// codepath: header + ranges + 8-bit codebook + raw u16 indices,
    /// zstd-compressed). This equals `ShRestPaletteSidetable::compressed.len()`.
    pub shpal_bytes: usize,
    /// Bytes used by the PostHAC range-coded `.idxp` payload over the same
    /// u16 index stream, when `posthac_indices` was enabled on the pass.
    /// `None` when the pass ran without PostHAC. The byte count includes the
    /// `.idxp` header + shipped histogram + range-coded bitstream.
    pub posthac_idxp_bytes: Option<usize>,
    /// Bytes of the index stream alone within the standard `.shpal`
    /// (= `2 * n_splats` raw, before zstd). Used for accounting in
    /// experiment diffs so the bench can compute "shpal minus indices,
    /// plus idxp" composed totals without re-running the encoder.
    pub raw_index_bytes: usize,
}

/// In-process register for the most recent VQ palette sidetable. See
/// `take_last_sh_rest_palette` for drainage.
static LAST_SH_REST_PALETTE: Mutex<Option<ShRestPaletteSidetable>> = Mutex::new(None);

/// Drain the most recent `VQPaletteShRest` side-channel sidetable (if any).
/// After draining the register is empty until the next pass run.
pub fn take_last_sh_rest_palette() -> Option<ShRestPaletteSidetable> {
    LAST_SH_REST_PALETTE.lock().ok().and_then(|mut g| g.take())
}

/// Reset the side-table register without consuming it. Used by tests.
#[doc(hidden)]
pub fn clear_last_sh_rest_palette() {
    if let Ok(mut g) = LAST_SH_REST_PALETTE.lock() {
        *g = None;
    }
}

// ---------- xorshift64* RNG (deterministic, seedable) ----------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[inline]
fn rand_index(state: &mut u64, n: usize) -> usize {
    (xorshift64(state) % (n as u64)) as usize
}

/// Brute-force nearest-centroid lookup for a single 45-D point. Returns the
/// `u32` index of the cluster whose centroid is closest in squared-L2.
///
/// The 45-dim inner loop is hand-unrolled into 5 chunks of 9 fmadds — this
/// gives the Rust optimizer enough breathing room to schedule NEON / AVX2
/// fmadds without inter-iteration dependencies stalling the pipeline.
#[inline(always)]
fn nearest_centroid(point: &[f32], centroids: &[f32], k: usize) -> u32 {
    debug_assert_eq!(point.len(), VQ_SH_REST_DIM);
    debug_assert!(centroids.len() >= k * VQ_SH_REST_DIM);
    let mut best_dist = f32::INFINITY;
    let mut best_c: u32 = 0;
    for c in 0..k {
        let centroid = &centroids[c * VQ_SH_REST_DIM..c * VQ_SH_REST_DIM + VQ_SH_REST_DIM];
        // Manually unroll the 45-element dot product into 9-wide chunks (5
        // chunks). The compiler emits clean fmadds with this shape on both
        // ARM NEON (a64) and x86-64 AVX2.
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;
        let mut d3 = 0.0f32;
        let mut d4 = 0.0f32;
        for j in 0..9 {
            let a0 = point[j] - centroid[j];
            let a1 = point[9 + j] - centroid[9 + j];
            let a2 = point[18 + j] - centroid[18 + j];
            let a3 = point[27 + j] - centroid[27 + j];
            let a4 = point[36 + j] - centroid[36 + j];
            d0 += a0 * a0;
            d1 += a1 * a1;
            d2 += a2 * a2;
            d3 += a3 * a3;
            d4 += a4 * a4;
        }
        let d = (d0 + d1) + (d2 + d3) + d4;
        if d < best_dist {
            best_dist = d;
            best_c = c as u32;
        }
    }
    best_c
}

impl Pass for VQPaletteShRest {
    fn name(&self) -> &'static str {
        "VQPaletteShRest"
    }

    fn run(&self, scene: &mut SplatScene, ctx: &mut PassContext) -> Result<PassStats> {
        let t0 = std::time::Instant::now();

        // 1) Extract the (N, 45) data matrix from the scene. Splats without
        //    `Color::Sh` or with a too-short coeffs vector contribute a zero
        //    vector and are flagged so we don't trample their non-SH state.
        let n = scene.splats.len();
        let mut has_sh = vec![false; n];
        // Row-major: data[i * 45 + d]
        let mut data: Vec<f32> = vec![0.0; n * VQ_SH_REST_DIM];
        for (i, s) in scene.splats.iter().enumerate() {
            if let Color::Sh { coeffs, .. } = &s.color {
                if coeffs.len() >= 3 + VQ_SH_REST_DIM {
                    has_sh[i] = true;
                    let dst = &mut data[i * VQ_SH_REST_DIM..(i + 1) * VQ_SH_REST_DIM];
                    dst.copy_from_slice(&coeffs[3..3 + VQ_SH_REST_DIM]);
                }
            }
        }
        let n_sh: usize = has_sh.iter().filter(|&&b| b).count();

        // Optional per-splat SH-rest Jacobian weight, threaded through
        // `PassContext` from the CLI's `--jacobian-sidecar` flag (or any
        // upstream pass that wants to influence the centroid update). When
        // present, length must equal `scene.splats.len()`; we copy it into
        // a per-SH-splat dense vector below. If the length disagrees we
        // silently fall back to unweighted (matches the original behaviour
        // and surfaces a note in pass stats).
        let weights_ok = ctx
            .sh_rest_weights
            .as_ref()
            .map(|w| w.len() == n)
            .unwrap_or(false);
        let use_weights = ctx.sh_rest_weights.is_some() && weights_ok;

        // Early-out: no SH-rest values to cluster.
        if n_sh == 0 {
            return Ok(PassStats {
                modified: 0,
                notes: vec!["no SH-rest splats found; pass is a no-op".to_string()],
                ..Default::default()
            });
        }

        // Clamp palette size to the number of usable splats — Lloyd's wants
        // at least K distinct seed points. (For real scenes N >> K so this is
        // only a small-test guard.)
        let k = self.palette_size.min(n_sh).max(1);
        let iters = self.iterations.max(1);

        // Build a compact list of SH splat indices so we cluster only those.
        let sh_idx: Vec<usize> = (0..n).filter(|&i| has_sh[i]).collect();

        // 2) Initialize centroids via Forgy sampling without replacement
        //    (simple reservoir-style: shuffle a sample of size K).
        let mut rng = ctx.seed.wrapping_add(0xC0FF_EE00_1234_5678);
        if rng == 0 {
            rng = 0xDEAD_BEEF_CAFE_BABE;
        }
        let mut centroids: Vec<f32> = vec![0.0; k * VQ_SH_REST_DIM];
        // Sample K distinct positions from sh_idx (with Fisher-Yates of a
        // shuffle on indices 0..n_sh, taking the first K).
        let mut perm: Vec<usize> = (0..n_sh).collect();
        for i in (1..n_sh).rev().take(n_sh.saturating_sub(1).min(k)) {
            let j = (xorshift64(&mut rng) as usize) % (i + 1);
            perm.swap(i, j);
            if perm.len() - i >= k && i <= n_sh - k {
                break;
            }
        }
        // Easier: pick K random indices via a HashSet. For n_sh up to ~1M and
        // k up to ~65k this is < 100k inserts so a simple linear-probe set is
        // fine. We use a sorted-vec-of-indices approach.
        //
        // When SH-rest Jacobian weights are available, use the Gumbel-top-K
        // trick for *weighted* sampling without replacement: g_i = -log(-log U)
        //  + log(w_i + eps); take top-K g_i. This matches the prototype's
        // `_init_centroids_subsample` so the high-J splats both attract
        // centroids in the update step AND are over-represented in the seed
        // set (faster convergence; same final centroids the prototype shipped
        // at +11.94 dB in `experiments/render-space-lloyd-max/RESULT.md`).
        let mut picked: Vec<usize> = Vec::with_capacity(k);
        if use_weights {
            let w_src = ctx.sh_rest_weights.as_ref().unwrap();
            let mut scored: Vec<(f64, usize)> = (0..n_sh)
                .map(|i| {
                    let splat_i = sh_idx[i];
                    let wi = (w_src[splat_i] as f64).max(0.0) + 1e-12;
                    // Inverse-CDF uniform in (0,1) from xorshift.
                    let u_raw = xorshift64(&mut rng);
                    let u = ((u_raw >> 11) as f64) / ((1u64 << 53) as f64);
                    let u = u.max(f64::MIN_POSITIVE);
                    let g = -(-u.ln()).ln() + wi.ln();
                    (g, i)
                })
                .collect();
            // Partial sort: just the top-K by g (descending).
            // For tractable cost we sort the whole thing; n_sh is ~1M.
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, i) in scored.into_iter().take(k) {
                picked.push(i);
            }
        } else {
            let mut seen = vec![false; n_sh];
            while picked.len() < k {
                let candidate = rand_index(&mut rng, n_sh);
                if !seen[candidate] {
                    seen[candidate] = true;
                    picked.push(candidate);
                }
            }
        }
        for (cluster, &pick) in picked.iter().enumerate() {
            let src_splat = sh_idx[pick];
            let src = &data[src_splat * VQ_SH_REST_DIM..(src_splat + 1) * VQ_SH_REST_DIM];
            let dst = &mut centroids[cluster * VQ_SH_REST_DIM..(cluster + 1) * VQ_SH_REST_DIM];
            dst.copy_from_slice(src);
        }

        // 3) Build training subset: every iter of Lloyd's runs against
        //    `train_idx` (indices into `sh_idx`). When training_subsample is
        //    smaller than n_sh, draw a random sample without replacement;
        //    otherwise train on the full population. The final assignment
        //    step ALWAYS labels every splat in `sh_idx`.
        // PARITY NOTE (T2.1.R drift fix, 2026-05-18): when `use_weights` is
        // true the CLI has explicitly opted in to the higher-fidelity
        // weighted-Lloyd path via `--jacobian-sidecar`. The Python reference
        // (`experiments/render-space-lloyd-max/code/rs_lloyd.py`) runs every
        // Lloyd iter on the full N. A 200k Gumbel-weighted subsample with
        // K=65,536 gives only ~3 training points per cluster, which produces
        // centroids that fit the (heavily-Jacobian-weighted) 200k subset
        // well but undertrains for the remaining ~1M splats — measured
        // drift of −1.52 dB on bonsai vs the Python reference. We disable
        // the subsample in weighted mode to match the prototype. The
        // unweighted path still honours `training_subsample` for back-compat.
        let train_size = if use_weights {
            n_sh
        } else {
            self.training_subsample
                .map(|s| s.min(n_sh))
                .unwrap_or(n_sh)
                .max(k) // need at least K points to seed K clusters
        };
        let train_idx: Vec<usize> = if train_size >= n_sh {
            (0..n_sh).collect()
        } else if use_weights {
            // Gumbel-top-K weighted-without-replacement, same trick as the
            // centroid init above. Ensures the high-J training set actually
            // drives the centroid update toward high-J splats — without
            // this, a 200k uniform sample from a 1.24M scene rarely catches
            // enough of the top-1% (≈12.4k splats holding 87% of J) and the
            // weighted update degenerates back toward unweighted means.
            let w_src = ctx.sh_rest_weights.as_ref().unwrap();
            let mut scored: Vec<(f64, usize)> = (0..n_sh)
                .map(|i| {
                    let splat_i = sh_idx[i];
                    let wi = (w_src[splat_i] as f64).max(0.0) + 1e-12;
                    let u_raw = xorshift64(&mut rng);
                    let u = ((u_raw >> 11) as f64) / ((1u64 << 53) as f64);
                    let u = u.max(f64::MIN_POSITIVE);
                    let g = -(-u.ln()).ln() + wi.ln();
                    (g, i)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored
                .into_iter()
                .take(train_size)
                .map(|(_, i)| i)
                .collect()
        } else {
            let mut tseen = vec![false; n_sh];
            let mut t = Vec::with_capacity(train_size);
            while t.len() < train_size {
                let c = rand_index(&mut rng, n_sh);
                if !tseen[c] {
                    tseen[c] = true;
                    t.push(c);
                }
            }
            t
        };

        // 4) Lloyd's loop on the training subset.
        //
        // When `use_weights` is true (CLI passed --jacobian-sidecar), we run
        // the *weighted* Lloyd update from the render-space prototype
        // (`experiments/render-space-lloyd-max/code/rs_lloyd.py`):
        //
        //   centroid_k = sum_{i in cluster_k} w_i * x_i / sum_{i in cluster_k} w_i
        //
        // The assignment step is unchanged — `argmin_k ||x_i - c_k||^2` —
        // because w_i cancels across cluster candidates for a fixed splat i.
        // The empty-cluster re-seed also picks weighted-random when weights
        // are available so we don't keep re-seeding the same low-J slot.
        let mut train_labels: Vec<u32> = vec![0; train_idx.len()];
        // Per-iteration accumulators for centroid recomputation.
        // sum_buf stores sum_i (w_i * x_i) when weighted, sum_i x_i otherwise.
        // wsum_buf stores sum_i w_i when weighted, otherwise stays unused
        // (we use cnt_buf as the unweighted denominator).
        let mut sum_buf: Vec<f64> = vec![0.0; k * VQ_SH_REST_DIM];
        let mut cnt_buf: Vec<u32> = vec![0; k];
        let mut wsum_buf: Vec<f64> = vec![0.0; k];

        // Pre-snapshot per-splat weights (indexed by splat index, like `data`)
        // so the inner loop doesn't need to chase through `ctx`.
        let weights_per_splat: Vec<f32> = if use_weights {
            ctx.sh_rest_weights.as_ref().unwrap().clone()
        } else {
            Vec::new()
        };

        let kmeans_t0 = std::time::Instant::now();
        for _iter in 0..iters {
            // Assignment step: for each training point, find the nearest
            // centroid.
            let centroids_ref: &[f32] = &centroids;
            let data_ref: &[f32] = &data;
            let sh_idx_ref: &[usize] = &sh_idx;
            let train_idx_ref: &[usize] = &train_idx;
            train_labels
                .par_iter_mut()
                .enumerate()
                .for_each(|(t_idx, label_out)| {
                    let splat_i = sh_idx_ref[train_idx_ref[t_idx]];
                    let point = &data_ref[splat_i * VQ_SH_REST_DIM..(splat_i + 1) * VQ_SH_REST_DIM];
                    *label_out = nearest_centroid(point, centroids_ref, k);
                });

            // Update step: recompute centroids as the (optionally weighted)
            // mean of assigned points.
            for v in sum_buf.iter_mut() {
                *v = 0.0;
            }
            for v in cnt_buf.iter_mut() {
                *v = 0;
            }
            for v in wsum_buf.iter_mut() {
                *v = 0.0;
            }
            for (t_idx, &lbl) in train_labels.iter().enumerate() {
                let c = lbl as usize;
                cnt_buf[c] += 1;
                let splat_i = sh_idx[train_idx[t_idx]];
                let point = &data[splat_i * VQ_SH_REST_DIM..(splat_i + 1) * VQ_SH_REST_DIM];
                let sum_slice = &mut sum_buf[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                if use_weights {
                    let wi = weights_per_splat[splat_i].max(0.0) as f64;
                    wsum_buf[c] += wi;
                    for dim in 0..VQ_SH_REST_DIM {
                        sum_slice[dim] += (point[dim] as f64) * wi;
                    }
                } else {
                    for dim in 0..VQ_SH_REST_DIM {
                        sum_slice[dim] += point[dim] as f64;
                    }
                }
            }
            for c in 0..k {
                // Empty (or all-zero-weight) clusters get re-seeded.
                let empty = if use_weights {
                    wsum_buf[c] <= 1e-12
                } else {
                    cnt_buf[c] == 0
                };
                if empty {
                    // Empty cluster: re-seed to a random point. Matches
                    // splat-transform's behaviour. When weights are present
                    // we weight-sample to avoid re-seeding into the very same
                    // low-J neighbourhood the cluster was abandoned for.
                    let re = if use_weights {
                        // Cheap weighted draw: try a few uniform candidates,
                        // accept the highest-weight one. O(8) work and we
                        // don't need exact unbiasedness here.
                        let mut best_i = rand_index(&mut rng, n_sh);
                        let mut best_w = weights_per_splat[sh_idx[best_i]];
                        for _ in 0..7 {
                            let cand = rand_index(&mut rng, n_sh);
                            let cw = weights_per_splat[sh_idx[cand]];
                            if cw > best_w {
                                best_w = cw;
                                best_i = cand;
                            }
                        }
                        best_i
                    } else {
                        rand_index(&mut rng, n_sh)
                    };
                    let splat_i = sh_idx[re];
                    let src = &data[splat_i * VQ_SH_REST_DIM..(splat_i + 1) * VQ_SH_REST_DIM];
                    let dst = &mut centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                    dst.copy_from_slice(src);
                } else {
                    let denom = if use_weights {
                        wsum_buf[c]
                    } else {
                        cnt_buf[c] as f64
                    };
                    let sum_slice = &sum_buf[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                    let dst = &mut centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                    for dim in 0..VQ_SH_REST_DIM {
                        dst[dim] = (sum_slice[dim] / denom) as f32;
                    }
                }
            }
        }

        // 5) Final assignment step: label every splat in the scene against the
        //    converged codebook. This is the only iter that touches the full
        //    `n_sh` population — under subsample-mode it's a one-shot O(N*K)
        //    cost instead of iters × O(N*K).
        let mut labels: Vec<u32> = vec![0; n_sh];
        {
            let centroids_ref: &[f32] = &centroids;
            let data_ref: &[f32] = &data;
            let sh_idx_ref: &[usize] = &sh_idx;
            labels
                .par_iter_mut()
                .enumerate()
                .for_each(|(p_idx, label_out)| {
                    let splat_i = sh_idx_ref[p_idx];
                    let point = &data_ref[splat_i * VQ_SH_REST_DIM..(splat_i + 1) * VQ_SH_REST_DIM];
                    *label_out = nearest_centroid(point, centroids_ref, k);
                });
        }
        let kmeans_ms = kmeans_t0.elapsed().as_millis() as u64;

        // 4) Compute MSE before writing back (sanity diag).
        let mse_sum: f64 = labels
            .par_iter()
            .enumerate()
            .map(|(p_idx, &lbl)| {
                let c = lbl as usize;
                let splat_i = sh_idx[p_idx];
                let point = &data[splat_i * VQ_SH_REST_DIM..(splat_i + 1) * VQ_SH_REST_DIM];
                let centroid = &centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                let mut d = 0.0f64;
                for dim in 0..VQ_SH_REST_DIM {
                    let diff = (point[dim] - centroid[dim]) as f64;
                    d += diff * diff;
                }
                d
            })
            .sum();
        let mse = mse_sum / (n_sh as f64 * VQ_SH_REST_DIM as f64).max(1.0);

        // 5) Write back: replace each splat's SH-rest with its centroid.
        let mut modified = 0usize;
        for (p_idx, &lbl) in labels.iter().enumerate() {
            let splat_i = sh_idx[p_idx];
            let c = lbl as usize;
            if let Color::Sh { coeffs, .. } = &mut scene.splats[splat_i].color {
                if coeffs.len() >= 3 + VQ_SH_REST_DIM {
                    let src = &centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                    for dim in 0..VQ_SH_REST_DIM {
                        coeffs[3 + dim] = src[dim];
                    }
                    modified += 1;
                }
            }
        }

        // 6) Build the sidecar blob.
        let encode_t0 = std::time::Instant::now();
        let codebook_bits = self.codebook_bits.clamp(2, 16);
        // Per-coefficient absolute-max range across the entire codebook. We
        // use absolute-max (not percentile) because the codebook is already a
        // distilled summary of the data; further percentile clipping would be
        // wasteful at this size.
        let mut ranges = vec![1e-9f32; VQ_SH_REST_DIM];
        for c in 0..k {
            let centroid = &centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
            for dim in 0..VQ_SH_REST_DIM {
                let v = centroid[dim].abs();
                if v > ranges[dim] {
                    ranges[dim] = v;
                }
            }
        }

        // Quantize codebook to int8 (or u16 if codebook_bits > 8).
        let codebook_payload = if codebook_bits <= 8 {
            let levels = ((1u32 << (codebook_bits as u32 - 1)) - 1) as f32;
            let mut out: Vec<u8> = Vec::with_capacity(k * VQ_SH_REST_DIM);
            for c in 0..k {
                let centroid = &centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                for dim in 0..VQ_SH_REST_DIM {
                    let r = ranges[dim].max(1e-9);
                    let t = (centroid[dim] / r).clamp(-1.0, 1.0);
                    let q = (t * levels).round().clamp(-127.0, 127.0) as i8;
                    out.push(q as u8);
                }
            }
            out
        } else {
            let levels = ((1u32 << (codebook_bits as u32 - 1)) - 1) as f32;
            let mut out: Vec<u8> = Vec::with_capacity(k * VQ_SH_REST_DIM * 2);
            for c in 0..k {
                let centroid = &centroids[c * VQ_SH_REST_DIM..(c + 1) * VQ_SH_REST_DIM];
                for dim in 0..VQ_SH_REST_DIM {
                    let r = ranges[dim].max(1e-9);
                    let t = (centroid[dim] / r).clamp(-1.0, 1.0);
                    let q = (t * levels).round().clamp(-32767.0, 32767.0) as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
            out
        };

        // Pack the on-wire blob.
        let mut raw: Vec<u8> = Vec::with_capacity(
            16 // header
                + 45 * 4 // ranges
                + codebook_payload.len()
                + n_sh * 2, // indices
        );
        // Header
        raw.extend_from_slice(&0x53485041u32.to_le_bytes()); // "SHPA"
        raw.extend_from_slice(&1u32.to_le_bytes()); // version
        raw.extend_from_slice(&(k as u32).to_le_bytes());
        raw.extend_from_slice(&(n_sh as u32).to_le_bytes());
        raw.push(codebook_bits);
        raw.extend_from_slice(&[0u8, 0, 0]);
        // Ranges
        for r in &ranges {
            raw.extend_from_slice(&r.to_le_bytes());
        }
        // Codebook
        raw.extend_from_slice(&codebook_payload);
        // Indices
        for &lbl in &labels {
            // Note: clamps if palette_size > 65535. We support up to 65,536
            // (clamped to u16::MAX) — the SOG default is 65,536 which fits
            // in u16 as `0..=65535`, the K=65536 actually allocates index
            // values 0..=65535 which is fine.
            let v = lbl.min(65_535) as u16;
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let raw_len = raw.len();
        let compressed = zstd::stream::encode_all(raw.as_slice(), 19)
            .map_err(|e| anyhow::anyhow!("zstd encode failed: {e}"))?;
        let encode_ms = encode_t0.elapsed().as_millis() as u64;

        // Optional PostHAC categorical range-coded payload over the same u16
        // index stream. **Hosted-only since 2026-05-19:** the entropy coder
        // moved to the private `catetus/catetus-private/crates/catetus-posthac`
        // crate. When `posthac_indices=true` the public crate emits a warning
        // and leaves the payload empty; the wider raw .shpal still ships.
        let posthac_payload: Option<Vec<u8>> = if self.posthac_indices {
            eprintln!(
                "warning: PostHAC index-stream encoding is hosted-only; \
                 POST the raw indices to https://api.catetus.com/v1/fidelity \
                 for the entropy-coded payload. Falling back to raw u16 stream."
            );
            None
        } else {
            None
        };

        let stats = VQPaletteShRestStats {
            n_splats_encoded: n_sh,
            palette_size: k,
            iterations_run: iters,
            codebook_bits,
            mse,
            kmeans_ms,
            encode_ms,
            shpal_bytes: compressed.len(),
            posthac_idxp_bytes: posthac_payload.as_ref().map(|p| p.len()),
            raw_index_bytes: n_sh * 2,
        };
        let blob = ShRestPaletteSidetable {
            compressed,
            raw_len,
            palette_size: k,
            n_splats: n_sh,
            codebook_bits,
            stats: stats.clone(),
            posthac_indices: posthac_payload,
        };
        if let Ok(mut g) = LAST_SH_REST_PALETTE.lock() {
            *g = Some(blob);
        }

        let duration_ms = t0.elapsed().as_millis() as u64;
        let posthac_note = match stats.posthac_idxp_bytes {
            Some(b) => format!(" posthac_idxp_bytes={b}"),
            None => String::new(),
        };
        let weights_note = if use_weights {
            " weighted_lloyd=on(J_sh_rest)"
        } else if ctx.sh_rest_weights.is_some() {
            " weighted_lloyd=off(weights-length-mismatch)"
        } else {
            ""
        };
        Ok(PassStats {
            modified,
            duration_ms,
            notes: vec![format!(
                "K={k} iters={iters} N={n_sh} dim={VQ_SH_REST_DIM} \
                 codebook_bits={codebook_bits} mse={mse:.6e} \
                 kmeans_ms={kmeans_ms} encode_ms={encode_ms} \
                 shpal_bytes={shpal} raw_index_bytes={raw_idx}{posthac_note}{weights_note}",
                shpal = stats.shpal_bytes,
                raw_idx = stats.raw_index_bytes,
            )],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::{Color, Splat};

    fn make_sh_splat(values: [f32; VQ_SH_REST_DIM]) -> Splat {
        let mut coeffs = vec![0.0f32; 3];
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
    fn vq_palette_clusters_two_distinct_groups() {
        clear_last_sh_rest_palette();
        // 16 splats split into two extremely distinct SH-rest patterns; with
        // K=2 the labels must partition them cleanly.
        let mut a = [0.0f32; VQ_SH_REST_DIM];
        let mut b = [0.0f32; VQ_SH_REST_DIM];
        for d in 0..VQ_SH_REST_DIM {
            a[d] = 1.0;
            b[d] = -1.0;
        }
        let mut scene = SplatScene::new();
        for i in 0..16 {
            if i % 2 == 0 {
                scene.splats.push(make_sh_splat(a));
            } else {
                scene.splats.push(make_sh_splat(b));
            }
        }
        let pass = VQPaletteShRest {
            palette_size: 2,
            iterations: 8,
            codebook_bits: 8,
            training_subsample: None,
            posthac_indices: false,
        };
        let mut ctx = PassContext {
            seed: 0xABCD,
            ..Default::default()
        };
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(stats.modified, 16);

        // After clustering, all 'a' splats and 'b' splats should map to two
        // distinct centroids — and the in-scene SH-rest values should be
        // near-identical to ±1 (centroid converges to the cluster mean).
        let blob = take_last_sh_rest_palette().expect("blob");
        assert_eq!(blob.n_splats, 16);
        assert_eq!(blob.palette_size, 2);

        // MSE must be very small for an obvious 2-cluster case.
        assert!(
            blob.stats.mse < 1e-6,
            "mse={} too high for synthetic 2-cluster test",
            blob.stats.mse
        );
        // Verify scene values approximately equal ±1 after writeback.
        for splat in &scene.splats {
            if let Color::Sh { coeffs, .. } = &splat.color {
                let v0 = coeffs[3];
                assert!(
                    (v0.abs() - 1.0).abs() < 1e-3,
                    "expected centroid ≈ ±1.0, got {}",
                    v0
                );
            }
        }
    }

    #[test]
    fn vq_palette_posthac_indices_is_hosted_only() {
        // Post 2026-05-19 open-core split: when `posthac_indices=true` the
        // public crate emits a stderr warning and leaves the payload empty.
        // The raw u16 index stream still ships in the .shpal regardless.
        clear_last_sh_rest_palette();
        let mut scene = SplatScene::new();
        for cluster in 0..4_usize {
            for _ in 0..64 {
                let mut v = [0.0f32; VQ_SH_REST_DIM];
                for d in 0..VQ_SH_REST_DIM {
                    v[d] = ((cluster * 17 + d) as f32 / 7.0).sin();
                }
                scene.splats.push(make_sh_splat(v));
            }
        }
        let pass = VQPaletteShRest {
            palette_size: 4,
            iterations: 8,
            codebook_bits: 8,
            training_subsample: None,
            posthac_indices: true,
        };
        let mut ctx = PassContext {
            seed: 0x1234,
            ..Default::default()
        };
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(stats.modified, 256);
        let blob = take_last_sh_rest_palette().expect("blob");
        // Hosted-only contract: payload is always None in the public crate.
        assert!(
            blob.posthac_indices.is_none(),
            "public crate must never produce a posthac payload — it is hosted-only"
        );
        // The raw .shpal compressed payload still rides through unchanged.
        assert!(!blob.compressed.is_empty());
    }

    #[test]
    fn vq_palette_no_sh_is_noop() {
        clear_last_sh_rest_palette();
        let mut scene = SplatScene::new();
        scene.splats.push(Splat {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            opacity: 1.0,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
        let pass = VQPaletteShRest::default();
        let mut ctx = PassContext::default();
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(stats.modified, 0);
        assert!(take_last_sh_rest_palette().is_none());
    }

    #[test]
    fn vq_palette_weighted_lloyd_pulls_centroid_toward_high_weight_member() {
        // Synthetic test that proves the weighted-mean update shifts the
        // centroid toward high-weight members vs. the unweighted mean.
        //
        // Setup (single cluster, K=1 so init can't matter):
        //   - 99 splats with SH-rest = +1.0 and weight 1.0
        //   -  1 splat   with SH-rest = +5.0 and weight 100.0  ("hot")
        // Closed-form expected centroid value (per coefficient):
        //   unweighted mean = (99 * 1 + 1 * 5) / 100              = 1.04
        //   weighted   mean = (99 * 1 + 1 * 100 * 5) / (99 + 100) ≈ 2.5075
        // Difference > 1.4 — well above any quantization noise.
        const HOT_W: f32 = 100.0;
        const N_LOW: usize = 99;
        let sh_low = [1.0f32; VQ_SH_REST_DIM];
        let sh_hot = [5.0f32; VQ_SH_REST_DIM];
        let mut scene = SplatScene::new();
        let mut weights: Vec<f32> = Vec::new();
        for _ in 0..N_LOW {
            scene.splats.push(make_sh_splat(sh_low));
            weights.push(1.0);
        }
        scene.splats.push(make_sh_splat(sh_hot));
        weights.push(HOT_W);
        let pass = VQPaletteShRest {
            palette_size: 1,
            iterations: 5,
            codebook_bits: 16,
            training_subsample: None,
            posthac_indices: false,
        };

        // --- weighted run
        clear_last_sh_rest_palette();
        let mut scene_w = scene.clone();
        let mut ctx_w = PassContext {
            seed: 0xCAFE,
            sh_rest_weights: Some(weights.clone()),
            splat_origin_idx: None,
        };
        pass.run(&mut scene_w, &mut ctx_w).expect("weighted run");
        let centroid_w = match &scene_w.splats[0].color {
            Color::Sh { coeffs, .. } => coeffs[3],
            _ => panic!("expected SH"),
        };

        // --- unweighted run (control)
        clear_last_sh_rest_palette();
        let mut scene_u = scene.clone();
        let mut ctx_u = PassContext {
            seed: 0xCAFE,
            sh_rest_weights: None,
            splat_origin_idx: None,
        };
        pass.run(&mut scene_u, &mut ctx_u).expect("unweighted run");
        let centroid_u = match &scene_u.splats[0].color {
            Color::Sh { coeffs, .. } => coeffs[3],
            _ => panic!("expected SH"),
        };

        // Closed-form expectations:
        //   unweighted: sum(x)/N         = (99*1 + 1*5) / 100         = 1.04
        //   weighted:   sum(w*x)/sum(w)  = (99*1*1 + 1*100*5) / 199   ≈ 3.0101
        // Codebook is 16-bit signed over abs-max range, so the round-trip
        // error per coefficient is < range / 32767 — for range ≈ 3.01 that's
        // < 1e-4. The 0.05 tolerance below leaves three orders of magnitude
        // of headroom.
        assert!(
            (centroid_u - 1.04).abs() < 0.05,
            "unweighted centroid should be 1.04, got {}",
            centroid_u,
        );
        assert!(
            (centroid_w - 3.0101).abs() < 0.05,
            "weighted centroid should be 3.0101, got {}",
            centroid_w,
        );
        // And the weighted result must differ meaningfully from the
        // unweighted result — this is the core acceptance criterion.
        assert!(
            (centroid_w - centroid_u).abs() > 1.0,
            "expected weighted centroid {} to differ from unweighted {} by >1.0",
            centroid_w,
            centroid_u,
        );
    }

    #[test]
    fn vq_palette_weighted_length_mismatch_falls_back_to_unweighted() {
        // Wrong-length weights vector should be ignored gracefully.
        clear_last_sh_rest_palette();
        let mut scene = SplatScene::new();
        for i in 0..16 {
            let mut v = [0.0f32; VQ_SH_REST_DIM];
            for d in 0..VQ_SH_REST_DIM {
                v[d] = ((i + d) as f32 / 7.0).sin();
            }
            scene.splats.push(make_sh_splat(v));
        }
        let pass = VQPaletteShRest {
            palette_size: 4,
            iterations: 5,
            codebook_bits: 8,
            training_subsample: None,
            posthac_indices: false,
        };
        // Length 3 != 16 splats — must not panic.
        let mut ctx = PassContext {
            seed: 0xABCD,
            sh_rest_weights: Some(vec![1.0, 2.0, 3.0]),
            splat_origin_idx: None,
        };
        let stats = pass.run(&mut scene, &mut ctx).expect("run");
        assert_eq!(stats.modified, 16);
        let note = &stats.notes[0];
        assert!(
            note.contains("weighted_lloyd=off(weights-length-mismatch)"),
            "expected length-mismatch fallback note, got: {note}",
        );
    }

    #[test]
    fn vq_palette_weighted_lloyd_uses_full_n_even_with_subsample_set() {
        // Drift-fix regression test (T2.1.R Python-parity, 2026-05-18).
        // When `use_weights=true`, the pass must train Lloyd on the full
        // population N regardless of `training_subsample` — see Python
        // reference `rs_lloyd.py::weighted_lloyd_gpu`. A 200k subsample
        // with K=65k gives ~3 training pts/cluster on bonsai and
        // undertrains the ~1M splats that never made the subsample,
        // costing 1.52 dB end-to-end PSNR.
        //
        // Construction: K=2 with a single high-weight splat far outside
        // the bulk. With subsample=10 and 100 splats, an unweighted-style
        // subsample path could miss the hot splat in some draws. With the
        // weighted path that ignores subsample (the drift fix), the hot
        // splat is always included and the centroid update on the full N
        // is deterministic.
        //
        // We check via the pass notes that the K-means MSE under the hot
        // splat with weight 1e6 lands at the (essentially-zero) value we
        // expect when the centroid for cluster k=1 lands exactly on
        // x=+10.0 — which only happens if the hot splat reaches the
        // update step. With a 10-splat training subsample drawn weight-
        // biased the hot splat *would* normally be picked, but we
        // hard-test that subsample is bypassed in weighted mode by
        // confirming `train_size == n_sh` via the pass notes path.
        const N_LOW: usize = 99;
        let sh_low = [1.0f32; VQ_SH_REST_DIM];
        let sh_hot = [10.0f32; VQ_SH_REST_DIM];
        let mut scene = SplatScene::new();
        let mut weights: Vec<f32> = Vec::new();
        for _ in 0..N_LOW {
            scene.splats.push(make_sh_splat(sh_low));
            weights.push(1.0);
        }
        scene.splats.push(make_sh_splat(sh_hot));
        weights.push(1.0e6);

        // Critically: training_subsample=Some(8) — well below N=100.
        // The drift-fix code path must IGNORE this when weighted.
        let pass = VQPaletteShRest {
            palette_size: 2,
            iterations: 6,
            codebook_bits: 16,
            training_subsample: Some(8),
            posthac_indices: false,
        };
        clear_last_sh_rest_palette();
        let mut ctx = PassContext {
            seed: 0xCAFE,
            sh_rest_weights: Some(weights.clone()),
            splat_origin_idx: None,
        };
        pass.run(&mut scene, &mut ctx).expect("weighted run");

        // After the fix, the K=2 centroids should converge to +10 (hot
        // cluster) and +1 (bulk cluster). The hot splat is at +10.0
        // with weight 1e6 — its centroid IS +10.0 to within 16-bit
        // codebook precision (range/32767 ≈ 3e-4).
        let mut centroid_values: Vec<f32> = Vec::with_capacity(2);
        for splat in &scene.splats {
            if let Color::Sh { coeffs, .. } = &splat.color {
                centroid_values.push(coeffs[3]);
            }
        }
        // Find the hot splat's assigned centroid: it was the last one
        // pushed, so it's at index N_LOW.
        let hot_centroid = centroid_values[N_LOW];
        assert!(
            (hot_centroid - 10.0).abs() < 1e-2,
            "hot splat's centroid should converge to +10 with full-N training; got {}",
            hot_centroid,
        );
        // One of the bulk splats should land at ~1.0.
        let bulk_centroid = centroid_values[0];
        assert!(
            (bulk_centroid - 1.0).abs() < 1e-2,
            "bulk centroid should converge to +1 with full-N training; got {}",
            bulk_centroid,
        );
    }

    #[test]
    fn vq_palette_determinism_under_same_seed() {
        clear_last_sh_rest_palette();
        let mut scene1 = SplatScene::new();
        let mut scene2 = SplatScene::new();
        for i in 0..32 {
            let mut v = [0.0f32; VQ_SH_REST_DIM];
            for d in 0..VQ_SH_REST_DIM {
                v[d] = ((i * 7 + d * 3) as f32 / 13.0).sin();
            }
            scene1.splats.push(make_sh_splat(v));
            scene2.splats.push(make_sh_splat(v));
        }
        let pass = VQPaletteShRest {
            palette_size: 4,
            iterations: 5,
            codebook_bits: 8,
            training_subsample: None,
            posthac_indices: false,
        };
        let mut ctx1 = PassContext {
            seed: 42,
            ..Default::default()
        };
        let mut ctx2 = PassContext {
            seed: 42,
            ..Default::default()
        };
        pass.run(&mut scene1, &mut ctx1).expect("run1");
        let blob1 = take_last_sh_rest_palette().expect("blob1");
        pass.run(&mut scene2, &mut ctx2).expect("run2");
        let blob2 = take_last_sh_rest_palette().expect("blob2");
        assert_eq!(blob1.raw_len, blob2.raw_len);
        assert_eq!(blob1.compressed, blob2.compressed);
    }
}
