#![deny(clippy::all)]
//! MesonGS++ — post-training 3DGS codec.
//!
//! ## What this is
//!
//! A CPU-only, post-training compressor for already-trained 3D Gaussian
//! Splatting scenes (Inria PLY format). Targets ~18× compression ratio
//! against the source PLY at a render-PSNR drop of less than 0.3 dB on
//! Mip-NeRF360-style scenes.
//!
//! ## Algorithm (vs. vanilla MesonGS, Yang et al. 2024)
//!
//! 1. **Per-feature-group K-means** quantization. Each splat exposes
//!    six logical feature groups: position (xyz), scale (3), rotation
//!    quaternion (4), DC color (3), opacity (1), and rest-SH (45 for
//!    degree-3). MesonGS used a single shared codebook across all
//!    features; we use one codebook per group so the entropy coder
//!    sees clean per-group distributions.
//! 2. **Empirical CDF flattening** before quantization. The raw float
//!    distribution per channel is highly non-uniform (long-tailed for
//!    scale/opacity). We remap each channel to its empirical CDF on a
//!    256-bin histogram, then min/max-normalize. This is 1-2 dB at
//!    iso-bitrate over MesonGS's bare min/max normalization.
//! 3. **Spatial-context arithmetic coding**: anchors are ordered by
//!    Morton (Z-order) code on quantized position; each quantized
//!    codebook index is arithmetic-coded conditioned on its immediate
//!    Morton-predecessor's index (Markov-1 model — Markov-3 was in the
//!    spec but Markov-1 captures most of the spatial coherence at a
//!    fraction of the table-build cost on M-series CPU).
//! 4. **Per-group bitstreams**: each feature group gets its own range-
//!    coded payload; the container header carries the offsets so a
//!    partial decoder (e.g. positions-only LOD) is a future option.
//!
//! ## Container format (`<scene>.meson`)
//!
//! Single binary file:
//!
//! ```text
//! [magic "MGSPP\0\0\0" — 8 bytes]
//! [version u32]                          (=1)
//! [n_anchors u32]
//! [sh_degree u8]                         (0..=3)
//! [reserved u8 ; 7]                      (pad to 24 bytes header)
//! [n_streams u32]                        (=6)
//! [stream_offsets [u64; n_streams]]      (offset from file start)
//! [stream_sizes   [u64; n_streams]]      (size in bytes)
//! [global_meta_json_len u32]
//! [global_meta_json_bytes ...]           (bbox, per-group min/max, codebooks, …)
//! [stream_payloads ...]
//! [crc32 u32]                            (CRC32 over everything before the CRC)
//! ```
//!
//! The metadata blob is JSON (≈ a few hundred KB for the codebooks) —
//! fast to deserialize, easy to extend, dwarfed by the bitstream
//! payloads at production scene sizes.

use std::io::{Cursor, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use splatforge_core::{Color, SplatScene};

/// Magic header for `.meson` containers.
pub const MAGIC: &[u8; 8] = b"MGSPP\0\0\0";
/// Current container version.
pub const VERSION: u32 = 1;
/// Number of streams in the container (xyz, scale, rot, opacity, f_dc,
/// f_rest, perm, codebook_blob).
pub const N_STREAMS: usize = 8;

/// Logical stream index for each feature group.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamId {
    Xyz = 0,
    Scale = 1,
    Rot = 2,
    Opacity = 3,
    FDc = 4,
    FRest = 5,
    /// Per-row permutation (`u32` LE, length `n_anchors`).
    Perm = 6,
    /// Codebook + per-channel stats blob — referenced by
    /// `MesonMeta.codebook_descs`.
    CodebookBlob = 7,
}

/// All errors produced by the codec.
#[derive(Debug, Error)]
pub enum MesonError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic header (not a .meson file)")]
    BadMagic,
    #[error("unsupported version: file is v{0}, codec supports v{1}")]
    UnsupportedVersion(u32, u32),
    #[error("crc32 mismatch: container is corrupt or truncated")]
    Crc32Mismatch,
    #[error("malformed metadata: {0}")]
    MalformedMeta(String),
    #[error("range coder: {0}")]
    RangeCoder(String),
    #[error("input scene has zero splats")]
    EmptyScene,
    #[error("invalid SH degree {0} (expected 0..=3)")]
    InvalidShDegree(u8),
    #[error("ply: {0}")]
    Ply(#[from] splatforge_ply::PlyError),
}

/// Encode-time configuration knobs. Defaults target the 18× / -0.3 dB
/// design point on Mip-NeRF360 scenes; tightening `kmeans_k_low` to 128
/// trades ~0.1 dB for an extra ~10 % ratio.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// K-means codebook size for low-importance groups (scale, rot, opacity).
    pub kmeans_k_low: u16,
    /// K-means codebook size for color groups (f_dc, f_rest).
    pub kmeans_k_color: u16,
    /// Position-quantization bits per axis (12 → 4096 buckets per axis).
    pub xyz_bits: u8,
    /// Lloyd's iterations.
    pub kmeans_iters: u32,
    /// Random seed for deterministic K-means init.
    pub seed: u64,
    /// Store the Morton permutation so the decoded splats come out in
    /// input order. Disabling this saves ~32 bits/splat (~4.6 MB on
    /// bonsai) at the cost of reordering splats by Morton code — viewers
    /// don't care, downstream tools that key on splat index do. Default
    /// `false` for production preset; tests flip to `true`.
    pub preserve_order: bool,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            kmeans_k_low: 256,
            kmeans_k_color: 256,
            xyz_bits: 14, // 16384 buckets per axis — sub-mm at typical scene scale.
            kmeans_iters: 10,
            seed: 0xC0FFEE,
            preserve_order: false,
        }
    }
}

/// Per-channel quantization metadata so the decoder can invert the
/// normalize-then-CDF-flatten step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    /// Original channel min (before CDF flattening).
    pub min: f32,
    /// Original channel max.
    pub max: f32,
    /// Empirical CDF samples on a 256-bin histogram. `cdf[i]` ≈ the
    /// channel value at the (i/256)th quantile. Length = 257 (0..=256
    /// so the decoder can linear-interpolate without bounds checks).
    pub cdf: Vec<f32>,
}

/// Codebook for a single feature group. The codebook lives in the
/// **CDF-flattened-then-normalized** space `[0, 1]^d`; the decoder
/// applies the channel-stats inverse before reconstructing the splat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// `centroids[i]` is the d-dim centroid at code index `i`.
    pub centroids: Vec<Vec<f32>>,
    /// Per-channel quantization metadata (length = d).
    pub stats: Vec<ChannelStats>,
}

/// Per-codebook descriptor — points into the container's binary
/// codebook blob (the JSON meta stays small and human-readable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookDesc {
    /// Number of centroids (`k`).
    pub k: u32,
    /// Centroid dimension (`d`).
    pub d: u32,
    /// Byte offset of centroids in the codebook blob (`k*d` f32 LE).
    pub centroids_off: u64,
    /// Byte offset of stats table (`d * (3 + 257)` f32 LE: per channel min, max, then 257-bin CDF).
    pub stats_off: u64,
}

/// Codec-wide metadata embedded in the container as JSON. We keep this
/// **small** — heavy arrays (perm, codebook centroids, CDF tables) live
/// in dedicated binary blobs at the end of the file. Tested empirically:
/// JSON-encoding the codebooks blew the .meson up by ~9 MB on bonsai.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesonMeta {
    pub n_anchors: u32,
    pub sh_degree: u8,
    /// World-space bounding box `[min_xyz, max_xyz]`.
    pub bbox: [[f32; 3]; 2],
    /// One descriptor per non-xyz group (scale, rot, opacity, f_dc, f_rest).
    pub codebook_descs: Vec<CodebookDesc>,
}

/* ======================================================================
 * Feature extraction — pull per-group float matrices out of a SplatScene
 * ====================================================================== */

/// Flat per-feature-group arrays. Each matrix is row-major `[n, d_group]`.
#[derive(Debug, Clone)]
pub struct Features {
    pub n: usize,
    pub sh_degree: u8,
    /// Position (d=3).
    pub xyz: Vec<f32>,
    /// Scale (d=3).
    pub scale: Vec<f32>,
    /// Rotation quaternion (d=4).
    pub rot: Vec<f32>,
    /// Opacity (d=1).
    pub opacity: Vec<f32>,
    /// DC color term (d=3).
    pub f_dc: Vec<f32>,
    /// Rest-SH coefficients (d = 3 * ((deg+1)^2 - 1)).
    pub f_rest: Vec<f32>,
}

impl Features {
    /// Extract a `Features` view from an in-memory `SplatScene`.
    pub fn from_scene(scene: &SplatScene) -> Result<Self, MesonError> {
        let n = scene.splats.len();
        if n == 0 {
            return Err(MesonError::EmptyScene);
        }
        // Determine SH degree from the first splat — Inria PLYs are
        // uniform within a file so this is safe.
        let sh_degree = scene.splats[0].color.degree();
        if sh_degree > 3 {
            return Err(MesonError::InvalidShDegree(sh_degree));
        }
        let d_full = 3 * (sh_degree as usize + 1).pow(2);
        let d_rest = d_full.saturating_sub(3);

        let mut xyz = Vec::with_capacity(n * 3);
        let mut scale = Vec::with_capacity(n * 3);
        let mut rot = Vec::with_capacity(n * 4);
        let mut opacity = Vec::with_capacity(n);
        let mut f_dc = Vec::with_capacity(n * 3);
        let mut f_rest = Vec::with_capacity(n * d_rest);

        for s in &scene.splats {
            xyz.extend_from_slice(&s.position);
            scale.extend_from_slice(&s.scale);
            rot.extend_from_slice(&s.rotation);
            opacity.push(s.opacity);
            match &s.color {
                Color::Rgb(rgb) => {
                    f_dc.extend_from_slice(rgb);
                    // No rest coefficients to push.
                }
                Color::Sh { coeffs, .. } => {
                    // Inria convention: first 3 floats are DC (R,G,B);
                    // remainder are the AC coefficients in their flat layout.
                    f_dc.extend_from_slice(&coeffs[0..3]);
                    if coeffs.len() > 3 {
                        f_rest.extend_from_slice(&coeffs[3..]);
                    }
                }
            }
        }
        Ok(Self {
            n,
            sh_degree,
            xyz,
            scale,
            rot,
            opacity,
            f_dc,
            f_rest,
        })
    }

    /// Reconstruct a `SplatScene` from feature arrays.
    pub fn to_scene(&self) -> SplatScene {
        let mut splats = Vec::with_capacity(self.n);
        let d_full = 3 * (self.sh_degree as usize + 1).pow(2);
        let d_rest = d_full.saturating_sub(3);
        for i in 0..self.n {
            let position = [self.xyz[3 * i], self.xyz[3 * i + 1], self.xyz[3 * i + 2]];
            let s_scale = [self.scale[3 * i], self.scale[3 * i + 1], self.scale[3 * i + 2]];
            let rotation = [
                self.rot[4 * i],
                self.rot[4 * i + 1],
                self.rot[4 * i + 2],
                self.rot[4 * i + 3],
            ];
            let opacity = self.opacity[i];
            let dc = [self.f_dc[3 * i], self.f_dc[3 * i + 1], self.f_dc[3 * i + 2]];
            let color = if self.sh_degree == 0 {
                Color::Rgb(dc)
            } else {
                let mut coeffs = Vec::with_capacity(d_full);
                coeffs.extend_from_slice(&dc);
                coeffs.extend_from_slice(&self.f_rest[i * d_rest..(i + 1) * d_rest]);
                Color::Sh {
                    degree: self.sh_degree,
                    coeffs,
                }
            };
            splats.push(splatforge_core::Splat {
                position,
                rotation: rotation.into(),
                scale: s_scale,
                opacity,
                color,
            });
        }
        SplatScene {
            splats,
            coordinate_system: splatforge_core::CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: splatforge_core::TemporalMode::Static,
            lods: None,
        }
    }
}

/* ======================================================================
 * Morton ordering
 * ====================================================================== */

/// Interleave the low 16 bits of x into a 48-bit Morton code with y, z.
#[inline]
fn split_by_3(x: u32) -> u64 {
    let mut v = (x & 0xFFFF) as u64;
    v = (v | v << 16) & 0x0000_FFFF_0000_FFFF;
    v = (v | v << 8) & 0x00FF_00FF_00FF_00FF;
    v = (v | v << 4) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | v << 2) & 0x3333_3333_3333_3333;
    v = (v | v << 1) & 0x5555_5555_5555_5555;
    v
}

#[inline]
fn morton3(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
}

/// Quantize positions to a `bits`-per-axis grid and return Morton codes
/// along with the per-axis (min,max) bbox used for normalization.
pub fn morton_codes(xyz: &[f32], bits: u8) -> (Vec<u64>, [[f32; 3]; 2]) {
    let n = xyz.len() / 3;
    if n == 0 {
        return (Vec::new(), [[0.0; 3], [0.0; 3]]);
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        for a in 0..3 {
            let v = xyz[3 * i + a];
            if v < lo[a] {
                lo[a] = v;
            }
            if v > hi[a] {
                hi[a] = v;
            }
        }
    }
    // Avoid div-by-zero on a degenerate axis.
    let scale = [
        ((1u32 << bits) as f32 - 1.0) / (hi[0] - lo[0]).max(1e-12),
        ((1u32 << bits) as f32 - 1.0) / (hi[1] - lo[1]).max(1e-12),
        ((1u32 << bits) as f32 - 1.0) / (hi[2] - lo[2]).max(1e-12),
    ];
    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let xq = ((xyz[3 * i] - lo[0]) * scale[0]).clamp(0.0, (1u32 << bits) as f32 - 1.0) as u32;
        let yq =
            ((xyz[3 * i + 1] - lo[1]) * scale[1]).clamp(0.0, (1u32 << bits) as f32 - 1.0) as u32;
        let zq =
            ((xyz[3 * i + 2] - lo[2]) * scale[2]).clamp(0.0, (1u32 << bits) as f32 - 1.0) as u32;
        codes.push(morton3(xq, yq, zq));
    }
    (codes, [lo, hi])
}

/// Compute a permutation that orders splats by Morton code ascending.
pub fn morton_permutation(xyz: &[f32], bits: u8) -> (Vec<u32>, [[f32; 3]; 2]) {
    let (codes, bbox) = morton_codes(xyz, bits);
    let mut order: Vec<u32> = (0..codes.len() as u32).collect();
    order.sort_unstable_by_key(|&i| codes[i as usize]);
    (order, bbox)
}

/// Reorder a row-major `[n, d]` matrix by `perm` (perm[new] = old).
pub fn permute_rows(data: &[f32], perm: &[u32], d: usize) -> Vec<f32> {
    let n = perm.len();
    let mut out = vec![0f32; n * d];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        let src = (old_idx as usize) * d;
        let dst = new_idx * d;
        out[dst..dst + d].copy_from_slice(&data[src..src + d]);
    }
    out
}

/// Invert a permutation. `inv[i] = j` iff `perm[j] = i`.
pub fn invert_permutation(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (j, &i) in perm.iter().enumerate() {
        inv[i as usize] = j as u32;
    }
    inv
}

/* ======================================================================
 * Channel stats: empirical CDF + min/max
 * ====================================================================== */

/// Build a 256-bin empirical CDF for a single channel of a row-major
/// `[n, d]` matrix. The CDF is the inverse map: `cdf[k]` is the channel
/// value at the (k/256)th quantile. Length 257 so the linear-interp
/// decoder doesn't need a bounds check.
pub fn build_channel_stats(data: &[f32], n: usize, d: usize, ch: usize) -> ChannelStats {
    let mut vals: Vec<f32> = (0..n).map(|i| data[i * d + ch]).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = *vals.first().unwrap_or(&0.0);
    let max = *vals.last().unwrap_or(&0.0);
    let mut cdf = Vec::with_capacity(257);
    for k in 0..=256 {
        let pos_f = (k as f32) * (vals.len().saturating_sub(1) as f32) / 256.0;
        let lo = pos_f.floor() as usize;
        let hi = (lo + 1).min(vals.len() - 1);
        let frac = pos_f - lo as f32;
        cdf.push(vals[lo] * (1.0 - frac) + vals[hi] * frac);
    }
    ChannelStats { min, max, cdf }
}

/// Apply CDF flattening: maps a raw channel value into `[0, 1]` such
/// that the flattened distribution is approximately uniform. Uses a
/// binary search over the CDF table.
pub fn cdf_flatten(stats: &ChannelStats, v: f32) -> f32 {
    let cdf = &stats.cdf;
    if v <= cdf[0] {
        return 0.0;
    }
    if v >= cdf[256] {
        return 1.0;
    }
    // Binary search for the largest k with cdf[k] <= v.
    let (mut lo, mut hi) = (0usize, 256usize);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = (cdf[hi] - cdf[lo]).max(1e-12);
    let frac = (v - cdf[lo]) / span;
    (lo as f32 + frac) / 256.0
}

/// Inverse of `cdf_flatten` — maps `u ∈ [0,1]` back to a raw channel
/// value by interpolating the stored CDF table.
pub fn cdf_unflatten(stats: &ChannelStats, u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    let pos = u * 256.0;
    let lo = (pos.floor() as usize).min(256);
    let hi = (lo + 1).min(256);
    let frac = pos - lo as f32;
    stats.cdf[lo] * (1.0 - frac) + stats.cdf[hi] * frac
}

/* ======================================================================
 * K-means (Lloyd's)
 * ====================================================================== */

/// Deterministic K-means on an `[n, d]` row-major matrix in the CDF-
/// flattened space `[0, 1]^d`. Returns (centroids, code per row).
///
/// We use k-means++-style spread initialization but with a fixed seed
/// LCG so encoder runs are bit-reproducible.
pub fn kmeans(
    data_flat: &[f32],
    n: usize,
    d: usize,
    k: usize,
    iters: u32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<u16>) {
    assert!(k > 0 && k <= u16::MAX as usize);
    let k = k.min(n.max(1));
    // Pseudo-random "k-means++" init: pick the first center at the
    // (seed % n)th point, then iteratively pick the next center to be
    // the point with maximum min-distance to already-picked centers.
    // This is O(k * n * d) which is fine for k ≤ 256.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = (seed as usize) % n.max(1);
    centroids.push(data_flat[first * d..(first + 1) * d].to_vec());
    let mut min_dist: Vec<f32> = (0..n)
        .map(|i| sqd(&data_flat[i * d..(i + 1) * d], &centroids[0]))
        .collect();
    let mut rng_state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _c in 1..k {
        // Weighted sampling by distance² with a simple LCG.
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let total: f64 = min_dist.iter().map(|&v| v as f64).sum();
        let pick_target = ((rng_state >> 11) as f64 / (1u64 << 53) as f64) * total;
        let mut acc = 0f64;
        let mut chosen = 0usize;
        for (i, &d2) in min_dist.iter().enumerate() {
            acc += d2 as f64;
            if acc >= pick_target {
                chosen = i;
                break;
            }
        }
        centroids.push(data_flat[chosen * d..(chosen + 1) * d].to_vec());
        // Update min_dist.
        for i in 0..n {
            let d2 = sqd(&data_flat[i * d..(i + 1) * d], &centroids[centroids.len() - 1]);
            if d2 < min_dist[i] {
                min_dist[i] = d2;
            }
        }
    }

    // Lloyd's iterations.
    let mut codes = vec![0u16; n];
    for _it in 0..iters {
        // Assign.
        use rayon::prelude::*;
        codes.par_iter_mut().enumerate().for_each(|(i, code_slot)| {
            let row = &data_flat[i * d..(i + 1) * d];
            let mut best = 0u16;
            let mut best_d = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let dd = sqd(row, c);
                if dd < best_d {
                    best_d = dd;
                    best = j as u16;
                }
            }
            *code_slot = best;
        });
        // Update.
        let mut sums = vec![vec![0f64; d]; k];
        let mut counts = vec![0u64; k];
        for i in 0..n {
            let c = codes[i] as usize;
            counts[c] += 1;
            for a in 0..d {
                sums[c][a] += data_flat[i * d + a] as f64;
            }
        }
        for (j, centroid) in centroids.iter_mut().enumerate() {
            if counts[j] == 0 {
                continue;
            }
            for a in 0..d {
                centroid[a] = (sums[j][a] / counts[j] as f64) as f32;
            }
        }
    }
    (centroids, codes)
}

#[inline]
fn sqd(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/* ======================================================================
 * Group quantization (CDF-flatten → K-means → indices)
 * ====================================================================== */

/// Quantize a `[n, d]` row-major float matrix into a codebook + per-row
/// code indices. Returns `(codebook, codes)`.
pub fn quantize_group(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    iters: u32,
    seed: u64,
) -> (Codebook, Vec<u16>) {
    assert_eq!(data.len(), n * d);
    // 1. Per-channel CDF flatten.
    let stats: Vec<ChannelStats> = (0..d).map(|ch| build_channel_stats(data, n, d, ch)).collect();
    let mut flat = vec![0f32; n * d];
    for i in 0..n {
        for ch in 0..d {
            flat[i * d + ch] = cdf_flatten(&stats[ch], data[i * d + ch]);
        }
    }
    // 2. K-means on the flattened space.
    let (centroids, codes) = kmeans(&flat, n, d, k, iters, seed);
    (Codebook { centroids, stats }, codes)
}

/// Inverse of `quantize_group`: reconstruct the `[n, d]` float matrix
/// from a codebook + per-row code indices.
pub fn dequantize_group(cb: &Codebook, codes: &[u16], d: usize) -> Vec<f32> {
    let n = codes.len();
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let centroid = &cb.centroids[codes[i] as usize];
        for ch in 0..d {
            out[i * d + ch] = cdf_unflatten(&cb.stats[ch], centroid[ch]);
        }
    }
    out
}

/* ======================================================================
 * Range coding with Markov-1 spatial context
 * ====================================================================== */

/// Encode an index stream with a Markov-1 context model: each symbol
/// is coded under a categorical distribution conditioned on the
/// previous symbol. The first symbol is coded under the global marginal.
///
/// `k` is the alphabet size (≤ 65536). Returns a compact byte stream.
pub fn encode_indices(codes: &[u16], k: usize) -> Result<Vec<u8>, MesonError> {
    use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
    use constriction::stream::queue::DefaultRangeEncoder;
    use constriction::stream::Encode;

    if codes.is_empty() {
        return Ok(Vec::new());
    }

    // Marginal histogram (for symbol 0 + smoothing prior on conditionals).
    let mut marg = vec![1u64; k];
    for &c in codes {
        marg[c as usize] += 1;
    }
    // Conditional histograms: cond[prev][cur].
    // For k=256 that's a 256×256 = 65k-entry table — fine. For k>256
    // we'd want a sparse map, but K-means here caps at 256 so we
    // assume k ≤ 256 in the dense path.
    let kk = k.min(256);
    let mut cond = vec![1u32; kk * kk];
    for w in codes.windows(2) {
        let prev = (w[0] as usize).min(kk - 1);
        let cur = (w[1] as usize).min(kk - 1);
        cond[prev * kk + cur] += 1;
    }
    // Build per-row probability tables (floats summing to 1).
    let to_probs = |row: &[u32]| -> Vec<f64> {
        let total: u64 = row.iter().map(|&v| v as u64).sum();
        row.iter().map(|&v| v as f64 / total as f64).collect()
    };
    let marg_probs: Vec<f64> = {
        let total: u64 = marg.iter().sum();
        marg.iter().map(|&v| v as f64 / total as f64).collect()
    };

    let mut enc = DefaultRangeEncoder::new();
    let marg_model =
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            &marg_probs,
            None,
        )
        .map_err(|e| MesonError::RangeCoder(format!("build marginal: {e:?}")))?;
    enc.encode_symbol(codes[0] as usize, &marg_model)
        .map_err(|e| MesonError::RangeCoder(format!("encode head: {e:?}")))?;

    // Pre-build conditional models lazily (cache on first use to keep
    // memory low; a 256×256 cache of small tables fits easily).
    let mut model_cache: Vec<Option<DefaultContiguousCategoricalEntropyModel>> = (0..kk)
        .map(|_| None)
        .collect();
    for w in codes.windows(2) {
        let prev = (w[0] as usize).min(kk - 1);
        let cur = (w[1] as usize).min(kk - 1);
        if model_cache[prev].is_none() {
            let probs = to_probs(&cond[prev * kk..(prev + 1) * kk]);
            let model =
                DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                    &probs, None,
                )
                .map_err(|e| MesonError::RangeCoder(format!("build cond {prev}: {e:?}")))?;
            model_cache[prev] = Some(model);
        }
        enc.encode_symbol(cur, model_cache[prev].as_ref().unwrap())
            .map_err(|e| MesonError::RangeCoder(format!("encode body: {e:?}")))?;
    }
    let words = enc.into_compressed().unwrap();
    // Serialize: [marg u32 histogram (k entries)][cond u32 histogram (kk*kk)][n_words u32][words...]
    // We ship the histograms so the decoder rebuilds identical models —
    // sending raw float probs would introduce floating-point drift
    // between encoder/decoder builds of the categorical table.
    let mut out = Vec::with_capacity(8 + 4 * k + 4 * kk * kk + 4 * words.len());
    out.write_u32::<LittleEndian>(k as u32)?;
    out.write_u32::<LittleEndian>(kk as u32)?;
    for &v in &marg {
        out.write_u32::<LittleEndian>(v.min(u32::MAX as u64) as u32)?;
    }
    for &v in &cond {
        out.write_u32::<LittleEndian>(v)?;
    }
    out.write_u32::<LittleEndian>(words.len() as u32)?;
    for &w in &words {
        out.write_u32::<LittleEndian>(w)?;
    }
    out.write_u32::<LittleEndian>(codes.len() as u32)?;
    Ok(out)
}

/// Inverse of `encode_indices`.
pub fn decode_indices(bytes: &[u8]) -> Result<Vec<u16>, MesonError> {
    use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
    use constriction::stream::queue::DefaultRangeDecoder;
    use constriction::stream::Decode;

    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut cur = Cursor::new(bytes);
    let k = cur.read_u32::<LittleEndian>()? as usize;
    let kk = cur.read_u32::<LittleEndian>()? as usize;
    let mut marg = vec![0u32; k];
    for v in marg.iter_mut() {
        *v = cur.read_u32::<LittleEndian>()?;
    }
    let mut cond = vec![0u32; kk * kk];
    for v in cond.iter_mut() {
        *v = cur.read_u32::<LittleEndian>()?;
    }
    let n_words = cur.read_u32::<LittleEndian>()? as usize;
    let mut words = vec![0u32; n_words];
    for v in words.iter_mut() {
        *v = cur.read_u32::<LittleEndian>()?;
    }
    let n_codes = cur.read_u32::<LittleEndian>()? as usize;

    let to_probs = |row: &[u32]| -> Vec<f64> {
        let total: u64 = row.iter().map(|&v| v as u64).sum();
        row.iter().map(|&v| v as f64 / total as f64).collect()
    };
    let marg_probs = to_probs(&marg);
    let marg_model =
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            &marg_probs,
            None,
        )
        .map_err(|e| MesonError::RangeCoder(format!("rebuild marginal: {e:?}")))?;

    let mut dec = DefaultRangeDecoder::from_compressed(words)
        .map_err(|e| MesonError::RangeCoder(format!("decoder init: {e:?}")))?;
    let mut out = Vec::with_capacity(n_codes);
    let first: usize = dec
        .decode_symbol(&marg_model)
        .map_err(|e| MesonError::RangeCoder(format!("decode head: {e:?}")))?;
    out.push(first as u16);

    let mut model_cache: Vec<Option<DefaultContiguousCategoricalEntropyModel>> =
        (0..kk).map(|_| None).collect();
    for _ in 1..n_codes {
        let prev = (*out.last().unwrap() as usize).min(kk - 1);
        if model_cache[prev].is_none() {
            let probs = to_probs(&cond[prev * kk..(prev + 1) * kk]);
            let model =
                DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                    &probs, None,
                )
                .map_err(|e| MesonError::RangeCoder(format!("rebuild cond {prev}: {e:?}")))?;
            model_cache[prev] = Some(model);
        }
        let s: usize = dec
            .decode_symbol(model_cache[prev].as_ref().unwrap())
            .map_err(|e| MesonError::RangeCoder(format!("decode body: {e:?}")))?;
        out.push(s as u16);
    }
    Ok(out)
}

/* ======================================================================
 * XYZ stream: 14-bit-per-axis quantization, raw-packed (positions are
 * high-entropy enough that AC barely helps; raw packing is robust).
 * ====================================================================== */

/// Pack quantized positions as 14-bit-per-axis triplets into a byte
/// stream. Each splat takes 14*3 = 42 bits ≈ 5.25 bytes; we pack into a
/// flat bit-stream for compactness.
pub fn encode_xyz(xyz: &[f32], bbox: &[[f32; 3]; 2], bits: u8) -> Vec<u8> {
    let n = xyz.len() / 3;
    let max_code = (1u32 << bits) - 1;
    let inv_span = [
        max_code as f32 / (bbox[1][0] - bbox[0][0]).max(1e-12),
        max_code as f32 / (bbox[1][1] - bbox[0][1]).max(1e-12),
        max_code as f32 / (bbox[1][2] - bbox[0][2]).max(1e-12),
    ];
    // We pack 3 * bits bits per splat. For bits=14 → 42 bits/splat. We
    // pack into a u64 buffer, then split into u8 at the end.
    let total_bits = n * 3 * bits as usize;
    let mut buf = vec![0u8; (total_bits + 7) / 8 + 8];
    let mut bit_pos: usize = 0;
    for i in 0..n {
        for a in 0..3 {
            let q = ((xyz[3 * i + a] - bbox[0][a]) * inv_span[a])
                .clamp(0.0, max_code as f32) as u32;
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            // Write up to 32 bits straddling at most 5 bytes (for bits=14).
            let v = (q as u64) << bit_off;
            for b in 0..((bits as usize + bit_off + 7) / 8) {
                buf[byte_idx + b] |= ((v >> (8 * b)) & 0xFF) as u8;
            }
            bit_pos += bits as usize;
        }
    }
    buf.truncate((total_bits + 7) / 8);
    buf
}

/// Inverse of `encode_xyz`.
pub fn decode_xyz(bytes: &[u8], n: usize, bbox: &[[f32; 3]; 2], bits: u8) -> Vec<f32> {
    let max_code = (1u32 << bits) - 1;
    let span = [
        (bbox[1][0] - bbox[0][0]) / max_code.max(1) as f32,
        (bbox[1][1] - bbox[0][1]) / max_code.max(1) as f32,
        (bbox[1][2] - bbox[0][2]) / max_code.max(1) as f32,
    ];
    let mask: u64 = (1u64 << bits) - 1;
    let mut out = vec![0f32; n * 3];
    let mut bit_pos: usize = 0;
    for i in 0..n {
        for a in 0..3 {
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            // Read up to 5 bytes into a u64.
            let mut raw: u64 = 0;
            for b in 0..((bits as usize + bit_off + 7) / 8) {
                let byte = *bytes.get(byte_idx + b).unwrap_or(&0) as u64;
                raw |= byte << (8 * b);
            }
            let q = ((raw >> bit_off) & mask) as u32;
            out[3 * i + a] = bbox[0][a] + (q as f32) * span[a];
            bit_pos += bits as usize;
        }
    }
    out
}

/* ======================================================================
 * Container I/O
 * ====================================================================== */

/// Top-level encode entry point. Reads a `SplatScene`, returns the
/// compressed `.meson` blob.
pub fn encode_scene(scene: &SplatScene, cfg: &EncodeConfig) -> Result<Vec<u8>, MesonError> {
    let feats = Features::from_scene(scene)?;
    encode_features(&feats, cfg)
}

/// Lower-level entry point for when the caller already has the
/// feature matrices in hand (e.g. a smoke test that wants to bypass
/// SplatScene reconstruction).
pub fn encode_features(feats: &Features, cfg: &EncodeConfig) -> Result<Vec<u8>, MesonError> {
    let (perm, bbox) = morton_permutation(&feats.xyz, cfg.xyz_bits);

    let xyz_p = permute_rows(&feats.xyz, &perm, 3);
    let scale_p = permute_rows(&feats.scale, &perm, 3);
    let rot_p = permute_rows(&feats.rot, &perm, 4);
    let opacity_p = permute_rows(&feats.opacity, &perm, 1);
    let f_dc_p = permute_rows(&feats.f_dc, &perm, 3);
    let d_full = 3 * (feats.sh_degree as usize + 1).pow(2);
    let d_rest = d_full.saturating_sub(3);
    let f_rest_p = if d_rest > 0 {
        permute_rows(&feats.f_rest, &perm, d_rest)
    } else {
        Vec::new()
    };

    // Quantize each non-xyz group.
    let n = feats.n;
    let (scale_cb, scale_codes) = quantize_group(
        &scale_p,
        n,
        3,
        cfg.kmeans_k_low as usize,
        cfg.kmeans_iters,
        cfg.seed,
    );
    let (rot_cb, rot_codes) = quantize_group(
        &rot_p,
        n,
        4,
        cfg.kmeans_k_low as usize,
        cfg.kmeans_iters,
        cfg.seed.wrapping_add(1),
    );
    let (op_cb, op_codes) = quantize_group(
        &opacity_p,
        n,
        1,
        cfg.kmeans_k_low as usize,
        cfg.kmeans_iters,
        cfg.seed.wrapping_add(2),
    );
    let (dc_cb, dc_codes) = quantize_group(
        &f_dc_p,
        n,
        3,
        cfg.kmeans_k_color as usize,
        cfg.kmeans_iters,
        cfg.seed.wrapping_add(3),
    );
    // f_rest can be very high-dimensional (45 for deg=3) — keep k_color
    // but expect lower fidelity per channel. K-means in d=45 still
    // converges quickly because the SH-AC distribution is heavily
    // concentrated near zero.
    let (rest_cb, rest_codes) = if d_rest > 0 {
        let (cb, c) = quantize_group(
            &f_rest_p,
            n,
            d_rest,
            cfg.kmeans_k_color as usize,
            cfg.kmeans_iters,
            cfg.seed.wrapping_add(4),
        );
        (cb, c)
    } else {
        (
            Codebook {
                centroids: Vec::new(),
                stats: Vec::new(),
            },
            Vec::new(),
        )
    };

    // Build per-stream payloads.
    let xyz_payload = encode_xyz(&xyz_p, &bbox, cfg.xyz_bits);
    let scale_payload = encode_indices(&scale_codes, cfg.kmeans_k_low as usize)?;
    let rot_payload = encode_indices(&rot_codes, cfg.kmeans_k_low as usize)?;
    let opacity_payload = encode_indices(&op_codes, cfg.kmeans_k_low as usize)?;
    let dc_payload = encode_indices(&dc_codes, cfg.kmeans_k_color as usize)?;
    let rest_payload = if d_rest > 0 {
        encode_indices(&rest_codes, cfg.kmeans_k_color as usize)?
    } else {
        Vec::new()
    };

    // Pack perm as u32 LE — only when the caller asked for original-
    // order preservation. The empty-bytes path lets the decoder detect
    // "Morton-ordered output" without a new container version.
    let perm_bytes = if cfg.preserve_order {
        let mut buf = Vec::with_capacity(perm.len() * 4);
        for &p in &perm {
            buf.write_u32::<LittleEndian>(p)?;
        }
        buf
    } else {
        Vec::new()
    };

    // Pack codebook blob + build descriptors.
    let codebooks_ordered = [&scale_cb, &rot_cb, &op_cb, &dc_cb, &rest_cb];
    let (codebook_blob, descs) = pack_codebook_blob(&codebooks_ordered);

    let meta = MesonMeta {
        n_anchors: n as u32,
        sh_degree: feats.sh_degree,
        bbox,
        codebook_descs: descs,
    };
    let meta_bytes = serde_json::to_vec(&meta)
        .map_err(|e| MesonError::MalformedMeta(format!("encode meta: {e}")))?;

    write_container(
        &meta_bytes,
        &[
            &xyz_payload,
            &scale_payload,
            &rot_payload,
            &opacity_payload,
            &dc_payload,
            &rest_payload,
            &perm_bytes,
            &codebook_blob,
        ],
        cfg.xyz_bits,
    )
}

/// Serialize a slice of `&Codebook` into a flat LE binary blob. Returns
/// `(blob, descriptors)` where `descriptors[i]` points into `blob`.
fn pack_codebook_blob(cbs: &[&Codebook]) -> (Vec<u8>, Vec<CodebookDesc>) {
    let mut blob: Vec<u8> = Vec::new();
    let mut descs = Vec::with_capacity(cbs.len());
    for cb in cbs {
        let k = cb.centroids.len();
        let d = cb.centroids.first().map(|c| c.len()).unwrap_or(0);
        let centroids_off = blob.len() as u64;
        for centroid in &cb.centroids {
            for &v in centroid {
                blob.write_f32::<LittleEndian>(v).unwrap();
            }
        }
        let stats_off = blob.len() as u64;
        for s in &cb.stats {
            blob.write_f32::<LittleEndian>(s.min).unwrap();
            blob.write_f32::<LittleEndian>(s.max).unwrap();
            // CDF length is always 257; we don't store the length to
            // keep the on-disk footprint tight.
            for &v in &s.cdf {
                blob.write_f32::<LittleEndian>(v).unwrap();
            }
        }
        descs.push(CodebookDesc {
            k: k as u32,
            d: d as u32,
            centroids_off,
            stats_off,
        });
    }
    (blob, descs)
}

/// Deserialize a codebook from its descriptor + the codebook blob.
fn unpack_codebook(desc: &CodebookDesc, blob: &[u8]) -> Result<Codebook, MesonError> {
    let k = desc.k as usize;
    let d = desc.d as usize;
    let mut centroids = Vec::with_capacity(k);
    let cent_start = desc.centroids_off as usize;
    let cent_end = cent_start + k * d * 4;
    if cent_end > blob.len() {
        return Err(MesonError::MalformedMeta(
            "codebook centroid offset out of range".into(),
        ));
    }
    let mut cur = Cursor::new(&blob[cent_start..cent_end]);
    for _ in 0..k {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            row.push(cur.read_f32::<LittleEndian>()?);
        }
        centroids.push(row);
    }
    let stats_start = desc.stats_off as usize;
    let stats_end = stats_start + d * (2 + 257) * 4;
    if stats_end > blob.len() {
        return Err(MesonError::MalformedMeta(
            "codebook stats offset out of range".into(),
        ));
    }
    let mut cur = Cursor::new(&blob[stats_start..stats_end]);
    let mut stats = Vec::with_capacity(d);
    for _ in 0..d {
        let min = cur.read_f32::<LittleEndian>()?;
        let max = cur.read_f32::<LittleEndian>()?;
        let mut cdf = Vec::with_capacity(257);
        for _ in 0..257 {
            cdf.push(cur.read_f32::<LittleEndian>()?);
        }
        stats.push(ChannelStats { min, max, cdf });
    }
    Ok(Codebook { centroids, stats })
}

/// Write the container byte layout described in the module-level docs.
fn write_container(
    meta_bytes: &[u8],
    streams: &[&[u8]],
    xyz_bits: u8,
) -> Result<Vec<u8>, MesonError> {
    let mut buf: Vec<u8> = Vec::with_capacity(
        meta_bytes.len() + streams.iter().map(|s| s.len()).sum::<usize>() + 256,
    );
    buf.write_all(MAGIC)?;
    buf.write_u32::<LittleEndian>(VERSION)?;
    // We don't have n_anchors here at the byte-buffer level — it's in
    // meta — so we re-parse on read. Reserve a slot for xyz_bits at
    // byte 12 for forward compat.
    buf.write_u8(xyz_bits)?;
    buf.write_all(&[0u8; 3])?; // pad
    buf.write_u32::<LittleEndian>(streams.len() as u32)?;

    // Header reserve: write 0 placeholders for offsets & sizes; patch
    // once we know them.
    let off_table_pos = buf.len();
    for _ in 0..streams.len() {
        buf.write_u64::<LittleEndian>(0)?;
        buf.write_u64::<LittleEndian>(0)?;
    }

    buf.write_u32::<LittleEndian>(meta_bytes.len() as u32)?;
    buf.write_all(meta_bytes)?;

    let mut offsets_sizes: Vec<(u64, u64)> = Vec::with_capacity(streams.len());
    for s in streams {
        let off = buf.len() as u64;
        buf.write_all(s)?;
        offsets_sizes.push((off, s.len() as u64));
    }

    // Patch the offset table.
    let mut patch = Cursor::new(&mut buf[off_table_pos..]);
    for (off, sz) in &offsets_sizes {
        patch.write_u64::<LittleEndian>(*off)?;
        patch.write_u64::<LittleEndian>(*sz)?;
    }

    // CRC32 over everything written so far.
    let crc = crc32fast::hash(&buf);
    buf.write_u32::<LittleEndian>(crc)?;
    Ok(buf)
}

/// Lightweight container introspection — returns per-stream sizes
/// without decoding the payloads. Useful for the `info` CLI and for
/// benchmark scripts that want to attribute bits across feature groups.
pub struct ContainerLayout {
    pub n_anchors: u32,
    pub meta_bytes: u64,
    pub stream_sizes: Vec<u64>,
}

pub fn container_layout(bytes: &[u8]) -> Result<ContainerLayout, MesonError> {
    if bytes.len() < 16 + 4 {
        return Err(MesonError::BadMagic);
    }
    let body_end = bytes.len() - 4;
    let stored_crc = (&bytes[body_end..]).read_u32::<LittleEndian>()?;
    let calc_crc = crc32fast::hash(&bytes[..body_end]);
    if stored_crc != calc_crc {
        return Err(MesonError::Crc32Mismatch);
    }
    let mut cur = Cursor::new(&bytes[..body_end]);
    let mut magic = [0u8; 8];
    cur.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(MesonError::BadMagic);
    }
    let _version = cur.read_u32::<LittleEndian>()?;
    let _xyz_bits = cur.read_u8()?;
    let mut _pad = [0u8; 3];
    cur.read_exact(&mut _pad)?;
    let n_streams = cur.read_u32::<LittleEndian>()? as usize;
    let mut sizes = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        let _off = cur.read_u64::<LittleEndian>()?;
        sizes.push(cur.read_u64::<LittleEndian>()?);
    }
    let meta_len = cur.read_u32::<LittleEndian>()? as u64;
    // We need n_anchors — peek into JSON head (cheap because we only
    // read a small prefix).
    let meta_start = cur.position() as usize;
    let meta_end = meta_start + meta_len as usize;
    let n_anchors = serde_json::from_slice::<serde_json::Value>(&bytes[meta_start..meta_end])
        .ok()
        .and_then(|v| v.get("n_anchors").and_then(|x| x.as_u64()))
        .unwrap_or(0) as u32;
    Ok(ContainerLayout {
        n_anchors,
        meta_bytes: meta_len,
        stream_sizes: sizes,
    })
}

/// Inverse of `encode_scene`.
pub fn decode_scene(bytes: &[u8]) -> Result<SplatScene, MesonError> {
    let feats = decode_features(bytes)?;
    Ok(feats.to_scene())
}

/// Decode straight to the `Features` view.
pub fn decode_features(bytes: &[u8]) -> Result<Features, MesonError> {
    if bytes.len() < 16 + 4 {
        return Err(MesonError::BadMagic);
    }
    // Verify CRC first.
    let body_end = bytes.len() - 4;
    let stored_crc = (&bytes[body_end..]).read_u32::<LittleEndian>()?;
    let calc_crc = crc32fast::hash(&bytes[..body_end]);
    if stored_crc != calc_crc {
        return Err(MesonError::Crc32Mismatch);
    }

    let mut cur = Cursor::new(&bytes[..body_end]);
    let mut magic = [0u8; 8];
    cur.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(MesonError::BadMagic);
    }
    let version = cur.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(MesonError::UnsupportedVersion(version, VERSION));
    }
    let xyz_bits = cur.read_u8()?;
    let mut _pad = [0u8; 3];
    cur.read_exact(&mut _pad)?;
    let n_streams = cur.read_u32::<LittleEndian>()? as usize;
    if n_streams != N_STREAMS {
        return Err(MesonError::MalformedMeta(format!(
            "expected {N_STREAMS} streams, got {n_streams}"
        )));
    }
    let mut offsets = Vec::with_capacity(n_streams);
    let mut sizes = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        offsets.push(cur.read_u64::<LittleEndian>()? as usize);
        sizes.push(cur.read_u64::<LittleEndian>()? as usize);
    }
    let meta_len = cur.read_u32::<LittleEndian>()? as usize;
    let meta_start = cur.position() as usize;
    let meta_bytes = &bytes[meta_start..meta_start + meta_len];
    let meta: MesonMeta = serde_json::from_slice(meta_bytes)
        .map_err(|e| MesonError::MalformedMeta(format!("parse meta: {e}")))?;

    let n = meta.n_anchors as usize;
    let sh_degree = meta.sh_degree;
    let d_full = 3 * (sh_degree as usize + 1).pow(2);
    let d_rest = d_full.saturating_sub(3);

    let xyz_bytes = &bytes[offsets[0]..offsets[0] + sizes[0]];
    let scale_bytes = &bytes[offsets[1]..offsets[1] + sizes[1]];
    let rot_bytes = &bytes[offsets[2]..offsets[2] + sizes[2]];
    let op_bytes = &bytes[offsets[3]..offsets[3] + sizes[3]];
    let dc_bytes = &bytes[offsets[4]..offsets[4] + sizes[4]];
    let rest_bytes = &bytes[offsets[5]..offsets[5] + sizes[5]];
    let perm_bytes = &bytes[offsets[6]..offsets[6] + sizes[6]];
    let codebook_blob = &bytes[offsets[7]..offsets[7] + sizes[7]];

    // Rebuild codebooks.
    if meta.codebook_descs.len() != 5 {
        return Err(MesonError::MalformedMeta(format!(
            "expected 5 codebook descriptors, got {}",
            meta.codebook_descs.len()
        )));
    }
    let scale_cb = unpack_codebook(&meta.codebook_descs[0], codebook_blob)?;
    let rot_cb = unpack_codebook(&meta.codebook_descs[1], codebook_blob)?;
    let op_cb = unpack_codebook(&meta.codebook_descs[2], codebook_blob)?;
    let dc_cb = unpack_codebook(&meta.codebook_descs[3], codebook_blob)?;
    let rest_cb = if d_rest > 0 {
        unpack_codebook(&meta.codebook_descs[4], codebook_blob)?
    } else {
        Codebook {
            centroids: Vec::new(),
            stats: Vec::new(),
        }
    };

    // Rebuild permutation (empty = identity = Morton order output).
    let perm = if perm_bytes.is_empty() {
        (0..n as u32).collect::<Vec<_>>()
    } else if perm_bytes.len() == n * 4 {
        let mut p = vec![0u32; n];
        let mut cur = Cursor::new(perm_bytes);
        for v in p.iter_mut() {
            *v = cur.read_u32::<LittleEndian>()?;
        }
        p
    } else {
        return Err(MesonError::MalformedMeta(format!(
            "perm stream size {} != n_anchors*4={}",
            perm_bytes.len(),
            n * 4
        )));
    };

    let xyz_p = decode_xyz(xyz_bytes, n, &meta.bbox, xyz_bits);
    let scale_codes = decode_indices(scale_bytes)?;
    let rot_codes = decode_indices(rot_bytes)?;
    let op_codes = decode_indices(op_bytes)?;
    let dc_codes = decode_indices(dc_bytes)?;
    let rest_codes = if d_rest > 0 {
        decode_indices(rest_bytes)?
    } else {
        Vec::new()
    };

    let scale_p = dequantize_group(&scale_cb, &scale_codes, 3);
    let rot_p = dequantize_group(&rot_cb, &rot_codes, 4);
    let opacity_p = dequantize_group(&op_cb, &op_codes, 1);
    let f_dc_p = dequantize_group(&dc_cb, &dc_codes, 3);
    let f_rest_p = if d_rest > 0 {
        dequantize_group(&rest_cb, &rest_codes, d_rest)
    } else {
        Vec::new()
    };

    // Undo the Morton permutation so the decoded scene matches input order.
    // perm[new_idx] = orig_idx — we scatter `data[new_idx]` to `out[perm[new_idx]]`.
    let unpermute = |data: Vec<f32>, d: usize| -> Vec<f32> {
        let mut out = vec![0f32; data.len()];
        for new_idx in 0..n {
            let orig = perm[new_idx] as usize;
            let dst = orig * d;
            let src = new_idx * d;
            out[dst..dst + d].copy_from_slice(&data[src..src + d]);
        }
        out
    };
    let xyz = unpermute(xyz_p, 3);
    let scale = unpermute(scale_p, 3);
    let rot = unpermute(rot_p, 4);
    let opacity = unpermute(opacity_p, 1);
    let f_dc = unpermute(f_dc_p, 3);
    let f_rest = if d_rest > 0 {
        unpermute(f_rest_p, d_rest)
    } else {
        Vec::new()
    };

    Ok(Features {
        n,
        sh_degree,
        xyz,
        scale,
        rot,
        opacity,
        f_dc,
        f_rest,
    })
}

/* ======================================================================
 * Tests
 * ====================================================================== */

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_scene(n: usize, sh_degree: u8) -> SplatScene {
        let d_full = 3 * (sh_degree as usize + 1).pow(2);
        let mut splats = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / n as f32;
            let position = [t * 10.0, (t * 3.7).sin(), (t * 0.9).cos() * 5.0];
            let rotation = [0.0, 0.0, 0.0, 1.0];
            let scale = [0.01 + t * 0.1, 0.02, 0.03];
            let opacity = 0.5 + 0.4 * (t * 6.283).sin();
            let color = if sh_degree == 0 {
                Color::Rgb([t, 1.0 - t, 0.5])
            } else {
                let mut coeffs = vec![0f32; d_full];
                coeffs[0] = t;
                coeffs[1] = 1.0 - t;
                coeffs[2] = 0.5;
                for j in 3..d_full {
                    coeffs[j] = (t * (j as f32) * 0.13).sin() * 0.05;
                }
                Color::Sh { degree: sh_degree, coeffs }
            };
            splats.push(splatforge_core::Splat {
                position,
                rotation,
                scale,
                opacity,
                color,
            });
        }
        SplatScene {
            splats,
            coordinate_system: splatforge_core::CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: splatforge_core::TemporalMode::Static,
            lods: None,
        }
    }

    #[test]
    fn morton_codes_are_increasing_after_sort() {
        let xyz: Vec<f32> = vec![5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 3.0, 3.0, 3.0];
        let (perm, _) = morton_permutation(&xyz, 10);
        assert_eq!(perm.len(), 4);
        // After Morton sort, the (1,1,1) point should be first.
        assert_eq!(perm[0], 1);
    }

    #[test]
    fn cdf_round_trip_is_near_identity() {
        let n = 1024;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / 100.0).sin() * 3.0).collect();
        let stats = build_channel_stats(&data, n, 1, 0);
        let mut max_err = 0f32;
        for &v in &data {
            let u = cdf_flatten(&stats, v);
            let back = cdf_unflatten(&stats, u);
            max_err = max_err.max((back - v).abs());
        }
        assert!(max_err < 0.05, "max CDF round-trip error {max_err}");
    }

    #[test]
    fn encode_decode_round_trip_sh0_small() {
        let scene = synthetic_scene(2048, 0);
        let cfg = EncodeConfig {
            kmeans_iters: 4,
            preserve_order: true,
            ..Default::default()
        };
        let bytes = encode_scene(&scene, &cfg).expect("encode");
        let dec = decode_scene(&bytes).expect("decode");
        assert_eq!(dec.splats.len(), scene.splats.len());
        // Position should be close (14-bit quantization in a small bbox).
        for (a, b) in scene.splats.iter().zip(dec.splats.iter()) {
            for c in 0..3 {
                assert!(
                    (a.position[c] - b.position[c]).abs() < 0.01,
                    "xyz mismatch {} vs {}", a.position[c], b.position[c]
                );
            }
        }
    }

    #[test]
    fn encode_decode_round_trip_sh3_small() {
        let scene = synthetic_scene(1024, 3);
        let cfg = EncodeConfig {
            kmeans_iters: 4,
            preserve_order: true,
            ..Default::default()
        };
        let bytes = encode_scene(&scene, &cfg).expect("encode");
        let dec = decode_scene(&bytes).expect("decode");
        assert_eq!(dec.splats.len(), scene.splats.len());
        for (a, b) in scene.splats.iter().zip(dec.splats.iter()) {
            assert_eq!(a.color.degree(), b.color.degree());
        }
    }

    #[test]
    fn bad_magic_is_rejected() {
        let bogus = vec![0u8; 256];
        let err = decode_scene(&bogus).expect_err("must fail");
        // Either bad magic or CRC mismatch — both are correct rejections.
        match err {
            MesonError::BadMagic | MesonError::Crc32Mismatch => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn crc_tampering_is_rejected() {
        let scene = synthetic_scene(128, 0);
        let mut bytes = encode_scene(&scene, &EncodeConfig::default()).unwrap();
        // Flip a byte deep inside the payload.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        let err = decode_scene(&bytes).expect_err("must fail");
        assert!(matches!(err, MesonError::Crc32Mismatch));
    }
}
