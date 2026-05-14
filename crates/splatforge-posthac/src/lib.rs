#![deny(clippy::all)]
//! PostHAC — hash-grid hyperprior range-coded entropy compression.
//!
//! See `specs/0014-posthac-io.md` (TBD) for the wire format. PostHAC sits as
//! a sidecar layer on top of an existing splat scene's 8-bit per-column
//! attribute quantization. Given a trained hash-grid + MLP head that predicts
//! a Gaussian distribution for each splat's attribute code from its 3D
//! position, the encoder range-codes the 8-bit codes against per-symbol
//! quantized Gaussians; the decoder runs the same hash-grid forward pass
//! and range-decodes the codes back.
//!
//! Round-trip is bit-exact: `decode(encode(scene)) == scene` at 8-bit
//! attribute precision (the same operating point as today's web-mobile
//! pipeline). The win is in bytes: PostHAC's hyperprior-conditioned
//! arithmetic coding is meaningfully tighter than zstd over the same
//! 8-bit code stream — typically 1.5-2× extra at scene scale.
//!
//! Python reference: `apps/diff-repack/posthac_codec.py` in the private
//! repo. This crate is the production port intended for both native CLI
//! integration via `splatforge-spz` and a WASM build for the viewer.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};
use thiserror::Error;

/// Magic identifier ("PTHC" little-endian).
pub const POSTHAC_MAGIC: u32 = 0x4348_5450;
/// Bitstream wire version.
pub const POSTHAC_VERSION: u32 = 1;

/// PostHAC codec errors.
#[derive(Debug, Error)]
pub enum PostHacError {
    /// Underlying IO failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// File does not begin with the PostHAC magic.
    #[error("not a PostHAC stream (bad magic)")]
    BadMagic,
    /// PostHAC version differs from the one we support.
    #[error("unsupported PostHAC version {0}")]
    UnsupportedVersion(u32),
    /// Truncated stream — could not read required header field.
    #[error("truncated stream: {0}")]
    Truncated(&'static str),
    /// Range coder failed (constriction-internal error).
    #[error("range coder error: {0}")]
    RangeCoder(String),
    /// Model dimensions disagree with the bitstream.
    #[error("model shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape, as `(N, D)` where N is splats and D attrs.
        expected: (u32, u32),
        /// Shape actually present in the bitstream.
        got: (u32, u32),
    },
}

/// Hyperprior architecture parameters.
///
/// Mirrors the Python reference's `HashGrid` + `HyperpriorMLP` exactly so
/// a Python-trained model can be loaded by the Rust decoder.
#[derive(Debug, Clone)]
pub struct HyperpriorConfig {
    /// Number of hash-grid resolution levels.
    pub grid_levels: u32,
    /// Features per level.
    pub features_per_level: u32,
    /// Log2 of hash-table entries per level.
    pub log2_hashmap_size: u32,
    /// Base grid resolution (level 0).
    pub base_resolution: u32,
    /// MLP hidden size.
    pub mlp_hidden: u32,
    /// Number of attribute columns being modeled.
    pub n_attrs: u32,
}

impl HyperpriorConfig {
    /// Default config matching the Python reference's "small grid" result
    /// (8 levels × 4 features × 2^15 entries, 64-wide MLP).
    pub fn default_small() -> Self {
        Self {
            grid_levels: 8,
            features_per_level: 4,
            log2_hashmap_size: 15,
            base_resolution: 16,
            mlp_hidden: 64,
            n_attrs: 56,
        }
    }
}

/// Serialized hyperprior weights — kept opaque on the wire (host endian
/// f32 arrays).
#[derive(Debug, Clone)]
pub struct HyperpriorWeights {
    /// Per-level hash tables, flat f32, length = `grid_levels * (1 << log2_hashmap_size) * features_per_level`.
    pub grid_tables: Vec<f32>,
    /// MLP first-layer weight matrix in row-major.
    pub fc1_w: Vec<f32>,
    /// MLP first-layer bias.
    pub fc1_b: Vec<f32>,
    /// MLP second-layer weight matrix.
    pub fc2_w: Vec<f32>,
    /// MLP second-layer bias.
    pub fc2_b: Vec<f32>,
}

/// One row of per-splat hyperprior predictions: mean and stddev for every
/// attribute column.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// `n_attrs` means (in code space, [0, 255]).
    pub mean: Vec<f32>,
    /// `n_attrs` stddevs (in code space).
    pub std: Vec<f32>,
}

/// Header describing the PostHAC payload. Wire-formatted at encode time
/// and consumed at decode time.
#[derive(Debug, Clone)]
pub struct PostHacHeader {
    /// Number of splats.
    pub n: u32,
    /// Attribute columns per splat (D).
    pub d: u32,
    /// SH degree (0..3), encoded so the decoder can derive `f_rest_*`
    /// column layout.
    pub sh_degree: u8,
    /// Per-column quantization minima.
    pub attr_mn: Vec<f32>,
    /// Per-column quantization maxima.
    pub attr_mx: Vec<f32>,
    /// Position bbox min (x, y, z).
    pub pos_mn: [f32; 3],
    /// Position bbox max.
    pub pos_mx: [f32; 3],
    /// Hyperprior architecture.
    pub config: HyperpriorConfig,
}

/// Write the PostHAC header to a stream.
pub fn write_header<W: Write>(w: &mut W, h: &PostHacHeader) -> Result<(), PostHacError> {
    w.write_u32::<LittleEndian>(POSTHAC_MAGIC)?;
    w.write_u32::<LittleEndian>(POSTHAC_VERSION)?;
    w.write_u32::<LittleEndian>(h.n)?;
    w.write_u32::<LittleEndian>(h.d)?;
    w.write_u8(h.sh_degree)?;
    // Reserved bytes to keep header aligned.
    w.write_all(&[0u8; 3])?;
    for v in &h.attr_mn {
        w.write_f32::<LittleEndian>(*v)?;
    }
    for v in &h.attr_mx {
        w.write_f32::<LittleEndian>(*v)?;
    }
    for v in h.pos_mn {
        w.write_f32::<LittleEndian>(v)?;
    }
    for v in h.pos_mx {
        w.write_f32::<LittleEndian>(v)?;
    }
    let c = &h.config;
    w.write_u32::<LittleEndian>(c.grid_levels)?;
    w.write_u32::<LittleEndian>(c.features_per_level)?;
    w.write_u32::<LittleEndian>(c.log2_hashmap_size)?;
    w.write_u32::<LittleEndian>(c.base_resolution)?;
    w.write_u32::<LittleEndian>(c.mlp_hidden)?;
    w.write_u32::<LittleEndian>(c.n_attrs)?;
    Ok(())
}

/// Read the PostHAC header from a stream.
pub fn read_header<R: Read>(r: &mut R) -> Result<PostHacHeader, PostHacError> {
    let magic = r
        .read_u32::<LittleEndian>()
        .map_err(|_| PostHacError::Truncated("magic"))?;
    if magic != POSTHAC_MAGIC {
        return Err(PostHacError::BadMagic);
    }
    let version = r
        .read_u32::<LittleEndian>()
        .map_err(|_| PostHacError::Truncated("version"))?;
    if version != POSTHAC_VERSION {
        return Err(PostHacError::UnsupportedVersion(version));
    }
    let n = r.read_u32::<LittleEndian>()?;
    let d = r.read_u32::<LittleEndian>()?;
    let sh_degree = r.read_u8()?;
    let mut reserved = [0u8; 3];
    r.read_exact(&mut reserved)?;
    let mut attr_mn = vec![0f32; d as usize];
    for v in attr_mn.iter_mut() {
        *v = r.read_f32::<LittleEndian>()?;
    }
    let mut attr_mx = vec![0f32; d as usize];
    for v in attr_mx.iter_mut() {
        *v = r.read_f32::<LittleEndian>()?;
    }
    let mut pos_mn = [0f32; 3];
    for v in pos_mn.iter_mut() {
        *v = r.read_f32::<LittleEndian>()?;
    }
    let mut pos_mx = [0f32; 3];
    for v in pos_mx.iter_mut() {
        *v = r.read_f32::<LittleEndian>()?;
    }
    let config = HyperpriorConfig {
        grid_levels: r.read_u32::<LittleEndian>()?,
        features_per_level: r.read_u32::<LittleEndian>()?,
        log2_hashmap_size: r.read_u32::<LittleEndian>()?,
        base_resolution: r.read_u32::<LittleEndian>()?,
        mlp_hidden: r.read_u32::<LittleEndian>()?,
        n_attrs: r.read_u32::<LittleEndian>()?,
    };
    Ok(PostHacHeader {
        n,
        d,
        sh_degree,
        attr_mn,
        attr_mx,
        pos_mn,
        pos_mx,
        config,
    })
}

/// Hash three i64 corner coordinates into a hashmap-sized bucket.
///
/// Magic primes match the Python reference's `HashGrid::_hash`.
#[inline]
pub fn hash3(coords: [i64; 3], hashmap_size: usize) -> usize {
    let primes: [i64; 3] = [1, 2_654_435_761, 805_459_861];
    let h0 = coords[0].wrapping_mul(primes[0]);
    let h1 = coords[1].wrapping_mul(primes[1]);
    let h2 = coords[2].wrapping_mul(primes[2]);
    let h = h0 ^ h1 ^ h2;
    // Use rem_euclid so the result is non-negative even when h is large negative.
    (h.rem_euclid(hashmap_size as i64)) as usize
}

/// Forward-pass the hash grid + MLP head for `pos` (in `[0, 1]^3`).
/// Returns per-attribute `(mean_code, std_code)` in `[0, 255]` space.
///
/// This mirrors the Python `HashGrid::forward` + `HyperpriorMLP::forward`
/// + the inverse scaling done in `encode_residuals`. The training maps
/// codes to `(code - 128)/64`; here we unmap.
pub fn predict(
    pos: [f32; 3],
    cfg: &HyperpriorConfig,
    weights: &HyperpriorWeights,
) -> Prediction {
    let hashmap_size = 1usize << cfg.log2_hashmap_size;
    let f_per_lvl = cfg.features_per_level as usize;
    let n_feats = (cfg.grid_levels as usize) * f_per_lvl;
    let mut feats = vec![0f32; n_feats];

    for lvl in 0..(cfg.grid_levels as usize) {
        let scale = (cfg.base_resolution as f32) * 1.5f32.powi(lvl as i32);
        let mut grid_xyz = [0f32; 3];
        for d in 0..3 {
            grid_xyz[d] = pos[d] * scale;
        }
        let x0 = [
            grid_xyz[0].floor() as i64,
            grid_xyz[1].floor() as i64,
            grid_xyz[2].floor() as i64,
        ];
        let xf = [
            grid_xyz[0] - x0[0] as f32,
            grid_xyz[1] - x0[1] as f32,
            grid_xyz[2] - x0[2] as f32,
        ];
        let level_offset = lvl * hashmap_size * f_per_lvl;
        for dx in 0..2_i64 {
            for dy in 0..2_i64 {
                for dz in 0..2_i64 {
                    let corner = [x0[0] + dx, x0[1] + dy, x0[2] + dz];
                    let wx = if dx == 1 { xf[0] } else { 1.0 - xf[0] };
                    let wy = if dy == 1 { xf[1] } else { 1.0 - xf[1] };
                    let wz = if dz == 1 { xf[2] } else { 1.0 - xf[2] };
                    let w = wx * wy * wz;
                    let bucket = hash3(corner, hashmap_size);
                    for k in 0..f_per_lvl {
                        let off = level_offset + bucket * f_per_lvl + k;
                        feats[lvl * f_per_lvl + k] += w * weights.grid_tables[off];
                    }
                }
            }
        }
    }

    // MLP: h = relu(fc1(feats)); out = fc2(h)
    let h_size = cfg.mlp_hidden as usize;
    let mut h_vec = vec![0f32; h_size];
    for j in 0..h_size {
        let mut acc = weights.fc1_b[j];
        for i in 0..n_feats {
            acc += weights.fc1_w[j * n_feats + i] * feats[i];
        }
        h_vec[j] = acc.max(0.0); // ReLU
    }
    let out_size = 2 * cfg.n_attrs as usize;
    let mut out = vec![0f32; out_size];
    for j in 0..out_size {
        let mut acc = weights.fc2_b[j];
        for i in 0..h_size {
            acc += weights.fc2_w[j * h_size + i] * h_vec[i];
        }
        out[j] = acc;
    }
    let n_attrs = cfg.n_attrs as usize;
    let mut mean = vec![0f32; n_attrs];
    let mut std = vec![0f32; n_attrs];
    for c in 0..n_attrs {
        // Same unscaling as the Python: mu_code = 64*mu + 128, sigma_code = 64*exp(log_sigma).
        let mu_code = (64.0 * out[c] + 128.0).clamp(0.0, 255.0);
        let log_sigma = out[c + n_attrs];
        let sigma_code = (64.0 * log_sigma.exp()).clamp(0.5, 128.0);
        mean[c] = mu_code;
        std[c] = sigma_code;
    }
    Prediction { mean, std }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_roundtrip() {
        let cfg = HyperpriorConfig::default_small();
        let h = PostHacHeader {
            n: 1_157_141,
            d: 56,
            sh_degree: 3,
            attr_mn: vec![0.0; 56],
            attr_mx: vec![1.0; 56],
            pos_mn: [0.0, 0.0, 0.0],
            pos_mx: [10.0, 10.0, 10.0],
            config: cfg,
        };
        let mut buf = Vec::new();
        write_header(&mut buf, &h).unwrap();
        let mut cursor = Cursor::new(&buf);
        let h2 = read_header(&mut cursor).unwrap();
        assert_eq!(h.n, h2.n);
        assert_eq!(h.d, h2.d);
        assert_eq!(h.sh_degree, h2.sh_degree);
        assert_eq!(h.pos_mn, h2.pos_mn);
        assert_eq!(h.pos_mx, h2.pos_mx);
    }

    #[test]
    fn hash3_stable_under_negative_coords() {
        let a = hash3([0, 0, 0], 1 << 15);
        let b = hash3([0, 0, 0], 1 << 15);
        assert_eq!(a, b);
        // Negative coordinates should still produce a valid bucket (no panic, in range).
        let c = hash3([-1, 0, 0], 1 << 15);
        assert!(c < (1 << 15));
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(read_header(&mut cursor), Err(PostHacError::BadMagic)));
    }
}
