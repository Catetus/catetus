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
/// attribute column. f64 because the constriction range coder consumes
/// `Gaussian::new(f64, f64)` internally; round-trip determinism requires
/// passing the same f64 bits the Python encoder used.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// `n_attrs` means (in code space, [0, 255]).
    pub mean: Vec<f64>,
    /// `n_attrs` stddevs (in code space).
    pub std: Vec<f64>,
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
pub fn predict(pos: [f32; 3], cfg: &HyperpriorConfig, weights: &HyperpriorWeights) -> Prediction {
    let hashmap_size = 1usize << cfg.log2_hashmap_size;
    let f_per_lvl = cfg.features_per_level as usize;
    let n_feats = (cfg.grid_levels as usize) * f_per_lvl;
    let mut feats = vec![0f32; n_feats];

    for lvl in 0..(cfg.grid_levels as usize) {
        // Match Python `int(base_resolution * per_level_scale ** lvl)` — truncate
        // the per-level resolution to an integer before applying it. Failing to
        // truncate breaks Python↔Rust bitstream interop because the corner-vertex
        // hashes differ at level boundaries.
        let scale_f = (cfg.base_resolution as f32) * 1.5f32.powi(lvl as i32);
        let scale = scale_f.trunc();
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
    let mut mean = vec![0f64; n_attrs];
    let mut std = vec![0f64; n_attrs];
    const MLP_GRID: f32 = 1024.0;
    const GRID: f64 = (1u64 << 20) as f64;
    for c in 0..n_attrs {
        // Round MLP output (f32) to 1/1024 BEFORE casting to f64. This absorbs
        // any 1-ULP drift between PyTorch BLAS and the naive Rust matmul.
        let mu_raw = (out[c] * MLP_GRID).round() / MLP_GRID;
        let log_sigma_raw = (out[c + n_attrs].clamp(-7.0, 7.0) * MLP_GRID).round() / MLP_GRID;
        // Then cast to f64 for the exp() step (IEEE-stable across platforms).
        let mu_code = (64.0 * (mu_raw as f64) + 128.0).clamp(0.0, 255.0);
        let sigma_code = (64.0 * (log_sigma_raw as f64).exp()).clamp(0.5, 128.0);
        // Snap one more time to 1/2^20 f64 grid for safety.
        let mu_q = (mu_code * GRID).round() / GRID;
        let sigma_q = (sigma_code * GRID).round() / GRID;
        mean[c] = mu_q;
        std[c] = sigma_q;
    }
    Prediction { mean, std }
}

/// Write hyperprior weights after the header. Layout (all little-endian
/// f32): grid_tables (sum of `levels * 2^log2 * features` floats),
/// fc1_w, fc1_b, fc2_w, fc2_b. Sizes are derived from the
/// `HyperpriorConfig` in the header so no extra length fields are
/// needed.
pub fn write_weights<W: Write>(
    w: &mut W,
    cfg: &HyperpriorConfig,
    wts: &HyperpriorWeights,
) -> Result<(), PostHacError> {
    let table_len = (cfg.grid_levels as usize)
        * (1usize << cfg.log2_hashmap_size)
        * (cfg.features_per_level as usize);
    debug_assert_eq!(wts.grid_tables.len(), table_len);
    for v in &wts.grid_tables {
        w.write_f32::<LittleEndian>(*v)?;
    }
    let n_feats = (cfg.grid_levels as usize) * (cfg.features_per_level as usize);
    let h_size = cfg.mlp_hidden as usize;
    let out_size = 2 * cfg.n_attrs as usize;
    debug_assert_eq!(wts.fc1_w.len(), h_size * n_feats);
    debug_assert_eq!(wts.fc1_b.len(), h_size);
    debug_assert_eq!(wts.fc2_w.len(), out_size * h_size);
    debug_assert_eq!(wts.fc2_b.len(), out_size);
    for arr in [&wts.fc1_w, &wts.fc1_b, &wts.fc2_w, &wts.fc2_b] {
        for v in arr {
            w.write_f32::<LittleEndian>(*v)?;
        }
    }
    Ok(())
}

/// Read hyperprior weights from a stream (the inverse of `write_weights`).
pub fn read_weights<R: Read>(
    r: &mut R,
    cfg: &HyperpriorConfig,
) -> Result<HyperpriorWeights, PostHacError> {
    let table_len = (cfg.grid_levels as usize)
        * (1usize << cfg.log2_hashmap_size)
        * (cfg.features_per_level as usize);
    let mut grid_tables = vec![0f32; table_len];
    for v in grid_tables.iter_mut() {
        *v = r.read_f32::<LittleEndian>()?;
    }
    let n_feats = (cfg.grid_levels as usize) * (cfg.features_per_level as usize);
    let h_size = cfg.mlp_hidden as usize;
    let out_size = 2 * cfg.n_attrs as usize;
    let mut fc1_w = vec![0f32; h_size * n_feats];
    let mut fc1_b = vec![0f32; h_size];
    let mut fc2_w = vec![0f32; out_size * h_size];
    let mut fc2_b = vec![0f32; out_size];
    for arr in [&mut fc1_w, &mut fc1_b, &mut fc2_w, &mut fc2_b] {
        for v in arr.iter_mut() {
            *v = r.read_f32::<LittleEndian>()?;
        }
    }
    Ok(HyperpriorWeights {
        grid_tables,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
    })
}

/// Encode a stream of 8-bit attribute codes against per-splat predicted
/// Gaussian distributions. `codes` is laid out as `[n, d]` row-major
/// (one splat at a time). `predictions[i]` corresponds to splat `i`.
///
/// Returns the compressed bitstream as a `Vec<u32>` (constriction's
/// native output format).
pub fn encode_codes(
    codes: &[u8],
    n: usize,
    d: usize,
    predictions: &[Prediction],
) -> Result<Vec<u32>, PostHacError> {
    use constriction::stream::model::{DefaultLeakyQuantizer, LeakyQuantizer};
    use constriction::stream::queue::DefaultRangeEncoder;
    use constriction::stream::Encode;
    use probability::distribution::Gaussian;

    if codes.len() != n * d {
        return Err(PostHacError::ShapeMismatch {
            expected: (n as u32, d as u32),
            got: ((codes.len() / d) as u32, d as u32),
        });
    }
    if predictions.len() != n {
        return Err(PostHacError::ShapeMismatch {
            expected: (n as u32, d as u32),
            got: (predictions.len() as u32, d as u32),
        });
    }

    let quantizer: DefaultLeakyQuantizer<f64, i32> = LeakyQuantizer::new(-1..=256);
    let mut encoder = DefaultRangeEncoder::new();

    // Symbol order matches Python: attr-major (loop attr, then splat), so
    // when the decoder reverses it gets the same sequence.
    for c in 0..d {
        for s in 0..n {
            let symbol = codes[s * d + c] as i32;
            let mean = predictions[s].mean[c];
            let std = predictions[s].std[c].max(0.5);
            let model = quantizer.quantize(Gaussian::new(mean, std));
            encoder
                .encode_symbol(symbol, model)
                .map_err(|e| PostHacError::RangeCoder(format!("encode {s},{c}: {e:?}")))?;
        }
    }
    Ok(encoder.into_compressed().unwrap())
}

/// Inverse of `encode_codes`. Returns `[n, d]` codes in row-major order.
pub fn decode_codes(
    compressed: &[u32],
    n: usize,
    d: usize,
    predictions: &[Prediction],
) -> Result<Vec<u8>, PostHacError> {
    use constriction::stream::model::{DefaultLeakyQuantizer, LeakyQuantizer};
    use constriction::stream::queue::DefaultRangeDecoder;
    use constriction::stream::Decode;
    use probability::distribution::Gaussian;

    let quantizer: DefaultLeakyQuantizer<f64, i32> = LeakyQuantizer::new(-1..=256);
    let mut decoder = DefaultRangeDecoder::from_compressed(compressed.to_vec())
        .map_err(|e| PostHacError::RangeCoder(format!("decoder init: {e:?}")))?;

    let mut out = vec![0u8; n * d];
    for c in 0..d {
        for s in 0..n {
            let mean = predictions[s].mean[c];
            let std = predictions[s].std[c].max(0.5);
            let model = quantizer.quantize(Gaussian::new(mean, std));
            let symbol = decoder
                .decode_symbol(model)
                .map_err(|e| PostHacError::RangeCoder(format!("decode {s},{c}: {e:?}")))?;
            out[s * d + c] = symbol.clamp(0, 255) as u8;
        }
    }
    Ok(out)
}

/// High-level container reader: parse a `.pthc` file produced by the
/// Python encoder (or the Rust `write_container` below) into all the
/// pieces a decoder needs.
pub struct Container {
    /// Bitstream header (scene metadata + hyperprior architecture).
    pub header: PostHacHeader,
    /// Hyperprior weights (hash grid tables + MLP).
    pub weights: HyperpriorWeights,
    /// Per-splat positions as flat `[x0, y0, z0, x1, y1, z1, ...]`.
    pub positions: Vec<f32>,
    /// Range-coded attribute codes.
    pub compressed: Vec<u32>,
}

/// Parse the entire `.pthc` container from a byte slice.
pub fn read_container(bytes: &[u8]) -> Result<Container, PostHacError> {
    let mut cur = std::io::Cursor::new(bytes);
    let header = read_header(&mut cur)?;
    let weights = read_weights(&mut cur, &header.config)?;
    let mut positions = vec![0f32; 3 * header.n as usize];
    for v in positions.iter_mut() {
        *v = cur.read_f32::<LittleEndian>()?;
    }
    let comp_len = cur.read_u32::<LittleEndian>()? as usize;
    let mut compressed = vec![0u32; comp_len];
    for v in compressed.iter_mut() {
        *v = cur.read_u32::<LittleEndian>()?;
    }
    Ok(Container {
        header,
        weights,
        positions,
        compressed,
    })
}

/// Write a `.pthc` container to a byte vector.
pub fn write_container<W: Write>(
    w: &mut W,
    header: &PostHacHeader,
    weights: &HyperpriorWeights,
    positions: &[f32],
    compressed: &[u32],
) -> Result<(), PostHacError> {
    write_header(w, header)?;
    write_weights(w, &header.config, weights)?;
    for v in positions {
        w.write_f32::<LittleEndian>(*v)?;
    }
    w.write_u32::<LittleEndian>(compressed.len() as u32)?;
    for v in compressed {
        w.write_u32::<LittleEndian>(*v)?;
    }
    Ok(())
}

/// Build a `Vec<Prediction>` for every splat by running `predict()` over
/// the normalized positions.
pub fn predict_all(
    positions: &[f32],
    pos_mn: [f32; 3],
    pos_mx: [f32; 3],
    cfg: &HyperpriorConfig,
    weights: &HyperpriorWeights,
) -> Vec<Prediction> {
    let n = positions.len() / 3;
    let mut out = Vec::with_capacity(n);
    let rng = [
        (pos_mx[0] - pos_mn[0]).max(1e-9),
        (pos_mx[1] - pos_mn[1]).max(1e-9),
        (pos_mx[2] - pos_mn[2]).max(1e-9),
    ];
    for i in 0..n {
        let p = [
            (positions[3 * i] - pos_mn[0]) / rng[0],
            (positions[3 * i + 1] - pos_mn[1]) / rng[1],
            (positions[3 * i + 2] - pos_mn[2]) / rng[2],
        ];
        out.push(predict(p, cfg, weights));
    }
    out
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

    fn small_cfg() -> HyperpriorConfig {
        HyperpriorConfig {
            grid_levels: 2,
            features_per_level: 2,
            log2_hashmap_size: 4,
            base_resolution: 4,
            mlp_hidden: 4,
            n_attrs: 3,
        }
    }

    fn weights_for(cfg: &HyperpriorConfig) -> HyperpriorWeights {
        let table_len = (cfg.grid_levels as usize)
            * (1usize << cfg.log2_hashmap_size)
            * (cfg.features_per_level as usize);
        let n_feats = (cfg.grid_levels as usize) * (cfg.features_per_level as usize);
        let h = cfg.mlp_hidden as usize;
        let out = 2 * cfg.n_attrs as usize;
        HyperpriorWeights {
            grid_tables: (0..table_len).map(|i| i as f32 * 1e-3).collect(),
            fc1_w: (0..h * n_feats).map(|i| (i as f32) * 1e-2).collect(),
            fc1_b: vec![0.1f32; h],
            fc2_w: (0..out * h).map(|i| (i as f32) * 1e-2).collect(),
            fc2_b: vec![0.0f32; out],
        }
    }

    #[test]
    fn weights_roundtrip() {
        let cfg = small_cfg();
        let wts = weights_for(&cfg);
        let mut buf = Vec::new();
        write_weights(&mut buf, &cfg, &wts).unwrap();
        let mut cursor = Cursor::new(&buf);
        let wts2 = read_weights(&mut cursor, &cfg).unwrap();
        assert_eq!(wts.grid_tables, wts2.grid_tables);
        assert_eq!(wts.fc1_w, wts2.fc1_w);
        assert_eq!(wts.fc1_b, wts2.fc1_b);
        assert_eq!(wts.fc2_w, wts2.fc2_w);
        assert_eq!(wts.fc2_b, wts2.fc2_b);
    }

    #[test]
    fn predict_is_deterministic() {
        let cfg = small_cfg();
        let wts = weights_for(&cfg);
        let p1 = predict([0.5, 0.5, 0.5], &cfg, &wts);
        let p2 = predict([0.5, 0.5, 0.5], &cfg, &wts);
        assert_eq!(p1.mean, p2.mean);
        assert_eq!(p1.std, p2.std);
        // All means clamped to [0, 255] and stds to [0.5, 128].
        for &m in &p1.mean {
            assert!((0.0..=255.0).contains(&m));
        }
        for &s in &p1.std {
            assert!((0.5..=128.0).contains(&s));
        }
    }

    #[test]
    fn roundtrip_small_scene() {
        // Encode 100 splats × 4 attrs with random codes + predictions,
        // then decode and check bit-exact equality.
        let n = 100;
        let d = 4;
        let mut codes = vec![0u8; n * d];
        for i in 0..codes.len() {
            codes[i] = ((i * 7 + 13) % 256) as u8;
        }
        let predictions: Vec<Prediction> = (0..n)
            .map(|i| Prediction {
                mean: (0..d).map(|c| ((i * 17 + c * 23) % 256) as f64).collect(),
                std: vec![32.0; d],
            })
            .collect();
        let compressed = encode_codes(&codes, n, d, &predictions).unwrap();
        let decoded = decode_codes(&compressed, n, d, &predictions).unwrap();
        assert_eq!(codes, decoded);
        // Compression ratio: 400 bytes raw → ~? compressed
        let raw_bytes = (n * d) as f64;
        let comp_bytes = (compressed.len() * 4) as f64;
        let ratio = raw_bytes / comp_bytes;
        // With well-predicted means + std=32, we should get some compression.
        // Don't assert specific ratio (it's small data); just sanity-check
        // that we didn't *expand* the stream by more than 2×.
        assert!(
            comp_bytes < raw_bytes * 2.0,
            "compression should not balloon: raw={raw_bytes}, comp={comp_bytes}, ratio={ratio}"
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            read_header(&mut cursor),
            Err(PostHacError::BadMagic)
        ));
    }
}
