#![deny(clippy::all)]
//! Minimal SPZ v2 writer/reader.
//!
//! See `specs/0003-spz-io.md` for the wire format. Everything after the fixed
//! header is zlib-compressed; integers are little-endian.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use splatforge_core::{Color, CoordinateSystem, Splat, SplatScene, TemporalMode};
use thiserror::Error;

/// "SNPS" little-endian magic.
pub const SPZ_MAGIC: u32 = 0x5053_4e47;
/// SPZ wire version.
pub const SPZ_VERSION: u32 = 2;

/// SPZ I/O errors.
#[derive(Debug, Error)]
pub enum SpzError {
    /// Underlying IO failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// File does not begin with the SPZ magic.
    #[error("not an SPZ file (bad magic)")]
    BadMagic,
    /// SPZ version differs from the one we support.
    #[error("unsupported SPZ version {0}")]
    UnsupportedVersion(u32),
    /// zlib decompression failed.
    #[error("decompression failed: {0}")]
    Decompress(String),
    /// Payload was shorter than the header declared.
    #[error("truncated payload")]
    Truncated,
}

/// Read an SPZ file from disk.
pub fn read_spz(path: &Path) -> Result<SplatScene, SpzError> {
    let bytes = fs::read(path)?;
    read_spz_bytes(&bytes)
}

/// Read an SPZ scene from an in-memory buffer.
pub fn read_spz_bytes(bytes: &[u8]) -> Result<SplatScene, SpzError> {
    if bytes.len() < 16 {
        return Err(SpzError::Truncated);
    }
    let mut cur = std::io::Cursor::new(bytes);
    let magic = cur.read_u32::<LittleEndian>()?;
    if magic != SPZ_MAGIC {
        return Err(SpzError::BadMagic);
    }
    let version = cur.read_u32::<LittleEndian>()?;
    if version != SPZ_VERSION {
        return Err(SpzError::UnsupportedVersion(version));
    }
    let splat_count = cur.read_u32::<LittleEndian>()? as usize;
    let sh_degree = cur.read_u8()?;
    let fractional_bits = cur.read_u8()?;
    let _flags = cur.read_u8()?;
    let _reserved = cur.read_u8()?;

    let pos = cur.position() as usize;
    let compressed = &bytes[pos..];
    let mut decoder = ZlibDecoder::new(compressed);
    let mut payload = Vec::new();
    decoder
        .read_to_end(&mut payload)
        .map_err(|e| SpzError::Decompress(e.to_string()))?;

    decode_payload(&payload, splat_count, sh_degree, fractional_bits)
}

/// Write a scene to an SPZ file on disk.
pub fn write_spz(path: &Path, scene: &SplatScene) -> Result<(), SpzError> {
    let bytes = encode_spz(scene)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Encode a scene to an in-memory SPZ byte buffer.
pub fn encode_spz(scene: &SplatScene) -> Result<Vec<u8>, SpzError> {
    let fractional_bits: u8 = 12;
    let sh_degree: u8 = scene
        .splats
        .iter()
        .map(|s| s.color.degree())
        .max()
        .unwrap_or(0)
        .min(1); // payload only carries up to SH degree 1 (15*3 per splat) per spec
    let mut header = Vec::with_capacity(16);
    header.write_u32::<LittleEndian>(SPZ_MAGIC)?;
    header.write_u32::<LittleEndian>(SPZ_VERSION)?;
    header.write_u32::<LittleEndian>(scene.splats.len() as u32)?;
    header.write_u8(sh_degree)?;
    header.write_u8(fractional_bits)?;
    header.write_u8(0)?;
    header.write_u8(0)?;

    let payload = encode_payload(scene, sh_degree, fractional_bits)?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&payload)?;
    let compressed = encoder.finish()?;

    let mut out = header;
    out.extend_from_slice(&compressed);
    Ok(out)
}

// ---------- payload ----------

fn pack_pos(x: f32, fractional_bits: u8) -> [u8; 3] {
    // 24-bit signed fixed-point.
    let scale = (1u32 << fractional_bits) as f32;
    let v = (x * scale).round() as i32;
    let clamped = v.clamp(-(1 << 23), (1 << 23) - 1);
    let u = (clamped as u32) & 0x00FF_FFFF;
    [
        (u & 0xFF) as u8,
        ((u >> 8) & 0xFF) as u8,
        ((u >> 16) & 0xFF) as u8,
    ]
}

fn unpack_pos(bytes: [u8; 3], fractional_bits: u8) -> f32 {
    let mut u = (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);
    // sign extend from 24 bits
    if u & 0x0080_0000 != 0 {
        u |= 0xFF00_0000;
    }
    let v = u as i32;
    let scale = (1u32 << fractional_bits) as f32;
    v as f32 / scale
}

fn pack_log(x: f32) -> u8 {
    // Log-quantize positive scale into 0..=255 mapping log(x) in [-10, 10].
    let l = x.max(1e-9).ln();
    let n = ((l + 10.0) * (255.0 / 20.0)).round();
    n.clamp(0.0, 255.0) as u8
}
fn unpack_log(b: u8) -> f32 {
    let l = (b as f32) * (20.0 / 255.0) - 10.0;
    l.exp()
}

fn pack_u8_unit(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn unpack_u8_unit(b: u8) -> f32 {
    (b as f32) / 255.0
}

/// Smallest-three quaternion encoding: drop the largest component, store the
/// sign separately, encode the remaining three as 8-bit ints in [-1/√2, 1/√2].
fn pack_quat(q: [f32; 4]) -> [u8; 3] {
    // Find largest magnitude component index.
    let mut idx = 0usize;
    let mut max = q[0].abs();
    for (i, v) in q.iter().enumerate().skip(1) {
        if v.abs() > max {
            max = v.abs();
            idx = i;
        }
    }
    let sign = if q[idx] < 0.0 { -1.0 } else { 1.0 };
    let mut other = [0.0f32; 3];
    let mut j = 0;
    for (i, v) in q.iter().enumerate() {
        if i != idx {
            other[j] = sign * *v;
            j += 1;
        }
    }
    // Each "other" lies in [-1/√2, 1/√2]; we encode the first into 6 bits + 2-bit idx; rest in 8 bits.
    // For simplicity: pack (idx in top 2 bits of byte0) + 6 bits of other[0], byte1 = other[1], byte2 = other[2].
    let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
    let q0 = ((other[0] / sqrt2_inv + 1.0) * 0.5 * 63.0)
        .round()
        .clamp(0.0, 63.0) as u8;
    let q1 = ((other[1] / sqrt2_inv + 1.0) * 0.5 * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let q2 = ((other[2] / sqrt2_inv + 1.0) * 0.5 * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    [((idx as u8) << 6) | (q0 & 0x3F), q1, q2]
}

fn unpack_quat(b: [u8; 3]) -> [f32; 4] {
    let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
    let idx = (b[0] >> 6) as usize;
    let q0_q = b[0] & 0x3F;
    let other0 = ((q0_q as f32 / 63.0) * 2.0 - 1.0) * sqrt2_inv;
    let other1 = ((b[1] as f32 / 255.0) * 2.0 - 1.0) * sqrt2_inv;
    let other2 = ((b[2] as f32 / 255.0) * 2.0 - 1.0) * sqrt2_inv;
    let mut q = [0.0f32; 4];
    let mut j = 0;
    let others = [other0, other1, other2];
    for (i, q_i) in q.iter_mut().enumerate() {
        if i == idx {
            continue;
        }
        *q_i = others[j];
        j += 1;
    }
    let sum_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let largest_sq = (1.0 - sum_sq).max(0.0);
    q[idx] = largest_sq.sqrt();
    q
}

fn pack_color(rgb: [f32; 3]) -> [u8; 3] {
    [
        pack_u8_unit(rgb[0]),
        pack_u8_unit(rgb[1]),
        pack_u8_unit(rgb[2]),
    ]
}
fn unpack_color(b: [u8; 3]) -> [f32; 3] {
    [
        unpack_u8_unit(b[0]),
        unpack_u8_unit(b[1]),
        unpack_u8_unit(b[2]),
    ]
}

fn encode_payload(
    scene: &SplatScene,
    sh_degree: u8,
    fractional_bits: u8,
) -> Result<Vec<u8>, SpzError> {
    let n = scene.splats.len();
    let mut out: Vec<u8> = Vec::with_capacity(n * (9 + 3 + 3 + 1 + 3 + sh_degree as usize * 45));
    // positions
    for s in &scene.splats {
        for &p in &s.position {
            out.extend_from_slice(&pack_pos(p, fractional_bits));
        }
    }
    // scales
    for s in &scene.splats {
        for &v in &s.scale {
            out.push(pack_log(v));
        }
    }
    // rotations
    for s in &scene.splats {
        out.extend_from_slice(&pack_quat(s.rotation));
    }
    // alpha
    for s in &scene.splats {
        out.push(pack_u8_unit(s.opacity));
    }
    // colors (DC)
    for s in &scene.splats {
        let dc = match &s.color {
            Color::Rgb(c) => *c,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        };
        out.extend_from_slice(&pack_color(dc));
    }
    // SH (if any) — emit 15 coeffs * 3 channels per splat for degree>=1.
    if sh_degree >= 1 {
        for s in &scene.splats {
            let coeffs: Vec<f32> = match &s.color {
                Color::Sh { coeffs, .. } => coeffs.clone(),
                Color::Rgb(_) => vec![0.0; 48],
            };
            // first 3 are DC; the remaining up to 45 (=15 per channel) we pack 8-bit.
            // Map [-1, 1] -> [0, 255]
            let rest = if coeffs.len() > 3 {
                &coeffs[3..]
            } else {
                &[][..]
            };
            for i in 0..45 {
                let v = if i < rest.len() { rest[i] } else { 0.0 };
                let b = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
                out.push(b);
            }
        }
    }
    Ok(out)
}

fn decode_payload(
    payload: &[u8],
    n: usize,
    sh_degree: u8,
    fractional_bits: u8,
) -> Result<SplatScene, SpzError> {
    let mut splats = Vec::with_capacity(n);
    // sections
    let pos_len = n * 9;
    let scale_len = n * 3;
    let rot_len = n * 3;
    let alpha_len = n;
    let color_len = n * 3;
    let sh_len = if sh_degree >= 1 { n * 45 } else { 0 };

    let total = pos_len + scale_len + rot_len + alpha_len + color_len + sh_len;
    if payload.len() < total {
        return Err(SpzError::Truncated);
    }
    let mut off = 0;
    let positions = &payload[off..off + pos_len];
    off += pos_len;
    let scales = &payload[off..off + scale_len];
    off += scale_len;
    let rotations = &payload[off..off + rot_len];
    off += rot_len;
    let alphas = &payload[off..off + alpha_len];
    off += alpha_len;
    let colors = &payload[off..off + color_len];
    off += color_len;
    let sh = if sh_len > 0 {
        Some(&payload[off..off + sh_len])
    } else {
        None
    };

    for i in 0..n {
        let p_off = i * 9;
        let p = [
            unpack_pos(
                [positions[p_off], positions[p_off + 1], positions[p_off + 2]],
                fractional_bits,
            ),
            unpack_pos(
                [
                    positions[p_off + 3],
                    positions[p_off + 4],
                    positions[p_off + 5],
                ],
                fractional_bits,
            ),
            unpack_pos(
                [
                    positions[p_off + 6],
                    positions[p_off + 7],
                    positions[p_off + 8],
                ],
                fractional_bits,
            ),
        ];
        let s = [
            unpack_log(scales[i * 3]),
            unpack_log(scales[i * 3 + 1]),
            unpack_log(scales[i * 3 + 2]),
        ];
        let r = unpack_quat([rotations[i * 3], rotations[i * 3 + 1], rotations[i * 3 + 2]]);
        let a = unpack_u8_unit(alphas[i]);
        let c = unpack_color([colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]]);

        let color = if let Some(sh_buf) = sh {
            let mut coeffs = Vec::with_capacity(48);
            coeffs.extend_from_slice(&c);
            for j in 0..45 {
                let b = sh_buf[i * 45 + j];
                let v = ((b as f32) / 255.0) * 2.0 - 1.0;
                coeffs.push(v);
            }
            Color::Sh { degree: 1, coeffs }
        } else {
            Color::Rgb(c)
        };
        splats.push(Splat {
            position: p,
            rotation: r,
            scale: s,
            opacity: a,
            color,
        });
    }

    Ok(SplatScene {
        splats,
        coordinate_system: CoordinateSystem::default(),
        semantic_labels: None,
        temporal_mode: TemporalMode::Static,
        lods: None,
    })
}
