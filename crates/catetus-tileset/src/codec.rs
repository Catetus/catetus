//! Per-tile payload encoding.
//!
//! The octree planner is agnostic to *how* a tile's splats are serialized: it
//! just needs something that turns a [`SplatScene`] into bytes + a file
//! extension. That is the [`TilePayloadCodec`] trait. In production the tile
//! payload will be an SF GLB or a SuperSplat-compatible SOG; for the MVP we
//! ship [`SfTileCodec`], a minimal self-describing binary (`.sftile`) that
//! round-trips losslessly and has **zero** external dependencies, so the whole
//! crate compiles and is testable without the GLB/SOG writers.
//!
//! ## `.sftile` v1 layout (little-endian)
//!
//! ```text
//!   offset  size  field
//!   0       4     magic "SFT1"
//!   4       4     u32  splat_count (N)
//!   8       1     u8   sh_degree
//!   9       3     pad (zero)
//!   12      N*?   splats, each:
//!                   3*f32 position
//!                   4*f32 rotation (x,y,z,w)   [IR order]
//!                   3*f32 scale (linear)
//!                   1*f32 opacity
//!                   C*f32 color coeffs (C = 3*(sh_degree+1)^2, constant per tile)
//! ```
//! For a DC-only tile (`sh_degree == 0`), `C == 3` (the RGB DC term). All
//! splats in a tile share `C`, so the per-splat stride is constant.

use catetus_core::ir::{Color, Splat, SplatScene};

/// Number of color coefficients (across RGB) for a given SH degree.
fn coeffs_for_degree(degree: u8) -> usize {
    let bands = (degree as usize + 1) * (degree as usize + 1);
    3 * bands
}

/// Flatten a splat's color into exactly `coeff_len` floats (DC first).
fn color_to_coeffs(color: &Color, coeff_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; coeff_len];
    match color {
        Color::Rgb(c) => {
            for k in 0..3.min(coeff_len) {
                out[k] = c[k];
            }
        }
        Color::Sh { coeffs, .. } => {
            for (k, v) in coeffs.iter().take(coeff_len).enumerate() {
                out[k] = *v;
            }
        }
    }
    out
}

/// Reconstruct a [`Color`] from `coeffs` at the given SH degree.
fn coeffs_to_color(coeffs: Vec<f32>, degree: u8) -> Color {
    if degree == 0 {
        Color::Rgb([
            coeffs.first().copied().unwrap_or(0.0),
            coeffs.get(1).copied().unwrap_or(0.0),
            coeffs.get(2).copied().unwrap_or(0.0),
        ])
    } else {
        Color::Sh { degree, coeffs }
    }
}

/// The SH degree shared by a scene's splats (0 if empty / DC-only).
fn scene_degree(scene: &SplatScene) -> u8 {
    scene.splats.first().map(|s| s.color.degree()).unwrap_or(0)
}

/// Encoded tile bytes plus the file extension to use (no leading dot).
///
/// Some codecs (notably the SF GLB codec when SH-rest is palette-coded) emit a
/// companion sidecar that must be written next to the tile, e.g. the `.shpal`
/// SH-palette sidecar for `VQPaletteShRest`. `sidecar` carries those bytes and
/// `sidecar_ext` the extension to append to the tile file name (so a tile
/// `tiles/7.glb` gets `tiles/7.glb.shpal`). DC-only / non-palette tiles leave
/// both `None`.
pub struct TileBytes {
    pub bytes: Vec<u8>,
    pub ext: &'static str,
    /// Optional companion bytes (e.g. the `.shpal` SH-palette sidecar).
    pub sidecar: Option<Vec<u8>>,
    /// Extension appended to the tile file name for the sidecar (no leading
    /// dot), e.g. `"shpal"` → `tiles/<i>.<ext>.shpal`. Required iff `sidecar`
    /// is `Some`.
    pub sidecar_ext: Option<&'static str>,
}

impl TileBytes {
    /// A tile with no sidecar (the common case).
    pub fn simple(bytes: Vec<u8>, ext: &'static str) -> Self {
        Self { bytes, ext, sidecar: None, sidecar_ext: None }
    }
}

/// Encodes a tile's splats to bytes. Implement this to plug in GLB or SOG.
pub trait TilePayloadCodec {
    /// Encode one tile.
    fn encode(&self, scene: &SplatScene) -> TileBytes;
    /// File extension (no dot) — used to name `tiles/<i>.<ext>`.
    fn extension(&self) -> &'static str;
}

/// Minimal dependency-free tile codec (`.sftile` v1). Lossless round-trip.
pub struct SfTileCodec;

const MAGIC: &[u8; 4] = b"SFT1";

impl SfTileCodec {
    /// Decode an `.sftile` v1 blob back into a [`SplatScene`] (used by tests
    /// and any consumer that wants to verify a tile).
    pub fn decode(bytes: &[u8]) -> Result<SplatScene, &'static str> {
        if bytes.len() < 12 || &bytes[0..4] != MAGIC {
            return Err("bad magic / too short");
        }
        let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let sh_degree = bytes[8];
        let mut off = 12usize;
        let mut scene = SplatScene::new();
        if n == 0 {
            return Ok(scene);
        }
        let coeff_len = coeffs_for_degree(sh_degree);
        let per_fixed = (3 + 4 + 3 + 1) * 4; // pos + rot + scale + opacity
        let per_splat = per_fixed + coeff_len * 4;
        let need = 12 + n * per_splat;
        if bytes.len() < need {
            return Err("payload shorter than declared splat count");
        }
        let rdf = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        scene.splats.reserve(n);
        for _ in 0..n {
            let position = [rdf(bytes, off), rdf(bytes, off + 4), rdf(bytes, off + 8)];
            off += 12;
            let rotation =
                [rdf(bytes, off), rdf(bytes, off + 4), rdf(bytes, off + 8), rdf(bytes, off + 12)];
            off += 16;
            let scale = [rdf(bytes, off), rdf(bytes, off + 4), rdf(bytes, off + 8)];
            off += 12;
            let opacity = rdf(bytes, off);
            off += 4;
            let mut coeffs = Vec::with_capacity(coeff_len);
            for _ in 0..coeff_len {
                coeffs.push(rdf(bytes, off));
                off += 4;
            }
            scene.splats.push(Splat {
                position,
                rotation,
                scale,
                opacity,
                color: coeffs_to_color(coeffs, sh_degree),
            });
        }
        Ok(scene)
    }
}

impl TilePayloadCodec for SfTileCodec {
    fn encode(&self, scene: &SplatScene) -> TileBytes {
        let n = scene.splats.len();
        let degree = scene_degree(scene);
        let coeff_len = coeffs_for_degree(degree);
        let per_splat = (3 + 4 + 3 + 1) * 4 + coeff_len * 4;
        let mut bytes = Vec::with_capacity(12 + n * per_splat);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(n as u32).to_le_bytes());
        bytes.push(degree);
        bytes.extend_from_slice(&[0u8, 0, 0]); // pad
        let put = |f: f32, b: &mut Vec<u8>| b.extend_from_slice(&f.to_le_bytes());
        for s in &scene.splats {
            for v in s.position {
                put(v, &mut bytes);
            }
            for v in s.rotation {
                put(v, &mut bytes);
            }
            for v in s.scale {
                put(v, &mut bytes);
            }
            put(s.opacity, &mut bytes);
            for v in color_to_coeffs(&s.color, coeff_len) {
                put(v, &mut bytes);
            }
        }
        TileBytes::simple(bytes, "sftile")
    }

    fn extension(&self) -> &'static str {
        "sftile"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use catetus_core::ir::{Color, Splat, SplatScene};

    fn sample_scene(n: usize, degree: u8) -> SplatScene {
        let coeff_len = coeffs_for_degree(degree);
        let mut s = SplatScene::new();
        for i in 0..n {
            let f = i as f32;
            let color = if degree == 0 {
                Color::Rgb([0.5, 0.4, 0.3])
            } else {
                Color::Sh { degree, coeffs: (0..coeff_len).map(|k| k as f32 + f).collect() }
            };
            s.splats.push(Splat {
                position: [f, f + 1.0, f + 2.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.1 * f + 0.01, 0.2, 0.3],
                opacity: (f % 10.0) / 10.0,
                color,
            });
        }
        s
    }

    #[test]
    fn sftile_roundtrip_dc_only() {
        let s = sample_scene(37, 0);
        let enc = SfTileCodec.encode(&s);
        assert_eq!(enc.ext, "sftile");
        let back = SfTileCodec::decode(&enc.bytes).unwrap();
        assert_eq!(back.len(), s.len());
        for i in 0..s.len() {
            assert_eq!(back.splats[i], s.splats[i]);
        }
    }

    #[test]
    fn sftile_roundtrip_with_sh() {
        let s = sample_scene(11, 1); // degree 1 -> 12 coeffs
        let enc = SfTileCodec.encode(&s);
        let back = SfTileCodec::decode(&enc.bytes).unwrap();
        assert_eq!(back.len(), s.len());
        for i in 0..s.len() {
            assert_eq!(back.splats[i].color, s.splats[i].color);
        }
    }

    #[test]
    fn empty_tile_roundtrips() {
        let s = SplatScene::new();
        let enc = SfTileCodec.encode(&s);
        let back = SfTileCodec::decode(&enc.bytes).unwrap();
        assert_eq!(back.len(), 0);
    }
}
