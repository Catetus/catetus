//! Stable BLAKE3 hashing of a `SplatScene`.
//!
//! The canonical byte stream concatenates, for every splat in input order:
//!
//! 1. position f32×3 little-endian
//! 2. rotation f32×4 little-endian
//! 3. scale f32×3 little-endian
//! 4. opacity f32 little-endian
//! 5. color DC f32×3 little-endian
//! 6. (if SH) degree u8, then flattened SH coeffs f32×N little-endian
//!
//! This produces a hash that is independent of import format but sensitive to
//! every meaningful change in the IR.

use crate::ir::{Color, SplatScene};

/// Compute the deterministic BLAKE3 hash of a `SplatScene`.
///
/// Returns a string of the form `"blake3:<hex>"`.
pub fn scene_hash(scene: &SplatScene) -> String {
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 4];

    for splat in &scene.splats {
        for v in splat.position {
            buf.copy_from_slice(&v.to_le_bytes());
            hasher.update(&buf);
        }
        for v in splat.rotation {
            buf.copy_from_slice(&v.to_le_bytes());
            hasher.update(&buf);
        }
        for v in splat.scale {
            buf.copy_from_slice(&v.to_le_bytes());
            hasher.update(&buf);
        }
        buf.copy_from_slice(&splat.opacity.to_le_bytes());
        hasher.update(&buf);

        match &splat.color {
            Color::Rgb(rgb) => {
                for v in rgb {
                    buf.copy_from_slice(&v.to_le_bytes());
                    hasher.update(&buf);
                }
            }
            Color::Sh { degree, coeffs } => {
                hasher.update(&[*degree]);
                for v in coeffs {
                    buf.copy_from_slice(&v.to_le_bytes());
                    hasher.update(&buf);
                }
            }
        }
    }

    let hex = hasher.finalize().to_hex().to_string();
    format!("blake3:{hex}")
}
