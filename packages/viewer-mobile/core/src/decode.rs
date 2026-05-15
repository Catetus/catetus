//! `.glb` (KHR_gaussian_splatting) -> [`SplatVertex`] decoder.
//!
//! Thin shim over `splatforge_gltf::read_glb_bytes`: that workspace crate
//! parses the glTF JSON + binary chunk; we flatten the resulting
//! [`splatforge_core::SplatScene`] into the packed render layout. Doing the
//! flatten here (instead of in the consumer) keeps the C ABI dead simple — the
//! caller hands us bytes, we hand back a contiguous `Vec<SplatVertex>`.

use splatforge_core::Color;
use splatforge_gltf::{read_glb_bytes, GltfError};
use thiserror::Error;

use crate::vertex::SplatVertex;

/// Errors that can be reported back across the C ABI.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Underlying glTF parser rejected the file.
    #[error("glb decode failed: {0}")]
    Gltf(#[from] GltfError),
    /// Asset parsed but had zero splats — refuse to upload an empty buffer.
    #[error("scene had no splats")]
    Empty,
}

/// Decode a `.glb` blob to a packed vertex buffer ready for upload.
pub fn decode_glb_bytes(bytes: &[u8]) -> Result<Vec<SplatVertex>, DecodeError> {
    let scene = read_glb_bytes(bytes)?;
    if scene.splats.is_empty() {
        return Err(DecodeError::Empty);
    }
    let mut out = Vec::with_capacity(scene.splats.len());
    for s in &scene.splats {
        let rgb = match &s.color {
            Color::Rgb(c) => *c,
            // For SH inputs we take the DC term (L=0 band, basis 1/(2*sqrt(pi))).
            // Full SH evaluation is on the follow-up shader port.
            Color::Sh { coeffs, .. } => {
                if coeffs.len() >= 3 {
                    let dc = 0.282_094_8_f32; // 1 / (2 * sqrt(pi))
                    [
                        coeffs[0] * dc + 0.5,
                        coeffs[1] * dc + 0.5,
                        coeffs[2] * dc + 0.5,
                    ]
                } else {
                    [1.0, 1.0, 1.0]
                }
            }
        };
        out.push(SplatVertex {
            position: s.position,
            rotation: s.rotation,
            scale: s.scale,
            opacity: s.opacity,
            color: rgb,
        });
    }
    Ok(out)
}
