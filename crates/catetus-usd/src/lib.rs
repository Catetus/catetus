#![deny(clippy::all)]
//! OpenUSD I/O for Catetus.
//!
//! This crate implements the SPEC-0011 IR ↔ `ParticleField3DGaussianSplat`
//! mapping with three concrete writers / readers:
//!
//! * **USDA** (text USD): [`write_usda`] / [`read_usda`].
//! * **USDC** (Pixar Crate binary, version 0.0.1): [`write_usdc`] /
//!   [`read_usdc`]. Round-trips bit-exact-as-USDA against `usdcat`
//!   (Apple USD Tools 0.25.2 verified). See `docs/openusd-conformance.md`.
//! * **Streaming** hooks (payload-arcs + `lod` variant set) per SPEC-0012 —
//!   feature-gated under `streaming`.
//!
//! Surface mirrors `catetus-gltf` deliberately so call-sites swap
//! between the two with minimal churn:
//!
//! ```text
//! catetus_gltf::write_gltf(scene, path, &opts)?;
//! catetus_usd::write_usda(scene, path, &opts)?;
//! catetus_usd::write_usdc(scene, path, &opts)?;
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod usda;
pub use usda::{parse_usda, read_usda, render_usda, write_usda};

mod usdc;
pub use usdc::{read_usdc, write_usdc};

/// USD I/O errors.
#[derive(Debug, Error)]
pub enum UsdError {
    /// Underlying IO failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed USDA input (unexpected token / encoding).
    #[error("malformed USDA: {0}")]
    Malformed(String),
    /// Malformed USDC binary (corrupt header, bad section, etc.).
    #[error("malformed USDC: {0}")]
    MalformedUsdc(String),
    /// Unsupported USDC version (we read 0.x with minor <= our writer; we
    /// always write 0.0.1 for maximum compatibility).
    #[error("unsupported USDC version: {0}.{1}.{2}")]
    UnsupportedUsdcVersion(u8, u8, u8),
    /// Unsupported USDC feature — e.g. a Value type the reader hasn't
    /// implemented yet. Distinct from `MalformedUsdc` so callers can route to
    /// usdcat fallback.
    #[error("unsupported USDC feature: {0}")]
    UnsupportedUsdcFeature(String),
    /// LZ4 decompression failed.
    #[error("LZ4 decompression failed: {0}")]
    Lz4(String),
    /// SH coefficient packing parity is not yet validated against a real
    /// USD toolchain. Tracked under SPEC-0011 §"Open questions".
    #[error("SH packing parity not yet validated; consult SPEC-0011 §Open questions")]
    UnverifiedShPacking,
}

/// Options controlling USD export. Shared between USDA and USDC writers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsdWriteOpts {
    /// Emit per-chunk prims via SPEC-0012 payloads (vs a single in-line prim).
    pub chunked: bool,
    /// Target splat count per chunk when `chunked` is true.
    pub chunk_target_splats: usize,
    /// LOD variant fractions — mirrors `catetus_gltf::WriteOpts.lod_fractions`.
    pub lod_fractions: Vec<f32>,
}
