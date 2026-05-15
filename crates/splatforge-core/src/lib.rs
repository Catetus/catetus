#![deny(clippy::all)]
//! `splatforge-core` — the canonical splat intermediate representation, hashing,
//! coordinate-system helpers, and the deterministic analyze-report types used by
//! every other SplatForge crate.

pub mod coords;
pub mod hash;
pub mod ir;
pub mod report;

pub use coords::{CoordinateSystem, Handedness, UpAxis};
pub use hash::scene_hash;
pub use ir::{Color, LodLevel, SemanticLabel, Splat, SplatScene, TemporalMode};
pub use report::{
    AnalyzeReport, Attributes, BoundingBox, CoordSystemReport, EstimatedMemory,
    OpacityDistribution, Recommendation, ScaleDistribution, Warning,
};

/// Detect a supported splat file format from its filename extension.
///
/// Returns one of `"ply"`, `"spz"`, `"gltf"`, `"glb"`, `"usda"`, `"usdc"`,
/// or `None` if the extension is unrecognized.
pub fn format_from_extension(path: &std::path::Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "ply" => Some("ply"),
        "spz" => Some("spz"),
        "gltf" => Some("gltf"),
        "glb" => Some("glb"),
        "usda" => Some("usda"),
        "usdc" => Some("usdc"),
        _ => None,
    }
}

/// Detect a supported splat file format by sniffing the first bytes of a file.
///
/// PLY files begin with the ASCII tag `ply\n`. SPZ files begin with our 32-bit
/// magic `0x5053_4e47`. glTF JSON files begin with `{`. GLB binary files begin
/// with the ASCII magic `glTF`.
pub fn format_from_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 4 {
        if &bytes[..4] == b"glTF" {
            return Some("glb");
        }
        if bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r") {
            return Some("ply");
        }
        // SPZ magic 0x5053_4e47 ('S','N','P','S') little-endian → bytes 47 4E 53 50
        if bytes[0] == 0x47 && bytes[1] == 0x4E && bytes[2] == 0x53 && bytes[3] == 0x50 {
            return Some("spz");
        }
        // USDC binary magic: "PXR-USDC" at offset 0.
        if bytes.len() >= 8 && &bytes[..8] == b"PXR-USDC" {
            return Some("usdc");
        }
        // USDA text starts with `#usda 1.0` (whitespace tolerant).
        let head = std::str::from_utf8(&bytes[..bytes.len().min(64)]).unwrap_or("");
        if head.trim_start().starts_with("#usda") {
            return Some("usda");
        }
        // glTF JSON heuristic
        if head.trim_start().starts_with('{') && head.contains("\"asset\"") {
            return Some("gltf");
        }
    }
    None
}
