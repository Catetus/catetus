//! Canonical splat intermediate representation (`SplatIR`).
//!
//! See `specs/0001-ir.md`. The IR is the single in-memory shape every
//! importer/exporter and optimization pass operates on.

use serde::{Deserialize, Serialize};

use crate::coords::CoordinateSystem;

/// A single Gaussian splat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Splat {
    /// World-space position `[x, y, z]`.
    pub position: [f32; 3],
    /// Unit quaternion stored as `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Per-axis linear scale (importers must convert from log-space if needed).
    pub scale: [f32; 3],
    /// Linear opacity in `[0, 1]` (importers must apply sigmoid as needed).
    pub opacity: f32,
    /// Either an RGB DC term or full SH coefficients.
    pub color: Color,
}

/// Color representation: either a flat RGB DC term or SH coefficients.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Color {
    /// Plain RGB DC color, linear sRGB.
    Rgb([f32; 3]),
    /// Spherical-harmonic coefficients. `coeffs.len() == 3 * (degree+1)^2`.
    Sh {
        /// SH band order (0..=3).
        degree: u8,
        /// Flattened SH coefficients (RGB interleaved per band).
        coeffs: Vec<f32>,
    },
}

impl Color {
    /// Return the SH degree (0 for a flat RGB color).
    pub fn degree(&self) -> u8 {
        match self {
            Color::Rgb(_) => 0,
            Color::Sh { degree, .. } => *degree,
        }
    }
}

/// Optional per-splat semantic tag (e.g. "product", "background").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticLabel(pub String);

/// A single LOD level: a subsampled view into the main `splats` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LodLevel {
    /// Fraction of the full splat count this level represents (e.g. `0.5`).
    pub fraction: f32,
    /// Indices into `SplatScene.splats` defining the level's contents.
    pub indices: Vec<u32>,
}

/// Temporal/4D mode. v1 only supports static scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemporalMode {
    /// A single non-animated scene.
    Static,
    /// Reserved for future 4D content; not yet implemented.
    Dynamic,
}

/// A whole scene: an ordered list of splats plus metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplatScene {
    /// Splats in canonical order. Importers must preserve input ordering.
    pub splats: Vec<Splat>,
    /// Coordinate-system convention for the scene.
    pub coordinate_system: CoordinateSystem,
    /// Optional per-splat semantic labels (parallel to `splats`).
    pub semantic_labels: Option<Vec<SemanticLabel>>,
    /// Temporal mode; always `Static` in v1.
    pub temporal_mode: TemporalMode,
    /// Optional precomputed LOD levels referencing `splats` by index.
    #[serde(default)]
    pub lods: Option<Vec<LodLevel>>,
}

impl SplatScene {
    /// Create an empty scene with default Y-up right-handed coordinates.
    pub fn new() -> Self {
        Self {
            splats: Vec::new(),
            coordinate_system: CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: TemporalMode::Static,
            lods: None,
        }
    }

    /// Iterate over splats by reference.
    pub fn iter(&self) -> std::slice::Iter<'_, Splat> {
        self.splats.iter()
    }

    /// Number of splats in the scene.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    /// Whether the scene contains zero splats.
    pub fn is_empty(&self) -> bool {
        self.splats.is_empty()
    }
}

impl Default for SplatScene {
    fn default() -> Self {
        Self::new()
    }
}
