//! Coordinate-system metadata helpers.

use serde::{Deserialize, Serialize};

/// The "up" axis a scene was authored against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpAxis {
    /// Y-axis points up (glTF, three.js convention).
    Y,
    /// Z-axis points up (Blender, OpenUSD convention).
    Z,
}

/// Handedness of the coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Handedness {
    /// Right-handed (glTF, standard math).
    Right,
    /// Left-handed (DirectX, Unity).
    Left,
}

/// Per-scene coordinate-system convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinateSystem {
    /// Which axis points "up".
    pub up: UpAxis,
    /// Handedness of the basis.
    pub handedness: Handedness,
}

impl Default for CoordinateSystem {
    fn default() -> Self {
        Self {
            up: UpAxis::Y,
            handedness: Handedness::Right,
        }
    }
}

impl CoordinateSystem {
    /// Return the canonical short-form label for the up axis (`"Y"` or `"Z"`).
    pub fn up_label(&self) -> &'static str {
        match self.up {
            UpAxis::Y => "Y",
            UpAxis::Z => "Z",
        }
    }

    /// Return the canonical short-form label for handedness.
    pub fn handedness_label(&self) -> &'static str {
        match self.handedness {
            Handedness::Right => "right",
            Handedness::Left => "left",
        }
    }
}
