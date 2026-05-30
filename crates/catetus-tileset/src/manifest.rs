//! Manifest types for both output formats.
//!
//! - [`LodMeta`] mirrors SuperSplat's `lod-meta.json` exactly (verified against
//!   `d28zzqy0iyovbz.cloudfront.net/b11e45d1/v1/lod-meta.json`, the Koriyama
//!   Castle scene): a top-level `{lodLevels, environment, filenames, tree}`
//!   where `tree` is a node of `{bound:{min,max}, lods:[{file,count}], children}`.
//! - [`TilesetManifest`] is a 3D-Tiles-1.1-shaped manifest for Cesium/generic
//!   tooling and the Catetus WebGPU viewer.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SuperSplat lod-meta.json shape (interop target)
// ---------------------------------------------------------------------------

/// Axis-aligned bound as serialized in SuperSplat's `lod-meta.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// One LOD reference inside a node: an index into the top-level `filenames`
/// array plus the splat count that tile contains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LodRef {
    pub file: usize,
    pub count: usize,
}

/// A node in the SuperSplat LOD tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LodMetaNode {
    pub bound: Aabb,
    /// Coarse → fine LOD tiles for this node (SuperSplat stores exactly 3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lods: Vec<LodRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<LodMetaNode>,
}

/// Top-level `lod-meta.json`, byte-shape-compatible with SuperSplat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LodMeta {
    #[serde(rename = "lodLevels")]
    pub lod_levels: usize,
    /// Path to an environment SOG tile, if any (SuperSplat ships `env/meta.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Tileset-root shared SH-rest codebook sidecar, relative to this manifest
    /// (e.g. `"palette.shpal"`), written ONCE for the whole tileset. When set,
    /// every tile's SH-rest is index-coded against this one codebook: a loader
    /// fetches it once, then for each tile reads the tile GLB (DC + geometry)
    /// plus a `<tile>.shpalx` index sidecar and reconstructs SH-rest =
    /// `codebook[index]`. Absent for FP32 / per-tile-palette tilesets — those
    /// keep working unchanged, so this is purely additive (D-WIRE).
    #[serde(rename = "sharedPalette", skip_serializing_if = "Option::is_none")]
    pub shared_palette: Option<String>,
    /// Flat list of tile payload paths; `LodRef::file` indexes into this.
    pub filenames: Vec<String>,
    pub tree: LodMetaNode,
}

// ---------------------------------------------------------------------------
// 3D Tiles 1.1 shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilesetAsset {
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
}

/// 3D-Tiles bounding volume. We emit the `box` form (12-float OBB: center +
/// 3 half-axis vectors), which is what Cesium prefers for axis-aligned boxes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingVolume {
    #[serde(rename = "box")]
    pub box_: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileContent {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileNode {
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<TileContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TileNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilesetManifest {
    pub asset: TilesetAsset,
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,
    pub root: TileNode,
}

/// Build a 3D-Tiles `box` bounding volume (center + 3 half-axes) from an AABB.
pub fn aabb_to_obb_box(min: [f32; 3], max: [f32; 3]) -> Vec<f64> {
    let cx = 0.5 * (min[0] + max[0]) as f64;
    let cy = 0.5 * (min[1] + max[1]) as f64;
    let cz = 0.5 * (min[2] + max[2]) as f64;
    let hx = 0.5 * (max[0] - min[0]) as f64;
    let hy = 0.5 * (max[1] - min[1]) as f64;
    let hz = 0.5 * (max[2] - min[2]) as f64;
    vec![
        cx, cy, cz, // center
        hx, 0.0, 0.0, // x half-axis
        0.0, hy, 0.0, // y half-axis
        0.0, 0.0, hz, // z half-axis
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_meta_roundtrips_supersplat_shape() {
        // A minimal tree matching the SuperSplat schema exactly.
        let meta = LodMeta {
            lod_levels: 7,
            environment: Some("env/meta.json".into()),
            shared_palette: None,
            filenames: vec!["0_0/meta.json".into(), "1_0/meta.json".into()],
            tree: LodMetaNode {
                bound: Aabb { min: [-1.0, -1.0, -1.0], max: [1.0, 1.0, 1.0] },
                lods: vec![],
                children: vec![LodMetaNode {
                    bound: Aabb { min: [-1.0, -1.0, -1.0], max: [0.0, 0.0, 0.0] },
                    lods: vec![
                        LodRef { file: 0, count: 7456 },
                        LodRef { file: 1, count: 31960 },
                    ],
                    children: vec![],
                }],
            },
        };
        let json = serde_json::to_string(&meta).unwrap();
        // Field names must match SuperSplat's exactly.
        assert!(json.contains("\"lodLevels\":7"));
        assert!(json.contains("\"environment\":\"env/meta.json\""));
        assert!(json.contains("\"bound\""));
        assert!(json.contains("\"lods\""));
        assert!(json.contains("\"file\":0"));
        assert!(json.contains("\"count\":7456"));
        let back: LodMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn parses_real_supersplat_node_fragment() {
        // Verbatim fragment from the Koriyama lod-meta.json.
        let frag = r#"{
            "bound": { "min": [-254.9435, -26.79508, -405.4218],
                       "max": [-15.44089, 42.23647, -111.9123] },
            "lods": [ { "file": 6, "count": 7456 },
                      { "file": 33, "count": 31960 },
                      { "file": 62, "count": 124648 } ],
            "children": []
        }"#;
        let node: LodMetaNode = serde_json::from_str(frag).unwrap();
        assert_eq!(node.lods.len(), 3);
        assert_eq!(node.lods[2].file, 62);
        assert_eq!(node.lods[2].count, 124648);
    }

    #[test]
    fn obb_box_is_12_floats() {
        let b = aabb_to_obb_box([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
        assert_eq!(b.len(), 12);
        assert_eq!(b[0], 0.0); // center x
        assert_eq!(b[3], 1.0); // x half extent
        assert_eq!(b[7], 2.0); // y half extent
        assert_eq!(b[11], 3.0); // z half extent
    }
}
