//! Cesium 3D Tiles 1.1 tileset emitter for SplatForge.
//!
//! Consumes a `SplatScene` whose `lods` field has been populated by the
//! `BuildLOD` pass (e.g. via the `geospatial` preset) and emits a directory
//! containing:
//!
//! * `tileset.json` — a standards-compliant 3D Tiles 1.1 hierarchy.
//! * One `lod{i}.glb` per LOD level. Each GLB is a self-contained binary glTF
//!   with the `KHR_gaussian_splatting` extension carrying that LOD's splats.
//!
//! ### Hierarchy
//!
//! 3D Tiles renders coarser tiles first and refines into children as the
//! screen-space error budget allows. We therefore put the COARSEST LOD (fewest
//! splats) at the root and chain finer LODs as children using
//! `refine: "REPLACE"`. A single-tile-per-level chain matches the Cesium April
//! 2026 "3D Gaussian Splats with LOD" reference layout for a single-region
//! asset.
//!
//! ### `geometricError`
//!
//! `geometricError` is the world-space error introduced by **not refining**
//! into a tile's children. Cesium decides whether to refine by comparing
//! `geometricError / distance` to the viewer's pixel-error budget.
//!
//! Our heuristic: the root's error is half the bounding-box diagonal (the
//! "skip all refinement" error is roughly the size of the asset). Each child
//! halves the error of its parent, matching the splat-count halving in
//! `BuildLOD`. Leaves (the full-resolution LOD0) get `geometricError = 0`,
//! signaling that no further refinement is possible.
//!
//! ### Bounding volume
//!
//! We emit a `box` bounding volume (the simpler of the two allowed forms;
//! `region` would require a geodetic CRS that SplatForge's IR does not carry).
//! The box is axis-aligned in the asset's local frame; Cesium ion picks up the
//! ECEF transform from upload-time geolocation, so the local frame is correct.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use splatforge_core::{LodLevel, SplatScene};
use splatforge_gltf::{write_glb, WriteOpts};

/// Caller-supplied knobs for `write_tileset`. The defaults match the
/// `geospatial` preset.
#[derive(Debug, Clone)]
pub struct TilesetOpts {
    /// File-name prefix for per-LOD GLBs. Final names are
    /// `<prefix>{i}.glb` (e.g. `lod0.glb`).
    pub glb_prefix: String,
    /// Whether to apply `KHR_mesh_quantization` to the per-tile GLBs. The
    /// `geospatial` preset always sets this to `true` to keep tile bytes
    /// close to the SPZ payload size.
    pub quantize_glb: bool,
    /// Optional override for the root `geometricError`. When `None` we derive
    /// it from the scene bounding box (see module docs).
    pub root_geometric_error: Option<f64>,
}

impl Default for TilesetOpts {
    fn default() -> Self {
        Self {
            glb_prefix: "lod".to_string(),
            quantize_glb: true,
            root_geometric_error: None,
        }
    }
}

// ---------- tileset.json wire format (3D Tiles 1.1 subset) ----------

#[derive(Serialize, Debug, Clone)]
struct TilesetAsset {
    /// 3D Tiles spec version — must be `"1.1"` for KHR_gaussian_splatting tiles.
    version: String,
    /// Generator string. Stable so the JSON is byte-deterministic.
    #[serde(rename = "tilesetVersion", skip_serializing_if = "Option::is_none")]
    tileset_version: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
struct BoundingVolume {
    /// 12-float OBB representation: `[cx, cy, cz, hx, 0, 0, 0, hy, 0, 0, 0, hz]`
    /// per the 3D Tiles 1.1 spec § Bounding Volume.
    #[serde(rename = "box")]
    box_: [f64; 12],
}

#[derive(Serialize, Debug, Clone)]
struct Content {
    /// Relative path to a glTF/GLB resource — Cesium resolves it against the
    /// tileset.json URL.
    uri: String,
}

#[derive(Serialize, Debug, Clone)]
struct Tile {
    /// Axis-aligned box (12 floats).
    #[serde(rename = "boundingVolume")]
    bounding_volume: BoundingVolume,
    /// Error introduced by NOT refining into this tile's children.
    #[serde(rename = "geometricError")]
    geometric_error: f64,
    /// `"REPLACE"` — children fully replace the parent's content.
    #[serde(skip_serializing_if = "Option::is_none")]
    refine: Option<String>,
    /// Content URI (the per-LOD GLB).
    content: Content,
    /// Child tiles in the LOD chain.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    children: Vec<Tile>,
}

#[derive(Serialize, Debug, Clone)]
struct Tileset {
    asset: TilesetAsset,
    /// Root `geometricError` — the error when nothing is loaded.
    #[serde(rename = "geometricError")]
    geometric_error: f64,
    /// Required extensions a runtime must understand to render this tileset.
    #[serde(rename = "extensionsRequired", skip_serializing_if = "Vec::is_empty")]
    extensions_required: Vec<String>,
    /// Extensions the tileset uses (superset of `extensionsRequired`).
    #[serde(rename = "extensionsUsed", skip_serializing_if = "Vec::is_empty")]
    extensions_used: Vec<String>,
    root: Tile,
}

// ---------- helpers ----------

fn bbox_for(scene: &SplatScene, indices: &[u32]) -> ([f64; 3], [f64; 3]) {
    if indices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for &i in indices {
        let p = &scene.splats[i as usize].position;
        for axis in 0..3 {
            let v = p[axis] as f64;
            if v < mn[axis] {
                mn[axis] = v;
            }
            if v > mx[axis] {
                mx[axis] = v;
            }
        }
    }
    (mn, mx)
}

fn bounding_volume_from(mn: &[f64; 3], mx: &[f64; 3]) -> BoundingVolume {
    let cx = 0.5 * (mn[0] + mx[0]);
    let cy = 0.5 * (mn[1] + mx[1]);
    let cz = 0.5 * (mn[2] + mx[2]);
    // Half-extents. The 12-float OBB is (center, x-half, y-half, z-half).
    // Empty box: keep half-extents at zero — Cesium tolerates degenerate volumes
    // for empty tiles, and our writer never emits an empty tileset in practice
    // because BuildLOD always seeds LOD0 with the full scene.
    let hx = 0.5 * (mx[0] - mn[0]).max(0.0);
    let hy = 0.5 * (mx[1] - mn[1]).max(0.0);
    let hz = 0.5 * (mx[2] - mn[2]).max(0.0);
    BoundingVolume {
        box_: [cx, cy, cz, hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz],
    }
}

fn diag(mn: &[f64; 3], mx: &[f64; 3]) -> f64 {
    let dx = mx[0] - mn[0];
    let dy = mx[1] - mn[1];
    let dz = mx[2] - mn[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Materialize a subscene that contains only the splats referenced by `level`.
/// We strip `lods` and `semantic_labels` from the subscene; downstream GLBs are
/// flat single-LOD assets, and a parent tileset.json already encodes the
/// hierarchy.
fn subscene_for(scene: &SplatScene, level: &LodLevel) -> SplatScene {
    let mut out = SplatScene::new();
    out.coordinate_system = scene.coordinate_system;
    out.temporal_mode = scene.temporal_mode;
    out.splats = level
        .indices
        .iter()
        .map(|&i| scene.splats[i as usize].clone())
        .collect();
    out
}

/// Per-tile metadata returned for diagnostics.
#[derive(Debug, Clone)]
pub struct TileReport {
    /// LOD index (0 = finest / leaf).
    pub lod_index: usize,
    /// Splat fraction encoded by `BuildLOD`.
    pub fraction: f32,
    /// Splat count actually written into the GLB.
    pub splat_count: usize,
    /// `geometricError` assigned to the tile.
    pub geometric_error: f64,
    /// GLB filename (relative to the tileset directory).
    pub glb: String,
}

/// Outcome of `write_tileset`.
#[derive(Debug, Clone)]
pub struct TilesetReport {
    /// Absolute path to the written `tileset.json`.
    pub tileset_json: PathBuf,
    /// Per-tile metadata in coarse-to-fine order (root first).
    pub tiles: Vec<TileReport>,
}

/// Write a Cesium 3D Tiles 1.1 tileset rooted at `output_dir`.
///
/// Preconditions:
///
/// * `output_dir` will be created (recursively) if it does not exist.
/// * `scene.lods` must be `Some` with at least one level. Use the
///   `geospatial` preset (or any preset that ends with `BuildLOD`) to populate
///   it.
///
/// The function is deterministic: identical inputs produce byte-identical
/// `tileset.json` and per-LOD GLBs across runs and machines.
pub fn write_tileset(
    scene: &SplatScene,
    output_dir: &Path,
    opts: &TilesetOpts,
) -> Result<TilesetReport> {
    let lods = scene
        .lods
        .as_ref()
        .context("scene has no LODs — run the BuildLOD pass first (preset=geospatial)")?;
    if lods.is_empty() {
        anyhow::bail!("scene.lods is empty — at least LOD0 is required");
    }

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating tileset directory {}", output_dir.display()))?;

    // Scene-wide bbox = bbox of LOD0 (the full scene).
    let (root_mn, root_mx) = bbox_for(scene, &lods[0].indices);
    let scene_diag = diag(&root_mn, &root_mx);
    // Root geometric error heuristic. Half the diagonal puts the "no LODs
    // loaded at all" error at roughly the radius of the asset — the right
    // ballpark for Cesium's SSE model where a fully-unrefined tile is
    // expected to occupy on the order of the asset's bounding sphere.
    let root_error = opts
        .root_geometric_error
        .unwrap_or_else(|| (scene_diag * 0.5).max(1e-6));

    // Build the LOD chain. `lods[0]` is the finest (full count); the COARSEST
    // becomes our root, with finer levels chained as children. Concretely we
    // walk `lods` in reverse to produce the coarse-to-fine list and then build
    // the tile tree bottom-up.
    let glb_opts = WriteOpts {
        chunked: false,
        chunk_target_splats: 100_000,
        lod_fractions: vec![1.0],
        quantize: opts.quantize_glb,
    };

    // Emit one GLB per LOD level. We name them by their original BuildLOD
    // index so `lod0.glb` is always the finest, matching the rest of the
    // SplatForge tooling.
    let mut tile_reports_fine_to_coarse: Vec<TileReport> = Vec::with_capacity(lods.len());
    for (i, level) in lods.iter().enumerate() {
        let sub = subscene_for(scene, level);
        let glb_name = format!("{}{}.glb", opts.glb_prefix, i);
        let glb_path = output_dir.join(&glb_name);
        write_glb(&sub, &glb_path, &glb_opts)
            .with_context(|| format!("writing tile GLB {}", glb_path.display()))?;
        tile_reports_fine_to_coarse.push(TileReport {
            lod_index: i,
            fraction: level.fraction,
            splat_count: sub.splats.len(),
            // `geometric_error` filled in below once we know the chain depth.
            geometric_error: 0.0,
            glb: glb_name,
        });
    }

    // Assign `geometricError` per tile. Cesium evaluates the parent's error to
    // decide whether to refine — so the coarsest tile needs the largest error
    // and the leaf (LOD0) gets 0 (no further refinement possible).
    //
    // Walk coarse → fine. Coarsest = `root_error`, then halve at each step.
    // The leaf overrides to 0.0.
    let n = tile_reports_fine_to_coarse.len();
    // tile_reports_fine_to_coarse[0] = finest (LOD0), [n-1] = coarsest.
    // Coarse-to-fine index `j` ∈ [0, n-1] maps to fine-to-coarse `n-1-j`.
    for j in 0..n {
        let fine_idx = n - 1 - j;
        let err = if fine_idx == 0 {
            0.0
        } else {
            // j == 0 (coarsest) → root_error.
            // Each step toward finer halves the error.
            root_error * (0.5f64).powi(j as i32)
        };
        tile_reports_fine_to_coarse[fine_idx].geometric_error = err;
    }

    // Build tile tree. Start from the finest (leaf, no children) and wrap each
    // coarser level around its predecessor.
    let mut current: Option<Tile> = None;
    // Walk fine → coarse so the leaf is constructed first.
    for (i, rep) in tile_reports_fine_to_coarse.iter().enumerate() {
        let (mn, mx) = bbox_for(scene, &lods[i].indices);
        let bv = bounding_volume_from(&mn, &mx);
        let tile = Tile {
            bounding_volume: bv,
            geometric_error: rep.geometric_error,
            // Only non-leaf tiles need an explicit `refine`; leaves omit it.
            refine: if current.is_some() {
                Some("REPLACE".to_string())
            } else {
                None
            },
            content: Content {
                uri: rep.glb.clone(),
            },
            children: current.take().into_iter().collect(),
        };
        current = Some(tile);
    }
    let root_tile = current.expect("at least one LOD; checked above");

    let tileset = Tileset {
        asset: TilesetAsset {
            version: "1.1".to_string(),
            tileset_version: Some("splatforge-geospatial-1".to_string()),
        },
        geometric_error: root_error,
        extensions_required: vec!["KHR_gaussian_splatting".to_string()],
        extensions_used: vec!["KHR_gaussian_splatting".to_string()],
        root: root_tile,
    };

    // Pretty-print with stable key order. `serde_json::to_string_pretty`
    // emits fields in struct-declaration order, which we control above —
    // the result is byte-deterministic.
    let json = serde_json::to_string_pretty(&tileset)
        .context("serializing tileset.json")?;
    let tileset_json_path = output_dir.join("tileset.json");
    std::fs::write(&tileset_json_path, json)
        .with_context(|| format!("writing {}", tileset_json_path.display()))?;

    // Reorder report to coarse-to-fine (root first) for the caller.
    let mut tiles_coarse_to_fine: Vec<TileReport> = tile_reports_fine_to_coarse
        .iter()
        .rev()
        .cloned()
        .collect();
    // The clone above lost the geometric_error we wrote into the original;
    // re-copy from the source list.
    for (j, t) in tiles_coarse_to_fine.iter_mut().enumerate() {
        let fine_idx = n - 1 - j;
        t.geometric_error = tile_reports_fine_to_coarse[fine_idx].geometric_error;
    }

    Ok(TilesetReport {
        tileset_json: tileset_json_path,
        tiles: tiles_coarse_to_fine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splatforge_core::{Color, LodLevel, Splat};

    fn tiny_scene(n: usize) -> SplatScene {
        let mut scene = SplatScene::new();
        for i in 0..n {
            let f = i as f32;
            scene.splats.push(Splat {
                position: [f * 0.1, -f * 0.05, f * 0.02],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.01, 0.01, 0.01],
                opacity: 0.9,
                color: Color::Rgb([0.5, 0.5, 0.5]),
            });
        }
        scene.lods = Some(vec![
            LodLevel {
                fraction: 1.0,
                indices: (0..n as u32).collect(),
            },
            LodLevel {
                fraction: 0.5,
                indices: (0..n as u32).step_by(2).collect(),
            },
        ]);
        scene
    }

    #[test]
    fn write_tileset_produces_expected_files() {
        let tmp = tempdir();
        let scene = tiny_scene(20);
        let report = write_tileset(&scene, &tmp, &TilesetOpts::default()).unwrap();
        assert!(report.tileset_json.exists());
        assert_eq!(report.tiles.len(), 2);
        // Root (coarsest) should be the larger geometric error.
        assert!(report.tiles[0].geometric_error > report.tiles[1].geometric_error);
        // Leaf is the finest LOD, error = 0.
        assert_eq!(report.tiles[1].geometric_error, 0.0);
        // GLBs exist.
        assert!(tmp.join("lod0.glb").exists());
        assert!(tmp.join("lod1.glb").exists());
    }

    #[test]
    fn write_tileset_is_deterministic() {
        let a = tempdir();
        let b = tempdir();
        let scene = tiny_scene(20);
        write_tileset(&scene, &a, &TilesetOpts::default()).unwrap();
        write_tileset(&scene, &b, &TilesetOpts::default()).unwrap();
        let json_a = std::fs::read(a.join("tileset.json")).unwrap();
        let json_b = std::fs::read(b.join("tileset.json")).unwrap();
        assert_eq!(json_a, json_b);
        let glb_a = std::fs::read(a.join("lod0.glb")).unwrap();
        let glb_b = std::fs::read(b.join("lod0.glb")).unwrap();
        assert_eq!(glb_a, glb_b);
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "splatforge-tileset-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        // Mix in a thread-local counter so two adjacent calls in the same
        // nanosecond still differ.
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        t.wrapping_add(N.fetch_add(1, Ordering::Relaxed))
    }
}
