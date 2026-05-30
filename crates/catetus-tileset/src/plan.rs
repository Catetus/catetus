//! Tileset planning + writing: octree → per-node LODs → manifests → disk.

use std::path::Path;

use catetus_core::ir::{Splat, SplatScene};

use crate::codec::TilePayloadCodec;
use crate::manifest::{
    aabb_to_obb_box, Aabb, BoundingVolume, LodMeta, LodMetaNode, LodRef, TileContent, TileNode,
    TilesetAsset, TilesetManifest,
};
use crate::octree::{Octree, OctreeConfig, OctreeNode};
use crate::shared_palette::{SharedCodebook, SharedPaletteTileCodec, SHARED_PALETTE_FILENAME};
use crate::LODS_PER_NODE;

/// Knobs for the scene-global shared SH-rest codebook used by
/// [`write_tileset_shared`]. Mirror [`SharedCodebook::build`]'s args.
#[derive(Debug, Clone, Copy)]
pub struct SharedPaletteConfig {
    /// Number of codebook centroids (K). Production single-file path uses 4096.
    pub palette_size: usize,
    /// k-means iterations over the scene SH-rest.
    pub iterations: usize,
    /// Deterministic seed for k-means init.
    pub seed: u64,
}

impl Default for SharedPaletteConfig {
    fn default() -> Self {
        // Matches the bench (`tile_codec_bench.rs`): K=4096, 5 iters, seed 0xC0DE,
        // which produced the 95,439,411 B reference total on bonsai.
        Self { palette_size: 4096, iterations: 5, seed: 0xC0DE }
    }
}

/// Configuration for a full tileset build.
#[derive(Debug, Clone, Copy)]
pub struct TilesetConfig {
    pub octree: OctreeConfig,
    /// LOD tiles carried per node (coarse→fine). Default [`LODS_PER_NODE`].
    pub lods_per_node: usize,
    /// Target splat count for the *coarsest* LOD of a node. Finer LODs scale
    /// up geometrically toward the node's proxy pool (capped at `proxy_cap`).
    pub coarsest_target: usize,
    /// Hard upper bound on the splat count of ANY single tile, including the
    /// root's coarsest *and* finest proxy. This is the STREAM-5 fixed-size
    /// invariant: a node's proxy pool is built bottom-up by importance-merging
    /// its children's proxies and capping at `proxy_cap`, so the root proxy is
    /// ~`proxy_cap` regardless of total scene size (the old builder grew the
    /// root proxy as `coarsest_target·(N/base)^(1/3)`, ~17k for 1.24M splats
    /// but ~58k for 48M — unbounded). SuperSplat's coarse proxy is a fixed
    /// ~5.8 MB at any scale; this is the splat-count analog. Default 20_000.
    pub proxy_cap: usize,
}

impl Default for TilesetConfig {
    fn default() -> Self {
        Self {
            octree: OctreeConfig::default(),
            lods_per_node: LODS_PER_NODE,
            coarsest_target: 2_000,
            proxy_cap: 20_000,
        }
    }
}

/// One tile payload: a node's splats at one LOD, ready to encode.
pub struct TilePayload {
    /// Index into the manifest `filenames` array (and tile file name).
    pub file_index: usize,
    /// LOD level within the owning node (0 = coarsest).
    pub lod: usize,
    pub scene: SplatScene,
    /// Per-tile-splat ORIGIN index: the row each tile splat had in the
    /// **original input scene** passed to [`plan_tileset`]. `origins[j]` is the
    /// global scene index of `scene.splats[j]`. Length equals `scene.len()`.
    ///
    /// This is what lets a scene-global shared codebook (the VQ45 SH-rest
    /// palette built once over the whole scene) be threaded into the tiles:
    /// the tile codec looks up `shared_indices[origins[j]]` to find each tile
    /// splat's palette index without re-running k-means per tile. See
    /// [`crate::SharedPaletteTileCodec`].
    pub origins: Vec<u32>,
}

/// The result of planning: both manifests + the per-tile payloads.
pub struct TilesetPlan {
    pub lod_meta: LodMeta,
    pub tileset: TilesetManifest,
    pub payloads: Vec<TilePayload>,
}

/// Errors from planning/writing a tileset.
#[derive(Debug, thiserror::Error)]
pub enum TilesetError {
    #[error("scene has no splats to tile")]
    EmptyScene,
    #[error("lods_per_node must be >= 1")]
    InvalidLods,
    #[error("serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shared-palette codebook build failed: {0}")]
    SharedPalette(String),
}

/// Importance of a splat = opacity × geometric footprint (cube-root scale
/// volume). Used to keep the most visually-significant splats in coarse LODs.
/// Scale is linear in the IR, so no `exp` is needed.
fn importance(s: &Splat) -> f32 {
    let vol = (s.scale[0].abs() * s.scale[1].abs() * s.scale[2].abs()).cbrt();
    s.opacity.max(0.0) * vol.max(1e-6)
}

/// Importance-decimate `indices` down to `target` entries, biased toward
/// high-importance splats while preserving spatial coverage. Deterministic.
///
/// Strategy: sort indices by importance (desc), keep the top half, then stride
/// the remainder — coarse proxies keep the big/bright splats without collapsing
/// to only the top-K. Returns the kept index list (a *subset* of the input, so
/// each returned index is still a valid index into `scene.splats`). If
/// `indices.len() <= target`, returns a copy unchanged.
fn decimate_indices(scene: &SplatScene, indices: &[u32], target: usize) -> Vec<u32> {
    if indices.len() <= target {
        return indices.to_vec();
    }
    let mut sorted: Vec<u32> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        let ia = importance(&scene.splats[a as usize]);
        let ib = importance(&scene.splats[b as usize]);
        ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let keep_top = target / 2;
    let mut chosen: Vec<u32> = sorted[..keep_top].to_vec();
    let rest = &sorted[keep_top..];
    let need = target - keep_top;
    if need > 0 && !rest.is_empty() {
        let stride = (rest.len() as f64 / need as f64).max(1.0);
        let mut acc = 0.0f64;
        while (acc as usize) < rest.len() && chosen.len() < target {
            chosen.push(rest[acc as usize]);
            acc += stride;
        }
    }
    chosen
}

/// Materialize a sub-scene from a list of `scene` indices, preserving metadata.
fn scene_from_indices(scene: &SplatScene, indices: &[u32]) -> SplatScene {
    let mut out = SplatScene::new();
    out.coordinate_system = scene.coordinate_system;
    out.temporal_mode = scene.temporal_mode;
    out.splats = indices.iter().map(|&i| scene.splats[i as usize].clone()).collect();
    out
}

/// Plan a tileset from an optimized splat scene.
pub fn plan_tileset(scene: &SplatScene, config: &TilesetConfig) -> Result<TilesetPlan, TilesetError> {
    if scene.is_empty() {
        return Err(TilesetError::EmptyScene);
    }
    if config.lods_per_node < 1 {
        return Err(TilesetError::InvalidLods);
    }

    let positions: Vec<[f32; 3]> = scene.splats.iter().map(|s| s.position).collect();
    let tree = Octree::build(&positions, config.octree);

    let mut payloads: Vec<TilePayload> = Vec::new();
    let mut filenames: Vec<String> = Vec::new();

    let (lod_root, _root_pool) =
        build_lod_node(&tree.root, scene, config, &mut payloads, &mut filenames);
    let ts_root = build_tileset_node(&tree.root);

    let lod_meta = LodMeta {
        lod_levels: tree.depth_reached + 1,
        environment: None,
        shared_palette: None,
        filenames: filenames.clone(),
        tree: lod_root,
    };

    let root_geo_error = tree.root.bounds.diagonal() as f64;
    let tileset = TilesetManifest {
        asset: TilesetAsset {
            version: "1.1".into(),
            generator: Some(format!("catetus-tileset {}", env!("CARGO_PKG_VERSION"))),
        },
        geometric_error: root_geo_error,
        root: ts_root,
    };

    Ok(TilesetPlan { lod_meta, tileset, payloads })
}

/// Compute the per-node LOD targets (coarse→fine) for a proxy pool of
/// `pool_len` splats. The finest target is `pool_len` (the whole, already-
/// bounded proxy pool — NOT the full subtree), so every tile is ≤ `proxy_cap`.
/// Coarser targets interpolate geometrically from `coarsest_target` up to
/// `pool_len`. Counts are clamped monotonic non-decreasing.
fn lod_targets(pool_len: usize, config: &TilesetConfig) -> Vec<usize> {
    let levels = config.lods_per_node;
    let mut targets = Vec::with_capacity(levels);
    for l in 0..levels {
        if l == levels - 1 {
            targets.push(pool_len); // finest = the whole (capped) proxy pool
        } else {
            let frac = (l + 1) as f64 / levels as f64;
            let base = config.coarsest_target.max(1) as f64;
            let t = (base * (pool_len as f64 / base).powf(frac)).round() as usize;
            targets.push(t.clamp(1, pool_len.max(1)));
        }
    }
    for i in 1..targets.len() {
        if targets[i] < targets[i - 1] {
            targets[i] = targets[i - 1];
        }
    }
    targets
}

/// Post-order build of one LOD-meta node plus its emitted tile payloads.
///
/// Returns `(LodMetaNode, proxy_pool)` where `proxy_pool` is this node's
/// representative index set, bounded by `config.proxy_cap`:
///   - **Leaf**: the pool is the leaf's full `splat_indices` (already bounded
///     by `max_splats_per_leaf`), capped at `proxy_cap` if a leaf is somehow
///     larger.
///   - **Internal**: recurse children FIRST, concatenate their returned proxy
///     pools, then importance-decimate the concatenation down to `proxy_cap`.
///     This is what keeps the root proxy fixed-size: the root pool ≤
///     `proxy_cap` no matter how many splats are beneath it.
///
/// The node's `lods_per_node` tiles are then importance-decimations of the pool
/// (coarse→fine), with the FINEST tile = the whole pool (≤ `proxy_cap`).
///
/// Memory: the pool returned at each level is freed as soon as the parent has
/// merged+capped it, so peak transient memory is bounded by (siblings' pools +
/// one merged buffer) ≈ `O(8·proxy_cap)` per level — independent of N.
fn build_lod_node(
    node: &OctreeNode,
    scene: &SplatScene,
    config: &TilesetConfig,
    payloads: &mut Vec<TilePayload>,
    filenames: &mut Vec<String>,
) -> (LodMetaNode, Vec<u32>) {
    let cap = config.proxy_cap.max(1);

    // 1. Build children first (post-order) and gather this node's proxy pool.
    let mut children = Vec::with_capacity(node.children.len());
    let pool: Vec<u32> = if node.is_leaf() {
        // Leaf pool = its own indices, capped (leaves are already bounded by
        // max_splats_per_leaf, but cap defensively so no tile exceeds proxy_cap).
        decimate_indices(scene, &node.splat_indices, cap)
    } else {
        let mut merged: Vec<u32> = Vec::new();
        for c in &node.children {
            let (child_meta, child_pool) = build_lod_node(c, scene, config, payloads, filenames);
            children.push(child_meta);
            merged.extend_from_slice(&child_pool);
            // child_pool dropped here.
        }
        // Cap the merged pool by importance so internal-node pools never grow
        // past proxy_cap as we ascend the tree.
        decimate_indices(scene, &merged, cap)
    };

    // 2. Emit this node's LOD tiles by decimating the (bounded) pool.
    let targets = lod_targets(pool.len(), config);
    let mut lods = Vec::with_capacity(targets.len());
    for (lod, &target) in targets.iter().enumerate() {
        let sub_indices = decimate_indices(scene, &pool, target);
        let sub = scene_from_indices(scene, &sub_indices);
        let count = sub.len();
        let file_index = filenames.len();
        filenames.push(format!("tiles/{file_index}.sftile"));
        // `sub_indices` are indices into the original `scene` (the octree leaf
        // indices flow through `decimate_indices` unchanged), so they ARE the
        // origin indices for this tile's splats — capture them so a scene-global
        // shared palette can be looked up per tile splat.
        payloads.push(TilePayload { file_index, lod, scene: sub, origins: sub_indices });
        lods.push(LodRef { file: file_index, count });
    }

    let meta =
        LodMetaNode { bound: Aabb { min: node.bounds.min, max: node.bounds.max }, lods, children };
    (meta, pool)
}

/// Build the 3D-Tiles node tree (same shape as the LOD-meta tree). Each node's
/// `content.uri` is left empty here and filled at write time by
/// [`fill_tileset_content`], which walks both trees in lockstep and points each
/// node at its coarsest-LOD proxy (REPLACE refinement; codec's real file ext).
fn build_tileset_node(node: &OctreeNode) -> TileNode {
    let geo_error = node.bounds.diagonal() as f64;
    let children = node.children.iter().map(build_tileset_node).collect();
    TileNode {
        bounding_volume: BoundingVolume { box_: aabb_to_obb_box(node.bounds.min, node.bounds.max) },
        geometric_error: geo_error,
        refine: Some("REPLACE".into()),
        content: None,
        children,
    }
}

/// Write a planned tileset to `out_dir`: `lod-meta.json`, `tileset.json`, and
/// `tiles/<i>.<ext>`. Returns the total bytes written across all tiles.
pub fn write_tileset<C: TilePayloadCodec>(
    plan: &TilesetPlan,
    codec: &C,
    out_dir: &Path,
) -> Result<u64, TilesetError> {
    let tiles_dir = out_dir.join("tiles");
    std::fs::create_dir_all(&tiles_dir)?;

    let mut total_tile_bytes = 0u64;
    let mut filenames: Vec<String> = Vec::with_capacity(plan.payloads.len());
    for p in &plan.payloads {
        let tb = codec.encode(&p.scene);
        let name = format!("tiles/{}.{}", p.file_index, tb.ext);
        std::fs::write(out_dir.join(&name), &tb.bytes)?;
        total_tile_bytes += tb.bytes.len() as u64;
        // Companion sidecar (e.g. `.shpal` for palette-coded SH-rest): written
        // next to the tile as `tiles/<i>.<ext>.<sidecar_ext>`. The loader pairs
        // them by name (mirrors the single-file CLI's `<output>.glb.shpal`).
        if let (Some(side), Some(side_ext)) = (&tb.sidecar, tb.sidecar_ext) {
            let side_name = format!("{name}.{side_ext}");
            std::fs::write(out_dir.join(&side_name), side)?;
            total_tile_bytes += side.len() as u64;
        }
        filenames.push(name);
    }

    let mut lod_meta = plan.lod_meta.clone();
    lod_meta.filenames = filenames.clone();

    let mut tileset = plan.tileset.clone();
    fill_tileset_content(&mut tileset.root, &lod_meta.tree, &filenames);

    std::fs::write(out_dir.join("lod-meta.json"), serde_json::to_string_pretty(&lod_meta)?)?;
    std::fs::write(out_dir.join("tileset.json"), serde_json::to_string_pretty(&tileset)?)?;

    Ok(total_tile_bytes)
}

/// On-disk total + layout summary returned by [`write_tileset_shared`].
#[derive(Debug, Clone)]
pub struct SharedTilesetWritten {
    /// Total bytes across the root codebook + all tile GLBs + all `.shpalx`.
    pub total_bytes: u64,
    /// Bytes of the single root `palette.shpal`.
    pub codebook_bytes: u64,
    /// Summed bytes of all tile GLBs (SH-rest elided).
    pub tile_glb_bytes: u64,
    /// Summed bytes of all per-tile `.shpalx` index sidecars.
    pub index_bytes: u64,
    /// Number of tiles written (== number of `.shpalx` sidecars).
    pub n_tiles: usize,
    /// Codebook centroid count (K). `None` for a DC-only scene (no palette).
    pub palette_size: Option<usize>,
}

/// Write a planned tileset using the **scene-global shared SH-rest codebook**
/// (front D / D-WIRE). This is the streaming-compression path: instead of FP32
/// SH-rest per tile (180 B/splat) every tile stores only u16 palette indices
/// (2 B/splat) into ONE codebook written once at the tileset root.
///
/// Layout written under `out_dir`:
/// ```text
///   out_dir/
///     ├── palette.shpal           ← shared VQ45 codebook, written ONCE
///     ├── lod-meta.json           ← LodMeta.sharedPalette = "palette.shpal"
///     ├── tileset.json
///     └── tiles/
///           ├── <i>.glb           ← geometry + DC, SH-rest accessors elided
///           └── <i>.glb.shpalx    ← this tile's u16 palette indices (zstd-19)
/// ```
///
/// If the scene is DC-only (no SH-rest to share), this transparently falls back
/// to the FP32 [`GlbTileCodec`](crate::GlbTileCodec) path so the call still
/// produces a valid tileset (no `palette.shpal`, `sharedPalette` absent).
///
/// The original `scene` is needed (not just `plan`) because the codebook is
/// built once over the WHOLE scene; `plan.payloads[*].origins` then index into
/// it per tile. Returns a [`SharedTilesetWritten`] byte breakdown.
pub fn write_tileset_shared(
    plan: &TilesetPlan,
    scene: &SplatScene,
    out_dir: &Path,
    cfg: &SharedPaletteConfig,
) -> Result<SharedTilesetWritten, TilesetError> {
    let tiles_dir = out_dir.join("tiles");
    std::fs::create_dir_all(&tiles_dir)?;

    // 1. Build the scene-global codebook ONCE. DC-only → no palette to share.
    let built = SharedCodebook::build(scene, cfg.palette_size, cfg.iterations, cfg.seed)
        .map_err(TilesetError::SharedPalette)?;

    // DC-only: fall back to the FP32 GLB codec so the call still yields a valid
    // (just not palette-compressed) tileset.
    let codebook = if let Some(cb) = built {
        cb
    } else {
        let total =
            write_tileset(plan, &crate::GlbTileCodec::new(crate::TilePreset::Balanced), out_dir)?;
        return Ok(SharedTilesetWritten {
            total_bytes: total,
            codebook_bytes: 0,
            tile_glb_bytes: total,
            index_bytes: 0,
            n_tiles: plan.payloads.len(),
            palette_size: None,
        });
    };

    // 2. Write the shared codebook ONCE at the tileset root.
    let codebook_bytes = codebook.root_sidecar_bytes().len() as u64;
    std::fs::write(out_dir.join(SHARED_PALETTE_FILENAME), codebook.root_sidecar_bytes())?;

    // 3. Write each tile via the shared-palette codec: GLB (SH elided) +
    //    `.shpalx` index sidecar.
    let shp = SharedPaletteTileCodec::new(&codebook);
    let mut tile_glb_bytes = 0u64;
    let mut index_bytes = 0u64;
    let mut filenames: Vec<String> = Vec::with_capacity(plan.payloads.len());
    for p in &plan.payloads {
        let tb = shp.encode_tile(&p.scene, &p.origins);
        let name = format!("tiles/{}.{}", p.file_index, tb.ext);
        std::fs::write(out_dir.join(&name), &tb.bytes)?;
        tile_glb_bytes += tb.bytes.len() as u64;
        // Every shared-palette tile carries a `.shpalx` index sidecar.
        if let (Some(side), Some(side_ext)) = (&tb.sidecar, tb.sidecar_ext) {
            let side_name = format!("{name}.{side_ext}");
            std::fs::write(out_dir.join(&side_name), side)?;
            index_bytes += side.len() as u64;
        }
        filenames.push(name);
    }

    // 4. Manifests, with the shared-codebook reference threaded into LodMeta.
    let mut lod_meta = plan.lod_meta.clone();
    lod_meta.filenames = filenames.clone();
    lod_meta.shared_palette = Some(SHARED_PALETTE_FILENAME.to_string());

    let mut tileset = plan.tileset.clone();
    fill_tileset_content(&mut tileset.root, &lod_meta.tree, &filenames);

    std::fs::write(out_dir.join("lod-meta.json"), serde_json::to_string_pretty(&lod_meta)?)?;
    std::fs::write(out_dir.join("tileset.json"), serde_json::to_string_pretty(&tileset)?)?;

    Ok(SharedTilesetWritten {
        total_bytes: codebook_bytes + tile_glb_bytes + index_bytes,
        codebook_bytes,
        tile_glb_bytes,
        index_bytes,
        n_tiles: plan.payloads.len(),
        palette_size: Some(codebook.palette_size()),
    })
}

/// Walk the 3D-Tiles tree and the LOD-meta tree in lockstep (identical shape)
/// and set each 3D-Tiles node's `content.uri` to its finest LOD file.
fn fill_tileset_content(ts: &mut TileNode, lm: &LodMetaNode, filenames: &[String]) {
    // REPLACE refinement: a node's content is its COARSEST LOD (a small proxy);
    // children refine it as the camera approaches. (Previously used lods.last()
    // = finest, which made the root's first-paint tile the ENTIRE scene — a
    // streaming-killer for big scenes; see STREAM-6.)
    if let Some(coarsest) = lm.lods.first() {
        if let Some(name) = filenames.get(coarsest.file) {
            ts.content = Some(TileContent { uri: name.clone() });
        }
    }
    for (tc, lc) in ts.children.iter_mut().zip(lm.children.iter()) {
        fill_tileset_content(tc, lc, filenames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::SfTileCodec;
    use catetus_core::ir::{Color, Splat};

    fn cube_scene(n_per_axis: usize) -> SplatScene {
        let mut s = SplatScene::new();
        for x in 0..n_per_axis {
            for y in 0..n_per_axis {
                for z in 0..n_per_axis {
                    s.splats.push(Splat {
                        position: [x as f32, y as f32, z as f32],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [0.5, 0.5, 0.5],
                        opacity: 0.8,
                        color: Color::Rgb([0.5, 0.5, 0.5]),
                    });
                }
            }
        }
        s
    }

    #[test]
    fn empty_scene_errors() {
        let s = SplatScene::new();
        assert!(matches!(plan_tileset(&s, &TilesetConfig::default()), Err(TilesetError::EmptyScene)));
    }

    #[test]
    fn plan_produces_multilevel_octree() {
        let s = cube_scene(16); // 4096 splats
        let cfg = TilesetConfig {
            octree: OctreeConfig { max_depth: 4, max_splats_per_leaf: 256 },
            lods_per_node: 3,
            coarsest_target: 100,
            proxy_cap: 20_000,
        };
        let plan = plan_tileset(&s, &cfg).unwrap();
        assert!(plan.lod_meta.lod_levels >= 2, "expected a multi-level octree");
        assert!(!plan.lod_meta.tree.children.is_empty());
        fn check(n: &LodMetaNode, k: usize) {
            assert_eq!(n.lods.len(), k);
            for w in n.lods.windows(2) {
                assert!(w[1].count >= w[0].count);
            }
            for c in &n.children {
                check(c, k);
            }
        }
        check(&plan.lod_meta.tree, 3);
    }

    #[test]
    fn finest_lod_of_root_leaf_holds_all_splats() {
        let s = cube_scene(8); // 512 splats
        let cfg = TilesetConfig {
            octree: OctreeConfig { max_depth: 3, max_splats_per_leaf: 1024 },
            lods_per_node: 3,
            coarsest_target: 50,
            proxy_cap: 20_000,
        };
        let plan = plan_tileset(&s, &cfg).unwrap();
        let finest = plan.lod_meta.tree.lods.last().unwrap();
        assert_eq!(finest.count, 512);
    }

    #[test]
    fn write_tileset_emits_files_and_loadable_tiles() {
        let s = cube_scene(16);
        let cfg = TilesetConfig {
            octree: OctreeConfig { max_depth: 4, max_splats_per_leaf: 256 },
            lods_per_node: 3,
            coarsest_target: 100,
            proxy_cap: 20_000,
        };
        let plan = plan_tileset(&s, &cfg).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let total = write_tileset(&plan, &SfTileCodec, dir.path()).unwrap();
        assert!(total > 0);

        let lod_meta_txt = std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap();
        let lod_meta: LodMeta = serde_json::from_str(&lod_meta_txt).unwrap();
        assert!(lod_meta.filenames.len() > 1);
        assert!(lod_meta.lod_levels >= 2);

        let ts_txt = std::fs::read_to_string(dir.path().join("tileset.json")).unwrap();
        let ts: TilesetManifest = serde_json::from_str(&ts_txt).unwrap();
        assert_eq!(ts.asset.version, "1.1");
        // REPLACE refinement: root content must be the root's OWN COARSEST proxy
        // (its first LOD), NOT the finest (whole scene) — else first-paint
        // downloads everything. Regression guard for STREAM-6 fix (A).
        //
        // NOTE: post-order tile emission (STREAM-5 bottom-up proxy build) means
        // the root's tiles are allocated LAST, so the root coarsest is no longer
        // `filenames[0]`; assert against the root node's own first LOD file.
        let root_coarsest_file = lod_meta.tree.lods.first().unwrap().file;
        assert_eq!(
            ts.root.content.as_ref().unwrap().uri,
            lod_meta.filenames[root_coarsest_file],
            "root should point at its own coarsest proxy tile, not the full scene"
        );
        // And it must be the SMALLEST root LOD (coarsest), never the finest.
        assert!(
            lod_meta.tree.lods.first().unwrap().count <= lod_meta.tree.lods.last().unwrap().count,
            "root coarsest proxy must not exceed root finest"
        );

        for (i, name) in lod_meta.filenames.iter().enumerate() {
            let bytes = std::fs::read(dir.path().join(name)).unwrap();
            let decoded = SfTileCodec::decode(&bytes).unwrap();
            let declared = find_count(&lod_meta.tree, i).expect("file index in tree");
            assert_eq!(decoded.len(), declared, "tile {name} count mismatch");
        }
    }

    fn find_count(n: &LodMetaNode, file: usize) -> Option<usize> {
        for l in &n.lods {
            if l.file == file {
                return Some(l.count);
            }
        }
        for c in &n.children {
            if let Some(v) = find_count(c, file) {
                return Some(v);
            }
        }
        None
    }

    /// A deterministic pseudo-random spread scene of `n` splats in a unit cube.
    /// Uses a cheap LCG so positions are well-distributed (the octree actually
    /// subdivides) without a dependency. Importance varies so decimation has
    /// something to rank.
    fn synthetic_scene(n: usize) -> SplatScene {
        let mut s = SplatScene::new();
        s.splats.reserve(n);
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as f32) / (u32::MAX as f32)
        };
        for _ in 0..n {
            let p = [next() * 100.0, next() * 100.0, next() * 100.0];
            let sc = 0.1 + next() * 0.9;
            s.splats.push(Splat {
                position: p,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [sc, sc, sc],
                opacity: 0.1 + next() * 0.9,
                color: Color::Rgb([next(), next(), next()]),
            });
        }
        s
    }

    /// Max splat count of ANY tile across the whole tree (the global tile cap).
    fn max_tile_count(n: &LodMetaNode) -> usize {
        let mut m = n.lods.iter().map(|l| l.count).max().unwrap_or(0);
        for c in &n.children {
            m = m.max(max_tile_count(c));
        }
        m
    }

    /// STREAM-5 core invariant: the ROOT node's coarsest *and* finest proxy are
    /// FIXED-SIZE (≤ proxy_cap) and do NOT scale with total scene size. The old
    /// builder set the root's finest LOD to the entire scene and its coarsest to
    /// `coarsest_target·(N/base)^(1/3)`, both of which grew with N. Here we build
    /// a tiny (~10k) and a large (~800k) scene with the SAME config and assert
    /// the large-scene root proxies are ≤ 2× the small-scene ones (well under any
    /// N-proportional growth — 800k/10k = 80×).
    #[test]
    fn root_proxy_is_fixed_size_independent_of_scene_size() {
        let cfg = TilesetConfig {
            octree: OctreeConfig { max_depth: 8, max_splats_per_leaf: 4_000 },
            lods_per_node: 3,
            coarsest_target: 2_000,
            proxy_cap: 20_000,
        };

        let small = synthetic_scene(10_000);
        let large = synthetic_scene(800_000);

        let plan_small = plan_tileset(&small, &cfg).unwrap();
        let plan_large = plan_tileset(&large, &cfg).unwrap();

        // Hard cap holds for BOTH at the root and everywhere.
        let s_root = &plan_small.lod_meta.tree.lods;
        let l_root = &plan_large.lod_meta.tree.lods;
        let s_coarse = s_root.first().unwrap().count;
        let s_fine = s_root.last().unwrap().count;
        let l_coarse = l_root.first().unwrap().count;
        let l_fine = l_root.last().unwrap().count;

        assert!(
            l_fine <= cfg.proxy_cap,
            "large root finest proxy {l_fine} exceeds proxy_cap {}",
            cfg.proxy_cap
        );
        assert!(
            l_coarse <= cfg.proxy_cap,
            "large root coarsest proxy {l_coarse} exceeds proxy_cap {}",
            cfg.proxy_cap
        );
        assert!(max_tile_count(&plan_large.lod_meta.tree) <= cfg.proxy_cap, "some tile > proxy_cap");

        // Fixed-size, NOT N-proportional: scene grew 80×, proxies must not.
        assert!(
            l_fine <= 2 * s_fine.max(1),
            "root finest grew with N: small={s_fine} large={l_fine} (80× scene)"
        );
        assert!(
            l_coarse <= 2 * s_coarse.max(1),
            "root coarsest grew with N: small={s_coarse} large={l_coarse} (80× scene)"
        );
    }

    /// Conservation under the bottom-up design: every input splat lands in
    /// exactly one LEAF (octree invariant), independent of the proxy logic.
    #[test]
    fn conservation_every_splat_in_one_leaf() {
        let scene = synthetic_scene(50_000);
        let positions: Vec<[f32; 3]> = scene.splats.iter().map(|s| s.position).collect();
        let tree = Octree::build(&positions, OctreeConfig { max_depth: 8, max_splats_per_leaf: 500 });
        assert_eq!(
            tree.leaf_index_total(),
            scene.len(),
            "every splat must be in exactly one leaf"
        );
        // Internal nodes must NOT retain indices (the OOM fix).
        fn assert_internal_empty(n: &OctreeNode) {
            if !n.is_leaf() {
                assert!(
                    n.splat_indices.is_empty(),
                    "internal node retained {} indices — STREAM-5 regression",
                    n.splat_indices.len()
                );
                for c in &n.children {
                    assert_internal_empty(c);
                }
            }
        }
        assert_internal_empty(&tree.root);
    }

    /// Memory measurement harness. Builds a tileset from an N-splat synthetic
    /// scene; intended to be run under `/usr/bin/time -l` to read peak RSS.
    /// Ignored by default (slow + only meaningful under an external profiler).
    /// Set `STREAM5_N` to choose N (default 2_000_000).
    ///
    ///   STREAM5_N=2000000 /usr/bin/time -l \
    ///     cargo test -p catetus-tileset --release \
    ///     mem_build_scales_with_n -- --ignored --nocapture 2>&1 | grep -E 'maximum|N='
    #[test]
    #[ignore = "memory measurement harness; run under /usr/bin/time -l"]
    fn mem_build_scales_with_n() {
        let n: usize =
            std::env::var("STREAM5_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
        let scene = synthetic_scene(n);
        let cfg = TilesetConfig {
            octree: OctreeConfig { max_depth: 8, max_splats_per_leaf: 50_000 },
            lods_per_node: 3,
            coarsest_target: 2_000,
            proxy_cap: 20_000,
        };
        let plan = plan_tileset(&scene, &cfg).unwrap();
        // Touch the result so the optimizer can't elide the build.
        let root_fine = plan.lod_meta.tree.lods.last().unwrap().count;
        eprintln!(
            "N={n} tiles={} lod_levels={} root_finest_proxy={root_fine} (proxy_cap={})",
            plan.payloads.len(),
            plan.lod_meta.lod_levels,
            cfg.proxy_cap
        );
        assert!(root_fine <= cfg.proxy_cap);
    }
}
