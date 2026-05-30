//! End-to-end test: build a real GLB tileset and verify every tile is a valid,
//! loadable SF GLB whose splat count matches the manifest, and that both
//! manifests are valid JSON with correct cross-references.
//!
//! This is the integration test the STREAM-2 contract requires: it proves the
//! `GlbTileCodec` emits genuine `KHR_gaussian_splatting` GLBs (decoded by the
//! same `catetus_gltf::read_glb_bytes` the CLI/viewer use), not placeholders.

use catetus_core::ir::{Color, Splat, SplatScene};
use catetus_gltf::read_glb_bytes;
use catetus_tileset::{
    plan_tileset, write_tileset, GlbTileCodec, LodMeta, LodMetaNode, OctreeConfig, TilePreset,
    TilesetConfig, TilesetManifest,
};

/// A spatially-spread scene so the octree actually subdivides into many nodes.
fn spread_scene(n_per_axis: usize) -> SplatScene {
    let mut s = SplatScene::new();
    for x in 0..n_per_axis {
        for y in 0..n_per_axis {
            for z in 0..n_per_axis {
                s.splats.push(Splat {
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.3, 0.3, 0.3],
                    opacity: 0.8,
                    color: Color::Rgb([0.5, 0.4, 0.3]),
                });
            }
        }
    }
    s
}

/// Collect (file_index -> declared count) from the LOD-meta tree.
fn collect_counts(node: &LodMetaNode, out: &mut Vec<(usize, usize)>) {
    for l in &node.lods {
        out.push((l.file, l.count));
    }
    for c in &node.children {
        collect_counts(c, out);
    }
}

#[test]
fn every_glb_tile_loads_and_matches_manifest() {
    let scene = spread_scene(20); // 8,000 splats
    let cfg = TilesetConfig {
        octree: OctreeConfig { max_depth: 4, max_splats_per_leaf: 400 },
        lods_per_node: 3,
        coarsest_target: 100,
        proxy_cap: 20_000,
    };
    let plan = plan_tileset(&scene, &cfg).unwrap();
    assert!(plan.lod_meta.lod_levels >= 2, "scene should produce a multi-level octree");
    assert!(plan.payloads.len() > 3, "should produce many tiles");

    let dir = tempfile::tempdir().unwrap();
    let codec = GlbTileCodec::new(TilePreset::Balanced);
    let total = write_tileset(&plan, &codec, dir.path()).unwrap();
    assert!(total > 0, "tiles must have bytes");

    // --- Manifests are valid JSON and well-formed. ---
    let lod_meta_txt = std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap();
    let lod_meta: LodMeta = serde_json::from_str(&lod_meta_txt).unwrap();
    assert!(lod_meta.lod_levels >= 2);
    assert!(lod_meta.filenames.iter().all(|f| f.ends_with(".glb")), "tiles are GLBs");

    let ts_txt = std::fs::read_to_string(dir.path().join("tileset.json")).unwrap();
    let ts: TilesetManifest = serde_json::from_str(&ts_txt).unwrap();
    assert_eq!(ts.asset.version, "1.1");
    assert!(ts.root.content.is_some(), "root must reference a tile");
    let root_uri = &ts.root.content.as_ref().unwrap().uri;
    assert!(root_uri.ends_with(".glb"));

    // --- Every referenced file exists, is a real GLB, and its decoded splat
    //     count matches the manifest. ---
    let mut declared: Vec<(usize, usize)> = Vec::new();
    collect_counts(&lod_meta.tree, &mut declared);
    assert_eq!(
        declared.len(),
        lod_meta.filenames.len(),
        "every filename should be referenced exactly once by the tree"
    );

    for (file_idx, count) in declared {
        let name = &lod_meta.filenames[file_idx];
        let bytes =
            std::fs::read(dir.path().join(name)).unwrap_or_else(|_| panic!("missing {name}"));
        // GLB container magic.
        assert_eq!(&bytes[0..4], b"glTF", "{name} is not a GLB");
        // The cardinal check: the SF GLB reader loads it back.
        let decoded =
            read_glb_bytes(&bytes).unwrap_or_else(|e| panic!("tile {name} failed to decode: {e}"));
        assert_eq!(decoded.len(), count, "tile {name} count != manifest");
    }
}

#[test]
fn quality_max_tileset_tiles_are_valid() {
    let scene = spread_scene(12); // 1,728 splats
    let cfg = TilesetConfig {
        octree: OctreeConfig { max_depth: 3, max_splats_per_leaf: 200 },
        lods_per_node: 2,
        coarsest_target: 64,
        proxy_cap: 20_000,
    };
    let plan = plan_tileset(&scene, &cfg).unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_tileset(&plan, &GlbTileCodec::new(TilePreset::QualityMax), dir.path()).unwrap();

    let lod_meta: LodMeta =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap())
            .unwrap();
    for name in &lod_meta.filenames {
        let glb = std::fs::read(dir.path().join(name)).unwrap();
        read_glb_bytes(&glb).unwrap_or_else(|e| panic!("quality-max tile {name} invalid: {e}"));
    }
}
