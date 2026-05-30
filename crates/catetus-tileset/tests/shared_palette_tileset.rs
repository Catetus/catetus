//! D-WIRE integration: plan → `write_tileset_shared` → verify the on-disk
//! shared-palette tileset is real and self-consistent:
//!   1. `palette.shpal` written ONCE at the tileset root.
//!   2. every tile GLB has a companion `.shpalx` index sidecar.
//!   3. `lod-meta.json` references the shared codebook (`sharedPalette`).
//!   4. a sample tile + the root codebook roundtrip SH-rest (the loader path).
//!
//! This is the proof that the proven shared-palette result (which previously
//! existed only inside the bench) is now a usable on-disk tileset.

use catetus_core::ir::{Color, Splat, SplatScene};
use catetus_tileset::{
    decode_tile_indices, dc_term, plan_tileset, write_tileset_shared, LodMeta, OctreeConfig,
    SharedCodebook, SharedPaletteConfig, TilesetConfig, SHARED_PALETTE_FILENAME, TILE_INDEX_EXT,
};

/// SH=3 scene with `n_clusters` distinct SH-rest patterns so the VQ codebook
/// has real structure to cluster, spread spatially so the octree subdivides.
fn sh3_scene(n: usize, n_clusters: usize) -> SplatScene {
    let mut s = SplatScene::new();
    for i in 0..n {
        let cluster = i % n_clusters;
        let mut coeffs = vec![0.0f32; 48]; // DC(3) + SH-rest(45)
        coeffs[0] = (i as f32 * 0.001).sin();
        coeffs[1] = (i as f32 * 0.002).cos();
        coeffs[2] = (i as f32 * 0.003).sin();
        for d in 0..45 {
            coeffs[3 + d] = ((cluster * 31 + d * 7) as f32 / 17.0).sin();
        }
        let f = i as f32;
        s.splats.push(Splat {
            position: [(f * 0.13).sin() * 50.0, (f * 0.17).cos() * 50.0, (f * 0.19).sin() * 50.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.05, 0.05, 0.05],
            opacity: 0.8,
            color: Color::Sh { degree: 3, coeffs },
        });
    }
    s
}

#[test]
fn shared_palette_tileset_is_real_on_disk_and_roundtrips() {
    let scene = sh3_scene(8_000, 16);
    let cfg = TilesetConfig {
        octree: OctreeConfig { max_depth: 5, max_splats_per_leaf: 1_000 },
        ..Default::default()
    };
    let plan = plan_tileset(&scene, &cfg).unwrap();
    let dir = tempfile::tempdir().unwrap();

    // Small but real codebook so the test is fast.
    let pal_cfg = SharedPaletteConfig { palette_size: 64, iterations: 4, seed: 0xD11E };
    let written = write_tileset_shared(&plan, &scene, dir.path(), &pal_cfg).unwrap();
    assert!(written.total_bytes > 0);
    assert_eq!(written.palette_size, Some(64));
    assert_eq!(written.n_tiles, plan.payloads.len());

    // (1) palette.shpal written exactly ONCE at the tileset root.
    let root_pal = dir.path().join(SHARED_PALETTE_FILENAME);
    assert!(root_pal.is_file(), "root palette.shpal must exist");
    let root_bytes = std::fs::read(&root_pal).unwrap();
    assert_eq!(root_bytes.len() as u64, written.codebook_bytes);
    assert!(!root_bytes.is_empty());

    // Load the manifest.
    let lod_meta_txt = std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap();
    let lod_meta: LodMeta = serde_json::from_str(&lod_meta_txt).unwrap();

    // (3) manifest references the shared codebook by its root-relative name.
    assert_eq!(
        lod_meta.shared_palette.as_deref(),
        Some(SHARED_PALETTE_FILENAME),
        "lod-meta.json must reference the shared codebook"
    );
    // And the raw JSON carries the camelCase key SuperSplat-style loaders read.
    assert!(
        lod_meta_txt.contains("\"sharedPalette\""),
        "manifest JSON must contain the sharedPalette key"
    );

    // (2) every tile GLB exists AND has a companion .shpalx index sidecar.
    assert!(!lod_meta.filenames.is_empty());
    for name in &lod_meta.filenames {
        let glb = dir.path().join(name);
        assert!(glb.is_file(), "tile GLB missing: {name}");
        let splx = dir.path().join(format!("{name}.{TILE_INDEX_EXT}"));
        assert!(splx.is_file(), "tile index sidecar missing: {name}.{TILE_INDEX_EXT}");
    }

    // The palette.shpal must NOT have been duplicated per-tile.
    let dup = std::fs::read_dir(dir.path().join("tiles"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".shpal"))
        .count();
    assert_eq!(dup, 0, "shared codebook must not be duplicated into tiles/");

    // (4) Loader contract roundtrip: pick a non-empty tile, decode its GLB for
    // DC terms, read its .shpalx indices, and reconstruct SH-rest against the
    // root codebook decoded from palette.shpal — exactly what a viewer does.
    let codebook = SharedCodebook::build(&scene, 64, 4, 0xD11E).unwrap().unwrap();

    let tile_name = lod_meta
        .filenames
        .iter()
        .find(|n| {
            std::fs::read(dir.path().join(n))
                .ok()
                .and_then(|b| catetus_gltf::read_glb_bytes(&b).ok())
                .map(|s| !s.is_empty())
                .unwrap_or(false)
        })
        .expect("at least one non-empty tile");

    let glb_bytes = std::fs::read(dir.path().join(tile_name)).unwrap();
    let decoded = catetus_gltf::read_glb_bytes(&glb_bytes).expect("tile GLB decodes");
    let dc_terms: Vec<[f32; 3]> = decoded.splats.iter().map(dc_term).collect();

    let splx_bytes = std::fs::read(dir.path().join(format!("{tile_name}.{TILE_INDEX_EXT}"))).unwrap();
    let indices = decode_tile_indices(&splx_bytes).unwrap();
    assert_eq!(indices.len(), decoded.len(), "index count == tile splat count");

    let recon = codebook
        .reconstruct_sh_rest(&splx_bytes, &dc_terms)
        .expect("reconstruct SH-rest from (codebook, indices)");
    assert_eq!(recon.len(), decoded.len());

    // Each reconstructed SH-rest must EXACTLY equal the codebook centroid for
    // that tile splat's stored index (the loader contract: SH = codebook[idx]).
    for (j, &idx) in indices.iter().enumerate() {
        let centroid = codebook.centroid(idx);
        if let Color::Sh { coeffs, .. } = &recon[j] {
            let mut max_err = 0.0f32;
            for d in 0..45 {
                max_err = max_err.max((coeffs[3 + d] - centroid[d]).abs());
            }
            assert!(max_err < 1e-5, "tile splat {j}: SH-rest != codebook centroid (err {max_err})");
        } else {
            panic!("expected SH color in reconstruction");
        }
    }
}

#[test]
fn shared_palette_dc_only_scene_falls_back_to_fp32() {
    // A DC-only scene has no SH-rest to share: write_tileset_shared must still
    // produce a valid tileset (FP32 fallback), with NO palette.shpal and NO
    // sharedPalette manifest reference.
    let mut scene = SplatScene::new();
    for i in 0..2_000 {
        scene.splats.push(Splat {
            position: [(i % 50) as f32, ((i / 50) % 50) as f32, (i / 2500) as f32],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1, 0.1, 0.1],
            opacity: 0.7,
            color: Color::Rgb([0.5, 0.4, 0.3]),
        });
    }
    let plan = plan_tileset(&scene, &TilesetConfig::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let written =
        write_tileset_shared(&plan, &scene, dir.path(), &SharedPaletteConfig::default()).unwrap();

    assert_eq!(written.palette_size, None, "DC-only → no codebook");
    assert!(written.total_bytes > 0);
    assert!(
        !dir.path().join(SHARED_PALETTE_FILENAME).exists(),
        "no palette.shpal for a DC-only scene"
    );

    let lod_meta_txt = std::fs::read_to_string(dir.path().join("lod-meta.json")).unwrap();
    let lod_meta: LodMeta = serde_json::from_str(&lod_meta_txt).unwrap();
    assert!(lod_meta.shared_palette.is_none(), "no sharedPalette ref for DC-only");
    assert!(!lod_meta_txt.contains("sharedPalette"));
}
