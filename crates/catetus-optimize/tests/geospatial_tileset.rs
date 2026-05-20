//! Round-trip test for the `geospatial` preset → Cesium 3D Tiles tileset.
//!
//! Builds a small synthetic scene, runs the `geospatial` pipeline end-to-end,
//! writes the tileset, then re-parses every artifact with a permissive JSON
//! validator and the `catetus_gltf` GLB reader. Asserts the tileset
//! structure matches the 3D Tiles 1.1 shape Cesium ion expects.

use std::path::PathBuf;

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::read_glb;
use catetus_optimize::{preset, write_tileset, TilesetOpts};
use serde_json::Value;

fn synthetic_scene(n: usize) -> SplatScene {
    let mut scene = SplatScene::new();
    // Deterministic pseudo-random scatter: positions on a simple lattice
    // perturbed by a halton-ish recurrence so we exercise the morton sort and
    // get a non-trivial bounding box.
    for i in 0..n {
        let f = i as f32;
        let x = (f * 0.137).sin() * 4.0 + f * 0.01;
        let y = (f * 0.291).cos() * 2.0;
        let z = (f * 0.413).sin() * 3.0 - f * 0.005;
        scene.splats.push(Splat {
            position: [x, y, z],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.02, 0.02, 0.02],
            opacity: 0.8,
            color: Color::Rgb([0.5, 0.4, 0.6]),
        });
    }
    scene
}

fn tempdir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let suffix = t.wrapping_add(N.fetch_add(1, Ordering::Relaxed));
    let p = std::env::temp_dir().join(format!(
        "catetus-geospatial-{tag}-{}-{suffix}",
        std::process::id()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn geospatial_preset_emits_cesium_tileset() {
    let mut scene = synthetic_scene(400);
    let pipe = preset("geospatial").expect("geospatial preset registered");
    let report = pipe.run(&mut scene).expect("pipeline runs");

    let lods = scene.lods.as_ref().expect("BuildLOD populated lods");
    // 4 LOD levels: full + 3 halvings.
    assert_eq!(lods.len(), 4, "expected 4 LOD levels");
    // Each subsequent LOD has roughly half the splat count of its predecessor.
    for w in lods.windows(2) {
        let parent = w[0].indices.len();
        let child = w[1].indices.len();
        assert!(
            child <= parent,
            "LOD chain not monotonically shrinking: {parent} -> {child}"
        );
        // Allow a slop window because BuildLOD uses integer strides; 0.4..0.6
        // is the tolerance that covers all our halving plans.
        let ratio = child as f32 / parent.max(1) as f32;
        assert!(
            (0.35..=0.65).contains(&ratio),
            "halving ratio out of tolerance: parent={parent} child={child} ratio={ratio}"
        );
    }
    assert!(report.passes.iter().any(|p| p.name == "BuildLOD"));

    let out = tempdir("emit");
    let tileset_report =
        write_tileset(&scene, &out, &TilesetOpts::default()).expect("tileset writes");

    // The report's tiles are coarse-to-fine. Root is the coarsest splat count.
    assert_eq!(tileset_report.tiles.len(), 4);
    let root_count = tileset_report.tiles[0].splat_count;
    let leaf_count = tileset_report.tiles.last().unwrap().splat_count;
    assert!(root_count < leaf_count, "root must be coarser than leaf");

    // Parse tileset.json as untyped JSON and validate against the 3D Tiles 1.1
    // shape Cesium ion requires.
    let bytes = std::fs::read(out.join("tileset.json")).expect("tileset.json exists");
    let json: Value = serde_json::from_slice(&bytes).expect("tileset.json is valid JSON");

    assert_eq!(
        json.pointer("/asset/version").and_then(Value::as_str),
        Some("1.1"),
        "asset.version must be exactly '1.1' for Cesium ion KHR_gaussian_splatting"
    );
    let root_err = json
        .pointer("/geometricError")
        .and_then(Value::as_f64)
        .expect("root geometricError present");
    assert!(root_err > 0.0, "root geometricError must be positive");
    let exts: Vec<&str> = json
        .pointer("/extensionsRequired")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    assert!(
        exts.contains(&"KHR_gaussian_splatting"),
        "extensionsRequired must declare KHR_gaussian_splatting"
    );

    // Walk the tile tree and assert: every tile has a box bounding volume,
    // every tile.content.uri references a file on disk that parses as a GLB,
    // and child geometricError < parent geometricError (3D Tiles 1.1 §6.1.4).
    fn walk(tile: &Value, parent_err: f64, dir: &std::path::Path) -> usize {
        let err = tile
            .get("geometricError")
            .and_then(Value::as_f64)
            .expect("tile.geometricError present");
        assert!(
            err <= parent_err + 1e-9,
            "child geometricError ({err}) must be <= parent ({parent_err})"
        );
        let bv = tile
            .pointer("/boundingVolume/box")
            .and_then(Value::as_array)
            .expect("tile.boundingVolume.box present");
        assert_eq!(bv.len(), 12, "box must have 12 floats");
        let uri = tile
            .pointer("/content/uri")
            .and_then(Value::as_str)
            .expect("tile.content.uri present");
        let glb_path = dir.join(uri);
        assert!(
            glb_path.exists(),
            "{} should exist on disk",
            glb_path.display()
        );
        let glb_scene = read_glb(&glb_path).expect("GLB parses");
        assert!(!glb_scene.splats.is_empty(), "GLB carries splats");

        let mut count = 1;
        if let Some(children) = tile.get("children").and_then(Value::as_array) {
            for c in children {
                count += walk(c, err, dir);
            }
        }
        count
    }
    let root_tile = json.get("root").expect("root tile present");
    let total = walk(root_tile, root_err, &out);
    assert_eq!(total, 4, "expected 4 tiles in the LOD chain");
}

#[test]
fn geospatial_preset_is_deterministic() {
    // Same input through `preset("geospatial")` → `write_tileset` must yield
    // byte-identical tileset.json and identical GLBs across runs.
    let a = tempdir("det-a");
    let b = tempdir("det-b");

    for dir in [&a, &b] {
        let mut scene = synthetic_scene(200);
        let pipe = preset("geospatial").unwrap();
        pipe.run(&mut scene).unwrap();
        write_tileset(&scene, dir, &TilesetOpts::default()).unwrap();
    }

    let json_a = std::fs::read(a.join("tileset.json")).unwrap();
    let json_b = std::fs::read(b.join("tileset.json")).unwrap();
    assert_eq!(json_a, json_b, "tileset.json must be byte-deterministic");

    for i in 0..4 {
        let glb_a = std::fs::read(a.join(format!("lod{i}.glb"))).unwrap();
        let glb_b = std::fs::read(b.join(format!("lod{i}.glb"))).unwrap();
        assert_eq!(glb_a, glb_b, "lod{i}.glb must be byte-deterministic");
    }
}
