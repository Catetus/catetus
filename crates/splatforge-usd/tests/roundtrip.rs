//! End-to-end USDA round-trip tests.
//!
//! Mirrors `crates/splatforge-gltf/tests/roundtrip.rs` so the two formats
//! converge on the same conformance bar — same scene goes in, same scene
//! comes back, attribute-by-attribute within `1e-5`.

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_usd::{read_usda, write_usda, UsdError, UsdWriteOpts};
use tempfile::tempdir;

fn three_splat_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..3u32 {
        let f = i as f32;
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.5 + f * 0.1,
            color: Color::Rgb([f * 0.1, 0.2, 0.3]),
        });
    }
    scene
}

#[test]
fn roundtrip_three_splats_usda() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usda");
    let scene = three_splat_scene();
    write_usda(&scene, &path, &UsdWriteOpts::default()).expect("write");
    let decoded = read_usda(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a.position[k] - b.position[k]).abs() < 1e-5,
                "splat {i} position mismatch on axis {k}"
            );
            assert!(
                (a.scale[k] - b.scale[k]).abs() < 1e-5,
                "splat {i} scale mismatch on axis {k}"
            );
        }
        for k in 0..4 {
            assert!(
                (a.rotation[k] - b.rotation[k]).abs() < 1e-5,
                "splat {i} rotation mismatch on component {k}"
            );
        }
        assert!(
            (a.opacity - b.opacity).abs() < 1e-5,
            "splat {i} opacity mismatch"
        );
    }
}

#[test]
fn empty_scene_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usda");
    let scene = SplatScene::new();
    let err = write_usda(&scene, &path, &UsdWriteOpts::default()).unwrap_err();
    assert!(matches!(err, UsdError::Malformed(_)));
}

#[test]
fn quat_order_flips_on_write() {
    // The IR stores rotations as `(x, y, z, w)`. USD uses `(w, x, y, z)`.
    // Verify the written USDA places `w` first inside the `orientations` tuple.
    let mut scene = SplatScene::new();
    scene.splats.push(Splat {
        position: [0.0, 0.0, 0.0],
        rotation: [0.1, 0.2, 0.3, 0.9],
        scale: [1.0, 1.0, 1.0],
        opacity: 1.0,
        color: Color::Rgb([0.0; 3]),
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usda");
    write_usda(&scene, &path, &UsdWriteOpts::default()).unwrap();
    let body = std::fs::read_to_string(&path).unwrap();
    // The written quaternion should be `(0.9, 0.1, 0.2, 0.3)` — w first.
    assert!(
        body.contains("quatf[] orientations = [(0.9, 0.1, 0.2, 0.3)]"),
        "expected w-first quaternion ordering; got body:\n{body}"
    );
}
