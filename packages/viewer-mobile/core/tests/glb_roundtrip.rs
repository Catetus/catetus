//! End-to-end round trip: write a small `KHR_gaussian_splatting` `.glb` via the
//! workspace `splatforge-gltf` writer, feed the bytes into our decoder, and
//! assert the resulting `SplatVertex` buffer matches the input scene.
//!
//! When `bonsai-7k.glb` is present at the configured fixture path we also
//! decode it for a real-asset smoke check.

use std::fs;

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{write_glb, WriteOpts};
use splatforge_viewer_mobile::{decode_glb_bytes, sort_by_depth, Camera};
use tempfile::tempdir;

fn synth_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..16u32 {
        let f = i as f32;
        scene.splats.push(Splat {
            position: [f * 0.1, (f * 0.07).sin(), -(f * 0.2)],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.05, 0.05, 0.05],
            opacity: 0.6,
            color: Color::Rgb([f * 0.05, 0.5, 1.0 - f * 0.04]),
        });
    }
    scene
}

#[test]
fn decode_synthetic_glb_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("synth.glb");
    let scene = synth_scene();
    write_glb(&scene, &path, &WriteOpts::default()).expect("write_glb");
    let bytes = fs::read(&path).expect("read bytes");
    let verts = decode_glb_bytes(&bytes).expect("decode_glb_bytes");
    assert_eq!(verts.len(), scene.splats.len());
    for (v, s) in verts.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((v.position[i] - s.position[i]).abs() < 1e-4);
            assert!((v.scale[i] - s.scale[i]).abs() < 1e-4);
        }
        assert!((v.opacity - s.opacity).abs() < 1e-4);
    }
}

#[test]
fn sort_orders_back_to_front() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("synth.glb");
    write_glb(&synth_scene(), &path, &WriteOpts::default()).expect("write");
    let bytes = fs::read(&path).unwrap();
    let verts = decode_glb_bytes(&bytes).unwrap();

    let cam = Camera {
        position: [0.0, 0.0, 5.0],
        target: [0.0, 0.0, 0.0],
        ..Camera::default()
    };
    let view = cam.view();
    let idx = sort_by_depth(&verts, &view);
    assert_eq!(idx.len(), verts.len());

    // After sorting back-to-front the projected view-z values must be
    // monotonically non-decreasing.
    let v0 = view[2];
    let v1 = view[6];
    let v2 = view[10];
    let v3 = view[14];
    let mut prev = f32::NEG_INFINITY;
    for &i in &idx {
        let p = verts[i as usize].position;
        let z = v0 * p[0] + v1 * p[1] + v2 * p[2] + v3;
        assert!(z >= prev - 1e-5, "non-monotonic depth at idx={i}");
        prev = z;
    }
}

/// Optional bonsai smoke test. Skipped unless `BONSAI_GLB` env var points at
/// a real asset (the file is too large to vendor into git).
#[test]
fn decode_bonsai_when_available() {
    let Some(path) = std::env::var_os("BONSAI_GLB") else {
        eprintln!("skipping bonsai smoke (set BONSAI_GLB=<path> to enable)");
        return;
    };
    let bytes = fs::read(&path).expect("read bonsai");
    let verts = decode_glb_bytes(&bytes).expect("decode bonsai");
    assert!(
        verts.len() > 1000,
        "expected >1k splats, got {}",
        verts.len()
    );
}
