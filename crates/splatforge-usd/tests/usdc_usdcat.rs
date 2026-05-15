//! `usdcat` integration tests — gated by the `usdcat-validation` Cargo feature.
//!
//! Prove that Pixar's reference implementation accepts the binaries we emit.
//! We shell out to `usdcat <our.usdc> -o tmp.usda` and re-parse the result,
//! asserting the IR matches the source scene.

#![cfg(feature = "usdcat-validation")]

use std::process::Command;

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_usd::{read_usda, write_usdc, UsdWriteOpts};
use tempfile::tempdir;

fn require_usdcat() {
    let probe = Command::new("usdcat").arg("--help").output();
    assert!(
        probe.is_ok() && probe.unwrap().status.success(),
        "this test requires Pixar `usdcat` on PATH; install via `brew install usd-tools` or build OpenUSD"
    );
}

fn scene_with_orientation(rotation: [f32; 4]) -> SplatScene {
    let mut s = SplatScene::new();
    s.splats.push(Splat {
        position: [0.1, 0.2, 0.3],
        rotation,
        scale: [0.5, 0.5, 0.5],
        opacity: 0.42,
        color: Color::Rgb([0.7, 0.2, 0.1]),
    });
    s
}

#[test]
fn usdcat_can_read_our_usdc() {
    require_usdcat();
    let scene = scene_with_orientation([0.1, 0.2, 0.3, 0.9]);
    let dir = tempdir().unwrap();
    let usdc = dir.path().join("scene.usdc");
    let via = dir.path().join("via_usdcat.usda");
    write_usdc(&scene, &usdc, &UsdWriteOpts::default()).unwrap();
    let out = Command::new("usdcat")
        .arg(&usdc)
        .arg("-o")
        .arg(&via)
        .output()
        .expect("spawn usdcat");
    assert!(
        out.status.success(),
        "usdcat rejected our USDC: stdout={:?} stderr={:?}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let recovered = read_usda(&via).expect("parse usdcat's USDA");
    assert_eq!(recovered.len(), scene.len());
    for (a, b) in recovered.splats.iter().zip(scene.splats.iter()) {
        for k in 0..3 {
            assert!((a.position[k] - b.position[k]).abs() < 1e-4);
            assert!((a.scale[k] - b.scale[k]).abs() < 1e-4);
        }
        for k in 0..4 {
            assert!(
                (a.rotation[k] - b.rotation[k]).abs() < 1e-4,
                "rotation channel {k}: {} vs {}",
                a.rotation[k],
                b.rotation[k]
            );
        }
        assert!((a.opacity - b.opacity).abs() < 1e-4);
    }
}

#[test]
fn usdcat_can_read_dense_fixture() {
    require_usdcat();

    let mut scene = SplatScene::new();
    for i in 0..64 {
        let xi = i % 4;
        let yi = (i / 4) % 4;
        let zi = i / 16;
        scene.splats.push(Splat {
            position: [xi as f32, yi as f32, zi as f32],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.25, 0.25, 0.25],
            opacity: 0.5,
            color: Color::Rgb([
                (xi as f32) * 0.25,
                (yi as f32) * 0.25,
                (zi as f32) * 0.25,
            ]),
        });
    }
    let dir = tempdir().unwrap();
    let usdc = dir.path().join("dense.usdc");
    let via = dir.path().join("dense_via_usdcat.usda");
    write_usdc(&scene, &usdc, &UsdWriteOpts::default()).unwrap();
    let out = Command::new("usdcat")
        .arg(&usdc)
        .arg("-o")
        .arg(&via)
        .output()
        .expect("spawn usdcat");
    assert!(
        out.status.success(),
        "usdcat rejected our USDC: stderr={:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    let recovered = read_usda(&via).expect("parse usdcat's USDA");
    assert_eq!(recovered.len(), 64);
    for (i, (a, b)) in recovered.splats.iter().zip(scene.splats.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a.position[k] - b.position[k]).abs() < 1e-4,
                "splat {i} position mismatch"
            );
        }
    }
}
