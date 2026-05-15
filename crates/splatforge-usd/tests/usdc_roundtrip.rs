#![allow(clippy::needless_range_loop)]
//! In-process USDC round-trip tests.
//!
//! These tests do not require Pixar's `usdcat` to be installed. They write
//! USDC bytes with our encoder and decode them with our reader, asserting
//! that the IR survives the trip within 1e-5 per component.
//!
//! The "real" conformance test — proving that *Pixar* can read what we
//! emit — is in `usdc_usdcat.rs` behind the `usdcat-validation` feature.

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_usd::{read_usdc, write_usdc, UsdWriteOpts};
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
fn roundtrip_three_splats_usdc() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usdc");
    let scene = three_splat_scene();
    write_usdc(&scene, &path, &UsdWriteOpts::default()).expect("write");
    let decoded = read_usdc(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a.position[k] - b.position[k]).abs() < 1e-5,
                "splat {i} position mismatch on axis {k}: {:?} vs {:?}",
                a.position,
                b.position
            );
            assert!(
                (a.scale[k] - b.scale[k]).abs() < 1e-5,
                "splat {i} scale mismatch on axis {k}"
            );
        }
        for k in 0..4 {
            assert!(
                (a.rotation[k] - b.rotation[k]).abs() < 1e-5,
                "splat {i} rotation mismatch on component {k}: {:?} vs {:?}",
                a.rotation,
                b.rotation
            );
        }
        assert!(
            (a.opacity - b.opacity).abs() < 1e-5,
            "splat {i} opacity mismatch"
        );
        match (&a.color, &b.color) {
            (Color::Rgb(ac), Color::Rgb(bc)) => {
                for k in 0..3 {
                    assert!(
                        (ac[k] - bc[k]).abs() < 1e-5,
                        "splat {i} color mismatch on channel {k}"
                    );
                }
            }
            _ => panic!("splat {i} color variant mismatch"),
        }
    }
}

#[test]
fn quat_roundtrip_preserves_xyzw() {
    let mut scene = SplatScene::new();
    scene.splats.push(Splat {
        position: [0.0, 0.0, 0.0],
        rotation: [0.1, 0.2, 0.3, 0.9],
        scale: [1.0, 1.0, 1.0],
        opacity: 1.0,
        color: Color::Rgb([0.0; 3]),
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usdc");
    write_usdc(&scene, &path, &UsdWriteOpts::default()).unwrap();
    let decoded = read_usdc(&path).unwrap();
    let q = decoded.splats[0].rotation;
    assert!((q[0] - 0.1).abs() < 1e-5);
    assert!((q[1] - 0.2).abs() < 1e-5);
    assert!((q[2] - 0.3).abs() < 1e-5);
    assert!((q[3] - 0.9).abs() < 1e-5);
}

#[test]
fn usdc_magic_and_version_are_001() {
    let scene = three_splat_scene();
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usdc");
    write_usdc(&scene, &path, &UsdWriteOpts::default()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"PXR-USDC");
    assert_eq!(bytes[8], 0, "major");
    assert_eq!(bytes[9], 0, "minor");
    assert_eq!(bytes[10], 1, "patch");
    let toc = i64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    assert!(toc > 88 && toc < bytes.len());
}

#[test]
fn empty_scene_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usdc");
    let scene = SplatScene::new();
    let err = write_usdc(&scene, &path, &UsdWriteOpts::default()).unwrap_err();
    assert!(matches!(err, splatforge_usd::UsdError::Malformed(_)));
}

#[test]
fn sh_scene_roundtrip() {
    let mut scene = SplatScene::new();
    let mut coeffs = vec![0.0f32; 48];
    coeffs[0] = 0.4;
    coeffs[1] = 0.5;
    coeffs[2] = 0.6;
    for i in 3..48 {
        coeffs[i] = (i as f32) * 0.01;
    }
    scene.splats.push(Splat {
        position: [1.0, 2.0, 3.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.5, 0.5, 0.5],
        opacity: 0.75,
        color: Color::Sh { degree: 3, coeffs },
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.usdc");
    write_usdc(&scene, &path, &UsdWriteOpts::default()).unwrap();
    let decoded = read_usdc(&path).unwrap();
    match &decoded.splats[0].color {
        Color::Sh { degree, coeffs } => {
            assert_eq!(*degree, 3);
            assert_eq!(coeffs.len(), 48);
            assert!((coeffs[0] - 0.4).abs() < 1e-5);
            assert!((coeffs[47] - 0.47).abs() < 1e-5);
        }
        _ => panic!("expected SH color"),
    }
}

#[test]
fn deterministic_encoding() {
    // Two encodes of the same scene must produce byte-identical output —
    // no HashMap iteration leaks (cf. the determinism rule in CLAUDE.md).
    let scene = three_splat_scene();
    let dir1 = tempdir().unwrap();
    let dir2 = tempdir().unwrap();
    let p1 = dir1.path().join("a.usdc");
    let p2 = dir2.path().join("b.usdc");
    write_usdc(&scene, &p1, &UsdWriteOpts::default()).unwrap();
    write_usdc(&scene, &p2, &UsdWriteOpts::default()).unwrap();
    let b1 = std::fs::read(&p1).unwrap();
    let b2 = std::fs::read(&p2).unwrap();
    assert_eq!(b1, b2, "USDC encoder must be deterministic");
}
