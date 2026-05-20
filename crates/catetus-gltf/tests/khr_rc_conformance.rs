//! Live integration of `catetus-gltf::write_glb` / `write_gltf` against
//! the `catetus-khr-conformance` validator. Locks in that the production
//! writer emits output that satisfies the Khronos RC text at commit
//! `63770cc7` (2026-04-15).
//!
//! Before this test landed, the conformance crate validated *its own* hand-
//! built JSON fixtures and the writer was only ever exercised by an opaque
//! round-trip — so a regression in the writer (e.g. dropping the namespaced
//! `KHR_gaussian_splatting:` prefix) would not have been caught.

use std::collections::HashMap;

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{write_glb, write_gltf, SpecVersion, WriteOpts};
use catetus_khr_conformance::{validate_path, Status};
use tempfile::tempdir;

fn deterministic_scene(n: usize, with_sh: bool) -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..n {
        let f = i as f32 / (n.max(2) - 1) as f32;
        let color = if with_sh {
            // 48 = 3 DC + 45 SH scalars. Fill with monotone ramps so the
            // round-trip equality check downstream stays meaningful.
            let mut coeffs = Vec::with_capacity(48);
            coeffs.extend_from_slice(&[f, 1.0 - f, 0.5]);
            for k in 0..45 {
                coeffs.push((k as f32) * 0.001 + f * 0.01);
            }
            Color::Sh { degree: 3, coeffs }
        } else {
            Color::Rgb([f, 1.0 - f, 0.5])
        };
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.05, 0.05, 0.05],
            opacity: 0.2 + 0.6 * f,
            color,
        });
    }
    scene
}

fn assert_all_clauses_pass(path: &std::path::Path) {
    let report = validate_path(path).expect("validator ran");
    let failures: Vec<_> = report
        .clauses
        .iter()
        .filter(|c| matches!(c.status, Status::Fail))
        .collect();
    assert!(
        failures.is_empty(),
        "writer output at {} failed clauses: {:#?}",
        path.display(),
        failures
    );
    // Sanity: at least PASS the structural clauses.
    let by_id: HashMap<_, _> = report
        .clauses
        .iter()
        .map(|c| (c.id.as_str(), &c.status))
        .collect();
    for required in [
        "EXT_USED",
        "ASSET_VERSION",
        "PRIM_EXT",
        "PRIM_MODE_POINTS",
        "EXT_KERNEL",
        "EXT_COLOR_SPACE",
        "ATTR_POSITION",
        "ATTR_ROTATION",
        "ATTR_SCALE",
        "ATTR_OPACITY",
        "ATTR_SH_DC",
        "ACC_POSITION_MINMAX",
        "ACC_COUNTS_AGREE",
        "BUFFERVIEW_BOUNDS",
        "ATTRS_KNOWN_ONLY",
    ] {
        assert_eq!(
            by_id.get(required).copied(),
            Some(&Status::Pass),
            "clause {required} did not pass for {}",
            path.display()
        );
    }
}

#[test]
fn rc_default_glb_passes_validator() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(8, false);
    write_glb(&scene, &path, &WriteOpts::default()).expect("write");
    assert_all_clauses_pass(&path);
}

#[test]
fn rc_default_gltf_passes_validator() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = deterministic_scene(6, false);
    write_gltf(&scene, &path, &WriteOpts::default()).expect("write");
    assert_all_clauses_pass(&path);
}

#[test]
fn rc_quantized_glb_passes_validator() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(8, false);
    let opts = WriteOpts {
        quantize: true,
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write");
    assert_all_clauses_pass(&path);
}

#[test]
fn rc_quantized_rotation_short_passes_validator() {
    // Exercises the RC quaternion-quantization table: normalized signed
    // SHORT for KHR_gaussian_splatting:ROTATION.
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(8, false);
    let opts = WriteOpts {
        quantize: true,
        quantize_rotation: true,
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write");
    assert_all_clauses_pass(&path);
}

#[test]
fn rc_sh_glb_passes_validator() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(4, true);
    write_glb(&scene, &path, &WriteOpts::default()).expect("write");
    let report = validate_path(&path).expect("validator ran");
    let by_id: HashMap<_, _> = report
        .clauses
        .iter()
        .map(|c| (c.id.as_str(), &c.status))
        .collect();
    // With SH present, ACC_SH_COEF and SH_DEGREES_FULL must be evaluated
    // (not skipped) and must pass.
    assert_eq!(by_id.get("ACC_SH_COEF").copied(), Some(&Status::Pass));
    assert_eq!(by_id.get("SH_DEGREES_FULL").copied(), Some(&Status::Pass));
    assert_eq!(
        report
            .clauses
            .iter()
            .filter(|c| matches!(c.status, Status::Fail))
            .count(),
        0,
        "SH GLB had failing clauses: {:#?}",
        report.clauses
    );
}

#[test]
fn rc_writer_roundtrip_preserves_splats() {
    // Writing under RC and reading back must reproduce the input scene
    // within FLOAT precision (no quantization).
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(6, false);
    write_glb(&scene, &path, &WriteOpts::default()).expect("write");
    let decoded = catetus_gltf::read_glb(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-5);
            assert!((a.scale[i] - b.scale[i]).abs() < 1e-5);
        }
        assert!((a.opacity - b.opacity).abs() < 1e-5);
    }
}

#[test]
fn pre_rc_writer_roundtrip_preserves_splats() {
    // Backwards-compat path: writing under Pre2026 still round-trips through
    // the reader (which sniffs the schema on load).
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(6, true);
    let opts = WriteOpts {
        spec_version: SpecVersion::Pre2026,
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write");
    let decoded = catetus_gltf::read_glb(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-5);
        }
    }
}

#[test]
fn rc_sh_roundtrip_preserves_coefficients() {
    // The biggest schema delta vs pre-RC is SH layout: VEC3 FLOAT per
    // coefficient instead of one SCALAR-of-45 blob. Round-trip the full
    // 48-scalar coefficient vector to make sure the per-coef packing /
    // unpacking stays in sync.
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = deterministic_scene(3, true);
    write_glb(&scene, &path, &WriteOpts::default()).expect("write");
    let decoded = catetus_gltf::read_glb(&path).expect("read");
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        match (&a.color, &b.color) {
            (Color::Sh { coeffs: ac, .. }, Color::Sh { coeffs: bc, .. }) => {
                assert_eq!(ac.len(), bc.len(), "splat {i}: coeff count mismatch");
                for (k, (av, bv)) in ac.iter().zip(bc.iter()).enumerate() {
                    assert!((av - bv).abs() < 1e-5, "splat {i} coef {k}: {av} vs {bv}");
                }
            }
            _ => panic!("splat {i}: color variant mismatch"),
        }
    }
}
