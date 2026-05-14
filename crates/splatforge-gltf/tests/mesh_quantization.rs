//! SPEC-0013 — KHR_mesh_quantization integration tests.
//!
//! These tests fail today (the writer hasn't been wired yet) and serve as
//! the executable specification for the feature. They are marked `#[ignore]`
//! so the workspace test count stays at 43 green while they sit unimplemented;
//! flipping the ignore attribute drives the implementation.
//!
//! See [`specs/0013-gltf-mesh-quantization.md`](../../../specs/0013-gltf-mesh-quantization.md)
//! for the full design.

use std::path::Path;

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{read_gltf, write_gltf, WriteOpts};
use tempfile::tempdir;

fn small_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..100u32 {
        let f = i as f32 / 100.0;
        scene.splats.push(Splat {
            position: [f - 0.5, f * 0.25, (1.0 - f) * 0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.05 + f * 0.02, 0.05, 0.05],
            opacity: 0.2 + f * 0.6,
            color: Color::Rgb([f, 1.0 - f, 0.5]),
        });
    }
    scene
}

fn read_gltf_json(path: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(path).expect("read gltf");
    serde_json::from_str(&raw).expect("parse gltf json")
}

/// SPEC-0013 §"Acceptance criteria" — Scenario 1.
///
/// `web-mobile` is the preset that should opt in to quantized integer
/// accessors. The POSITION accessor must report `UNSIGNED_SHORT` with
/// `normalized: true` and a min/max bracketing the bbox.
#[test]
#[ignore = "blocked on SPEC-0013 implementation; see specs/0013-gltf-mesh-quantization.md"]
fn web_mobile_position_is_uint16_normalized() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    // Today this WriteOpts has no `quantize` field — implementing SPEC-0013
    // adds it and flips it on for the `web-mobile`/`size-min`/`quest-browser`
    // presets in splatforge-optimize.
    let opts = WriteOpts::default();
    write_gltf(&scene, &path, &opts).expect("write");

    let g = read_gltf_json(&path);
    let pos = &g["accessors"][0];
    assert_eq!(
        pos["componentType"].as_u64().unwrap_or(0),
        5123,
        "POSITION should be UNSIGNED_SHORT (5123) when quantize=true"
    );
    assert_eq!(
        pos["normalized"].as_bool().unwrap_or(false),
        true,
        "POSITION should be normalized for KHR_mesh_quantization dequant"
    );
    let min = pos["min"].as_array().expect("min");
    let max = pos["max"].as_array().expect("max");
    assert_eq!(min.len(), 3);
    assert_eq!(max.len(), 3);
}

/// SPEC-0013 §"Acceptance criteria" — Scenario 2.
///
/// `lossless-repack` must keep f32 accessors and MUST NOT list
/// `KHR_mesh_quantization` in `extensionsUsed`.
#[test]
#[ignore = "blocked on SPEC-0013 implementation; see specs/0013-gltf-mesh-quantization.md"]
fn lossless_repack_keeps_f32() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    let opts = WriteOpts::default();
    write_gltf(&scene, &path, &opts).expect("write");

    let g = read_gltf_json(&path);
    let pos = &g["accessors"][0];
    assert_eq!(
        pos["componentType"].as_u64().unwrap_or(0),
        5126,
        "lossless-repack POSITION must stay FLOAT (5126)"
    );
    let used: Vec<String> = g["extensionsUsed"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !used.iter().any(|e| e == "KHR_mesh_quantization"),
        "lossless-repack must not advertise KHR_mesh_quantization, found: {used:?}"
    );
}

/// SPEC-0013 §"Acceptance criteria" — Scenario 3.
///
/// Round-trip: positions must be preserved to 1 / 32768 of the bbox span.
#[test]
#[ignore = "blocked on SPEC-0013 implementation; see specs/0013-gltf-mesh-quantization.md"]
fn roundtrip_position_within_quantization_step() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    let opts = WriteOpts::default();
    write_gltf(&scene, &path, &opts).expect("write");
    let decoded = read_gltf(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());

    // The bbox of the small_scene.
    let span_x = (-0.5f32..0.49f32).end - (-0.5f32..0.49f32).start;
    let step = span_x / 32768.0;
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        for k in 0..3 {
            let delta = (a.position[k] - b.position[k]).abs();
            assert!(
                delta < step * 2.0,
                "splat {i} axis {k}: position drift {delta} exceeds 2 * quantization step {step}"
            );
        }
    }
}
