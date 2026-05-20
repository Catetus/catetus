//! SPEC-0013 — KHR_mesh_quantization integration tests.
//!
//! Executable acceptance criteria for SPEC-0013. See
//! [`specs/0013-gltf-mesh-quantization.md`](../../../specs/0013-gltf-mesh-quantization.md)
//! for the full design.

use std::path::Path;

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{read_gltf, write_gltf, WriteOpts};
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
/// When `WriteOpts::quantize == true` (the `web-mobile` / `quest-browser` /
/// `size-min` preset path), the POSITION accessor must report
/// `UNSIGNED_SHORT` with `normalized: true` and a min/max bracketing the bbox.
#[test]
fn web_mobile_position_is_uint16_normalized() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    let opts = WriteOpts {
        quantize: true,
        ..Default::default()
    };
    write_gltf(&scene, &path, &opts).expect("write");

    let g = read_gltf_json(&path);
    let pos = &g["accessors"][0];
    assert_eq!(
        pos["componentType"].as_u64().unwrap_or(0),
        5123,
        "POSITION should be UNSIGNED_SHORT (5123) when quantize=true"
    );
    assert!(
        pos["normalized"].as_bool().unwrap_or(false),
        "POSITION should be normalized for KHR_mesh_quantization dequant"
    );
    let min = pos["min"].as_array().expect("min");
    let max = pos["max"].as_array().expect("max");
    assert_eq!(min.len(), 3);
    assert_eq!(max.len(), 3);

    // `KHR_mesh_quantization` must be advertised as used (non-required).
    let used: Vec<String> = g["extensionsUsed"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        used.iter().any(|e| e == "KHR_mesh_quantization"),
        "quantize=true must advertise KHR_mesh_quantization, found: {used:?}"
    );
    let required: Vec<String> = g["extensionsRequired"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !required.iter().any(|e| e == "KHR_mesh_quantization"),
        "KHR_mesh_quantization must NOT be in extensionsRequired"
    );
}

/// SPEC-0013 §"Acceptance criteria" — Scenario 2.
///
/// `lossless-repack` (and the default) keeps f32 accessors and MUST NOT list
/// `KHR_mesh_quantization` in `extensionsUsed`.
#[test]
fn lossless_repack_keeps_f32() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    let opts = WriteOpts::default(); // quantize: false by default
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
/// Round-trip: positions must be preserved to ~1 / 32768 of the bbox span
/// (we accept up to 2× the theoretical step to absorb rounding).
#[test]
fn roundtrip_position_within_quantization_step() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = small_scene();
    let opts = WriteOpts {
        quantize: true,
        ..Default::default()
    };
    write_gltf(&scene, &path, &opts).expect("write");
    let decoded = read_gltf(&path).expect("read");
    assert_eq!(decoded.len(), scene.len());

    // small_scene positions span x ∈ [-0.5, 0.49], y ∈ [0, 0.2475],
    // z ∈ [0.005, 0.5]. The widest axis is x with span 0.99.
    let span = 1.0f32;
    let step = span / 32768.0;
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        for k in 0..3 {
            let delta = (a.position[k] - b.position[k]).abs();
            assert!(
                delta < step * 2.0,
                "splat {i} axis {k}: position drift {delta} exceeds 2 * quantization step {step}"
            );
        }
    }

    // Scales are 8-bit quantized — assert per-axis error bounded by 1/256 of
    // the per-axis scale range. The small_scene scale spans x ∈ [0.05, 0.07].
    let scale_step_x = (0.07f32 - 0.05f32) / 256.0;
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        let delta = (a.scale[0] - b.scale[0]).abs();
        assert!(
            delta < scale_step_x * 2.0,
            "splat {i} scale x drift {delta} exceeds 2 * 8-bit step {scale_step_x}"
        );
    }
}
