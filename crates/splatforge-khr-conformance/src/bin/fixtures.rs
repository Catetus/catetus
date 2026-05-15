//! `splatforge-khr-fixtures` — deterministic generator for the
//! `KHR_gaussian_splatting` conformance fixture corpus.
//!
//! Usage:
//!     splatforge-khr-fixtures <out_dir>
//!
//! Writes ten fixtures to `<out_dir>`:
//!
//!   01_valid_baseline.glb              valid 4-splat scene, FLOAT accessors
//!   02_valid_baseline.gltf             valid 4-splat scene, FLOAT accessors, ext .gltf
//!   03_valid_quantized.glb             KHR_mesh_quantization integer accessors
//!   04_valid_with_sh.glb               valid, includes _COLOR_SH accessor
//!   05_valid_spz_stub.glb              declares the SPZ sub-extension
//!   06_invalid_missing_ext_used.glb    extensionsUsed lacks KHR_gaussian_splatting
//!   07_invalid_no_rotation.gltf        attributes object missing _ROTATION
//!   08_invalid_rotation_vec3.gltf      _ROTATION accessor is VEC3 instead of VEC4
//!   09_invalid_position_no_minmax.gltf POSITION accessor missing min/max
//!   10_invalid_count_mismatch.gltf     per-splat accessors disagree on count
//!
//! The generator is deterministic: the same input always produces the same
//! bytes (no clocks, no entropy).
//!
//! The negative fixtures are produced by mutating a JSON document that was
//! itself emitted by `splatforge-gltf`, so the validator always operates on
//! realistic glTF shapes rather than synthetic stubs.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{write_glb, write_gltf, WriteOpts};

fn deterministic_scene(n: usize, with_sh: bool) -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..n {
        let f = i as f32;
        let color = if with_sh {
            // 48 = 3 DC + 45 SH coeffs (degree 3).
            let mut coeffs = Vec::with_capacity(48);
            coeffs.push((f * 0.1).fract().abs());
            coeffs.push(0.2);
            coeffs.push(0.3);
            for j in 0..45 {
                coeffs.push((f + j as f32) * 0.001);
            }
            Color::Sh { degree: 3, coeffs }
        } else {
            Color::Rgb([(f * 0.1).fract().abs(), 0.2, 0.3])
        };
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.5,
            color,
        });
    }
    scene
}

fn write_negative_gltf(path: &Path, mutate: impl FnOnce(&mut Value)) -> std::io::Result<()> {
    // Build a valid baseline in an isolated staging directory so we never
    // touch the canonical baseline's `buffers/` sidecar. Then mutate the
    // staged JSON and write only the .gltf to the destination — the negative
    // fixture is validator-input only, the external buffer is unused.
    let staging_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            "_staging_{}",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("neg")
        ));
    let _ = fs::remove_dir_all(&staging_dir);
    fs::create_dir_all(&staging_dir)?;
    let staged = staging_dir.join("scene.gltf");
    let scene = deterministic_scene(4, false);
    write_gltf(&scene, &staged, &WriteOpts::default()).expect("baseline write");
    let raw = fs::read_to_string(&staged)?;
    let _ = fs::remove_dir_all(&staging_dir);

    let mut value: Value = serde_json::from_str(&raw).expect("parse baseline");
    mutate(&mut value);
    let pretty = serde_json::to_string_pretty(&value).expect("ser");
    fs::write(path, pretty)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("fixtures"));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // 01: valid GLB baseline.
    {
        let scene = deterministic_scene(4, false);
        let p = out_dir.join("01_valid_baseline.glb");
        write_glb(&scene, &p, &WriteOpts::default()).expect("write 01");
    }

    // 02: valid .gltf baseline (with external buffer).
    {
        let scene = deterministic_scene(4, false);
        let p = out_dir.join("02_valid_baseline.gltf");
        write_gltf(&scene, &p, &WriteOpts::default()).expect("write 02");
    }

    // 03: valid quantized GLB.
    {
        let scene = deterministic_scene(4, false);
        let p = out_dir.join("03_valid_quantized.glb");
        let opts = WriteOpts {
            quantize: true,
            ..Default::default()
        };
        write_glb(&scene, &p, &opts).expect("write 03");
    }

    // 04: valid GLB with SH.
    {
        let scene = deterministic_scene(4, true);
        let p = out_dir.join("04_valid_with_sh.glb");
        write_glb(&scene, &p, &WriteOpts::default()).expect("write 04");
    }

    // 05: valid GLB that *also* declares KHR_gaussian_splatting_compression_spz.
    //     This fixture is a structural test of the SPZ clauses — the actual
    //     SPZ-compressed binary format is not yet RC. We declare the
    //     extension on both extensionsUsed and the primitive so the validator
    //     can prove the consistency check fires correctly.
    {
        // Reuse the baseline-glb pipeline, then rewrite its JSON chunk.
        let scene = deterministic_scene(4, false);
        let staging = out_dir.join("05_staging.glb");
        write_glb(&scene, &staging, &WriteOpts::default()).expect("write 05 staging");
        let bytes = fs::read(&staging).expect("read 05 staging");
        let _ = fs::remove_file(&staging);
        let bytes = rewrite_glb_json(&bytes, |v| {
            let used = v
                .get_mut("extensionsUsed")
                .and_then(|a| a.as_array_mut())
                .expect("extensionsUsed array");
            used.push(Value::String(
                "KHR_gaussian_splatting_compression_spz".to_string(),
            ));
            let prim_ext = v
                .get_mut("meshes")
                .and_then(|m| m.get_mut(0))
                .and_then(|m| m.get_mut("primitives"))
                .and_then(|p| p.get_mut(0))
                .and_then(|p| p.get_mut("extensions"))
                .and_then(|e| e.as_object_mut())
                .expect("primitive extensions");
            prim_ext.insert(
                "KHR_gaussian_splatting_compression_spz".to_string(),
                serde_json::json!({ "bufferView": 0 }),
            );
        });
        fs::write(out_dir.join("05_valid_spz_stub.glb"), bytes).expect("write 05");
    }

    // 06: invalid GLB — extensionsUsed lacks KHR_gaussian_splatting.
    {
        let scene = deterministic_scene(4, false);
        let staging = out_dir.join("06_staging.glb");
        write_glb(&scene, &staging, &WriteOpts::default()).expect("write 06 staging");
        let bytes = fs::read(&staging).expect("read 06 staging");
        let _ = fs::remove_file(&staging);
        let bytes = rewrite_glb_json(&bytes, |v| {
            v["extensionsUsed"] = serde_json::json!([]);
        });
        fs::write(out_dir.join("06_invalid_missing_ext_used.glb"), bytes).expect("write 06");
    }

    // 07: invalid .gltf — _ROTATION attribute removed.
    write_negative_gltf(&out_dir.join("07_invalid_no_rotation.gltf"), |v| {
        let attrs = v["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            ["attributes"]
            .as_object_mut()
            .expect("attrs");
        attrs.remove("_ROTATION");
    })
    .expect("write 07");

    // 08: invalid .gltf — _ROTATION accessor type set to VEC3 instead of VEC4.
    write_negative_gltf(&out_dir.join("08_invalid_rotation_vec3.gltf"), |v| {
        let rot_idx = v["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            ["attributes"]["_ROTATION"]
            .as_u64()
            .expect("rot idx") as usize;
        v["accessors"][rot_idx]["type"] = Value::String("VEC3".to_string());
    })
    .expect("write 08");

    // 09: invalid .gltf — POSITION accessor missing min/max.
    write_negative_gltf(&out_dir.join("09_invalid_position_no_minmax.gltf"), |v| {
        let pos_idx = v["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            ["attributes"]["POSITION"]
            .as_u64()
            .expect("pos idx") as usize;
        let acc = v["accessors"][pos_idx].as_object_mut().expect("acc obj");
        acc.remove("min");
        acc.remove("max");
    })
    .expect("write 09");

    // 10: invalid .gltf — per-splat accessor counts disagree.
    write_negative_gltf(&out_dir.join("10_invalid_count_mismatch.gltf"), |v| {
        let op_idx = v["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            ["attributes"]["_OPACITY"]
            .as_u64()
            .expect("op idx") as usize;
        v["accessors"][op_idx]["count"] = serde_json::json!(7);
    })
    .expect("write 10");

    println!("wrote 10 fixtures to {}", out_dir.display());
}

/// Rewrite the JSON chunk of a GLB file, leaving the BIN chunk untouched.
/// Re-emits the GLB header with the new total length.
fn rewrite_glb_json(bytes: &[u8], mutate: impl FnOnce(&mut Value)) -> Vec<u8> {
    assert!(bytes.len() >= 12, "glb too small");
    assert_eq!(&bytes[..4], b"glTF");
    let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    // First chunk is JSON.
    let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let json_ty = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert_eq!(json_ty, 0x4E4F_534A, "expected JSON chunk first");
    let json_start = 20usize;
    let json_end = json_start + json_len;
    let bin_start = json_end;
    let bin_chunk = &bytes[bin_start..total];

    let raw_json = {
        let mut end = json_end;
        while end > json_start && (bytes[end - 1] == b' ' || bytes[end - 1] == 0) {
            end -= 1;
        }
        std::str::from_utf8(&bytes[json_start..end])
            .expect("utf8")
            .to_string()
    };
    let mut value: Value = serde_json::from_str(&raw_json).expect("parse json chunk");
    mutate(&mut value);
    let new_json = serde_json::to_string(&value).expect("ser json");
    let mut new_json_bytes = new_json.into_bytes();
    while new_json_bytes.len() % 4 != 0 {
        new_json_bytes.push(b' ');
    }

    let new_total = 12 + 8 + new_json_bytes.len() + bin_chunk.len();
    let mut out = Vec::with_capacity(new_total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(new_total as u32).to_le_bytes());
    out.extend_from_slice(&(new_json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&new_json_bytes);
    out.extend_from_slice(bin_chunk);
    out
}
