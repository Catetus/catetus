//! `splatforge-khr-fixtures` — deterministic generator for the
//! `KHR_gaussian_splatting` conformance fixture corpus.
//!
//! The corpus targets the Release Candidate spec at
//! `extensions/2.0/Khronos/KHR_gaussian_splatting` (commit
//! `63770cc70a3709cf101a42cece0bdf602b37e2e7`, dated 2026-04-15 — the
//! "Editorial review" merge that defines the RC attribute layout used here:
//! a `points` primitive, KHR-namespaced semantics (`KHR_gaussian_splatting:ROTATION`,
//! `:SCALE`, `:OPACITY`, `:SH_DEGREE_0_COEF_0`, ...), and a required
//! `kernel` + `colorSpace` on the extension object.
//!
//! Usage:
//!     splatforge-khr-fixtures <out_dir>
//!
//! Writes ten fixtures to `<out_dir>`:
//!
//!   01_valid_baseline.glb              valid 4-splat scene, FLOAT accessors
//!   02_valid_baseline.gltf             valid 4-splat scene, FLOAT accessors, ext .gltf
//!   03_valid_quantized.glb             KHR_mesh_quantization integer ROTATION + SCALE
//!   04_valid_with_sh.glb               valid, includes SH degree-1 accessors
//!   05_valid_default_methods.glb       valid, omits optional projection+sortingMethod
//!   06_invalid_missing_ext_used.glb    extensionsUsed lacks KHR_gaussian_splatting
//!   07_invalid_no_rotation.gltf        attributes object missing ROTATION
//!   08_invalid_rotation_vec3.gltf      ROTATION accessor is VEC3 instead of VEC4
//!   09_invalid_position_no_minmax.gltf POSITION accessor missing min/max
//!   10_invalid_count_mismatch.gltf     per-splat accessors disagree on count
//!
//! The generator is deterministic: the same input always produces the same
//! bytes (no clocks, no entropy). All fixtures are emitted as synthetic
//! glTF JSON — they intentionally avoid the SplatForge writer, since the
//! writer still emits a pre-RC attribute layout. Fixtures are normative for
//! the validator, not vice versa.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

const N_SPLATS: usize = 4;

/// Returns a deterministic FLOAT-accessor 4-splat document. Buffer layout:
///   POSITION    VEC3 FLOAT  4*12 = 48
///   ROTATION    VEC4 FLOAT  4*16 = 64
///   SCALE       VEC3 FLOAT  4*12 = 48
///   OPACITY     SCAL FLOAT  4*4  = 16
///   SH_DC       VEC3 FLOAT  4*12 = 48
/// Total: 224 bytes.
fn baseline_json(buffer_uri: Option<&str>) -> Value {
    let mut buffer = json!({ "byteLength": 224 });
    if let Some(uri) = buffer_uri {
        buffer["uri"] = Value::String(uri.to_string());
    }
    json!({
        "asset": { "version": "2.0", "generator": "splatforge-khr-fixtures" },
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [buffer],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0,   "byteLength": 48 },
            { "buffer": 0, "byteOffset": 48,  "byteLength": 64 },
            { "buffer": 0, "byteOffset": 112, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 160, "byteLength": 16 },
            { "buffer": 0, "byteOffset": 176, "byteLength": 48 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": N_SPLATS, "type": "VEC3",
              "min": [0.0, 0.0, -3.0], "max": [3.0, 1.5, 0.0] },
            { "bufferView": 1, "componentType": 5126, "count": N_SPLATS, "type": "VEC4" },
            { "bufferView": 2, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" },
            { "bufferView": 3, "componentType": 5126, "count": N_SPLATS, "type": "SCALAR" },
            { "bufferView": 4, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": {
                    "POSITION": 0,
                    "KHR_gaussian_splatting:ROTATION": 1,
                    "KHR_gaussian_splatting:SCALE": 2,
                    "KHR_gaussian_splatting:OPACITY": 3,
                    "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "kernel": "ellipse",
                        "colorSpace": "srgb_rec709_display",
                        "projection": "perspective",
                        "sortingMethod": "cameraDistance"
                    }
                }
            }]
        }],
        "scenes": [{ "nodes": [0] }],
        "scene": 0,
        "nodes": [{ "mesh": 0 }]
    })
}

/// Deterministic FLOAT byte buffer for the baseline scene.
/// Layout matches `baseline_json`'s bufferViews.
fn baseline_bin() -> Vec<u8> {
    let mut out = Vec::with_capacity(224);
    // POSITION
    for i in 0..N_SPLATS {
        let f = i as f32;
        for c in [f, f * 0.5, -f] {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    // ROTATION (identity quaternion, xyzw)
    for _ in 0..N_SPLATS {
        for c in [0.0_f32, 0.0, 0.0, 1.0] {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    // SCALE
    for _ in 0..N_SPLATS {
        for c in [1.0_f32, 1.0, 1.0] {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    // OPACITY
    for _ in 0..N_SPLATS {
        out.extend_from_slice(&0.5_f32.to_le_bytes());
    }
    // SH_DEGREE_0_COEF_0 (VEC3 FLOAT, diffuse-ish constant)
    for _ in 0..N_SPLATS {
        for c in [0.1_f32, 0.2, 0.3] {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    assert_eq!(out.len(), 224);
    out
}

/// Wrap a JSON document + binary chunk into a single .glb (binary glTF).
fn pack_glb(json_value: &Value, bin: &[u8]) -> Vec<u8> {
    // Stringify without trailing newline; padding to 4-byte alignment with
    // spaces (JSON chunk) / zeros (BIN chunk) is part of the GLB spec.
    let mut json_bytes = serde_json::to_vec(json_value).expect("ser json");
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let mut bin_padded = bin.to_vec();
    while bin_padded.len() % 4 != 0 {
        bin_padded.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin_padded.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin_padded.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin_padded);
    out
}

fn write_glb_with_mutator(path: &Path, mutate: impl FnOnce(&mut Value)) -> std::io::Result<()> {
    let mut value = baseline_json(None);
    mutate(&mut value);
    let bin = baseline_bin();
    let bytes = pack_glb(&value, &bin);
    fs::write(path, bytes)
}

fn write_gltf_with_mutator(path: &Path, mutate: impl FnOnce(&mut Value)) -> std::io::Result<()> {
    let buffer_uri = "buffers/chunk_0000.bin";
    let mut value = baseline_json(Some(buffer_uri));
    mutate(&mut value);
    let pretty = serde_json::to_string_pretty(&value).expect("ser");
    fs::write(path, pretty)?;
    // Sidecar bin (write only once per directory).
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let bin_path = parent.join(buffer_uri);
    if let Some(p) = bin_path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(bin_path, baseline_bin())?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("fixtures"));
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // 01: valid GLB baseline (FLOAT, all optional ext fields populated).
    write_glb_with_mutator(&out_dir.join("01_valid_baseline.glb"), |_| {}).expect("write 01");

    // 02: valid external .gltf baseline + buffers/chunk_0000.bin sidecar.
    write_gltf_with_mutator(&out_dir.join("02_valid_baseline.gltf"), |_| {}).expect("write 02");

    // 03: valid GLB with KHR_mesh_quantization — ROTATION as normalized
    //     signed short, SCALE as unsigned byte. We rewrite the entire
    //     buffer + accessor block to satisfy buffer-view bounds.
    {
        // Layout (quantized):
        //   POSITION    VEC3 FLOAT          4*12 = 48
        //   ROTATION    VEC4 SHORT(norm)    4*8  = 32
        //   SCALE       VEC3 UBYTE(norm)    4*3  = 12  (pad to 16)
        //   OPACITY     SCAL UBYTE(norm)    4*1  = 4   (pad to 4)
        //   SH_DC       VEC3 FLOAT          4*12 = 48
        // Total padded: 48 + 32 + 16 + 4 + 48 = 148. Round buffer to 148.
        let mut value = baseline_json(None);
        value["extensionsUsed"] = json!([
            "KHR_gaussian_splatting",
            "KHR_mesh_quantization"
        ]);
        value["buffers"][0]["byteLength"] = json!(148);
        value["bufferViews"] = json!([
            { "buffer": 0, "byteOffset": 0,   "byteLength": 48 },
            { "buffer": 0, "byteOffset": 48,  "byteLength": 32 },
            { "buffer": 0, "byteOffset": 80,  "byteLength": 16 },
            { "buffer": 0, "byteOffset": 96,  "byteLength": 4  },
            { "buffer": 0, "byteOffset": 100, "byteLength": 48 }
        ]);
        value["accessors"] = json!([
            { "bufferView": 0, "componentType": 5126, "count": N_SPLATS, "type": "VEC3",
              "min": [0.0, 0.0, -3.0], "max": [3.0, 1.5, 0.0] },
            { "bufferView": 1, "componentType": 5122, "count": N_SPLATS, "type": "VEC4",
              "normalized": true },
            { "bufferView": 2, "componentType": 5121, "count": N_SPLATS, "type": "VEC3",
              "normalized": true },
            { "bufferView": 3, "componentType": 5121, "count": N_SPLATS, "type": "SCALAR",
              "normalized": true },
            { "bufferView": 4, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" }
        ]);
        // Build a matching binary buffer.
        let mut bin = Vec::with_capacity(148);
        for i in 0..N_SPLATS {
            let f = i as f32;
            for c in [f, f * 0.5, -f] {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        // ROTATION (normalized signed shorts representing identity quat).
        for _ in 0..N_SPLATS {
            for c in [0_i16, 0, 0, i16::MAX] {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        // SCALE (normalized unsigned bytes — pad each VEC3 to 4 bytes).
        for _ in 0..N_SPLATS {
            bin.extend_from_slice(&[200u8, 200, 200, 0]);
        }
        // OPACITY (normalized unsigned byte) — pad each scalar group to 4.
        for _ in 0..N_SPLATS {
            bin.push(127);
        }
        // SH_DC.
        for _ in 0..N_SPLATS {
            for c in [0.1_f32, 0.2, 0.3] {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        // Trim/zero-pad to buffer.byteLength.
        while bin.len() < 148 {
            bin.push(0);
        }
        bin.truncate(148);
        let bytes = pack_glb(&value, &bin);
        fs::write(out_dir.join("03_valid_quantized.glb"), bytes).expect("write 03");
    }

    // 04: valid GLB with SH degree-1 accessors (4 extra VEC3 FLOAT accessors).
    {
        // Add SH_DEGREE_1_COEF_0/_1/_2 each VEC3 FLOAT 4*12=48 → 3*48=144 extra.
        // Baseline is 224; total 224+144=368. Each new bufferView is 48 bytes.
        let mut value = baseline_json(None);
        value["buffers"][0]["byteLength"] = json!(368);
        // Append three bufferViews: starting at 224, 272, 320.
        let bvs = value["bufferViews"].as_array_mut().unwrap();
        bvs.push(json!({ "buffer": 0, "byteOffset": 224, "byteLength": 48 }));
        bvs.push(json!({ "buffer": 0, "byteOffset": 272, "byteLength": 48 }));
        bvs.push(json!({ "buffer": 0, "byteOffset": 320, "byteLength": 48 }));
        let accs = value["accessors"].as_array_mut().unwrap();
        accs.push(json!({ "bufferView": 5, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" }));
        accs.push(json!({ "bufferView": 6, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" }));
        accs.push(json!({ "bufferView": 7, "componentType": 5126, "count": N_SPLATS, "type": "VEC3" }));
        let attrs = value["meshes"][0]["primitives"][0]["attributes"]
            .as_object_mut()
            .unwrap();
        attrs.insert("KHR_gaussian_splatting:SH_DEGREE_1_COEF_0".into(), json!(5));
        attrs.insert("KHR_gaussian_splatting:SH_DEGREE_1_COEF_1".into(), json!(6));
        attrs.insert("KHR_gaussian_splatting:SH_DEGREE_1_COEF_2".into(), json!(7));

        // Build buffer.
        let mut bin = baseline_bin();
        bin.reserve(144);
        for coef in 0..3 {
            for splat in 0..N_SPLATS {
                let f = splat as f32 + coef as f32 * 0.01;
                for c in [f * 0.01, f * 0.02, f * 0.03] {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        assert_eq!(bin.len(), 368);
        let bytes = pack_glb(&value, &bin);
        fs::write(out_dir.join("04_valid_with_sh.glb"), bytes).expect("write 04");
    }

    // 05: valid GLB that omits optional projection/sortingMethod fields,
    //     exercising the "default" branch of EXT_PROJECTION / EXT_SORTING.
    write_glb_with_mutator(&out_dir.join("05_valid_default_methods.glb"), |v| {
        let ext = v["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
            .as_object_mut()
            .unwrap();
        ext.remove("projection");
        ext.remove("sortingMethod");
    })
    .expect("write 05");

    // 06: invalid GLB — extensionsUsed lacks KHR_gaussian_splatting.
    write_glb_with_mutator(&out_dir.join("06_invalid_missing_ext_used.glb"), |v| {
        v["extensionsUsed"] = json!([]);
    })
    .expect("write 06");

    // 07: invalid .gltf — KHR_gaussian_splatting:ROTATION attribute removed.
    write_gltf_with_mutator(&out_dir.join("07_invalid_no_rotation.gltf"), |v| {
        let attrs = v["meshes"][0]["primitives"][0]["attributes"]
            .as_object_mut()
            .expect("attrs");
        attrs.remove("KHR_gaussian_splatting:ROTATION");
    })
    .expect("write 07");

    // 08: invalid .gltf — ROTATION accessor type set to VEC3 instead of VEC4.
    write_gltf_with_mutator(&out_dir.join("08_invalid_rotation_vec3.gltf"), |v| {
        let rot_idx = v["meshes"][0]["primitives"][0]["attributes"]
            ["KHR_gaussian_splatting:ROTATION"]
            .as_u64()
            .expect("rot idx") as usize;
        v["accessors"][rot_idx]["type"] = Value::String("VEC3".to_string());
    })
    .expect("write 08");

    // 09: invalid .gltf — POSITION accessor missing min/max.
    write_gltf_with_mutator(&out_dir.join("09_invalid_position_no_minmax.gltf"), |v| {
        let pos_idx = v["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
            .as_u64()
            .expect("pos idx") as usize;
        let acc = v["accessors"][pos_idx].as_object_mut().expect("acc obj");
        acc.remove("min");
        acc.remove("max");
    })
    .expect("write 09");

    // 10: invalid .gltf — per-splat accessor counts disagree.
    write_gltf_with_mutator(&out_dir.join("10_invalid_count_mismatch.gltf"), |v| {
        let op_idx = v["meshes"][0]["primitives"][0]["attributes"]
            ["KHR_gaussian_splatting:OPACITY"]
            .as_u64()
            .expect("op idx") as usize;
        v["accessors"][op_idx]["count"] = json!(7);
    })
    .expect("write 10");

    println!("wrote 10 fixtures to {}", out_dir.display());
}
