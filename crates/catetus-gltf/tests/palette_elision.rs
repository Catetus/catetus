//! Palette elision tests — verify that a `CT_gaussian_splatting_palette`-
//! aware `WriteOpts` produces a GLB with NO `SH_DEGREE_l_COEF_n` accessors
//! for l>=1, the root extension is populated with the sidecar pointer, and
//! the GLB still round-trips through `read_glb` (without SH-rest values,
//! since the writer suppressed those bytes intentionally — the decoder must
//! load the sidecar to recover them).

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{
    read_glb, read_glb_with_opts, write_glb, GltfError, ReadOpts, ShRestPaletteRef, WriteOpts,
};
use std::io::Write;
use tempfile::tempdir;

/// Synthesize a deterministic K=4 sh=3 8-bit `.shpal` sidecar. Mirrors
/// `synth_shpal_bytes` in `sf_extension_decoders.rs` but lives here so the
/// palette-elision suite is self-contained. Returns
/// `(zstd-compressed bytes, reconstructed codebook floats, per-splat indices)`.
fn synth_shpal_bytes(k: usize, n: usize, sh_dim: usize) -> (Vec<u8>, Vec<f32>, Vec<u16>) {
    let ranges: Vec<f32> = (0..sh_dim).map(|d| 0.1 + (d as f32) * 0.01).collect();
    let mut q_codebook = Vec::with_capacity(k * sh_dim);
    for c in 0..k {
        for d in 0..sh_dim {
            let q: i32 = (((c * 13 + d * 7) % 127) as i32) - 63;
            q_codebook.push(q.clamp(-127, 127) as i8);
        }
    }
    let codebook_f: Vec<f32> = (0..k * sh_dim)
        .map(|i| (q_codebook[i] as f32 / 127.0) * ranges[i % sh_dim])
        .collect();
    let indices: Vec<u16> = (0..n).map(|i| (i % k) as u16).collect();

    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&0x5348_5041u32.to_le_bytes()); // magic "SHPA"
    raw.extend_from_slice(&1u32.to_le_bytes()); // version
    raw.extend_from_slice(&(k as u32).to_le_bytes());
    raw.extend_from_slice(&(n as u32).to_le_bytes());
    raw.push(8u8); // codebookBits
    raw.extend_from_slice(&[0u8; 3]); // pad
    for r in &ranges {
        raw.extend_from_slice(&r.to_le_bytes());
    }
    for q in &q_codebook {
        raw.push(*q as u8);
    }
    for idx in &indices {
        raw.extend_from_slice(&idx.to_le_bytes());
    }
    let compressed = zstd::bulk::compress(&raw, 19).expect("zstd encode .shpal");
    (compressed, codebook_f, indices)
}

/// 8 splats each carrying a degree-3 SH coefficient vector (3 DC + 45 rest).
/// Deterministic so the test is reproducible.
fn sh3_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..8u32 {
        let mut coeffs = vec![0.0f32; 48];
        for c in 0..48 {
            // Distinct per-splat per-coef value so any accidental write-through
            // is detectable on read.
            coeffs[c] = ((i as f32 + 1.0) * 0.1) + (c as f32) * 0.01;
        }
        scene.splats.push(Splat {
            position: [i as f32 * 0.5, i as f32 * 0.25, -(i as f32)],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.5,
            color: Color::Sh { degree: 3, coeffs },
        });
    }
    scene
}

#[test]
fn palette_elision_drops_sh_rest_accessors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = sh3_scene();

    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: "scene.shpal".to_string(),
            palette_size: 4,
            n_splats: scene.splats.len(),
            codebook_bits: 8,
            sh_degree: 3,
        }),
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write_glb");

    // Pull the JSON chunk out of the GLB and verify:
    //   1. No `SH_DEGREE_l_COEF_n` attribute keys for l>=1.
    //   2. `extensionsUsed` mentions `CT_gaussian_splatting_palette`.
    //   3. The root `extensions` map carries the sidecar pointer.
    let bytes = std::fs::read(&path).expect("read glb");
    assert_eq!(&bytes[0..4], b"glTF");
    let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let json_chunk = &bytes[20..20 + json_len];
    let s = std::str::from_utf8(json_chunk).expect("utf8");
    // Trim trailing whitespace/null padding before JSON parse.
    let s = s.trim_end_matches(|c: char| c == ' ' || c == '\0');
    let root: serde_json::Value = serde_json::from_str(s).expect("json");

    // No SH-rest accessor names should appear anywhere in the JSON. DC
    // (`SH_DEGREE_0_COEF_0`) still appears — that's view-independent color
    // and stays in the GLB.
    for l in 1..=3 {
        for n in 0..=6 {
            let needle = format!("SH_DEGREE_{l}_COEF_{n}");
            assert!(
                !s.contains(&needle),
                "elided GLB unexpectedly contains '{needle}'"
            );
        }
    }

    let used = root["extensionsUsed"]
        .as_array()
        .expect("extensionsUsed array");
    assert!(used.iter().any(|v| v == "CT_gaussian_splatting_palette"));

    let pal_ext = &root["extensions"]["CT_gaussian_splatting_palette"];
    assert_eq!(pal_ext["uri"].as_str(), Some("scene.shpal"));
    assert_eq!(pal_ext["paletteSize"].as_u64(), Some(4));
    assert_eq!(pal_ext["splatCount"].as_u64(), Some(8));
    assert_eq!(pal_ext["codebookBits"].as_u64(), Some(8));
    assert_eq!(pal_ext["shDegree"].as_u64(), Some(3));

    // Round-trip the GLB through `read_glb_with_opts(allow_missing_palette)`.
    // The strict default (`read_glb`) would now refuse to silently emit
    // zero SH-rest when the GLB declares the palette extension but the
    // sidecar is absent — that hard-fail is what task #91 added to fix
    // the canonical-11 bench corruption. This test deliberately opts into
    // the permissive DC-only degradation because it asserts the elision
    // contract (writer drops SH-rest accessors), not the round-trip
    // correctness of SH-rest values.
    let decoded = read_glb_with_opts(
        &path,
        &ReadOpts {
            allow_missing_palette: true,
            allow_missing_tail: true,
        },
    )
    .expect("read_glb_with_opts(permissive) of elided asset");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-5);
        }
        // DC color must round-trip exactly (it's emitted as FP32 VEC3 in
        // the GLB). The Rust reader collapses "no SH-rest accessors" into
        // `Color::Rgb`, which carries the DC values directly — palette-
        // aware decoders (e.g. the Node-side bench harness) reconstruct
        // SH-rest separately from the `.shpal` sidecar.
        let dc_decoded = match &a.color {
            Color::Rgb(rgb) => *rgb,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        };
        let dc_orig = match &b.color {
            Color::Rgb(rgb) => *rgb,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        };
        for c in 0..3 {
            assert!((dc_decoded[c] - dc_orig[c]).abs() < 1e-5);
        }
    }
}

#[test]
fn palette_elision_shrinks_glb_bytes_vs_baseline() {
    // Sanity that the elision actually removes bytes — a 4096-splat sh=3
    // scene goes from `4096 * 12 * 15` = ~720 KB of SH-rest accessors to
    // zero (DC stays, plus per-splat pos/rot/scale/opacity overhead).
    let dir = tempdir().unwrap();
    let baseline = dir.path().join("baseline.glb");
    let elided = dir.path().join("elided.glb");

    let mut scene = SplatScene::new();
    for i in 0..4096u32 {
        let mut coeffs = vec![0.0f32; 48];
        for c in 0..48 {
            // Synthetic but compressible — k-means would happily cluster these.
            coeffs[c] = ((i as f32) * 0.0001 + (c as f32) * 0.001).sin();
        }
        scene.splats.push(Splat {
            position: [
                (i as f32) * 0.01,
                ((i as f32) * 0.013).sin(),
                ((i as f32) * 0.017).cos(),
            ],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1, 0.1, 0.1],
            opacity: 0.5,
            color: Color::Sh { degree: 3, coeffs },
        });
    }

    write_glb(&scene, &baseline, &WriteOpts::default()).expect("baseline write");
    let baseline_bytes = std::fs::metadata(&baseline).expect("stat baseline").len();

    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: "x.shpal".to_string(),
            palette_size: 256,
            n_splats: scene.splats.len(),
            codebook_bits: 8,
            sh_degree: 3,
        }),
        ..Default::default()
    };
    write_glb(&scene, &elided, &opts).expect("elided write");
    let elided_bytes = std::fs::metadata(&elided).expect("stat elided").len();

    // SH-rest at sh=3 contributes 45 * 4 = 180 bytes per splat (FP32 VEC3 x
    // 15 coefficients). For 4096 splats that's ~737 KB — elision should
    // strip at least ~500 KB after accounting for JSON/accessor overhead.
    assert!(
        baseline_bytes >= elided_bytes + 500_000,
        "elision didn't shrink the GLB enough: baseline={baseline_bytes} elided={elided_bytes}"
    );
}

/// Regression test for task #91: a GLB that advertises
/// `CT_gaussian_splatting_palette` but whose `.shpal` sidecar is missing
/// MUST fail loudly under the strict default `read_glb` rather than
/// silently returning a scene whose SH-rest coefficients are all zero.
/// Before this fix, `catetus convert --to ply` was emitting PLYs with
/// `f_rest_0..f_rest_44 = 0.0` for the 9 canonical-11 SF scenes that ship
/// without sidecars, deflating their PSNR by ~9 dB on the gsplat bench
/// (`experiments/v4-lod-multiscene/RESULT.md`, the 38.08 vs 47.42 anomaly).
#[test]
fn missing_palette_sidecar_is_hard_error_by_default() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = sh3_scene();
    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: "scene.shpal".to_string(),
            palette_size: 4,
            n_splats: scene.splats.len(),
            codebook_bits: 8,
            sh_degree: 3,
        }),
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write_glb");
    // Deliberately do NOT create `scene.shpal`. The default `read_glb`
    // must surface the missing sidecar instead of silently dropping SH.
    let err = read_glb(&path).expect_err("read_glb must hard-fail without sidecar");
    match err {
        GltfError::MissingPaletteSidecar { uri, .. } => {
            assert_eq!(uri, "scene.shpal");
        }
        other => panic!("expected MissingPaletteSidecar, got {other:?}"),
    }
}

/// Companion to the regression test above: callers that genuinely want
/// the legacy DC-only degradation can opt back in via `ReadOpts`.
#[test]
fn missing_palette_sidecar_allowed_when_opted_in() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = sh3_scene();
    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: "scene.shpal".to_string(),
            palette_size: 4,
            n_splats: scene.splats.len(),
            codebook_bits: 8,
            sh_degree: 3,
        }),
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write_glb");
    let decoded = read_glb_with_opts(
        &path,
        &ReadOpts {
            allow_missing_palette: true,
            allow_missing_tail: true,
        },
    )
    .expect("permissive read_glb must succeed");
    assert_eq!(decoded.len(), scene.len());
}

/// Positive round-trip: when an inline-shaped palette GLB ships alongside
/// its `.shpal` sidecar, the strict `read_glb` reconstructs `Color::Sh`
/// coefficients (NOT zero) from the codebook + indices. This is the
/// behaviour the canonical-11 leaderboard depended on — when the sidecar
/// existed at bench time, bonsai_sf hit 47.42 dB; with the sidecar deleted,
/// the same artifact decoded to zero SH-rest and re-benched at 38.08 dB.
#[test]
fn palette_sidecar_present_round_trips_nonzero_sh_rest() {
    let dir = tempdir().unwrap();
    let glb_path = dir.path().join("scene.glb");
    let shpal_path = dir.path().join("scene.shpal");

    let n_splats = 6usize;
    let k = 4usize;
    let sh_dim = 45usize;

    let mut scene = SplatScene::new();
    for i in 0..n_splats {
        let mut coeffs = vec![0.0f32; 48];
        coeffs[0] = i as f32 * 0.1;
        coeffs[1] = i as f32 * 0.2;
        coeffs[2] = i as f32 * 0.3;
        scene.splats.push(Splat {
            position: [i as f32, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.5,
            color: Color::Sh { degree: 3, coeffs },
        });
    }

    write_glb(
        &scene,
        &glb_path,
        &WriteOpts {
            palette: Some(ShRestPaletteRef {
                sidecar_uri: "scene.shpal".to_string(),
                palette_size: k,
                n_splats,
                codebook_bits: 8,
                sh_degree: 3,
            }),
            ..Default::default()
        },
    )
    .expect("write_glb");

    let (compressed, codebook_f, indices) = synth_shpal_bytes(k, n_splats, sh_dim);
    {
        let mut f = std::fs::File::create(&shpal_path).expect("create .shpal");
        f.write_all(&compressed).expect("write .shpal");
    }

    let decoded = read_glb(&glb_path).expect("strict read_glb with sidecar present");
    assert_eq!(decoded.len(), n_splats);

    // Sum |f_rest| across the whole scene — must be strictly > 0 (this is
    // the assertion that catches the silent-zero regression).
    let mut abs_sum = 0.0f64;
    for splat in &decoded.splats {
        let Color::Sh { coeffs, .. } = &splat.color else {
            panic!("expected Color::Sh after palette rebuild");
        };
        for c in &coeffs[3..48] {
            abs_sum += c.abs() as f64;
        }
    }
    assert!(
        abs_sum > 1e-3,
        "decoded SH-rest is all-zero — palette did not reconstruct"
    );

    // Spot-check exact equality against the codebook row that index points at.
    for (i, splat) in decoded.splats.iter().enumerate() {
        let Color::Sh { coeffs, .. } = &splat.color else {
            unreachable!()
        };
        let row = indices[i] as usize;
        for d in 0..sh_dim {
            let want = codebook_f[row * sh_dim + d];
            let got = coeffs[3 + d];
            assert!(
                (got - want).abs() < 1e-5,
                "splat {i} coef {d}: want {want} got {got}"
            );
        }
    }
}
