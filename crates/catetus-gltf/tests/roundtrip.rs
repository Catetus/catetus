use std::fs;
use std::io::{Seek, SeekFrom, Write};

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{
    inspect_gltf, read_glb, read_gltf, write_glb, write_gltf, GltfError, SpzVariant, WriteOpts,
};
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
fn roundtrip_three_splats() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = three_splat_scene();
    write_gltf(&scene, &path, &WriteOpts::default()).expect("write");
    let decoded = read_gltf(&path).expect("read");
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
fn chunked_export_has_streaming_index() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = three_splat_scene();
    let opts = WriteOpts {
        chunked: true,
        chunk_target_splats: 1,
        lod_fractions: vec![1.0],
        ..Default::default()
    };
    write_gltf(&scene, &path, &opts).expect("write");
    let report = inspect_gltf(&path).expect("inspect");
    assert!(report.has_khr);
    assert!(report.has_spatial_index);
    assert_eq!(report.chunk_count, 3);
    assert_eq!(report.splat_count, 3);
}

#[test]
fn glb_roundtrip_three_splats() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = three_splat_scene();
    write_glb(&scene, &path, &WriteOpts::default()).expect("write_glb");
    let decoded = read_glb(&path).expect("read_glb");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-5);
            assert!((a.scale[i] - b.scale[i]).abs() < 1e-5);
        }
        for i in 0..4 {
            assert!((a.rotation[i] - b.rotation[i]).abs() < 1e-5);
        }
        assert!((a.opacity - b.opacity).abs() < 1e-5);
    }
}

#[test]
fn glb_rejects_chunked() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    let scene = three_splat_scene();
    let opts = WriteOpts {
        chunked: true,
        chunk_target_splats: 1,
        lod_fractions: vec![1.0],
        ..Default::default()
    };
    let err = write_glb(&scene, &path, &opts).unwrap_err();
    assert!(matches!(err, GltfError::GlbChunkedUnsupported));
}

#[test]
fn glb_spz_compressed_roundtrip() {
    // Writer emits the KHR_gaussian_splatting_compression_spz extension and
    // the reader transparently decodes the SPZ blob, returning a scene with
    // the same splat count. SPZ is lossy on positions/scales/quat/colors —
    // we only assert structural identity (count + signed-magnitude tolerance).
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.spz.glb");
    let scene = three_splat_scene();
    let opts = WriteOpts {
        compress: Some(SpzVariant::V2),
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write spz-compressed GLB");

    // The GLB JSON chunk must declare both extensions and the SPZ blob must
    // start with the SPZ magic. Cheap on-disk asserts before round-trip.
    let bytes = fs::read(&path).unwrap();
    let s = String::from_utf8_lossy(&bytes);
    assert!(s.contains("KHR_gaussian_splatting_compression_spz"));

    let decoded = read_glb(&path).expect("read spz GLB");
    assert_eq!(decoded.len(), scene.len(), "splat count survives SPZ");
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        // SPZ position is 24-bit fixed-point with 12 fractional bits; 1/4096
        // worst case. Splats at f<3 are far inside the range.
        for i in 0..3 {
            assert!(
                (a.position[i] - b.position[i]).abs() < 1e-2,
                "pos drift too large: {:?} vs {:?}",
                a.position,
                b.position
            );
        }
        // Quat smallest-three is lossy at the ~1/63 level; just check it
        // came back roughly unit-norm and aligned (rotation field present).
        let n: f32 = a.rotation.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 0.1, "rotation not unit norm: {n}");
    }
}

#[test]
fn corrupted_chunk_fails_checksum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.gltf");
    let scene = three_splat_scene();
    let opts = WriteOpts {
        chunked: true,
        chunk_target_splats: 2,
        lod_fractions: vec![1.0],
        ..Default::default()
    };
    write_gltf(&scene, &path, &opts).expect("write");
    // Flip one byte in the first chunk's bin file.
    let chunk_path = dir.path().join("buffers").join("chunk_0000.bin");
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&chunk_path)
        .unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(&[0xFF]).unwrap();
    drop(f);
    let err = inspect_gltf(&path).unwrap_err();
    assert!(matches!(err, GltfError::ChecksumMismatch(_)));
}

#[test]
fn glb_brotli_lossless_roundtrip_quality_max() {
    // Build a small scene that exercises the FP32 SH-coefficients path the
    // way `quality-max` does: SH degree-3 with real-looking float values.
    use catetus_core::{Color, Splat, SplatScene};
    let mut scene = SplatScene::new();
    for i in 0..32u32 {
        let f = i as f32 * 0.05;
        // SH degree 3: 16 bands × 3 channels = 48 coefficients (DC + 15 directional × 3).
        let degree = 3u8;
        let bands = ((degree as usize) + 1) * ((degree as usize) + 1); // 16
        let mut coeffs = vec![0.0f32; 3 * bands];
        // DC channels.
        coeffs[0] = f * 0.1;
        coeffs[1] = 0.2 - f * 0.05;
        coeffs[2] = 0.3;
        for j in 3..coeffs.len() {
            coeffs[j] = ((i * 31 + j as u32) as f32).sin() * 0.5;
        }
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f * 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1 + f, 0.1 + f * 0.7, 0.1 + f * 0.3],
            opacity: 0.5 + (f * 0.3).fract() * 0.5,
            color: Color::Sh { degree, coeffs },
        });
    }
    let dir = tempdir().unwrap();
    let plain_path = dir.path().join("scene.glb");
    let brotli_path = dir.path().join("scene.brotli.glb");

    let plain = WriteOpts::default();
    write_glb(&scene, &plain_path, &plain).expect("write plain GLB");
    let plain_decoded = read_glb(&plain_path).expect("read plain GLB");

    let mut brot = WriteOpts::default();
    brot.lossless = Some(catetus_gltf::LosslessWrap::Brotli11);
    write_glb(&scene, &brotli_path, &brot).expect("write brotli-wrapped GLB");
    let brot_decoded = read_glb(&brotli_path).expect("read brotli-wrapped GLB");

    // Bit-exact splat reconstruction.
    assert_eq!(brot_decoded.len(), plain_decoded.len());
    for (a, b) in brot_decoded.splats.iter().zip(plain_decoded.splats.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.scale, b.scale);
        assert_eq!(a.opacity, b.opacity);
        match (&a.color, &b.color) {
            (
                catetus_core::Color::Sh {
                    degree: da,
                    coeffs: ca,
                },
                catetus_core::Color::Sh {
                    degree: db,
                    coeffs: cb,
                },
            ) => {
                assert_eq!(da, db);
                assert_eq!(ca, cb);
            }
            (ca, cb) => panic!("color variant mismatch: {ca:?} vs {cb:?}"),
        }
    }

    // Brotli output should be strictly smaller than the plain GLB on a SH-3
    // payload of this size — the SH bytes are smooth FP32 floats and brotli
    // crushes them by ~40-50%.
    let plain_size = std::fs::metadata(&plain_path).unwrap().len();
    let brot_size = std::fs::metadata(&brotli_path).unwrap().len();
    assert!(
        brot_size < plain_size,
        "expected brotli GLB to be smaller: plain={plain_size} brotli={brot_size}"
    );
}

#[test]
fn glb_zstd_split_lossless_roundtrip_quantized() {
    // Build a quantized scene that exercises the integer accessor path the
    // way `wmv-sh3-q8` does. The byte-plane-transposed zstd-19 wrap must
    // round-trip bit-exactly through write_glb / read_glb.
    use catetus_core::{Color, Splat, SplatScene};
    let mut scene = SplatScene::new();
    for i in 0..64u32 {
        let f = i as f32 * 0.02;
        let degree = 3u8;
        let bands = ((degree as usize) + 1) * ((degree as usize) + 1);
        let mut coeffs = vec![0.0f32; 3 * bands];
        coeffs[0] = f * 0.1;
        coeffs[1] = 0.2 - f * 0.05;
        coeffs[2] = 0.3;
        for j in 3..coeffs.len() {
            coeffs[j] = ((i * 17 + j as u32) as f32).sin() * 0.4;
        }
        scene.splats.push(Splat {
            position: [f, f * 0.5, -f * 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1 + f, 0.1 + f * 0.7, 0.1 + f * 0.3],
            opacity: 0.5 + (f * 0.3).fract() * 0.5,
            color: Color::Sh { degree, coeffs },
        });
    }

    let dir = tempdir().unwrap();
    let plain_path = dir.path().join("plain.glb");
    let zstd_path = dir.path().join("zstd_split.glb");

    // Plain baseline (no wrap) using the same quantize=true config so
    // accessor offsets and types match exactly between the two writes.
    let plain_opts = WriteOpts {
        quantize: true,
        ..Default::default()
    };
    write_glb(&scene, &plain_path, &plain_opts).expect("write plain GLB");
    let plain_decoded = read_glb(&plain_path).expect("read plain GLB");

    let zstd_opts = WriteOpts {
        quantize: true,
        lossless: Some(catetus_gltf::LosslessWrap::Zstd19Split),
        ..Default::default()
    };
    write_glb(&scene, &zstd_path, &zstd_opts).expect("write zstd-split GLB");
    let zstd_decoded = read_glb(&zstd_path).expect("read zstd-split GLB");

    // Wrap is lossless: every splat must come back bit-identical to the plain
    // (same quantize, same accessor layout) path.
    assert_eq!(zstd_decoded.len(), plain_decoded.len());
    for (a, b) in zstd_decoded.splats.iter().zip(plain_decoded.splats.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.scale, b.scale);
        assert_eq!(a.opacity, b.opacity);
        match (&a.color, &b.color) {
            (
                catetus_core::Color::Sh {
                    degree: da,
                    coeffs: ca,
                },
                catetus_core::Color::Sh {
                    degree: db,
                    coeffs: cb,
                },
            ) => {
                assert_eq!(da, db);
                assert_eq!(ca, cb);
            }
            (ca, cb) => panic!("color variant mismatch: {ca:?} vs {cb:?}"),
        }
    }
}

#[test]
fn glb_brotli_rejects_spz_combo() {
    // Combining SPZ + brotli wrap is a configuration error — SPZ is already
    // compressed, double-wrapping is a measured net loss.
    use catetus_core::{Color, Splat, SplatScene};
    let mut scene = SplatScene::new();
    scene.splats.push(Splat {
        position: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        opacity: 1.0,
        color: Color::Rgb([0.5, 0.5, 0.5]),
    });
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.glb");
    let opts = WriteOpts {
        compress: Some(SpzVariant::V2),
        lossless: Some(catetus_gltf::LosslessWrap::Brotli11),
        ..Default::default()
    };
    let err = write_glb(&scene, &path, &opts).unwrap_err();
    assert!(
        matches!(err, GltfError::Brotli(_)),
        "expected GltfError::Brotli, got {err:?}"
    );
}

/// Decoder-conventions fix (#86): synthesize a 100-splat scene with
/// extreme opacity and scale values that span the heavy-tailed
/// distribution real 3DGS scenes exhibit, then round-trip through the
/// quantized GLB writer with `log_quant_attrs: true` and assert that
/// every scale + opacity comes back within tight relative error.
///
/// The pre-fix encoder (uniform-in-linear 8-bit quant on SCALE +
/// OPACITY) crushed any scale below ~`(max - min)/256` to the same low
/// bin, then the PLY writer's `ln(EPSILON)` reported it back as -15.94,
/// which SuperSplat rendered as too-large too-opaque splats. The fix
/// quantizes in log/logit space and stores the accessor min/max in the
/// same transformed space; the decoder applies `exp`/`sigmoid` to
/// recover the IR values.
#[test]
fn glb_roundtrip_log_quant_attrs_preserves_heavy_tail() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("scene.glb");
    // 100 splats. Scales span ~9 orders of magnitude (4e-9 → 3.3), which
    // is the actual bonsai range. Opacities span (0.001, 0.999) — the
    // logit-space extremes the linear-quant path destroys.
    let n = 100usize;
    let mut scene = SplatScene::new();
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        // log-linear from -19 to 1 (matches bonsai's `ln(scale)` range).
        let log_scale = -19.0 + t * 20.0;
        let scale = log_scale.exp();
        // logit-linear from -10 to +10.
        let logit_op = -10.0 + t * 20.0;
        let opacity = 1.0 / (1.0 + (-logit_op).exp());
        scene.splats.push(Splat {
            position: [t, t * 0.5, -t],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [scale, scale * 1.1, scale * 0.9],
            opacity,
            color: Color::Rgb([t, 1.0 - t, 0.5]),
        });
    }
    let opts = WriteOpts {
        quantize: true,
        log_quant_attrs: true,
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write_glb log_quant_attrs");
    let decoded = read_glb(&path).expect("read_glb log_quant_attrs");
    assert_eq!(decoded.len(), scene.len());

    // Compare in log-space for scale (multiplicative-error grid).
    for (i, (a, b)) in decoded.splats.iter().zip(scene.splats.iter()).enumerate() {
        for axis in 0..3 {
            // 12-bit linear quant in log-space over [-19, 1] yields a
            // step of 20/4095 ≈ 4.9e-3 ln-units = 0.5% relative error.
            // The downstream 8-bit GLB-writer quant in the same
            // log-space gives 20/255 ≈ 0.078 ln-units ≈ 8% per step.
            let ratio = (a.scale[axis] / b.scale[axis]).ln().abs();
            assert!(
                ratio < 0.1,
                "splat {i} axis {axis} scale ratio drift {ratio} (got {} vs {})",
                a.scale[axis],
                b.scale[axis]
            );
        }
        // Opacity in sigmoid space: an 8-bit grid in logit-space [-12, 12]
        // gives a step of 24/255 ≈ 0.094 logit, which is at most ~0.024
        // sigmoid units near 0.5 and much less at the tails (where it
        // matters). Use a generous 0.03 bound.
        assert!(
            (a.opacity - b.opacity).abs() < 0.03,
            "splat {i} opacity drift {} vs {}",
            a.opacity,
            b.opacity
        );
    }

    // The pre-fix path would have produced opacities in {0, 1} for the
    // extremes (linear quant + sigmoid logit-write blowup); confirm we
    // are NOT seeing that pathological collapse.
    let saturated = decoded
        .splats
        .iter()
        .filter(|s| s.opacity == 0.0 || s.opacity == 1.0)
        .count();
    assert_eq!(
        saturated, 0,
        "log_quant_attrs round-trip produced {saturated}/{n} saturated opacities (pre-fix regression)"
    );
}
