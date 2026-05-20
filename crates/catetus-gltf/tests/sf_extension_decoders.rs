//! End-to-end decoder tests for the three Catetus vendor extensions that
//! the reader started supporting in this changeset:
//!
//!  * `SF_zstd_split_buffer` — byte-plane-transposed zstd-19 BIN wrap. The
//!    encoder is already covered by `roundtrip.rs`, but we add a focused
//!    palette + zstd-wrap test so a regression in either component would
//!    surface here rather than only in the cross-cutting tight preset.
//!  * `SF_gaussian_splatting_palette` — 45-D k-means SH-rest palette in a
//!    sidecar `.shpal` file. We exercise the public `decode_shpal_bytes`
//!    decoder directly with a synthetic K=4, sh=3, codebookBits=8 sidecar
//!    so the test is self-contained (no Rust encoder API for the sidecar
//!    lives in this crate — it's in `catetus-optimize::vq_palette`).
//!  * `SF_quat_smallest3` — SOG-style smallest-3 packed quaternion
//!    (`SCALAR UINT` accessor with `[q0|q1|q2|tag]` u32 payload). Exercised
//!    via a write_glb → read_glb round-trip so the wire-format wiring lines
//!    up with `pack_chunk_with`.
//!
//! Together these are the on-wire pieces the JS bench harness already
//! decodes (`packages/viewer/src/streaming/glb.ts::decompressZstdSplitBuffer`
//! and `decodeShPaletteSidecar`, plus the smallest3 unpacking the harness's
//! `cpu-fidelity.mjs` re-implements inline).

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{
    decode_shpal_bytes, read_glb, write_glb, LosslessWrap, RotationSmallest3Table,
    ShRestPaletteRef, WriteOpts,
};
use std::io::Write;
use tempfile::tempdir;

/// Deterministic K=4 sh=3 8-bit `.shpal` sidecar built byte-for-byte to the
/// wire format documented at
/// `catetus-optimize::vq_palette::ShRestPaletteSidetable`. We construct it
/// uncompressed in memory, then run it through zstd-19 so `decode_shpal_bytes`
/// sees exactly what a real sidecar looks like on disk.
fn synth_shpal_bytes(k: usize, n: usize, sh_dim: usize) -> (Vec<u8>, Vec<f32>, Vec<u16>) {
    // Per-coefficient range table (45 floats). Pick varied values so the
    // dequantization scale shows up if it's wrong.
    let ranges: Vec<f32> = (0..sh_dim).map(|d| 0.1 + (d as f32) * 0.01).collect();
    // Codebook in i8 quantized form. Pick `q = (c * 13 + d * 7) % 127 - 63`
    // so adjacent rows are perceptually distinct and we get both signs.
    let mut q_codebook = Vec::with_capacity(k * sh_dim);
    for c in 0..k {
        for d in 0..sh_dim {
            let q: i32 = (((c * 13 + d * 7) % 127) as i32) - 63;
            q_codebook.push(q.clamp(-127, 127) as i8);
        }
    }
    // Reconstructed float codebook for the assertion side.
    let codebook_f: Vec<f32> = (0..k * sh_dim)
        .map(|i| (q_codebook[i] as f32 / 127.0) * ranges[i % sh_dim])
        .collect();
    // Per-splat indices: a simple deterministic pattern that hits every
    // codebook entry.
    let indices: Vec<u16> = (0..n).map(|i| (i % k) as u16).collect();

    // Encode header + payload bit-for-bit per the wire spec.
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&0x5348_5041u32.to_le_bytes()); // magic "SHPA" (LE u32)
    raw.extend_from_slice(&1u32.to_le_bytes()); // version
    raw.extend_from_slice(&(k as u32).to_le_bytes());
    raw.extend_from_slice(&(n as u32).to_le_bytes());
    raw.push(8u8); // codebookBits
    raw.extend_from_slice(&[0u8; 3]); // alignment pad
    for r in &ranges {
        raw.extend_from_slice(&r.to_le_bytes());
    }
    for q in &q_codebook {
        raw.push(*q as u8);
    }
    for idx in &indices {
        raw.extend_from_slice(&idx.to_le_bytes());
    }
    let compressed = zstd::bulk::compress(&raw, 19).expect("zstd encode synthesised .shpal");
    (compressed, codebook_f, indices)
}

#[test]
fn shpal_decoder_roundtrips_synthetic_sidecar() {
    let k = 4;
    let n = 7;
    let sh_dim = 45;
    let (compressed, expected_codebook, expected_indices) = synth_shpal_bytes(k, n, sh_dim);

    let decoded =
        decode_shpal_bytes(&compressed, Some((k, n, 8)), 3).expect("decode synthetic .shpal");

    assert_eq!(decoded.k, k);
    assert_eq!(decoded.n, n);
    assert_eq!(decoded.codebook_bits, 8);
    assert_eq!(decoded.sh_degree, 3);
    assert_eq!(decoded.indices.len(), n);
    assert_eq!(decoded.codebook.len(), k * sh_dim);
    for (a, b) in decoded.indices.iter().zip(expected_indices.iter()) {
        assert_eq!(a, b);
    }
    // Dequant tolerance: 1/127 of one range step, comfortably under 1e-3.
    for (i, (got, want)) in decoded
        .codebook
        .iter()
        .zip(expected_codebook.iter())
        .enumerate()
    {
        assert!((got - want).abs() < 1e-5, "codebook[{i}] = {got} vs {want}");
    }
}

#[test]
fn shpal_decoder_rejects_bad_magic() {
    // 16-byte truncated zstd frame, wrong magic. Decoder should error rather
    // than panic — the JS harness throws on the same check.
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    raw.extend_from_slice(&1u32.to_le_bytes());
    raw.extend_from_slice(&4u32.to_le_bytes());
    raw.extend_from_slice(&1u32.to_le_bytes());
    raw.push(8u8);
    raw.extend_from_slice(&[0u8; 3]);
    let compressed = zstd::bulk::compress(&raw, 5).expect("zstd encode");
    let err = decode_shpal_bytes(&compressed, None, 3).expect_err("must reject bad magic");
    let msg = format!("{err}");
    assert!(msg.contains("magic"), "expected magic mismatch, got: {msg}");
}

#[test]
fn smallest3_rotation_roundtrips_through_glb() {
    // 8 random-ish unit quaternions exercise every `tag` slot (0..=3) at
    // least once. The encoder picks the largest |component| as tag, so we
    // shape the inputs to cover all four.
    let mut scene = SplatScene::new();
    let quats: [[f32; 4]; 8] = [
        [0.9, 0.1, 0.05, 0.05],     // tag = 0 (w-largest-equivalent layout: x)
        [0.1, 0.9, 0.05, 0.05],     // tag = 1
        [0.05, 0.05, 0.9, 0.1],     // tag = 2
        [0.05, 0.05, 0.1, 0.9],     // tag = 3
        [-0.95, 0.1, 0.1, 0.1],     // negative-largest sign branch (tag = 0)
        [0.5, 0.5, 0.5, 0.5],       // four-way tie — encoder picks index 0
        [0.7071, 0.0, 0.7071, 0.0], // identity-rotated 90deg around y-ish
        [0.0, 0.0, 0.0, 1.0],       // canonical identity
    ];
    for (i, q) in quats.iter().enumerate() {
        // Normalise so the encoder's input-normalisation step is a no-op
        // and the asserted tolerance reflects only the 10-bit quantisation.
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        scene.splats.push(Splat {
            position: [i as f32, 0.0, 0.0],
            rotation: [q[0] / n, q[1] / n, q[2] / n, q[3] / n],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.5,
            color: Color::Rgb([0.4, 0.5, 0.6]),
        });
    }

    let dir = tempdir().unwrap();
    let path = dir.path().join("smallest3.glb");
    let opts = WriteOpts {
        rotation_smallest3: Some(RotationSmallest3Table { component_bits: 10 }),
        ..Default::default()
    };
    write_glb(&scene, &path, &opts).expect("write_glb smallest3");

    let decoded = read_glb(&path).expect("read_glb smallest3");
    assert_eq!(decoded.len(), scene.len());

    // 10-bit per-component packing carries ~3 milli-radian noise; the writer
    // also performs sign-normalisation on the largest component, so we
    // compare on |q| and dot product, not raw components. A dot product of
    // ±1 means the quaternions encode the same rotation.
    for (orig, got) in scene.splats.iter().zip(decoded.splats.iter()) {
        // Unit-length within float epsilon.
        let n: f32 = got.rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "rotation not unit-length: |q|={n}");
        let dot: f32 = orig
            .rotation
            .iter()
            .zip(got.rotation.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot.abs() > 0.999,
            "smallest3 decode lost rotation: dot={dot}, orig={:?} got={:?}",
            orig.rotation,
            got.rotation
        );
    }
}

#[test]
fn palette_and_zstd_split_roundtrip_through_glb() {
    // End-to-end check the headline asset depends on:
    //   1. Write a GLB with palette elision (no SH-rest accessors) + a zstd-
    //      split BIN wrapper.
    //   2. Drop a synthetic `.shpal` next to it.
    //   3. `read_glb` must decompress the BIN, then load the sidecar, then
    //      reconstruct `Color::Sh` coefficients from the codebook.
    let dir = tempdir().unwrap();
    let glb_path = dir.path().join("scene.glb");
    let shpal_path = dir.path().join("scene.shpal");

    let n_splats = 6;
    let k = 4;
    let sh_dim = 45;
    let mut scene = SplatScene::new();
    for i in 0..n_splats {
        let mut coeffs = vec![0.0f32; 48];
        // DC: distinct per-splat so the FP32 DC path is exercised independently.
        coeffs[0] = i as f32 * 0.1;
        coeffs[1] = i as f32 * 0.2;
        coeffs[2] = i as f32 * 0.3;
        // SH-rest: deliberately doesn't have to match the .shpal — the
        // writer elides it anyway, and the reader rebuilds from the sidecar.
        scene.splats.push(Splat {
            position: [i as f32, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.5,
            color: Color::Sh { degree: 3, coeffs },
        });
    }

    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: "scene.shpal".to_string(),
            palette_size: k,
            n_splats,
            codebook_bits: 8,
            sh_degree: 3,
        }),
        lossless: Some(LosslessWrap::Zstd19Split),
        ..Default::default()
    };
    write_glb(&scene, &glb_path, &opts).expect("write_glb tight");

    // Drop a real-shape sidecar next to the GLB.
    let (compressed, codebook_f, indices) = synth_shpal_bytes(k, n_splats, sh_dim);
    {
        let mut f = std::fs::File::create(&shpal_path).expect("create .shpal");
        f.write_all(&compressed).expect("write .shpal");
    }

    let decoded = read_glb(&glb_path).expect("read_glb tight asset");
    assert_eq!(decoded.len(), n_splats);

    for (i, splat) in decoded.splats.iter().enumerate() {
        let Color::Sh { degree, coeffs } = &splat.color else {
            panic!("expected Color::Sh from palette rebuild, got Rgb at splat {i}");
        };
        assert_eq!(*degree, 3);
        assert_eq!(coeffs.len(), 48);
        // DC: float round-trip exact.
        assert!((coeffs[0] - i as f32 * 0.1).abs() < 1e-5, "dc[0] splat {i}");
        assert!((coeffs[1] - i as f32 * 0.2).abs() < 1e-5, "dc[1] splat {i}");
        assert!((coeffs[2] - i as f32 * 0.3).abs() < 1e-5, "dc[2] splat {i}");
        // SH-rest: every scalar must equal the codebook row pointed at by
        // `indices[i]`. Tolerance covers the i8 dequant.
        let row = indices[i] as usize;
        for d in 0..sh_dim {
            let want = codebook_f[row * sh_dim + d];
            let got = coeffs[3 + d];
            assert!(
                (got - want).abs() < 1e-5,
                "splat {i} coef {d}: got {got} want {want}"
            );
        }
    }
}
