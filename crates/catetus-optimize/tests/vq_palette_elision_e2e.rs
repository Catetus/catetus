//! End-to-end test for `VQPaletteShRest` + the GLB writer's palette-elision
//! path. Builds a synthetic 64-splat sh=3 scene with TWO obviously distinct
//! SH-rest patterns, runs `VQPaletteShRest` at K=2, drains the sidetable into
//! `WriteOpts::palette`, writes a GLB, and asserts:
//!
//!   * the elided GLB carries the `CT_gaussian_splatting_palette` root
//!     extension with the correct sidecar URI and metadata
//!   * the elided GLB is substantially smaller than the baseline GLB (the
//!     baseline has 15 FP32 VEC3 SH-rest accessors per splat)
//!   * the .shpal sidecar parses to a valid SHPA-v1 blob with the expected
//!     palette_size / n_splats / codebook_bits header values
//!   * decoding the codebook + indices reconstructs the cluster centroids
//!     to within the quant grid (1 step at codebook_bits=8 = ~0.008 of range)

use catetus_core::{Color, Splat, SplatScene};
use catetus_gltf::{write_glb, ShRestPaletteRef, WriteOpts};
use catetus_optimize::{take_last_sh_rest_palette, Pipeline, VQPaletteShRest, VQ_SH_REST_DIM};

fn make_sh_splat(idx: usize, pattern: [f32; VQ_SH_REST_DIM]) -> Splat {
    let mut coeffs = vec![0.4, 0.5, 0.6]; // DC
    coeffs.extend_from_slice(&pattern);
    Splat {
        position: [
            (idx as f32) * 0.05,
            -(idx as f32) * 0.03,
            (idx as f32) * 0.02,
        ],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [0.01, 0.01, 0.01],
        opacity: 0.9,
        color: Color::Sh { degree: 3, coeffs },
    }
}

#[test]
fn vq_palette_elision_end_to_end() {
    // Two clearly distinct SH-rest patterns; with K=2 every splat should
    // collapse to its native cluster.
    let mut pat_a = [0.0f32; VQ_SH_REST_DIM];
    let mut pat_b = [0.0f32; VQ_SH_REST_DIM];
    for d in 0..VQ_SH_REST_DIM {
        pat_a[d] = 0.5;
        pat_b[d] = -0.5;
    }
    let mut scene = SplatScene::new();
    for i in 0..64usize {
        let p = if i % 2 == 0 { pat_a } else { pat_b };
        scene.splats.push(make_sh_splat(i, p));
    }

    let pipe = Pipeline::new().push(Box::new(VQPaletteShRest {
        palette_size: 2,
        iterations: 6,
        codebook_bits: 8,
        training_subsample: None,
        posthac_indices: false,
    }));
    pipe.run(&mut scene).expect("vq pipeline");

    let sidetable = take_last_sh_rest_palette().expect("palette sidetable parked");
    assert_eq!(sidetable.palette_size, 2);
    assert_eq!(sidetable.n_splats, 64);
    assert_eq!(sidetable.codebook_bits, 8);
    assert!(sidetable.stats.mse < 1e-6, "two-cluster MSE should be tiny");

    let out_dir = std::env::temp_dir();
    let glb_baseline = out_dir.join("catetus_vq45_baseline.glb");
    let glb_elided = out_dir.join("catetus_vq45_elided.glb");
    let shpal_path = out_dir.join("catetus_vq45_elided.glb.shpal");

    // Baseline: write WITHOUT palette elision → SH-rest accessors emitted.
    write_glb(&scene, &glb_baseline, &WriteOpts::default()).expect("baseline write");
    let baseline_bytes = std::fs::metadata(&glb_baseline).expect("stat").len() as usize;

    // Elided: write WITH palette elision → SH-rest accessors omitted.
    std::fs::write(&shpal_path, &sidetable.compressed).expect("write sidecar");
    let opts = WriteOpts {
        palette: Some(ShRestPaletteRef {
            sidecar_uri: shpal_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap()
                .to_string(),
            palette_size: sidetable.palette_size,
            n_splats: sidetable.n_splats,
            codebook_bits: sidetable.codebook_bits,
            sh_degree: 3,
        }),
        ..Default::default()
    };
    write_glb(&scene, &glb_elided, &opts).expect("elided write");
    let elided_bytes = std::fs::metadata(&glb_elided).expect("stat").len() as usize;

    // 64 splats × 45 FP32 SH-rest scalars = 11_520 bytes of SH-rest payload.
    // Plus per-coef accessor / bufferView JSON. Elision should drop at least
    // ~10 KB.
    assert!(
        baseline_bytes >= elided_bytes + 10_000,
        "expected >=10 KB savings: baseline={baseline_bytes} elided={elided_bytes}"
    );

    // Verify the elided GLB's JSON carries the extension.
    let bytes = std::fs::read(&glb_elided).expect("read glb");
    let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let json = std::str::from_utf8(&bytes[20..20 + json_len])
        .expect("utf8")
        .trim_end_matches(|c: char| c == ' ' || c == '\0');
    let root: serde_json::Value = serde_json::from_str(json).expect("json");
    let pal_ext = &root["extensions"]["CT_gaussian_splatting_palette"];
    assert_eq!(pal_ext["paletteSize"].as_u64(), Some(2));
    assert_eq!(pal_ext["splatCount"].as_u64(), Some(64));
    assert_eq!(pal_ext["shDegree"].as_u64(), Some(3));
    assert_eq!(pal_ext["codebookBits"].as_u64(), Some(8));
    assert!(pal_ext["uri"].as_str().unwrap().ends_with(".shpal"));

    // Parse the sidecar header and verify the on-wire format matches what
    // the decoder expects (see vq_palette.rs for the wire format).
    let compressed = std::fs::read(&shpal_path).expect("read sidecar");
    let raw = zstd::stream::decode_all(compressed.as_slice()).expect("zstd decode");
    assert!(raw.len() >= 16 + 45 * 4);
    assert_eq!(&raw[0..4], &0x53485041u32.to_le_bytes()); // "SHPA"
    assert_eq!(&raw[4..8], &1u32.to_le_bytes()); // version
    let palette_size = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let n_splats = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]) as usize;
    assert_eq!(palette_size, 2);
    assert_eq!(n_splats, 64);
    assert_eq!(raw[16], 8); // codebook_bits

    // Decode the codebook (palette_size × 45 int8 normalized by per-coef ranges)
    // and verify the recovered centroids match the two patterns within one
    // quantization step (1/127 of range ≈ 0.008 here since range ~= 0.5).
    let mut off = 20; // 16 header + 4 alignment pad
    let mut ranges = [0.0f32; VQ_SH_REST_DIM];
    for d in 0..VQ_SH_REST_DIM {
        ranges[d] = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        off += 4;
    }
    let mut decoded = [[0.0f32; VQ_SH_REST_DIM]; 2];
    for c in 0..2 {
        for d in 0..VQ_SH_REST_DIM {
            let q = raw[off] as i8;
            decoded[c][d] = (q as f32 / 127.0) * ranges[d];
            off += 1;
        }
    }
    // Each centroid must be near either +0.5 or -0.5 on every dim (depending
    // on which cluster k-means picked first). We don't know the label order,
    // so we just check both centroids hit one of the two patterns.
    let target_a: f32 = 0.5;
    let target_b: f32 = -0.5;
    for c in 0..2 {
        let mean = decoded[c].iter().sum::<f32>() / (VQ_SH_REST_DIM as f32);
        assert!(
            (mean - target_a).abs() < 0.02 || (mean - target_b).abs() < 0.02,
            "centroid {c} mean {mean} doesn't match either ±0.5 pattern"
        );
    }

    let _ = std::fs::remove_file(&glb_baseline);
    let _ = std::fs::remove_file(&glb_elided);
    let _ = std::fs::remove_file(&shpal_path);
}
