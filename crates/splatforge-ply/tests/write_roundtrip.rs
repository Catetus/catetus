use splatforge_ply::{read_ply_bytes, write_ply_bytes};

#[test]
fn writer_then_reader_preserves_ir() {
    // Build a synthetic scene exactly as the binary roundtrip test does.
    let header = concat!(
        "ply\n",
        "format binary_little_endian 1.0\n",
        "element vertex 3\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float scale_0\n",
        "property float scale_1\n",
        "property float scale_2\n",
        "property float rot_0\n",
        "property float rot_1\n",
        "property float rot_2\n",
        "property float rot_3\n",
        "property float opacity\n",
        "property float f_dc_0\n",
        "property float f_dc_1\n",
        "property float f_dc_2\n",
        "end_header\n",
    );
    let mut buf = Vec::new();
    buf.extend_from_slice(header.as_bytes());
    for i in 0..3u32 {
        let f = i as f32;
        let record = [
            f,
            f * 0.5,
            -f, // pos
            0.0,
            0.0,
            0.0, // log-scale 0 -> 1
            1.0,
            0.0,
            0.0,
            0.0, // identity quat (w, x, y, z)
            0.0, // opacity logit 0 -> 0.5
            0.1,
            0.2,
            0.3,
        ];
        for v in record {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    let scene = read_ply_bytes(&buf).expect("read");

    // Re-encode and re-read; assert the IR survives the round-trip.
    let encoded = write_ply_bytes(&scene).expect("write");
    let decoded = read_ply_bytes(&encoded).expect("read back");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!(
                (a.position[i] - b.position[i]).abs() < 1e-5,
                "position[{i}] diverged: {} vs {}",
                a.position[i],
                b.position[i]
            );
            assert!((a.scale[i] - b.scale[i]).abs() < 1e-4);
        }
        for i in 0..4 {
            assert!((a.rotation[i] - b.rotation[i]).abs() < 1e-5);
        }
        assert!((a.opacity - b.opacity).abs() < 1e-4);
    }
}

#[test]
fn fixture_basic_binary_roundtrips() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("tiny")
        .join("basic_binary.ply");
    let bytes = std::fs::read(&path).expect("read fixture");
    let scene = read_ply_bytes(&bytes).expect("parse fixture");
    let encoded = write_ply_bytes(&scene).expect("write");
    let decoded = read_ply_bytes(&encoded).expect("read back");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-3);
            assert!((a.scale[i] - b.scale[i]).abs() < 1e-3);
        }
        assert!((a.opacity - b.opacity).abs() < 1e-3);
    }
}
