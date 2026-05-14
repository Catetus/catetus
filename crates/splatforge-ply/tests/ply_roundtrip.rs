use splatforge_ply::{read_ply_bytes, PlyError};

/// Build a tiny binary PLY with 3 splats in memory for testing.
fn synthesize_binary_ply() -> Vec<u8> {
    let mut buf = Vec::new();
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
    buf.extend_from_slice(header.as_bytes());
    for i in 0..3u32 {
        let f = i as f32;
        let record = [
            f,
            f * 0.5,
            -f, // pos
            0.0,
            0.0,
            0.0, // log scale 0 -> scale 1
            1.0,
            0.0,
            0.0,
            0.0, // rot (w, x, y, z) -> identity
            0.0, // opacity logit 0 -> sigmoid 0.5
            0.1,
            0.2,
            0.3, // f_dc
        ];
        for v in record {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

#[test]
fn parses_synthetic_binary_ply() {
    let bytes = synthesize_binary_ply();
    let scene = read_ply_bytes(&bytes).expect("parse ok");
    assert_eq!(scene.len(), 3);
    let first = &scene.splats[0];
    assert!((first.opacity - 0.5).abs() < 1e-5);
    assert!((first.scale[0] - 1.0).abs() < 1e-5);
    // identity quaternion in IR order (x, y, z, w)
    assert!((first.rotation[3] - 1.0).abs() < 1e-5);
}

#[test]
fn ascii_format_parses() {
    let header = concat!(
        "ply\n",
        "format ascii 1.0\n",
        "element vertex 1\n",
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
        "0 0 0 0 0 0 1 0 0 0 0 0.1 0.2 0.3\n",
    );
    let scene = read_ply_bytes(header.as_bytes()).expect("parse ok");
    assert_eq!(scene.len(), 1);
}

#[test]
fn rejects_big_endian() {
    let header = b"ply\nformat binary_big_endian 1.0\nelement vertex 0\nend_header\n";
    let err = read_ply_bytes(header).unwrap_err();
    assert!(matches!(err, PlyError::UnsupportedEndian));
}

#[test]
fn rejects_missing_rotation() {
    let header = concat!(
        "ply\n",
        "format ascii 1.0\n",
        "element vertex 0\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float scale_0\n",
        "property float scale_1\n",
        "property float scale_2\n",
        "property float opacity\n",
        "property float f_dc_0\n",
        "property float f_dc_1\n",
        "property float f_dc_2\n",
        "end_header\n",
    );
    let err = read_ply_bytes(header.as_bytes()).unwrap_err();
    assert!(matches!(err, PlyError::MissingRequiredField(_)));
}

#[test]
fn detects_truncated_payload() {
    let mut bytes = synthesize_binary_ply();
    // Truncate to a record-and-a-half.
    bytes.truncate(bytes.len() - 20);
    let err = read_ply_bytes(&bytes).unwrap_err();
    assert!(matches!(err, PlyError::TruncatedPayload));
}
