//! Public-reader compatibility for the SwVQ extension chunk: a base scene
//! encoded with a stub SwVQ payload between the header and the zlib stream
//! must still round-trip through the public reader at byte-level.

use catetus_core::{Color, Splat, SplatScene};
use catetus_spz::{encode_spz, read_spz_bytes, SPZ_FLAG_SWVQ_EXT};

fn tiny_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..7u32 {
        let f = i as f32 * 0.1;
        scene.splats.push(Splat {
            position: [f, -f, f * 0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.7,
            color: Color::Rgb([0.3, 0.4, 0.5]),
        });
    }
    scene
}

/// Splice a fake SwVQ chunk between the 16-byte SPZ header and the zlib
/// payload, flipping the `flags` byte to mark the extension. Header layout:
/// magic(4) version(4) splat_count(4) sh_degree(1) fractional_bits(1)
/// flags(1) reserved(1) — so `flags` is at offset 14, not 13.
fn splice_swvq_chunk(spz: &[u8], chunk: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(spz.len() + chunk.len());
    out.extend_from_slice(&spz[..16]);
    out[14] |= SPZ_FLAG_SWVQ_EXT;
    out.extend_from_slice(chunk);
    out.extend_from_slice(&spz[16..]);
    out
}

#[test]
fn public_reader_skips_swvq_chunk() {
    let scene = tiny_scene();
    let base = encode_spz(&scene).expect("encode");
    let baseline = read_spz_bytes(&base).expect("baseline decode");

    // Stub payload: length prefix (u32 LE) + arbitrary opaque bytes.
    let stub: [u8; 12] = [0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut chunk = (stub.len() as u32).to_le_bytes().to_vec();
    chunk.extend_from_slice(&stub);

    let extended = splice_swvq_chunk(&base, &chunk);
    let parsed = read_spz_bytes(&extended).expect("extended decode");

    assert_eq!(parsed.len(), baseline.len());
    for (a, b) in parsed.splats.iter().zip(baseline.splats.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.scale, b.scale);
        assert!((a.opacity - b.opacity).abs() < 1e-6);
    }
}

#[test]
fn public_reader_rejects_truncated_swvq_chunk() {
    let scene = tiny_scene();
    let base = encode_spz(&scene).expect("encode");
    // Claim a 1 MB payload but provide zero bytes.
    let chunk = (1_000_000u32).to_le_bytes().to_vec();
    let extended = splice_swvq_chunk(&base, &chunk);
    let err = read_spz_bytes(&extended).unwrap_err();
    assert!(matches!(err, catetus_spz::SpzError::Truncated));
}
