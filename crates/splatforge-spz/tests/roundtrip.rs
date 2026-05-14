use splatforge_core::{Color, Splat, SplatScene};
use splatforge_spz::{encode_spz, read_spz_bytes};

fn tiny_scene() -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..5u32 {
        let f = i as f32 * 0.1;
        scene.splats.push(Splat {
            position: [f, -f, f * 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            opacity: 0.7,
            color: Color::Rgb([0.5, 0.6, 0.7]),
        });
    }
    scene
}

#[test]
fn position_roundtrip_within_tolerance() {
    let scene = tiny_scene();
    let bytes = encode_spz(&scene).expect("encode");
    let decoded = read_spz_bytes(&bytes).expect("decode");
    assert_eq!(decoded.len(), scene.len());
    for (a, b) in decoded.splats.iter().zip(scene.splats.iter()) {
        for i in 0..3 {
            assert!((a.position[i] - b.position[i]).abs() < 1e-2);
        }
    }
}

#[test]
fn header_bytes_round_trip() {
    let scene = tiny_scene();
    let bytes = encode_spz(&scene).unwrap();
    // Magic at offset 0
    assert_eq!(&bytes[0..4], &[0x47, 0x4E, 0x53, 0x50]);
    // Version u32 LE = 2
    assert_eq!(&bytes[4..8], &[2, 0, 0, 0]);
}

#[test]
fn rejects_bad_magic() {
    let bytes = vec![0u8; 32];
    let err = read_spz_bytes(&bytes).unwrap_err();
    assert!(matches!(err, splatforge_spz::SpzError::BadMagic));
}
