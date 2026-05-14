use std::fs;
use std::io::{Seek, SeekFrom, Write};

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{
    inspect_gltf, read_glb, read_gltf, write_glb, write_gltf, GltfError, WriteOpts,
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
    };
    let err = write_glb(&scene, &path, &opts).unwrap_err();
    assert!(matches!(err, GltfError::GlbChunkedUnsupported));
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
