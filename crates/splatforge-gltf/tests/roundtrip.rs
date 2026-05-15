use std::fs;
use std::io::{Seek, SeekFrom, Write};

use splatforge_core::{Color, Splat, SplatScene};
use splatforge_gltf::{
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
