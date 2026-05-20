//! End-to-end tests for the V5.2 joint-tail residual sidecar — writer +
//! reader round-trip through an actual `.glb` + `.glb.v5tail` on disk.
//!
//! Phase B coverage points (from task #109):
//!   * Sidecar bytes round-trip through `decode_v5tail_bytes` and the
//!     scene-apply path.
//!   * GLB with `extensionsRequired: ["CT_v5_tail_residual"]` and missing
//!     sidecar → hard error.
//!   * Same GLB with `allow_missing_tail=true` → warn + render baseline.

use catetus_core::{Color, CoordinateSystem, Splat, SplatScene, TemporalMode};
use catetus_gltf::{
    read_glb_with_opts, v5_tail, write_glb, GltfError, ReadOpts, V5TailRef, WriteOpts,
};
use tempfile::tempdir;

fn make_synthetic_scene(n: usize, sh_rest_coefs: usize) -> SplatScene {
    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        // Spread positions along x so Morton order is monotone.
        let pos = [(i as f32) * 0.1, 0.0, 0.0];
        // SH degree 3 → 16 coefs / channel * 3 channels = 48 floats. We
        // want sh_rest_coefs=15 coefs/channel for a real V5.2 run; the
        // synthetic test uses a smaller sh_rest_coefs to keep things tight.
        let total = 3 + sh_rest_coefs * 3;
        let mut coeffs = vec![0.0f32; total];
        for c in 0..3 {
            coeffs[c] = (i as f32 + c as f32) * 0.01;
        }
        for c in 0..(sh_rest_coefs * 3) {
            coeffs[3 + c] = (i as f32 * sh_rest_coefs as f32 + c as f32) * 0.001;
        }
        splats.push(Splat {
            position: pos,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.1, 0.1, 0.1],
            opacity: 0.5,
            color: Color::Sh { degree: 3, coeffs },
        });
    }
    SplatScene {
        splats,
        coordinate_system: CoordinateSystem::default(),
        semantic_labels: None,
        temporal_mode: TemporalMode::Static,
        lods: None,
        codecgs: None,
    }
}

/// Build a synthetic sidecar that adds known residuals to splats {1, 3, 5}.
fn build_sidecar(scene: &SplatScene, sh_rest_coefs: usize) -> Vec<u8> {
    let n = scene.splats.len();
    let sel_idx = vec![1u32, 3, 5];
    let mut sel_bool = vec![false; n];
    for &i in &sel_idx {
        sel_bool[i as usize] = true;
    }
    let positions_selected: Vec<[f32; 3]> = sel_idx
        .iter()
        .map(|&i| scene.splats[i as usize].position)
        .collect();
    let morton_idx = v5_tail::morton_sort_indices(&positions_selected);
    let k = sel_idx.len();
    // Residual = 0.5 per channel for every group (round trip should
    // recover this to dequant precision).
    let pos: Vec<f32> = vec![0.5; k * 3];
    let rot: Vec<f32> = vec![0.05; k * 4];
    let opa: Vec<f32> = vec![0.1; k];
    let sca: Vec<f32> = vec![0.02; k * 3];
    let dc: Vec<f32> = vec![0.3; k * 3];
    let shr: Vec<f32> = vec![0.01; k * sh_rest_coefs * 3];
    let residuals = v5_tail::Residuals {
        k_selected: k,
        sh_rest_coefs,
        pos,
        rot,
        opa,
        sca,
        dc,
        shr,
    };
    let cell_offsets = v5_tail::build_cell_offsets(k, 2);
    let (bytes, _sizes) = v5_tail::encode_v5_2_sidecar(
        n,
        &sel_bool,
        &morton_idx,
        &residuals,
        v5_tail::BitDepths::v5_2(),
        &cell_offsets,
    )
    .expect("encode v5 tail");
    bytes
}

/// Sidecar bytes round-trip into a baseline scene → selected splats get
/// the additive residual; others are untouched.
#[test]
fn sidecar_apply_matches_residual() {
    let sh_rest_coefs = 2usize;
    let scene = make_synthetic_scene(8, sh_rest_coefs);
    let sidecar_bytes = build_sidecar(&scene, sh_rest_coefs);

    let decoded = v5_tail::decode_v5tail_bytes(&sidecar_bytes).expect("decode ok");
    assert_eq!(decoded.sel_idx, vec![1u32, 3, 5]);

    let mut applied = scene.clone();
    catetus_gltf::apply_v5tail_to_scene(&mut applied, &decoded).expect("apply ok");

    // Untouched splats stay identical.
    for &i in &[0usize, 2, 4, 6, 7] {
        assert_eq!(
            applied.splats[i].position, scene.splats[i].position,
            "splat {i} position drifted"
        );
        assert_eq!(applied.splats[i].opacity, scene.splats[i].opacity);
        if let (Color::Sh { coeffs: a, .. }, Color::Sh { coeffs: b, .. }) =
            (&applied.splats[i].color, &scene.splats[i].color)
        {
            assert_eq!(a, b, "splat {i} sh drifted");
        }
    }
    // Touched splats: position shifted by ~+0.5 per axis (dequant slack).
    // Note: opacity / scale residuals are applied in raw 3DGS-PLY space
    // (logit / log) — see the docstring on `apply_v5tail_to_scene`. For
    // base opacity = 0.5 the logit-space residual of 0.1 maps to an IR
    // shift of `sigmoid(logit(0.5) + 0.1) - 0.5 ≈ 0.025`, not 0.1.
    let expected_opa_delta = {
        let base = 0.5_f32;
        let raw = (base / (1.0 - base)).ln();
        let after = 1.0 / (1.0 + (-(raw + 0.1)).exp());
        after - base
    };
    for &i in &[1usize, 3, 5] {
        for c in 0..3 {
            let d = applied.splats[i].position[c] - scene.splats[i].position[c];
            assert!(
                (d - 0.5).abs() < 0.02,
                "splat {i} pos[{c}] residual was {d}",
            );
        }
        let dopa = applied.splats[i].opacity - scene.splats[i].opacity;
        assert!(
            (dopa - expected_opa_delta).abs() < 0.005,
            "splat {i} opacity residual was {dopa} expected ~{expected_opa_delta}",
        );
    }
}

/// Hard-fail: GLB with `extensionsRequired: ["CT_v5_tail_residual"]` and
/// missing sidecar -> returns `MissingTailSidecar`.
#[test]
fn missing_sidecar_required_hard_fails() {
    let dir = tempdir().expect("tempdir");
    let scene = make_synthetic_scene(6, 2);
    let glb_path = dir.path().join("scene.glb");
    let opts = WriteOpts {
        v5_tail: Some(V5TailRef {
            sidecar_uri: "scene.glb.v5tail".to_string(),
            n_splats: scene.len(),
            k_selected: 3,
            sh_rest_coefs: 2,
            n_cells: 2,
            required: true, // <-- hard requirement
        }),
        ..Default::default()
    };
    write_glb(&scene, &glb_path, &opts).expect("write glb with v5_tail ext");
    // No sidecar file written. Strict mode (default) must hard-fail.
    let err =
        read_glb_with_opts(&glb_path, &ReadOpts::default()).expect_err("strict read should error");
    match err {
        GltfError::MissingTailSidecar { uri, tried } => {
            assert_eq!(uri, "scene.glb.v5tail");
            assert!(tried.ends_with("scene.glb.v5tail"), "tried={tried}");
        }
        other => panic!("expected MissingTailSidecar, got {other:?}"),
    }
}

/// Permissive: same GLB but `allow_missing_tail = true` -> warns + returns
/// the baseline (un-residualed) scene.
#[test]
fn missing_sidecar_permissive_warns_and_returns_baseline() {
    let dir = tempdir().expect("tempdir");
    let scene = make_synthetic_scene(6, 2);
    let glb_path = dir.path().join("scene.glb");
    let opts = WriteOpts {
        v5_tail: Some(V5TailRef {
            sidecar_uri: "scene.glb.v5tail".to_string(),
            n_splats: scene.len(),
            k_selected: 3,
            sh_rest_coefs: 2,
            n_cells: 2,
            required: true,
        }),
        ..Default::default()
    };
    write_glb(&scene, &glb_path, &opts).expect("write glb");
    let read_opts = ReadOpts {
        allow_missing_tail: true,
        ..Default::default()
    };
    let loaded = read_glb_with_opts(&glb_path, &read_opts).expect("permissive read should succeed");
    assert_eq!(loaded.len(), scene.len());
    // Position is FP32 in the legacy quality-max path; should round-trip.
    for (a, b) in loaded.splats.iter().zip(scene.splats.iter()) {
        for c in 0..3 {
            assert!(
                (a.position[c] - b.position[c]).abs() < 1e-5,
                "position drift on baseline-only read",
            );
        }
    }
}

/// Full GLB round-trip with the sidecar physically present: `write_glb`
/// emits the extension + we write the `.v5tail` next to it, then
/// `read_glb` decodes and applies the residuals. The selected splats
/// end up shifted by ~+0.5 (residual amplitude) just like the
/// unit-level test above.
#[test]
fn end_to_end_glb_with_sidecar_applies_residuals() {
    let dir = tempdir().expect("tempdir");
    let sh_rest_coefs = 2usize;
    let scene = make_synthetic_scene(8, sh_rest_coefs);
    let glb_path = dir.path().join("scene.glb");
    let sidecar_path = dir.path().join("scene.glb.v5tail");

    let opts = WriteOpts {
        v5_tail: Some(V5TailRef {
            sidecar_uri: "scene.glb.v5tail".to_string(),
            n_splats: scene.len(),
            k_selected: 3,
            sh_rest_coefs: sh_rest_coefs as u8,
            n_cells: 2,
            required: false,
        }),
        ..Default::default()
    };
    write_glb(&scene, &glb_path, &opts).expect("write glb");
    let sidecar_bytes = build_sidecar(&scene, sh_rest_coefs);
    std::fs::write(&sidecar_path, &sidecar_bytes).expect("write sidecar");

    let loaded =
        read_glb_with_opts(&glb_path, &ReadOpts::default()).expect("read glb with sidecar applied");
    assert_eq!(loaded.len(), scene.len());

    // Touched splats: position shifted by ~+0.5 per axis.
    for &i in &[1usize, 3, 5] {
        for c in 0..3 {
            let d = loaded.splats[i].position[c] - scene.splats[i].position[c];
            assert!(
                (d - 0.5).abs() < 0.02,
                "splat {i} pos[{c}] residual was {d}",
            );
        }
    }
    // Untouched splats unchanged.
    for &i in &[0usize, 2, 4, 6, 7] {
        for c in 0..3 {
            assert!(
                (loaded.splats[i].position[c] - scene.splats[i].position[c]).abs() < 1e-5,
                "untouched splat {i} drifted",
            );
        }
    }
}
