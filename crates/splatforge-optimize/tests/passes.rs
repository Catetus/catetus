use splatforge_core::{Color, SemanticLabel, Splat, SplatScene};
use splatforge_optimize::{
    preset, AspectRatioPrune, BackgroundOverdrawPrune, BuildLOD, FloaterPrune, MortonSort,
    ObjectAwarePruneExperimental, OpacityPrune, Pass, PassContext, Pipeline, ReduceSHDegree,
    RemoveInvalidSplats,
};

fn make_scene(n: usize) -> SplatScene {
    let mut scene = SplatScene::new();
    for i in 0..n {
        let f = i as f32 * 0.01;
        scene.splats.push(Splat {
            position: [f, -f, f * 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: if i < 20 { 0.001 } else { 0.9 },
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
    }
    scene
}

#[test]
fn opacity_prune_removes_low_splats() {
    let mut scene = make_scene(100);
    let mut ctx = PassContext::default();
    let stats = OpacityPrune { threshold: 0.01 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 20);
    assert_eq!(scene.splats.len(), 80);
}

#[test]
fn morton_sort_is_deterministic() {
    let mut a = make_scene(50);
    let mut b = make_scene(50);
    let mut ctx = PassContext::default();
    MortonSort.run(&mut a, &mut ctx).unwrap();
    MortonSort.run(&mut b, &mut ctx).unwrap();
    let positions_a: Vec<_> = a.splats.iter().map(|s| s.position).collect();
    let positions_b: Vec<_> = b.splats.iter().map(|s| s.position).collect();
    assert_eq!(positions_a, positions_b);
}

#[test]
fn remove_invalid_drops_nan() {
    let mut scene = make_scene(5);
    scene.splats[2].position[0] = f32::NAN;
    let mut ctx = PassContext::default();
    let stats = RemoveInvalidSplats.run(&mut scene, &mut ctx).unwrap();
    assert_eq!(stats.removed, 1);
    assert_eq!(scene.splats.len(), 4);
}

#[test]
fn reduce_sh_collapses_to_rgb() {
    let mut scene = SplatScene::new();
    scene.splats.push(Splat {
        position: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        opacity: 0.5,
        color: Color::Sh {
            degree: 2,
            coeffs: vec![0.1; 27],
        },
    });
    let mut ctx = PassContext::default();
    ReduceSHDegree { target_degree: 0 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert!(matches!(scene.splats[0].color, Color::Rgb(_)));
}

#[test]
fn reduce_sh_to_degree_1_preserves_degree_1_coeffs() {
    // Degree-3 source: 3 DC + (3+5+7)*3 = 48 scalars total.
    let mut scene = SplatScene::new();
    let mut coeffs = Vec::with_capacity(48);
    for i in 0..48 {
        coeffs.push(0.01 * (i as f32));
    }
    scene.splats.push(Splat {
        position: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        opacity: 0.5,
        color: Color::Sh {
            degree: 3,
            coeffs: coeffs.clone(),
        },
    });
    let mut ctx = PassContext::default();
    ReduceSHDegree { target_degree: 1 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    // Expect degree=1 with 3 DC + 3*3 = 12 scalars, preserving the first 12.
    match &scene.splats[0].color {
        Color::Sh {
            degree,
            coeffs: out,
        } => {
            assert_eq!(*degree, 1, "target_degree=1 must reduce to degree 1");
            assert_eq!(out.len(), 12, "degree-1 SH has 4 bands × 3 channels = 12");
            for i in 0..12 {
                assert!(
                    (out[i] - coeffs[i]).abs() < 1e-9,
                    "coefficient {i} drifted: {} vs {}",
                    out[i],
                    coeffs[i]
                );
            }
        }
        other => panic!("expected SH degree 1, got {other:?}"),
    }
}

#[test]
fn preset_web_mobile_builds() {
    let pipe = preset("web-mobile").unwrap();
    assert!(!pipe.passes.is_empty());
    let mut scene = make_scene(100);
    let report = pipe.run(&mut scene).unwrap();
    assert!(report.splats_after <= report.splats_before);
}

#[test]
fn preset_unknown_errors() {
    assert!(preset("does-not-exist").is_err());
}

#[test]
fn build_lod_levels_are_subsamples() {
    let mut scene = make_scene(100);
    let mut ctx = PassContext::default();
    let stats = BuildLOD {
        levels: vec![0.5, 0.25],
    }
    .run(&mut scene, &mut ctx)
    .unwrap();
    assert!(!stats.notes.is_empty());
    let lods = scene.lods.as_ref().expect("lods populated");
    assert_eq!(lods.len(), 3);
    assert_eq!(lods[0].indices.len(), 100);
    // Stride 2 over 100 -> 50; stride 4 over 100 -> 25.
    assert_eq!(lods[1].indices.len(), 50);
    assert_eq!(lods[2].indices.len(), 25);

    // Determinism: re-run produces the same indices.
    let mut scene2 = make_scene(100);
    let mut ctx2 = PassContext::default();
    BuildLOD {
        levels: vec![0.5, 0.25],
    }
    .run(&mut scene2, &mut ctx2)
    .unwrap();
    assert_eq!(
        scene.lods.as_ref().unwrap()[1].indices,
        scene2.lods.as_ref().unwrap()[1].indices,
    );
}

#[test]
fn object_aware_prune_preserves_protected_labels() {
    // 100 splats. The first 10 are "product" placed at sparse positions; the
    // remaining 90 are tightly clustered around origin.
    let mut scene = SplatScene::new();
    let mut labels: Vec<SemanticLabel> = Vec::new();
    for i in 0..10 {
        let f = 100.0 + i as f32 * 5.0;
        scene.splats.push(Splat {
            position: [f, f, f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.05, // low opacity to exercise the pruning branch
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
        labels.push(SemanticLabel("product".to_string()));
    }
    for i in 0..90 {
        let f = i as f32 * 0.001;
        scene.splats.push(Splat {
            position: [f, -f, f * 0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.5,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
        labels.push(SemanticLabel("background".to_string()));
    }
    scene.semantic_labels = Some(labels);

    let mut ctx = PassContext::default();
    let pass = ObjectAwarePruneExperimental {
        protect_labels: vec!["product".to_string()],
    };
    pass.run(&mut scene, &mut ctx).unwrap();

    // No product-labeled splats should be missing.
    let labels = scene.semantic_labels.as_ref().expect("labels survive");
    let product_count = labels.iter().filter(|l| l.0 == "product").count();
    assert_eq!(product_count, 10, "all product splats preserved");
}

#[test]
fn object_aware_prune_flat_scene_is_conservative() {
    // Uniformly-spaced lattice — density is uniform, so the lowest decile is
    // small and pruning should be modest (and bounded by the decile size).
    let mut scene = SplatScene::new();
    let side = 5;
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                scene.splats.push(Splat {
                    position: [x as f32, y as f32, z as f32],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    opacity: 0.05, // low opacity to enable pruning candidacy
                    color: Color::Rgb([0.5, 0.5, 0.5]),
                });
            }
        }
    }
    let n = scene.splats.len();
    let mut ctx = PassContext::default();
    let stats = ObjectAwarePruneExperimental::default()
        .run(&mut scene, &mut ctx)
        .unwrap();
    // Strict upper bound: never prune more than ~the lowest decile.
    let decile = n.div_ceil(10);
    assert!(
        stats.removed <= decile,
        "pruned {} but decile is {}",
        stats.removed,
        decile
    );
}

#[test]
fn empty_pipeline_runs() {
    let pipe = Pipeline::new();
    let mut scene = make_scene(3);
    let r = pipe.run(&mut scene).unwrap();
    assert_eq!(r.splats_before, 3);
    assert_eq!(r.splats_after, 3);
}

// --- FloaterPrune (k-NN) -------------------------------------------------

fn splat_at(p: [f32; 3]) -> Splat {
    Splat {
        position: p,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        opacity: 0.9,
        color: Color::Rgb([0.5, 0.5, 0.5]),
    }
}

/// Build a deterministic "subject + halo" scene. The subject is a dense
/// 12×12×12 grid (1728 splats) inside a unit cube; the halo is 50 splats
/// scattered uniformly across a 10× larger volume. The halo MUST be pruned
/// (sparse, isolated) while the subject MUST survive intact.
fn subject_and_halo() -> (SplatScene, usize, usize) {
    let mut scene = SplatScene::new();
    let subject_n = 12usize;
    for i in 0..subject_n {
        for j in 0..subject_n {
            for k in 0..subject_n {
                let x = (i as f32 / (subject_n - 1) as f32) * 2.0 - 1.0;
                let y = (j as f32 / (subject_n - 1) as f32) * 2.0 - 1.0;
                let z = (k as f32 / (subject_n - 1) as f32) * 2.0 - 1.0;
                scene.splats.push(splat_at([x, y, z]));
            }
        }
    }
    let subject_count = scene.splats.len(); // 1728
                                              // Halo: deterministic LCG, scatter across [-10, 10]³, far from subject.
    let mut s: u32 = 0x12345678;
    let halo_n = 50usize;
    for _ in 0..halo_n {
        // Three independent u32 LCG draws → [-10, 10] each.
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 20.0 - 10.0
        };
        // Reject anything inside the subject ±1.5 envelope so we're guaranteed
        // a sparse-vs-dense contrast, not a tied near-neighbor.
        loop {
            let p = [next(), next(), next()];
            if p[0].abs() > 1.5 || p[1].abs() > 1.5 || p[2].abs() > 1.5 {
                scene.splats.push(splat_at(p));
                break;
            }
        }
    }
    (scene, subject_count, halo_n)
}

#[test]
fn floater_prune_removes_isolated_halo_but_keeps_subject() {
    let (mut scene, subject_count, halo_n) = subject_and_halo();
    let before = scene.splats.len();
    assert_eq!(before, subject_count + halo_n);

    let mut ctx = PassContext::default();
    let stats = FloaterPrune::default().run(&mut scene, &mut ctx).unwrap();

    // The halo should be entirely pruned; some grid-boundary points may also
    // get pruned because they have fewer than k=8 neighbors inside the
    // sphere of distance R (corner subject points have ~7-13 grid neighbors).
    // Accept any removal in [halo_n, halo_n + grid_corner_slack] but require
    // the dense interior of the subject to survive — at least 90% of it.
    assert!(
        stats.removed >= halo_n,
        "removed {} < halo_n {}; FloaterPrune didn't catch the halo",
        stats.removed,
        halo_n
    );
    let kept_subject_floor = (subject_count as f64 * 0.90) as usize;
    assert!(
        scene.splats.len() >= kept_subject_floor,
        "kept {} < {} (90% of subject); FloaterPrune over-pruned the dense subject",
        scene.splats.len(),
        kept_subject_floor
    );
}

#[test]
fn floater_prune_is_deterministic() {
    let (mut a, _, _) = subject_and_halo();
    let (mut b, _, _) = subject_and_halo();
    let mut ctx = PassContext::default();
    let sa = FloaterPrune::default().run(&mut a, &mut ctx).unwrap();
    let sb = FloaterPrune::default().run(&mut b, &mut ctx).unwrap();
    assert_eq!(sa.removed, sb.removed);
    let pa: Vec<_> = a.splats.iter().map(|s| s.position).collect();
    let pb: Vec<_> = b.splats.iter().map(|s| s.position).collect();
    assert_eq!(pa, pb);
}

#[test]
fn floater_prune_noop_below_k() {
    // Default k=8; n=5 must not panic and must not remove anything.
    let mut scene = SplatScene::new();
    for i in 0..5 {
        scene
            .splats
            .push(splat_at([i as f32 * 0.1, 0.0, 0.0]));
    }
    let mut ctx = PassContext::default();
    let stats = FloaterPrune::default().run(&mut scene, &mut ctx).unwrap();
    assert_eq!(stats.removed, 0);
    assert_eq!(scene.splats.len(), 5);
}

#[test]
fn floater_prune_keeps_uniform_grid() {
    // A uniform 10×10×10 grid (1000 splats) has near-zero MAD; the floor
    // clamp must prevent over-pruning. Accept up to 5% removal as the
    // grid-boundary artifact.
    let mut scene = SplatScene::new();
    let n = 10;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                scene.splats.push(splat_at([
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                    k as f32 * 0.1,
                ]));
            }
        }
    }
    let before = scene.splats.len();
    let mut ctx = PassContext::default();
    let stats = FloaterPrune::default().run(&mut scene, &mut ctx).unwrap();
    let removed_pct = stats.removed as f64 / before as f64;
    assert!(
        removed_pct <= 0.05,
        "uniform grid lost {:.1}% (expected ≤ 5%)",
        removed_pct * 100.0
    );
}

#[test]
fn floater_prune_reports_threshold_stats() {
    // After running on a real subject+halo, the notes must surface the
    // median / MAD / threshold so an operator can sanity-check the pass
    // without rebuilding the binary. Catches regressions where someone
    // strips the notes vec.
    let (mut scene, _, _) = subject_and_halo();
    let mut ctx = PassContext::default();
    let stats = FloaterPrune::default().run(&mut scene, &mut ctx).unwrap();
    let notes_blob = stats.notes.join(" ");
    assert!(
        notes_blob.contains("knn_median=")
            && notes_blob.contains("mad=")
            && notes_blob.contains("threshold="),
        "expected diagnostic notes, got: {notes_blob}"
    );
}

// --- AspectRatioPrune ----------------------------------------------------

fn splat_with_scale(scale: [f32; 3]) -> Splat {
    Splat {
        position: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale,
        opacity: 0.9,
        color: Color::Rgb([0.5, 0.5, 0.5]),
    }
}

#[test]
fn aspect_ratio_prune_drops_needle_splat() {
    // A "needle" splat: 1.0 / 0.001 = 1000.0 ratio, well above 5.0.
    let mut scene = SplatScene::new();
    scene.splats.push(splat_with_scale([1.0, 0.001, 0.001]));
    let mut ctx = PassContext::default();
    let stats = AspectRatioPrune { max_ratio: 5.0 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 1);
    assert_eq!(scene.splats.len(), 0);
}

#[test]
fn aspect_ratio_prune_keeps_normal_splat() {
    // Isotropic 1:1:1 scale → ratio 1.0, far below 5.0.
    let mut scene = SplatScene::new();
    scene.splats.push(splat_with_scale([1.0, 1.0, 1.0]));
    let mut ctx = PassContext::default();
    let stats = AspectRatioPrune { max_ratio: 5.0 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 0);
    assert_eq!(scene.splats.len(), 1);
}

#[test]
fn aspect_ratio_prune_is_deterministic() {
    // Same input → same output, twice in a row.
    let mut a = SplatScene::new();
    let mut b = SplatScene::new();
    for s in [
        [1.0, 1.0, 1.0],
        [1.0, 0.001, 0.001],
        [2.0, 1.0, 0.5],
        [10.0, 0.1, 0.1],
        [1.0, 0.5, 0.5],
    ] {
        a.splats.push(splat_with_scale(s));
        b.splats.push(splat_with_scale(s));
    }
    let mut ctx = PassContext::default();
    let sa = AspectRatioPrune { max_ratio: 8.0 }
        .run(&mut a, &mut ctx)
        .unwrap();
    let sb = AspectRatioPrune { max_ratio: 8.0 }
        .run(&mut b, &mut ctx)
        .unwrap();
    assert_eq!(sa.removed, sb.removed);
    let pa: Vec<_> = a.splats.iter().map(|s| s.scale).collect();
    let pb: Vec<_> = b.splats.iter().map(|s| s.scale).collect();
    assert_eq!(pa, pb);
}

#[test]
fn aspect_ratio_prune_surfaces_stats_notes() {
    // After dropping a known needle, the notes vec must report the count,
    // median dropped ratio, and max dropped ratio so an operator can
    // sanity-check the pass without rerunning the pipeline.
    let mut scene = SplatScene::new();
    scene.splats.push(splat_with_scale([1.0, 1.0, 1.0])); // keeper
    scene.splats.push(splat_with_scale([100.0, 1.0, 1.0])); // dropped ratio=100
    scene.splats.push(splat_with_scale([1.0, 0.01, 1.0])); // dropped ratio=100
    let mut ctx = PassContext::default();
    let stats = AspectRatioPrune { max_ratio: 8.0 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 2);
    let notes_blob = stats.notes.join(" ");
    assert!(
        notes_blob.contains("dropped=2")
            && notes_blob.contains("median_dropped_ratio=")
            && notes_blob.contains("max_dropped_ratio="),
        "expected diagnostic notes, got: {notes_blob}"
    );
}

#[test]
fn aspect_ratio_prune_threshold_boundary() {
    // Exact-ratio splat (ratio == threshold) is KEPT (strict >). One just
    // above the threshold is DROPPED. Catches off-by-epsilon regressions.
    let mut scene = SplatScene::new();
    scene.splats.push(splat_with_scale([5.0, 1.0, 1.0])); // ratio exactly 5.0 → keep
    scene.splats.push(splat_with_scale([5.01, 1.0, 1.0])); // ratio > 5.0 → drop
    let mut ctx = PassContext::default();
    let stats = AspectRatioPrune { max_ratio: 5.0 }
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 1);
    assert_eq!(scene.splats.len(), 1);
    assert_eq!(scene.splats[0].scale, [5.0, 1.0, 1.0]);
}

#[test]
fn aspect_ratio_prune_handles_degenerate_zero_scale() {
    // A zero-axis splat is degenerate (unrenderable). The pass drops it
    // rather than dividing by zero.
    let mut scene = SplatScene::new();
    scene.splats.push(splat_with_scale([1.0, 0.0, 1.0]));
    scene.splats.push(splat_with_scale([1.0, 1.0, 1.0]));
    let mut ctx = PassContext::default();
    let stats = AspectRatioPrune::default()
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 1);
    assert_eq!(scene.splats.len(), 1);
    assert_eq!(scene.splats[0].scale, [1.0, 1.0, 1.0]);
}

#[test]
fn aspect_ratio_prune_in_web_mobile_preset() {
    // The web-mobile preset must include AspectRatioPrune (between
    // OpacityPrune and FloaterPrune) — a stuffed-with-needles scene must
    // see them dropped before MortonSort/quantization.
    let mut scene = SplatScene::new();
    // 20 healthy splats spread in a small cube
    for i in 0..20 {
        let f = i as f32 * 0.05;
        scene.splats.push(Splat {
            position: [f, -f, f * 0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.05, 0.05, 0.05],
            opacity: 0.9,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
    }
    // 10 needles
    for i in 0..10 {
        let f = i as f32 * 0.05 + 1.0;
        scene.splats.push(Splat {
            position: [f, f, f],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 0.001, 0.001], // ratio 1000
            opacity: 0.9,
            color: Color::Rgb([0.5, 0.5, 0.5]),
        });
    }
    let before = scene.splats.len();
    let pipe = preset("web-mobile").unwrap();
    let report = pipe.run(&mut scene).unwrap();
    assert!(report.splats_after < before);
    // No survivor should have ratio > 10 (the preset threshold).
    for s in &scene.splats {
        let smin = s.scale[0].abs().min(s.scale[1].abs()).min(s.scale[2].abs());
        let smax = s.scale[0].abs().max(s.scale[1].abs()).max(s.scale[2].abs());
        assert!(
            smin > 0.0 && smax / smin <= 10.0 + 1e-3,
            "needle survived web-mobile: scale={:?}",
            s.scale
        );
    }
}

// ----------------------------------------------------------------------------
// BackgroundOverdrawPrune tests
// ----------------------------------------------------------------------------

fn splat_full(pos: [f32; 3], scale: [f32; 3], opacity: f32) -> Splat {
    Splat {
        position: pos,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale,
        opacity,
        color: Color::Rgb([0.5, 0.5, 0.5]),
    }
}

#[test]
fn background_overdraw_prune_drops_large_faint_splats() {
    // 100 small bright "subject" splats + 5 huge faint "background" splats.
    // top_fraction=0.05 => 5 candidates; all 5 should match the faint test.
    let mut scene = SplatScene::new();
    for _ in 0..100 {
        scene.splats.push(splat_full([0.0; 3], [0.01, 0.01, 0.01], 0.95));
    }
    for _ in 0..5 {
        scene.splats.push(splat_full([10.0; 3], [3.0, 3.0, 3.0], 0.1));
    }
    let mut ctx = PassContext::default();
    let pass = BackgroundOverdrawPrune::default();
    let stats = pass.run(&mut scene, &mut ctx).unwrap();
    assert_eq!(stats.removed, 5, "expected to drop 5 large faint splats");
    assert_eq!(scene.splats.len(), 100);
    // All survivors should be the small bright ones.
    for s in &scene.splats {
        assert!(s.opacity > 0.9);
    }
    assert!(!stats.notes.is_empty());
}

#[test]
fn background_overdraw_prune_protects_bright_overdrawers() {
    // 100 small bright subject splats + 5 huge BRIGHT splats (highlights).
    // The huge bright ones are top-by-cost but should NOT be dropped because
    // opacity >= opacity_keep_above (default 0.5).
    let mut scene = SplatScene::new();
    for _ in 0..100 {
        scene.splats.push(splat_full([0.0; 3], [0.01, 0.01, 0.01], 0.95));
    }
    for _ in 0..5 {
        scene.splats.push(splat_full([10.0; 3], [3.0, 3.0, 3.0], 0.9));
    }
    let mut ctx = PassContext::default();
    let stats = BackgroundOverdrawPrune::default()
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(
        stats.removed, 0,
        "bright splats should be immune even when cost is in top fraction"
    );
    assert_eq!(scene.splats.len(), 105);
}

#[test]
fn background_overdraw_prune_is_deterministic() {
    let make = || {
        let mut sc = SplatScene::new();
        for i in 0..200 {
            let big = i % 17 == 0;
            let scale = if big { 4.0 } else { 0.05 };
            let op = if big { 0.2 } else { 0.8 };
            sc.splats
                .push(splat_full([i as f32, 0.0, 0.0], [scale, scale, scale], op));
        }
        sc
    };
    let mut a = make();
    let mut b = make();
    let mut ctx = PassContext::default();
    BackgroundOverdrawPrune::default()
        .run(&mut a, &mut ctx)
        .unwrap();
    BackgroundOverdrawPrune::default()
        .run(&mut b, &mut ctx)
        .unwrap();
    let pa: Vec<_> = a.splats.iter().map(|s| s.position).collect();
    let pb: Vec<_> = b.splats.iter().map(|s| s.position).collect();
    assert_eq!(pa, pb);
}

#[test]
fn background_overdraw_prune_handles_empty_scene() {
    let mut scene = SplatScene::new();
    let mut ctx = PassContext::default();
    let stats = BackgroundOverdrawPrune::default()
        .run(&mut scene, &mut ctx)
        .unwrap();
    assert_eq!(stats.removed, 0);
}

#[test]
fn background_overdraw_prune_respects_top_fraction_zero() {
    let mut scene = SplatScene::new();
    for _ in 0..100 {
        scene.splats.push(splat_full([0.0; 3], [0.01, 0.01, 0.01], 0.95));
    }
    for _ in 0..5 {
        scene.splats.push(splat_full([10.0; 3], [3.0, 3.0, 3.0], 0.1));
    }
    let mut ctx = PassContext::default();
    let pass = BackgroundOverdrawPrune {
        top_fraction: 0.0,
        opacity_keep_above: 0.5,
    };
    let stats = pass.run(&mut scene, &mut ctx).unwrap();
    assert_eq!(stats.removed, 0, "top_fraction=0 should drop nothing");
}

#[test]
fn hero_quality_preset_runs_clean() {
    let pipe = preset("hero-quality").unwrap();
    assert!(!pipe.passes.is_empty());
    // Build a small mixed scene: bright subject + faint background overdraw +
    // a needle to exercise AspectRatioPrune within the pipeline.
    let mut scene = SplatScene::new();
    for i in 0..200 {
        scene
            .splats
            .push(splat_full([i as f32 * 0.01, 0.0, 0.0], [0.05; 3], 0.8));
    }
    for _ in 0..10 {
        scene.splats.push(splat_full([5.0; 3], [3.0, 3.0, 3.0], 0.1));
    }
    scene
        .splats
        .push(splat_full([0.0; 3], [10.0, 0.05, 0.05], 0.9));
    let report = pipe.run(&mut scene).unwrap();
    assert!(report.splats_after <= report.splats_before);
    assert!(report.splats_after > 0);
}
