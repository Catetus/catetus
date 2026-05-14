use splatforge_core::{Color, SemanticLabel, Splat, SplatScene};
use splatforge_optimize::{
    preset, BuildLOD, MortonSort, ObjectAwarePruneExperimental, OpacityPrune, Pass, PassContext,
    Pipeline, ReduceSHDegree, RemoveInvalidSplats,
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
    let decile = (n + 9) / 10;
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
