use splatforge_core::{scene_hash, AnalyzeReport, Color, Splat, SplatScene};

fn three_splats() -> SplatScene {
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
fn hash_is_stable() {
    let a = three_splats();
    let b = three_splats();
    assert_eq!(scene_hash(&a), scene_hash(&b));
    assert!(scene_hash(&a).starts_with("blake3:"));
}

#[test]
fn hash_changes_with_position() {
    let mut a = three_splats();
    let b = three_splats();
    a.splats[0].position[0] += 1.0;
    assert_ne!(scene_hash(&a), scene_hash(&b));
}

#[test]
fn ir_preserves_three_splats() {
    let scene = three_splats();
    assert_eq!(scene.len(), 3);
    assert_eq!(scene.iter().count(), 3);
}

#[test]
fn report_is_deterministic() {
    let scene = three_splats();
    let r1 = AnalyzeReport::from_scene(&scene, "ply", 1024).to_json(false);
    let r2 = AnalyzeReport::from_scene(&scene, "ply", 1024).to_json(false);
    assert_eq!(r1, r2);
    assert!(r1.contains("\"splatCount\": 3"));
}

#[test]
fn report_keys_are_sorted() {
    let scene = three_splats();
    let json = AnalyzeReport::from_scene(&scene, "ply", 0).to_json(false);
    // boundingBox key should appear before format key in sorted order.
    let bb = json.find("bounding").unwrap_or(usize::MAX);
    let fm = json.find("\"format\"").unwrap_or(0);
    assert!(bb < fm, "keys must be lex-sorted: {json}");
}
