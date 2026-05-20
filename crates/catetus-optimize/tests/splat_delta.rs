//! Integration tests for the `SplatDelta` pass. The unit tests in
//! `src/splat_delta.rs` cover the codec internals; these tests exercise the
//! preset wiring and the public reportable contract on a synthetic 1k-splat
//! scene.

use catetus_core::{Color, Splat, SplatScene};
use catetus_optimize::{preset, take_last_delta_stream, MortonSort, Pass, PassContext, SplatDelta};

fn make_synth_scene(n: usize, seed: u64) -> SplatScene {
    // Deterministic cluster + jitter so adjacent Morton-order splats share
    // most attributes (the codec's assumption).
    let mut x = seed.wrapping_mul(0xdeadbeef).wrapping_add(1);
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let mut scene = SplatScene::new();
    for i in 0..n {
        let cluster = i / 16;
        let cx = (cluster as f32 * 0.137) % 4.0 - 2.0;
        let cy = ((cluster as f32 * 0.211) % 4.0) - 2.0;
        let cz = ((cluster as f32 * 0.299) % 4.0) - 2.0;
        let jx = rng() * 0.05;
        let jy = rng() * 0.05;
        let jz = rng() * 0.05;
        let qa = rng();
        let qb = rng();
        let qc = rng();
        let qd = rng();
        let qn = (qa * qa + qb * qb + qc * qc + qd * qd).sqrt().max(1e-9);
        scene.splats.push(Splat {
            position: [cx + jx, cy + jy, cz + jz],
            rotation: [qa / qn, qb / qn, qc / qn, qd / qn],
            scale: [
                0.05 + 0.02 * rng().abs(),
                0.05 + 0.02 * rng().abs(),
                0.05 + 0.02 * rng().abs(),
            ],
            opacity: 0.5 + 0.5 * rng().abs(),
            color: Color::Rgb([0.5 + 0.4 * rng(), 0.5 + 0.4 * rng(), 0.5 + 0.4 * rng()]),
        });
    }
    scene
}

#[test]
fn splat_delta_after_morton_is_deterministic() {
    let mut a = make_synth_scene(1024, 1);
    let mut b = make_synth_scene(1024, 1);
    let mut ctx = PassContext::default();
    MortonSort.run(&mut a, &mut ctx).unwrap();
    MortonSort.run(&mut b, &mut ctx).unwrap();
    SplatDelta::default().run(&mut a, &mut ctx).unwrap();
    let blob_a = take_last_delta_stream().expect("delta blob a");
    SplatDelta::default().run(&mut b, &mut ctx).unwrap();
    let blob_b = take_last_delta_stream().expect("delta blob b");
    assert_eq!(blob_a.compressed, blob_b.compressed);
    for (sa, sb) in a.splats.iter().zip(b.splats.iter()) {
        assert_eq!(sa.position, sb.position);
        assert_eq!(sa.rotation, sb.rotation);
        assert_eq!(sa.scale, sb.scale);
        assert_eq!(sa.opacity, sb.opacity);
    }
}

#[test]
fn web_mobile_delta_preset_runs_end_to_end_and_emits_sidecar() {
    // The preset should: prune + Morton + SplatDelta + BuildLOD without
    // tripping any pass. The sidecar register should be populated.
    let _ = take_last_delta_stream(); // drain anything stale from earlier tests.
    let mut scene = make_synth_scene(1024, 2);
    let pipe = preset("web-mobile-delta").expect("preset known");
    let report = pipe.run(&mut scene).unwrap();
    // Expect SplatDelta in the pass list.
    let names: Vec<&str> = report.passes.iter().map(|p| p.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n == &"SplatDelta"),
        "SplatDelta missing from pipeline: {names:?}"
    );
    let blob = take_last_delta_stream().expect("sidecar must be drainable");
    assert!(blob.compressed.starts_with(b"SFDL"[..2].as_ref()) || blob.compressed.len() > 16);
    assert!(blob.stats.n_anchor > 0);
    assert!(blob.stats.n_anchor + blob.stats.n_nonanchor == blob.stats.n);
    // RMSE on synthetic-cluster data should be modest (well under 1.0).
    assert!(
        blob.stats.rmse_recon < 0.5,
        "rmse {} above sanity bound 0.5",
        blob.stats.rmse_recon
    );
}

#[test]
fn splat_delta_take_drains_only_once() {
    let mut scene = make_synth_scene(512, 3);
    let mut ctx = PassContext::default();
    SplatDelta::default().run(&mut scene, &mut ctx).unwrap();
    let first = take_last_delta_stream();
    let second = take_last_delta_stream();
    assert!(first.is_some());
    assert!(
        second.is_none(),
        "register should drain on first take_last_delta_stream"
    );
}
