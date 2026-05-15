//! Feature-vector extraction on a known fixture (bonsai_iter7000.ply).
//!
//! Sanity-checks that:
//!   - the candidate PLY parses through `splatforge-ply`
//!   - the resulting 22-vector has the documented shape
//!   - identity-mode (no baseline) leaves drift==0
//!   - baseline-mode produces a finite, in-range score

use std::path::PathBuf;

use splatforge_fidelity::{score_ply, NUM_FEATURES};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches")
        .join("scenes")
}

#[test]
fn feature_extraction_on_bonsai_real() {
    let cand = fixture_dir().join("real").join("bonsai_iter7000.ply");
    if !cand.exists() {
        eprintln!("[fixture] skipping — {} missing", cand.display());
        return;
    }
    let r = score_ply(&cand, None).expect("score_ply identity");
    assert_eq!(r.features.len(), NUM_FEATURES);
    assert_eq!(r.feature_names.len(), NUM_FEATURES);
    assert!(r.score.is_finite(), "score not finite: {}", r.score);
    assert!(
        (0.0..=1.0).contains(&r.score),
        "score out of [0,1]: {}",
        r.score
    );

    // identity-mode: pos_centroid_drift (feature 0) MUST be 0 because we
    // don't have a real baseline centroid.
    assert!(
        r.features[0].abs() < 1e-9,
        "identity drift must be 0: {}",
        r.features[0]
    );
}

#[test]
fn feature_extraction_with_baseline() {
    let cand = fixture_dir().join("real").join("bonsai_iter7000.ply");
    let base = fixture_dir().join("bonsai_mipnerf360_iter7k.ply");
    if !cand.exists() || !base.exists() {
        eprintln!("[fixture] skipping — bonsai fixtures missing");
        return;
    }
    let r = score_ply(&cand, Some(&base)).expect("score_ply baseline");
    assert!(r.score.is_finite() && (0.0..=1.0).contains(&r.score));
    assert!(r.baseline_used);
    // pos_extent_ratio (feature 1) should be > 0 with a real baseline.
    assert!(r.features[1] > 0.0);
    // bootstrap flag must be true at this point in the project (no real
    // ratings yet) — fail loudly if the metadata says otherwise so we
    // notice when production data flips the flag.
    assert!(r.bootstrap, "expected bootstrap=true with no real ratings");
}
