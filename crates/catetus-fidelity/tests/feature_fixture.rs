//! Feature-vector extraction on a known fixture (bonsai_iter7000.ply).
//!
//! Post-open-core-split (2026-05-19): the trained MLP head is hosted-only,
//! so this test no longer exercises `score_ply`. Instead it asserts that the
//! local 22-feature extractor still produces the documented shape + the
//! identity-mode invariant on `pos_centroid_drift`.

use std::path::PathBuf;

use catetus_fidelity::{extract_features, NUM_FEATURES};

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
    let feats = extract_features(&cand, None).expect("extract_features identity");
    assert_eq!(feats.len(), NUM_FEATURES);

    // identity-mode: pos_centroid_drift (feature 0) MUST be 0 because we
    // don't have a real baseline centroid.
    assert!(
        feats[0].abs() < 1e-9,
        "identity drift must be 0: {}",
        feats[0]
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
    let feats = extract_features(&cand, Some(&base)).expect("extract_features baseline");
    // pos_extent_ratio (feature 1) should be > 0 with a real baseline.
    assert!(feats[1] > 0.0);
}
