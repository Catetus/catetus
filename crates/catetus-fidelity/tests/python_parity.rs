#![allow(clippy::doc_lazy_continuation, clippy::manual_memcpy)]
//! Python ↔ Rust round-trip parity test — FEATURES-ONLY post 2026-05-19.
//!
//! Fixture `tests/data/parity_fixture.json` is produced by the private repo's
//! `research/fidelity-ml/dump_fixture.py` for a known PLY (bonsai_iter7000) and
//! contains the 22 feature values produced by the Python extractor and the
//! MLP score produced by the Python `forward_numpy` reference.
//!
//! Pre-open-core-split: this test asserted Python and Rust agree on the score
//! through the local MLP forward pass. The trained weights now live in the
//! private `catetus-fidelity-ml` crate and `score_ply` returns `Err` from
//! the public crate, so this test only validates feature-vector parity. The
//! MLP-forward parity check moves to the private repo where the weights
//! actually live.

use std::path::PathBuf;

use catetus_fidelity::{extract_features, NUM_FEATURES};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    ply_path: String,
    baseline_path: Option<String>,
    features: Vec<f32>,
    #[allow(dead_code)]
    score: f32,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("parity_fixture.json")
}

#[test]
fn python_rust_feature_parity() {
    let p = fixture_path();
    if !p.exists() {
        eprintln!("[parity] skipping — fixture missing at {}", p.display());
        return;
    }
    let raw = std::fs::read_to_string(&p).expect("read fixture");
    let fx: Fixture = serde_json::from_str(&raw).expect("parse fixture");
    let ply = PathBuf::from(&fx.ply_path);
    if !ply.exists() {
        eprintln!("[parity] skipping — PLY missing at {}", ply.display());
        return;
    }
    let baseline = fx.baseline_path.as_ref().map(PathBuf::from);
    if let Some(b) = &baseline {
        if !b.exists() {
            eprintln!("[parity] skipping — baseline missing at {}", b.display());
            return;
        }
    }
    let rust_feats = extract_features(&ply, baseline.as_deref()).expect("extract_features");

    assert_eq!(rust_feats.len(), NUM_FEATURES);
    assert_eq!(fx.features.len(), NUM_FEATURES);

    for (i, (rv, pv)) in rust_feats.iter().zip(fx.features.iter()).enumerate() {
        let d = (rv - pv).abs();
        assert!(
            d < 1e-3,
            "feature {} mismatch: rust={} py={} delta={}",
            i,
            rv,
            pv,
            d,
        );
    }
}
