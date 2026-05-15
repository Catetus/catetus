#![deny(clippy::all)]
#![allow(clippy::needless_range_loop)]
//! Predict-only fidelity scorer for Gaussian-splat scenes.
//!
//! v0.4 uses a small 22 → 64 → 32 → 1 MLP with ReLU + sigmoid, trained
//! against Bradley-Terry-aggregated human pairwise ratings (or, when the
//! ratings table has <5 rows, a synthetic bootstrap derived from the
//! splatbench corruption corpus — see `metadata-v0.4.json::bootstrap`).
//!
//! API
//! ---
//! ```ignore
//! use splatforge_fidelity::{score_ply, ScoreReport};
//! let r: ScoreReport = score_ply("scene.ply", Some("baseline.ply"))?;
//! println!("{}", r.score); // f32 in [0,1]
//! ```
//!
//! The Rust binary `splatforge-fidelity` (in `src/bin/`) is the CLI form
//! invoked by `splatforge fidelity-score`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use splatforge_ply::read_ply;

pub mod features;
pub mod weights_v04;

pub use features::{
    build_feature_vector, summarise, FEATURE_NAMES, IDENTITY, NUM_FEATURES,
};

/// JSON-serialisable score report. The CLI emits this verbatim.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreReport {
    pub version: &'static str,
    pub score: f32,
    pub baseline_used: bool,
    pub features: Vec<f32>,
    pub feature_names: Vec<&'static str>,
    pub bootstrap: bool,
}

const VERSION: &str = "0.4.0-mlp22";

/// Forward the 22-vector through the embedded MLP. Reference implementation —
/// every test in this crate's `tests/` directory eventually lands here.
pub fn forward(features: &[f32; NUM_FEATURES]) -> f32 {
    // (1) normalise.
    let mut x = [0.0f32; NUM_FEATURES];
    for i in 0..NUM_FEATURES {
        x[i] = (features[i] - weights_v04::FEATURE_MEANS[i]) / weights_v04::FEATURE_STDS[i];
    }

    // (2) layer 1: x @ W1 + b1, ReLU.
    let mut h1 = [0.0f32; weights_v04::HIDDEN_1];
    for j in 0..weights_v04::HIDDEN_1 {
        let mut s = weights_v04::B1[j];
        for i in 0..NUM_FEATURES {
            s += x[i] * weights_v04::W1[i][j];
        }
        h1[j] = if s > 0.0 { s } else { 0.0 };
    }

    // (3) layer 2: h1 @ W2 + b2, ReLU.
    let mut h2 = [0.0f32; weights_v04::HIDDEN_2];
    for j in 0..weights_v04::HIDDEN_2 {
        let mut s = weights_v04::B2[j];
        for i in 0..weights_v04::HIDDEN_1 {
            s += h1[i] * weights_v04::W2[i][j];
        }
        h2[j] = if s > 0.0 { s } else { 0.0 };
    }

    // (4) output: h2 @ W3 + b3, sigmoid.
    let mut z = weights_v04::B3[0];
    for i in 0..weights_v04::HIDDEN_2 {
        z += h2[i] * weights_v04::W3[i][0];
    }
    1.0 / (1.0 + (-z).exp())
}

/// Score a single PLY against an optional baseline. Returns the full report.
pub fn score_ply(cand: &Path, baseline: Option<&Path>) -> Result<ScoreReport> {
    let cand_scene = read_ply(cand)
        .with_context(|| format!("read candidate PLY {}", cand.display()))?;
    let cand_summary = summarise(&cand_scene);
    let base_summary = match baseline {
        Some(p) => {
            let s = read_ply(p)
                .with_context(|| format!("read baseline PLY {}", p.display()))?;
            Some(summarise(&s))
        }
        None => None,
    };
    let feats = build_feature_vector(&cand_summary, base_summary.as_ref());
    let score = forward(&feats);
    Ok(ScoreReport {
        version: VERSION,
        score,
        baseline_used: base_summary.is_some(),
        features: feats.to_vec(),
        feature_names: FEATURE_NAMES.to_vec(),
        bootstrap: is_bootstrap(),
    })
}

fn is_bootstrap() -> bool {
    // Cheap parse of the metadata sidecar — done once.
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        serde_json::from_str::<serde_json::Value>(weights_v04::METADATA_JSON)
            .ok()
            .and_then(|v| v.get("bootstrap").and_then(|b| b.as_bool()))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_in_unit_interval() {
        let f = [0.0f32; NUM_FEATURES];
        let s = forward(&f);
        assert!(s.is_finite());
        assert!((0.0..=1.0).contains(&s), "score out of [0,1]: {s}");
    }

    #[test]
    fn feature_names_match_weights() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
        assert_eq!(weights_v04::W1.len(), NUM_FEATURES);
    }

    #[test]
    fn metadata_parses() {
        let v: serde_json::Value = serde_json::from_str(weights_v04::METADATA_JSON)
            .expect("metadata JSON should parse");
        assert_eq!(v["version"], "0.4.0-mlp22");
        // bootstrap flag must be present.
        assert!(v["bootstrap"].is_boolean());
    }
}
