#![deny(clippy::all)]
#![allow(clippy::needless_range_loop)]
//! Predict-only fidelity scorer for Gaussian-splat scenes — PUBLIC half.
//!
//! This crate ships ONLY the deterministic 22-feature extractor (`features.rs`)
//! and a hosted-API client stub for ML scoring. The trained MLP weights and
//! the `forward()` implementation are part of the proprietary
//! `catetus-fidelity-ml` crate, which is hosted by `apps/api` on
//! `api.catetus.com`.
//!
//! Open-core split (see `RENAME_TO_CATETUS_PLAN.md` Section 1.5):
//!   - PUBLIC `catetus-fidelity` (this crate): features + types + HTTP client
//!     stub. Anyone can extract the 22-vector locally.
//!   - PRIVATE `catetus-fidelity-ml`: trained-MLP weights (v0.4 + v0.5) and
//!     the `forward()` numerical core. Used inside `apps/api` only.
//!
//! API
//! ---
//! ```ignore
//! use catetus_fidelity::{score_ply, ScoreReport};
//! // Local call now returns an Err pointing to the hosted endpoint.
//! match score_ply("scene.ply", Some("baseline.ply")) {
//!     Ok(r)  => println!("score = {}", r.score),
//!     Err(e) => eprintln!("ML scoring is hosted-only: {}", e),
//! }
//! ```
//!
//! For local-only consumers that need the feature vector but not the score:
//! ```ignore
//! use catetus_fidelity::{build_feature_vector, summarise};
//! let scene = catetus_ply::read_ply("scene.ply")?;
//! let summary = summarise(&scene);
//! let feats = build_feature_vector(&summary, None);
//! // feats: [f32; 22] — same vector the hosted MLP consumes.
//! ```

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use catetus_ply::read_ply;
use serde::Serialize;

pub mod features;

pub use features::{build_feature_vector, summarise, FEATURE_NAMES, IDENTITY, NUM_FEATURES};

/// Hosted API endpoint the score_ply* functions point at when called locally.
pub const HOSTED_FIDELITY_ENDPOINT: &str = "https://api.catetus.com/v1/fidelity";

/// JSON-serialisable score report. Returned by the hosted endpoint; this
/// crate keeps the type so downstream code can deserialize the API response.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreReport {
    pub version: &'static str,
    pub score: f32,
    pub baseline_used: bool,
    pub features: Vec<f32>,
    pub feature_names: Vec<&'static str>,
    pub bootstrap: bool,
}

/// Which MLP head the hosted API should use when scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpVersion {
    /// v0.4 — original Bradley-Terry + bootstrap synthetic head (default).
    V04,
    /// v0.5 — real-PSNR-calibrated head.
    V05,
}

impl MlpVersion {
    pub fn version_str(self) -> &'static str {
        match self {
            MlpVersion::V04 => "0.4.0-mlp22",
            MlpVersion::V05 => "0.5.0-mlp22",
        }
    }
}

/// Build a feature summary for a PLY without touching the hosted scorer.
/// This is the local-only fast path — extracts the 22-vector but does not
/// return a `ScoreReport.score`.
pub fn extract_features(cand: &Path, baseline: Option<&Path>) -> Result<[f32; NUM_FEATURES]> {
    let cand_scene =
        read_ply(cand).with_context(|| format!("read candidate PLY {}", cand.display()))?;
    let cand_summary = summarise(&cand_scene);
    let base_summary = match baseline {
        Some(p) => {
            let s = read_ply(p).with_context(|| format!("read baseline PLY {}", p.display()))?;
            Some(summarise(&s))
        }
        None => None,
    };
    Ok(build_feature_vector(&cand_summary, base_summary.as_ref()))
}

/// Score a single PLY against an optional baseline using the v0.4 head.
///
/// **Hosted-only.** This function used to run the v0.4 MLP locally. The
/// trained weights now live in the private `catetus-fidelity-ml` crate;
/// public consumers must POST to `HOSTED_FIDELITY_ENDPOINT`. This stub
/// returns an `Err` with the migration message so callers fail loudly
/// rather than silently producing wrong scores.
pub fn score_ply(_cand: &Path, _baseline: Option<&Path>) -> Result<ScoreReport> {
    Err(anyhow!(hosted_only_message(MlpVersion::V04)))
}

/// Score a single PLY against an optional baseline, choosing the MLP head.
///
/// Same hosted-only contract as `score_ply` — returns `Err` and points
/// callers at `HOSTED_FIDELITY_ENDPOINT`.
pub fn score_ply_with_version(
    _cand: &Path,
    _baseline: Option<&Path>,
    version: MlpVersion,
) -> Result<ScoreReport> {
    Err(anyhow!(hosted_only_message(version)))
}

fn hosted_only_message(version: MlpVersion) -> String {
    format!(
        "ML fidelity scoring (v{}) is hosted-only — POST to {} instead. \
         The PUBLIC catetus-fidelity crate ships feature extraction only; \
         trained weights live in the private catetus-fidelity-ml crate.",
        version.version_str(),
        HOSTED_FIDELITY_ENDPOINT,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_names_match_documented_length() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
    }

    #[test]
    fn score_ply_is_hosted_only() {
        let p = std::path::PathBuf::from("/nonexistent/scene.ply");
        let err = score_ply(&p, None).err().expect("must be Err");
        let msg = format!("{err}");
        assert!(msg.contains("hosted-only"));
        assert!(msg.contains("api.catetus.com"));
    }

    #[test]
    fn score_ply_with_version_is_hosted_only() {
        let p = std::path::PathBuf::from("/nonexistent/scene.ply");
        for v in [MlpVersion::V04, MlpVersion::V05] {
            let err = score_ply_with_version(&p, None, v)
                .err()
                .expect("must be Err");
            assert!(format!("{err}").contains("hosted-only"));
        }
    }

    #[test]
    fn version_strings() {
        assert_eq!(MlpVersion::V04.version_str(), "0.4.0-mlp22");
        assert_eq!(MlpVersion::V05.version_str(), "0.5.0-mlp22");
    }
}
