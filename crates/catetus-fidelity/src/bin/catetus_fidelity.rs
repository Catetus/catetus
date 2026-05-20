//! `catetus-fidelity` — predict-only fidelity scorer (binary form).
//!
//! Usage: `catetus-fidelity <candidate.ply> [--baseline <baseline.ply>]
//!                            [--mlp-version 0.4|0.5] [--pretty]`.
//!
//! **Hosted-only since 2026-05-19.** Trained MLP weights were moved to the
//! private `catetus-fidelity-ml` crate as part of the open-core split. This
//! binary now emits the local 22-feature vector and a JSON envelope pointing
//! callers at `https://api.catetus.com/v1/fidelity` for the score. The exit
//! code is non-zero so CI gates continue to fail loudly until the caller
//! either flips to the hosted endpoint or stops asking for an ML score.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use catetus_fidelity::{extract_features, MlpVersion, FEATURE_NAMES, HOSTED_FIDELITY_ENDPOINT};

fn parse() -> Result<(PathBuf, Option<PathBuf>, bool, MlpVersion)> {
    let mut args = std::env::args().skip(1).peekable();
    let mut cand: Option<PathBuf> = None;
    let mut baseline: Option<PathBuf> = None;
    let mut pretty = false;
    let mut version = MlpVersion::V04;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--baseline" | "-b" => {
                baseline =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--baseline requires a path")
                    })?));
            }
            "--mlp-version" => {
                let v = args.next().ok_or_else(|| {
                    anyhow::anyhow!("--mlp-version requires a value (0.4 or 0.5)")
                })?;
                version = match v.as_str() {
                    "0.4" | "v0.4" | "0.4.0" | "0.4.0-mlp22" => MlpVersion::V04,
                    "0.5" | "v0.5" | "0.5.0" | "0.5.0-mlp22" => MlpVersion::V05,
                    other => {
                        return Err(anyhow::anyhow!(
                            "unknown --mlp-version {other:?} (try 0.4 or 0.5)"
                        ))
                    }
                };
            }
            "--pretty" => pretty = true,
            "-h" | "--help" => {
                eprintln!("usage: catetus-fidelity <candidate.ply> [--baseline <baseline.ply>] [--mlp-version 0.4|0.5] [--pretty]");
                eprintln!();
                eprintln!("ML scoring is hosted-only. POST the emitted feature vector to:");
                eprintln!("  {HOSTED_FIDELITY_ENDPOINT}");
                std::process::exit(0);
            }
            _ if cand.is_none() => cand = Some(PathBuf::from(a)),
            other => return Err(anyhow::anyhow!("unexpected arg: {other}")),
        }
    }
    let cand = cand.ok_or_else(|| anyhow::anyhow!("missing <candidate.ply>"))?;
    Ok((cand, baseline, pretty, version))
}

fn main() -> ExitCode {
    let (cand, baseline, pretty, version) = match parse() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("catetus-fidelity: {e:#}");
            return ExitCode::from(2);
        }
    };
    let features = match extract_features(cand.as_path(), baseline.as_deref()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("catetus-fidelity: {e:#}");
            return ExitCode::from(1);
        }
    };
    let envelope = serde_json::json!({
        "kind": "feature_vector_only",
        "version": version.version_str(),
        "hosted_endpoint": HOSTED_FIDELITY_ENDPOINT,
        "features": features.to_vec(),
        "feature_names": FEATURE_NAMES.to_vec(),
        "score": null,
        "note": "ML scoring is hosted-only; POST features + version to hosted_endpoint",
    });
    let json = if pretty {
        serde_json::to_string_pretty(&envelope).unwrap()
    } else {
        serde_json::to_string(&envelope).unwrap()
    };
    println!("{json}");
    // Exit nonzero so CI gates that previously asserted on `.score` keep
    // failing until they migrate to the hosted endpoint.
    ExitCode::from(3)
}
