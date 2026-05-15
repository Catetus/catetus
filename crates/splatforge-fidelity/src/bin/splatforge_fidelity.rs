//! `splatforge-fidelity` — predict-only fidelity scorer (binary form).
//!
//! Usage: `splatforge-fidelity <candidate.ply> [--baseline <baseline.ply>]`.
//! Emits a JSON `ScoreReport` to stdout.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use splatforge_fidelity::score_ply;

fn parse() -> Result<(PathBuf, Option<PathBuf>, bool)> {
    let mut args = std::env::args().skip(1).peekable();
    let mut cand: Option<PathBuf> = None;
    let mut baseline: Option<PathBuf> = None;
    let mut pretty = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--baseline" | "-b" => {
                baseline =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--baseline requires a path")
                    })?));
            }
            "--pretty" => pretty = true,
            "-h" | "--help" => {
                eprintln!("usage: splatforge-fidelity <candidate.ply> [--baseline <baseline.ply>] [--pretty]");
                std::process::exit(0);
            }
            _ if cand.is_none() => cand = Some(PathBuf::from(a)),
            other => return Err(anyhow::anyhow!("unexpected arg: {other}")),
        }
    }
    let cand = cand.ok_or_else(|| anyhow::anyhow!("missing <candidate.ply>"))?;
    Ok((cand, baseline, pretty))
}

fn main() -> ExitCode {
    let (cand, baseline, pretty) = match parse() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("splatforge-fidelity: {e:#}");
            return ExitCode::from(2);
        }
    };
    let report = match score_ply(cand.as_path(), baseline.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("splatforge-fidelity: {e:#}");
            return ExitCode::from(1);
        }
    };
    let json = if pretty {
        serde_json::to_string_pretty(&report).unwrap()
    } else {
        serde_json::to_string(&report).unwrap()
    };
    println!("{json}");
    ExitCode::SUCCESS
}
