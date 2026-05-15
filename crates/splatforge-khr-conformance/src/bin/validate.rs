//! `splatforge-khr-validate` — CLI driver for the conformance suite.
//!
//! Usage:
//!     splatforge-khr-validate <file.gltf|file.glb> [--json] [--quiet]
//!
//! Exit codes:
//!   0 — every clause passed (skips allowed).
//!   1 — one or more clauses failed.
//!   2 — validator-level error (file not readable, malformed GLB, etc.).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use splatforge_khr_conformance::{validate_path, Clause, Status};

#[derive(Parser, Debug)]
#[command(
    name = "splatforge-khr-validate",
    about = "Validate a glTF asset against the KHR_gaussian_splatting conformance suite."
)]
struct Cli {
    /// glTF (.gltf) or GLB (.glb) file to inspect.
    file: PathBuf,
    /// Emit JSON instead of the human-readable table.
    #[arg(long)]
    json: bool,
    /// Suppress the per-clause table; print only the summary line.
    #[arg(long)]
    quiet: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let report = match validate_path(&cli.file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    if cli.json {
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        if !cli.quiet {
            println!(
                "KHR_gaussian_splatting conformance report for {} ({})",
                report.source, report.container
            );
            println!("{:<24} {:<6} detail", "clause", "status");
            println!("{}", "-".repeat(70));
            for c in &report.clauses {
                let status = match c.status {
                    Status::Pass => "PASS",
                    Status::Fail => "FAIL",
                    Status::Skip => "SKIP",
                };
                let detail = c.detail.as_deref().unwrap_or("");
                println!("{:<24} {:<6} {}", c.id, status, detail);
            }
            println!();
        }
        println!(
            "summary: {} pass, {} fail, {} skip (of {} clauses)",
            report.pass,
            report.fail,
            report.skip,
            Clause::all().len()
        );
    }

    if report.is_pass() {
        ExitCode::from(0)
    } else {
        ExitCode::from(1)
    }
}
