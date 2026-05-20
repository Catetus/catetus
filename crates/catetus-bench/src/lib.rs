#![deny(clippy::all)]
//! Tiny benchmark harness that times analyze + optimize on whatever fixtures
//! exist in `fixtures/tiny/`. Used by `catetus corpus run smoke`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use catetus_core::AnalyzeReport;
use catetus_optimize::preset;
use catetus_ply::read_ply;
use serde::Serialize;

/// A single fixture's timing record.
#[derive(Debug, Clone, Serialize)]
pub struct BenchRecord {
    /// Fixture filename.
    pub fixture: String,
    /// Splat count parsed from the fixture.
    pub splat_count: usize,
    /// Wall time to analyze, in milliseconds.
    pub analyze_ms: u128,
    /// Wall time to run a `lossless-repack` pipeline, in milliseconds.
    pub optimize_ms: u128,
}

/// Aggregate benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct BenchSuite {
    /// Suite identifier (e.g. `"smoke"`).
    pub suite: String,
    /// Per-fixture records in the order they were processed.
    pub records: Vec<BenchRecord>,
}

/// Run the default smoke suite over `fixtures/tiny/*.ply`.
pub fn run_smoke() -> Result<BenchSuite> {
    run_named("smoke", Path::new("fixtures/tiny"))
}

/// Run a benchmark suite of given name over PLY fixtures in `dir`.
pub fn run_named(name: &str, dir: &Path) -> Result<BenchSuite> {
    let mut records = Vec::new();
    let entries: Vec<PathBuf> = if dir.is_dir() {
        std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("ply"))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        Vec::new()
    };

    for path in entries {
        let fixture = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let bytes = std::fs::read(&path)?;
        let t0 = Instant::now();
        let scene = read_ply(&path)?;
        let _report = AnalyzeReport::from_scene(&scene, "ply", bytes.len() as u64);
        let analyze_ms = t0.elapsed().as_millis();

        let mut scene2 = scene.clone();
        let t1 = Instant::now();
        let pipe = preset("lossless-repack")?;
        let _pipeline_report = pipe.run(&mut scene2)?;
        let optimize_ms = t1.elapsed().as_millis();

        records.push(BenchRecord {
            fixture,
            splat_count: scene.splats.len(),
            analyze_ms,
            optimize_ms,
        });
    }

    Ok(BenchSuite {
        suite: name.to_string(),
        records,
    })
}

/// Convenience: render a BenchSuite to pretty JSON.
pub fn to_json(suite: &BenchSuite) -> Result<String> {
    Ok(serde_json::to_string_pretty(suite)?)
}
