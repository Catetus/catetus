//! End-to-end integration test: generate the canonical fixture corpus from
//! scratch into a temp directory, then assert the validator's verdict on
//! each.  Mirrors `catetus-khr-conformance/tests/fixtures.rs`.
//!
//! This is the gate the GitHub Action runs on every PR.

use std::path::PathBuf;
use std::process::Command;

use catetus_usd_conformance::{validate_path, Status};

fn fixtures_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_catetus-usd-fixtures"))
}

fn build_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let status = Command::new(fixtures_bin())
        .arg(dir.path())
        .status()
        .expect("run fixtures bin");
    assert!(status.success(), "fixture generator failed");
    dir
}

fn assert_clause(file: &std::path::Path, clause_id: &str, expected: Status) {
    let report = validate_path(file).expect("validate");
    let c = report
        .clauses
        .iter()
        .find(|c| c.id == clause_id)
        .unwrap_or_else(|| panic!("clause {clause_id} not in report for {}", file.display()));
    assert_eq!(
        c.status,
        expected,
        "clause {clause_id} for {} — detail={:?}",
        file.display(),
        c.detail
    );
}

#[test]
fn valid_minimal_usda_passes() {
    let corpus = build_corpus();
    let p = corpus.path().join("01_valid_minimal.usda");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "expected no failing clauses, got: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.container, "usda");
}

#[test]
fn valid_particle_field_passes() {
    let corpus = build_corpus();
    let p = corpus.path().join("02_valid_particle_field.usda");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
}

#[test]
fn valid_dense_passes() {
    let corpus = build_corpus();
    let p = corpus.path().join("03_valid_dense.usda");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "dense fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
}

#[test]
fn valid_with_sh_exercises_sh_clause() {
    let corpus = build_corpus();
    let p = corpus.path().join("04_valid_with_sh.usda");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "SH fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
    // SH count clause must have been evaluated (not skipped).
    let sh = report
        .clauses
        .iter()
        .find(|c| c.id == "SH_COEFFS_COUNT")
        .expect("SH_COEFFS_COUNT present");
    assert_ne!(sh.status, Status::Skip);
}

#[test]
fn valid_minimal_usdc_passes() {
    let corpus = build_corpus();
    let p = corpus.path().join("05_valid_minimal.usdc");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "USDC fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.container, "usdc");
}

#[test]
fn missing_orientations_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("06_invalid_no_orientations.usda");
    assert_clause(&p, "ATTR_ORIENTATIONS", Status::Fail);
}

#[test]
fn opacity_out_of_range_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("07_invalid_opacity_out_of_range.usda");
    assert_clause(&p, "ATTR_OPACITIES_RANGE", Status::Fail);
}

#[test]
fn count_mismatch_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("08_invalid_count_mismatch.usda");
    assert_clause(&p, "COUNTS_AGREE", Status::Fail);
}

#[test]
fn validate_returns_at_least_fifteen_clauses() {
    let corpus = build_corpus();
    let p = corpus.path().join("01_valid_minimal.usda");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.clauses.len() >= 15,
        "expected >=15 clauses, got {}",
        report.clauses.len()
    );
}

fn collect_files(root: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut out: Vec<(std::path::PathBuf, Vec<u8>)> = Vec::new();
    fn walk(
        dir: &std::path::Path,
        root: &std::path::Path,
        out: &mut Vec<(std::path::PathBuf, Vec<u8>)>,
    ) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_path_buf();
                out.push((rel, std::fs::read(&p).unwrap()));
            }
        }
    }
    walk(root, root, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn fixtures_are_byte_deterministic() {
    let a = build_corpus();
    let b = build_corpus();
    let fa = collect_files(a.path());
    let fb = collect_files(b.path());
    assert_eq!(fa.len(), fb.len(), "different file count between runs");
    for ((pa, ba), (pb, bb)) in fa.iter().zip(fb.iter()) {
        assert_eq!(pa, pb, "file ordering differs");
        assert_eq!(ba, bb, "non-deterministic fixture: {}", pa.display());
    }
}
