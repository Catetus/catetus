//! End-to-end integration test: generate the canonical fixture corpus from
//! scratch into a temp directory, then assert the validator's verdict on each.
//!
//! This is the gate the GitHub Action runs on every PR.

use std::path::PathBuf;
use std::process::Command;

use splatforge_khr_conformance::{validate_path, Status};

fn fixtures_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_splatforge-khr-fixtures"))
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
fn valid_baseline_glb_passes_all_clauses() {
    let corpus = build_corpus();
    let p = corpus.path().join("01_valid_baseline.glb");
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
    assert_eq!(report.container, "glb");
}

#[test]
fn valid_baseline_gltf_passes_all_clauses() {
    let corpus = build_corpus();
    let p = corpus.path().join("02_valid_baseline.gltf");
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
    assert_eq!(report.container, "gltf");
}

#[test]
fn valid_quantized_glb_passes() {
    let corpus = build_corpus();
    let p = corpus.path().join("03_valid_quantized.glb");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "quantized fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
}

#[test]
fn valid_with_sh_evaluates_sh_clauses() {
    let corpus = build_corpus();
    let p = corpus.path().join("04_valid_with_sh.glb");
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
    // SH-related clauses must be PASS (not SKIP) because the fixture
    // declares SH attributes.
    assert_clause(&p, "ACC_SH_COEF", Status::Pass);
    assert_clause(&p, "SH_DEGREES_FULL", Status::Pass);
}

#[test]
fn defaults_method_glb_passes_with_skips() {
    let corpus = build_corpus();
    let p = corpus.path().join("05_valid_default_methods.glb");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "default-methods fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
    assert_clause(&p, "EXT_PROJECTION", Status::Skip);
    assert_clause(&p, "EXT_SORTING", Status::Skip);
}

#[test]
fn missing_ext_used_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("06_invalid_missing_ext_used.glb");
    assert_clause(&p, "EXT_USED", Status::Fail);
}

#[test]
fn missing_rotation_attribute_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("07_invalid_no_rotation.gltf");
    assert_clause(&p, "ATTR_ROTATION", Status::Fail);
}

#[test]
fn rotation_vec3_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("08_invalid_rotation_vec3.gltf");
    assert_clause(&p, "ACC_ROTATION", Status::Fail);
}

#[test]
fn position_missing_minmax_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("09_invalid_position_no_minmax.gltf");
    assert_clause(&p, "ACC_POSITION_MINMAX", Status::Fail);
}

#[test]
fn accessor_count_mismatch_fails() {
    let corpus = build_corpus();
    let p = corpus.path().join("10_invalid_count_mismatch.gltf");
    assert_clause(&p, "ACC_COUNTS_AGREE", Status::Fail);
}

#[test]
fn validate_returns_twenty_three_clauses() {
    let corpus = build_corpus();
    let p = corpus.path().join("01_valid_baseline.glb");
    let report = validate_path(&p).expect("validate");
    assert_eq!(
        report.clauses.len(),
        23,
        "expected exactly 23 clauses, got {}",
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
