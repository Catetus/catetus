//! End-to-end integration test: generate the canonical fixture corpus from
//! scratch into a temp directory, then assert the validator's verdict on each.
//!
//! This is the gate the GitHub Action runs on every PR.

use std::path::PathBuf;
use std::process::Command;

use catetus_khr_conformance::{validate_path, Status};

fn fixtures_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_catetus-khr-fixtures"))
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
fn spz_stub_glb_passes_all_clauses() {
    // Fixture 05 is the preserved-from-pre-RC "spz stub" — a real SPZ blob
    // (via catetus_spz::encode_spz) attached to a RC-shaped baseline.
    // It must pass the validator with no failures.
    let corpus = build_corpus();
    let p = corpus.path().join("05_valid_spz_stub.glb");
    let report = validate_path(&p).expect("validate");
    assert!(
        report.is_pass(),
        "spz stub fixture failed: {:?}",
        report
            .clauses
            .iter()
            .filter(|c| c.status == Status::Fail)
            .collect::<Vec<_>>()
    );
    assert_clause(&p, "SPZ_EXT_PRESENT", Status::Pass);
    assert_clause(&p, "SPZ_BLOB_MAGIC", Status::Pass);
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
fn validate_returns_thirty_clauses() {
    // 23 core KHR_gaussian_splatting RC clauses + 7 SPZ sub-extension clauses.
    let corpus = build_corpus();
    let p = corpus.path().join("01_valid_baseline.glb");
    let report = validate_path(&p).expect("validate");
    assert_eq!(
        report.clauses.len(),
        30,
        "expected exactly 30 clauses (23 KHR + 7 SPZ), got {}",
        report.clauses.len()
    );
}

#[test]
fn spz_compressed_glb_passes_all_spz_clauses() {
    // Fixture 11 is a real KHR_gaussian_splatting_compression_spz asset.
    // Every SPZ-compression clause defined for that extension must pass.
    let corpus = build_corpus();
    let p = corpus.path().join("11_valid_spz_compressed.glb");
    let report = validate_path(&p).expect("validate");
    let fails: Vec<&catetus_khr_conformance::ClauseResult> = report
        .clauses
        .iter()
        .filter(|c| c.status == Status::Fail)
        .collect();
    assert!(fails.is_empty(), "unexpected failures: {fails:?}");
    for id in [
        "SPZ_EXT_PRESENT",
        "SPZ_VERSION",
        "SPZ_BUFFERVIEW",
        "SPZ_BLOB_MAGIC",
        "SPZ_DECODED_COUNT",
        "SPZ_DECLARED",
        "SPZ_CONSISTENT",
    ] {
        assert_clause(&p, id, Status::Pass);
    }
}

#[test]
fn spz_compressed_missing_ext_used_fails_spz_declared() {
    let corpus = build_corpus();
    let p = corpus.path().join("12_invalid_spz_missing_ext_used.glb");
    // The primitive declares SPZ but extensionsUsed does not — base
    // SPZ_DECLARED catches this. The new SPZ_EXT_PRESENT still passes
    // (it asks about the primitive-side declaration, which is present).
    assert_clause(&p, "SPZ_DECLARED", Status::Fail);
    assert_clause(&p, "SPZ_EXT_PRESENT", Status::Pass);
}

#[test]
fn spz_compressed_wrong_magic_fails_blob_magic() {
    let corpus = build_corpus();
    let p = corpus.path().join("13_invalid_spz_wrong_magic.glb");
    assert_clause(&p, "SPZ_BLOB_MAGIC", Status::Fail);
    // Other SPZ structural clauses (version, bufferView present, etc.)
    // should still pass — only the blob's first 4 bytes are wrong.
    assert_clause(&p, "SPZ_VERSION", Status::Pass);
    assert_clause(&p, "SPZ_EXT_PRESENT", Status::Pass);
    assert_clause(&p, "SPZ_BUFFERVIEW", Status::Pass);
}

#[test]
fn fixture_set_contains_thirteen_files() {
    let corpus = build_corpus();
    let mut count = 0usize;
    for entry in std::fs::read_dir(corpus.path()).expect("read_dir") {
        let entry = entry.expect("entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".glb") || name.ends_with(".gltf") {
            count += 1;
        }
    }
    assert!(count >= 13, "expected >=13 fixtures, got {count}");
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
