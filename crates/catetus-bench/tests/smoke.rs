use catetus_bench::{run_named, to_json};
use std::path::Path;

#[test]
fn empty_dir_returns_empty_suite() {
    let dir = tempfile::tempdir().unwrap();
    let suite = run_named("smoke", dir.path()).unwrap();
    assert_eq!(suite.records.len(), 0);
}

#[test]
fn missing_dir_returns_empty_suite() {
    let suite = run_named("smoke", Path::new("/path/that/does/not/exist")).unwrap();
    assert_eq!(suite.records.len(), 0);
    let json = to_json(&suite).unwrap();
    assert!(json.contains("\"smoke\""));
}
