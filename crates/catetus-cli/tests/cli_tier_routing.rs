//! Integration tests for SH-degree tier routing in `catetus optimize`.
//!
//! Proves the INGEST-1 product behavior end-to-end through the real CLI binary:
//!   * An SH=0 capture (no `f_rest_*` columns) routes to the SF baseline tier
//!     and emits the routing note + recapture upsell.
//!   * An SH=3 capture (45 `f_rest_*` columns) routes to the full quality tier
//!     and does NOT emit the upsell.
//!   * `--auto-jacobian` / `--emit-v5-tail` WARN (not error) on SH=0 input
//!     instead of silently producing identical output (the KORIYAMA-1 bug).

use std::io::Write;
use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

/// Minimal Inria-3DGS binary-LE PLY writer.
///
/// `f_rest_count` selects the SH degree the decoder will infer:
///   0 -> SH degree 0 (DC-only / RGB)
///   45 -> SH degree 3
/// Required fields (x,y,z, nx,ny,nz, f_dc_0..2, f_rest_*, opacity, scale_0..2,
/// rot_0..3) are all written so `catetus`'s PLY ingest accepts the file.
fn write_ply(path: &Path, n_splats: usize, f_rest_count: usize) {
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {n_splats}\n"));
    let mut props: Vec<String> = vec![
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for i in 0..f_rest_count {
        props.push(format!("f_rest_{i}"));
    }
    for p in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        props.push(p.to_string());
    }
    for p in &props {
        header.push_str(&format!("property float {p}\n"));
    }
    header.push_str("end_header\n");

    let mut bytes = header.into_bytes();
    let n_fields = props.len();
    for s in 0..n_splats {
        for (i, name) in props.iter().enumerate() {
            // Deterministic but non-degenerate values; scales positive, a
            // normalized-ish rotation, mid opacity.
            let v: f32 = match name.as_str() {
                "scale_0" | "scale_1" | "scale_2" => -2.0, // log-space => exp ~0.135
                "rot_0" => 1.0,
                "rot_1" | "rot_2" | "rot_3" => 0.0,
                "opacity" => 0.0, // logit 0 => sigmoid 0.5
                _ => 0.01 * (s * n_fields + i) as f32,
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(path).expect("create ply");
    f.write_all(&bytes).expect("write ply");
}

fn run_optimize(args: &[&str]) -> assert_cmd::assert::Assert {
    Command::cargo_bin("catetus")
        .expect("catetus bin")
        .arg("optimize")
        .args(args)
        .assert()
}

#[test]
fn sh0_explain_tier_emits_baseline_and_upsell() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("sh0.ply");
    write_ply(&ply, 5, 0);

    let out = run_optimize(&[
        "--preset",
        "web-mobile",
        "--explain-tier",
        ply.to_str().unwrap(),
    ])
    .success();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).into_owned();

    assert!(stderr.contains("SH degree 0"), "stderr: {stderr}");
    assert!(stderr.contains("SF baseline"), "stderr: {stderr}");
    assert!(
        stderr.contains("recapturing at SH=3") || stderr.contains("Recapturing at SH=3"),
        "expected recapture upsell; stderr: {stderr}"
    );
}

#[test]
fn sh3_explain_tier_emits_full_tier_without_upsell() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("sh3.ply");
    write_ply(&ply, 5, 45);

    let out = run_optimize(&[
        "--preset",
        "web-mobile",
        "--explain-tier",
        ply.to_str().unwrap(),
    ])
    .success();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).into_owned();

    assert!(stderr.contains("Full quality tiers"), "stderr: {stderr}");
    assert!(stderr.contains("SH degree 3"), "stderr: {stderr}");
    // SH=3 must route DIFFERENTLY from SH=0: no recapture upsell, no no-op warns.
    assert!(
        !stderr.contains("Recapturing at SH=3") && !stderr.contains("recapturing at SH=3"),
        "SH=3 input must not emit the recapture upsell; stderr: {stderr}"
    );
    assert!(
        !stderr.contains("NO EFFECT"),
        "SH=3 input must not warn about no-op tiers; stderr: {stderr}"
    );
}

#[test]
fn auto_jacobian_warns_on_sh0_but_succeeds() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("sh0.ply");
    let out_glb = dir.path().join("sh0.glb");
    write_ply(&ply, 5, 0);

    let out = run_optimize(&[
        "--preset",
        "web-mobile",
        "--target",
        "glb",
        "--auto-jacobian",
        "-o",
        out_glb.to_str().unwrap(),
        ply.to_str().unwrap(),
    ])
    .success(); // WARNING, not error.
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).into_owned();

    assert!(
        stderr.contains("--auto-jacobian (T2.1.R) has NO EFFECT"),
        "expected auto-jacobian no-op warning; stderr: {stderr}"
    );
}

#[test]
fn auto_jacobian_does_not_warn_on_sh3() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("sh3.ply");
    let out_glb = dir.path().join("sh3.glb");
    write_ply(&ply, 5, 45);

    let out = run_optimize(&[
        "--preset",
        "web-mobile",
        "--target",
        "glb",
        "--auto-jacobian",
        "-o",
        out_glb.to_str().unwrap(),
        ply.to_str().unwrap(),
    ])
    .success();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).into_owned();

    assert!(
        !stderr.contains("NO EFFECT"),
        "SH=3 input must not warn about no-op tiers; stderr: {stderr}"
    );
}
