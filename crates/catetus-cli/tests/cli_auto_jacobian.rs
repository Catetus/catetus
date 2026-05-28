//! End-to-end test for `catetus optimize --auto-jacobian`.
//!
//! Proves the full chain:
//!
//!   PLY → catetus-jacobian (CPU proxy) → PassContext.sh_rest_weights →
//!     VQPaletteShRest weighted Lloyd-Max → GLB output
//!
//! works without any external `.npz` file. The acceptance criterion is that
//! the output GLB BYTES from `--auto-jacobian` differ from the output GLB
//! bytes WITHOUT the flag (when the same preset is used). If they're
//! byte-identical, the Jacobian weights weren't actually consumed by
//! `VQPaletteShRest` and the wiring is broken.
//!
//! We also accept the case where the synthetic test fixture is too small
//! for VQPaletteShRest to behave differently (e.g. fewer splats than k);
//! in that case we still require the `--auto-jacobian` run to succeed and
//! to emit the expected `auto-jacobian:` log line — proof that the proxy
//! ran and was fed in.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

/// SH degree 3 = 1 DC band + 15 higher bands = 16 coefficients per channel
/// × 3 channels = 48 SH coefficients per splat, of which 3 are DC (f_dc_*)
/// and 45 are "rest" (f_rest_0..f_rest_44, channel-major: R-band1..R-band15,
/// G-band1..G-band15, B-band1..B-band15).
const SH_REST_PER_SPLAT: usize = 45;

/// Build a synthetic Inria-3DGS-style PLY with SH degree 3 and `n` splats.
///
/// The splats are arranged so that VQPaletteShRest sees:
///   - varying anisotropic scale  (drives `area_2d` in the proxy)
///   - varying opacity            (drives `α`         in the proxy)
///   - varying SH-rest magnitude  (drives `‖sh_rest‖₂` in the proxy)
///
/// So the per-splat Jacobian weights span ~3 orders of magnitude — enough
/// for weighted Lloyd-Max to pick visibly-different centroids than the
/// unweighted variant.
fn write_sh3_ply(path: &std::path::Path, n: usize) {
    let mut buf = Vec::new();
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {n}\n"));
    for p in [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
    ] {
        header.push_str(&format!("property float {p}\n"));
    }
    for k in 0..SH_REST_PER_SPLAT {
        header.push_str(&format!("property float f_rest_{k}\n"));
    }
    header.push_str("end_header\n");
    buf.extend_from_slice(header.as_bytes());

    for i in 0..n {
        let f = i as f32;
        // Position: spread on a cube.
        let x = (i % 8) as f32;
        let y = ((i / 8) % 8) as f32;
        let z = (i / 64) as f32;
        // Scale (log): every 16th splat is "big"; the rest are small. The
        // big splats dominate the Jacobian via `area_2d(scale)`.
        let big = i % 16 == 0;
        let s = if big { 0.0_f32 } else { -3.5_f32 };
        // Opacity (logit). Mostly transparent; the big splats are opaque.
        // The transparent splats get α≈0 → near-zero Jacobian → palette
        // ignores them.
        let op_logit = if big { 4.0_f32 } else { -3.0_f32 };
        // DC band: distinct color per splat so the encoder has something
        // non-trivial to compress.
        let dc_r = (i as f32 / n as f32) - 0.5;
        let dc_g = ((i * 7) % 13) as f32 / 13.0 - 0.5;
        let dc_b = ((i * 11) % 17) as f32 / 17.0 - 0.5;
        // SH-rest band: weight the big splats up hard. Small splats get
        // tiny but non-zero SH-rest so the proxy returns a finite weight.
        let sh_scale = if big { 0.4 } else { 0.005 };
        let mut record = vec![
            x, y, z, s, s, s, 1.0, 0.0, 0.0, 0.0, op_logit, dc_r, dc_g, dc_b,
        ];
        for k in 0..SH_REST_PER_SPLAT {
            // Make SH-rest unique per splat AND per band so VQ has clusters
            // to discover.
            let band = (k as f32 / SH_REST_PER_SPLAT as f32) - 0.5;
            let v = sh_scale * (((f + band * 7.3).sin() * 1.7) + ((f * 0.131).cos() * 0.7));
            record.push(v);
        }
        for v in record {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    fs::write(path, buf).unwrap();
}

/// Encode `ply` with the canonical T2.1.R preset, optionally with
/// `--auto-jacobian`. Returns the resulting GLB bytes.
fn run_optimize(ply: &std::path::Path, out: &std::path::Path, auto_jacobian: bool) -> Vec<u8> {
    let mut args: Vec<String> = vec![
        "optimize".into(),
        ply.to_str().unwrap().into(),
        "--preset".into(),
        "wmv-vq45-no-prune-tight".into(),
        "--target".into(),
        "glb".into(),
        "--out".into(),
        out.to_str().unwrap().into(),
    ];
    if auto_jacobian {
        args.push("--auto-jacobian".into());
    }
    let output = Command::cargo_bin("catetus")
        .unwrap()
        .args(&args)
        .output()
        .expect("run optimize");
    assert!(
        output.status.success(),
        "optimize {auto_jacobian:?} failed:\n  args={args:?}\n  stdout={}\n  stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(out.exists(), "GLB not written");
    // When --auto-jacobian was set the CLI must log the proxy computation
    // (provides operator-visible evidence the in-process Jacobian ran).
    if auto_jacobian {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("auto-jacobian:"),
            "--auto-jacobian missing operator log line. stdout=\n{stdout}",
        );
    }
    fs::read(out).unwrap()
}

#[test]
fn auto_jacobian_changes_glb_output_vs_uniform_weights() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    // 256 splats × SH degree 3. `wmv-vq45-no-prune-tight` uses
    // VQPaletteShRest with K=4096 by default, but the implementation
    // clamps K to splat count when oversized, so 256 still exercises the
    // weighted-Lloyd code path.
    write_sh3_ply(&ply, 256);

    let out_plain = dir.path().join("plain.glb");
    let out_auto = dir.path().join("auto.glb");

    let bytes_plain = run_optimize(&ply, &out_plain, false);
    let bytes_auto = run_optimize(&ply, &out_auto, true);

    // Headline gate: the two GLBs must NOT be byte-identical. If they
    // ARE byte-identical, that means VQPaletteShRest produced exactly the
    // same centroids/assignments in both runs, which would mean the
    // sh_rest_weights field never reached the pass.
    assert_ne!(
        bytes_plain, bytes_auto,
        "--auto-jacobian GLB is byte-identical to plain GLB ({} bytes each). \
         This means the Jacobian weights weren't actually consumed by \
         VQPaletteShRest — the wiring is broken.",
        bytes_plain.len()
    );

    // Sanity: both outputs declare KHR_gaussian_splatting (the GLB writer
    // wraps the SF payload in the public Khronos extension).
    let raw_plain = String::from_utf8_lossy(&bytes_plain);
    let raw_auto = String::from_utf8_lossy(&bytes_auto);
    assert!(
        raw_plain.contains("KHR_gaussian_splatting"),
        "plain GLB missing KHR_gaussian_splatting"
    );
    assert!(
        raw_auto.contains("KHR_gaussian_splatting"),
        "auto-jacobian GLB missing KHR_gaussian_splatting"
    );
}

#[test]
fn auto_jacobian_and_jacobian_sidecar_are_mutually_exclusive() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    write_sh3_ply(&ply, 16);
    let out = dir.path().join("out.glb");
    // Pass both flags. clap should reject before the encoder runs.
    let result = Command::cargo_bin("catetus")
        .unwrap()
        .args([
            "optimize",
            ply.to_str().unwrap(),
            "--preset",
            "wmv-vq45-no-prune-tight",
            "--target",
            "glb",
            "--auto-jacobian",
            "--jacobian-sidecar",
            "/tmp/does-not-exist.npz",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run optimize");
    assert!(
        !result.status.success(),
        "optimize should reject --auto-jacobian + --jacobian-sidecar (mutually exclusive)"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    // clap's standard "conflicts_with" diagnostic mentions both flag names.
    assert!(
        stderr.contains("--jacobian-sidecar") || stderr.contains("--auto-jacobian"),
        "expected mutual-exclusion diagnostic; stderr=\n{stderr}"
    );
}

#[test]
fn auto_jacobian_help_documents_t21r_tier() {
    let out = Command::cargo_bin("catetus")
        .unwrap()
        .args(["optimize", "--help"])
        .output()
        .expect("run optimize --help");
    assert!(out.status.success(), "optimize --help failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--auto-jacobian"),
        "optimize --help should list --auto-jacobian; stdout=\n{stdout}"
    );
    // The help text identifies the tier so the user understands the lift.
    assert!(
        stdout.contains("T2.1.R") || stdout.contains("SuperSplat"),
        "--auto-jacobian help text should reference the T2.1.R tier / SuperSplat lift; stdout=\n{stdout}"
    );
}
