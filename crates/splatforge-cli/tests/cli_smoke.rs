use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

/// Build a tiny binary PLY (3 splats) on disk for CLI tests.
fn write_tiny_ply(path: &std::path::Path) {
    let mut buf = Vec::new();
    let header = concat!(
        "ply\n",
        "format binary_little_endian 1.0\n",
        "element vertex 3\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float scale_0\n",
        "property float scale_1\n",
        "property float scale_2\n",
        "property float rot_0\n",
        "property float rot_1\n",
        "property float rot_2\n",
        "property float rot_3\n",
        "property float opacity\n",
        "property float f_dc_0\n",
        "property float f_dc_1\n",
        "property float f_dc_2\n",
        "end_header\n",
    );
    buf.extend_from_slice(header.as_bytes());
    for i in 0..3u32 {
        let f = i as f32;
        let record = [
            f, f * 0.5, -f, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3,
        ];
        for v in record {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    fs::write(path, buf).unwrap();
}

#[test]
fn analyze_emits_json() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    write_tiny_ply(&ply);
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args(["analyze", ply.to_str().unwrap()])
        .output()
        .expect("run analyze");
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"splatCount\": 3"));
    assert!(stdout.contains("\"format\": \"ply\""));
}

#[test]
fn optimize_writes_outputs() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    write_tiny_ply(&ply);
    let out_gltf = dir.path().join("opt.gltf");
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args([
            "optimize",
            ply.to_str().unwrap(),
            "--preset",
            "lossless-repack",
            "--out",
            out_gltf.to_str().unwrap(),
        ])
        .output()
        .expect("run optimize");
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    assert!(out_gltf.exists());
}

#[test]
fn convert_ply_to_ply_roundtrips() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    write_tiny_ply(&ply);
    let out_ply = dir.path().join("out.ply");
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args([
            "convert",
            ply.to_str().unwrap(),
            "--to",
            "ply",
            "--out",
            out_ply.to_str().unwrap(),
        ])
        .output()
        .expect("run convert");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_ply.exists());

    // analyze the round-tripped file: 3 splats survive.
    let analyze = Command::cargo_bin("splatforge")
        .unwrap()
        .args(["analyze", out_ply.to_str().unwrap()])
        .output()
        .expect("run analyze");
    assert!(analyze.status.success());
    let stdout = String::from_utf8_lossy(&analyze.stdout);
    assert!(stdout.contains("\"splatCount\": 3"));
}

#[test]
fn convert_ply_to_glb_inspect_succeeds() {
    let dir = tempdir().unwrap();
    let ply = dir.path().join("scene.ply");
    write_tiny_ply(&ply);
    let out_glb = dir.path().join("out.glb");
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args([
            "convert",
            ply.to_str().unwrap(),
            "--to",
            "glb",
            "--out",
            out_glb.to_str().unwrap(),
        ])
        .output()
        .expect("run convert");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_glb.exists());

    let inspect = Command::cargo_bin("splatforge")
        .unwrap()
        .args(["inspect", out_glb.to_str().unwrap()])
        .output()
        .expect("run inspect");
    assert!(
        inspect.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let stdout = String::from_utf8_lossy(&inspect.stdout);
    assert!(stdout.contains("splatCount=3"));
}

#[test]
fn diff_with_stub_helper_writes_json() {
    let dir = tempdir().unwrap();
    let before = dir.path().join("before.ply");
    let after = dir.path().join("after.ply");
    write_tiny_ply(&before);
    write_tiny_ply(&after);
    let out_dir = dir.path().join("out");
    fs::create_dir_all(&out_dir).unwrap();

    // A stub helper: just writes diff.json and exits 0. It must be a
    // standalone Node script because the CLI spawns `node <helper>`.
    let helper = dir.path().join("stub-helper.mjs");
    let helper_src = r#"
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
let out = 'reports/diff';
for (let i = 0; i < process.argv.length; i++) {
  if (process.argv[i] === '--out') out = process.argv[i + 1];
}
mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'diff.json'), JSON.stringify({ status: 'stub' }));
process.exit(0);
"#;
    fs::write(&helper, helper_src).unwrap();

    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .env("SPLATFORGE_DIFF_HELPER", &helper)
        .args([
            "diff",
            before.to_str().unwrap(),
            after.to_str().unwrap(),
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run diff");

    // If `node` isn't on PATH, the CLI should still print a clear hint and
    // exit non-zero — skip the rest of the test in that environment.
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("node not found") || stderr.contains("diff helper"),
            "unexpected failure stderr={stderr}"
        );
        return;
    }
    let diff_json = out_dir.join("diff.json");
    assert!(diff_json.exists(), "diff.json should exist");
    let contents = fs::read_to_string(&diff_json).unwrap();
    assert!(contents.contains("\"status\""));
}

#[test]
fn diff_missing_input_errors() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("nope.ply");
    let real = dir.path().join("real.ply");
    write_tiny_ply(&real);
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args([
            "diff",
            missing.to_str().unwrap(),
            real.to_str().unwrap(),
            "--out",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("run diff");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("before file does not exist"),
        "stderr={stderr}"
    );
}

#[test]
fn inspect_rejects_bogus_file() {
    let dir = tempdir().unwrap();
    let bad = dir.path().join("nope.ply");
    fs::write(&bad, b"not a ply").unwrap();
    let out = Command::cargo_bin("splatforge")
        .unwrap()
        .args(["inspect", bad.to_str().unwrap()])
        .output()
        .expect("run inspect");
    assert!(!out.status.success());
}
