//! End-to-end test for the public CLI's hosted SOG paths.
//!
//! Spawns a wiremock HTTP stub that mimics the two-call protocol of
//! `splatforge-private/apps/api/src/routes/encode.rs`:
//!
//!   - `POST /v1/encode?...` → 202 JSON `{ job_id, poll_url, poll_after_seconds }`
//!   - `GET  /v1/encode/<id>` → 200 + canned binary (the "SOG" payload)
//!   - `GET  /v1/encode/<id>/sidecar` → 200 + canned binary (the V5 sidecar)
//!   - `POST /v1/decode?...` → 200 + canned binary (reconstructed PLY)
//!
//! Then invokes the `catetus` CLI against `--api-url http://127.0.0.1:<port>`
//! and asserts the output bytes match the canned binaries.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a tiny binary PLY (3 splats) on disk so the encode-API client's
/// `looks_like_ply` magic check passes.
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

const CANNED_JOB_ID: &str = "00000000-0000-4000-8000-000000000001";
const CANNED_SOG_BYTES: &[u8] = b"<<canned-sog-bytes>>";
const CANNED_SIDECAR_BYTES: &[u8] = b"<<canned-v5tail-sidecar>>";
const CANNED_PLY_BYTES: &[u8] = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";

/// Wire up the encode endpoints (POST + GET + sidecar GET) against a
/// fresh `wiremock` server. Returns the server + its base URL.
async fn spawn_encode_stub() -> (MockServer, String) {
    let server = MockServer::start().await;
    let base = server.uri();
    let accepted_body = serde_json::json!({
        "job_id": CANNED_JOB_ID,
        "status": "queued",
        "target": "sog",
        "v5tail": true,
        "poll_url": format!("/v1/encode/{}", CANNED_JOB_ID),
        "poll_after_seconds": 1,
    });
    Mock::given(method("POST"))
        .and(path("/v1/encode"))
        .and(header("content-type", "application/octet-stream"))
        .respond_with(
            ResponseTemplate::new(202)
                .insert_header("retry-after", "1")
                .set_body_json(accepted_body),
        )
        .mount(&server)
        .await;

    // GET poll → 200 + canned SOG bytes immediately (no queued-then-done
    // sequence needed; the client's poll loop reaches Done on the first
    // GET after the initial sleep).
    Mock::given(method("GET"))
        .and(path(format!("/v1/encode/{}", CANNED_JOB_ID)))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(CANNED_SOG_BYTES),
        )
        .mount(&server)
        .await;

    // GET sidecar
    Mock::given(method("GET"))
        .and(path(format!(
            "/v1/encode/{}/sidecar",
            CANNED_JOB_ID
        )))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(CANNED_SIDECAR_BYTES),
        )
        .mount(&server)
        .await;

    (server, base)
}

/// Spin up a tokio runtime, build the wiremock stub, then return both so
/// the (sync) CLI assertion path can join the runtime when teardown runs.
fn with_encode_stub<F>(test: F)
where
    F: FnOnce(String) + Send + 'static,
{
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (server, base) = rt.block_on(spawn_encode_stub());
    test(base);
    // Drop the server inside the runtime context.
    drop(server);
    drop(rt);
}

#[test]
fn optimize_target_sog_round_trips_through_hosted_api() {
    with_encode_stub(|base| {
        let dir = tempdir().unwrap();
        let ply = dir.path().join("scene.ply");
        write_tiny_ply(&ply);
        let out = dir.path().join("scene.sog");
        let assert = Command::cargo_bin("catetus")
            .unwrap()
            .args([
                "optimize",
                ply.to_str().unwrap(),
                "--preset",
                "web-mobile",
                "--target",
                "sog",
                "-o",
                out.to_str().unwrap(),
                "--api-url",
                &base,
            ])
            // Make sure the env var doesn't override the flag.
            .env_remove("CATETUS_API_URL")
            .output()
            .expect("run catetus optimize --target sog");
        assert!(
            assert.status.success(),
            "expected success, got stderr={}",
            String::from_utf8_lossy(&assert.stderr)
        );
        let bytes = fs::read(&out).expect("output sog written");
        assert_eq!(bytes, CANNED_SOG_BYTES, "SOG output must match stub bytes");
        let stdout = String::from_utf8_lossy(&assert.stdout);
        assert!(
            stdout.contains("hosted-sog encode"),
            "stdout should announce the hosted encode (got: {})",
            stdout
        );
        assert!(
            stdout.contains(CANNED_JOB_ID),
            "stdout should include the stubbed job_id (got: {})",
            stdout
        );
    });
}

#[test]
fn optimize_target_sog_with_emit_v5_tail_writes_sidecar() {
    with_encode_stub(|base| {
        let dir = tempdir().unwrap();
        let ply = dir.path().join("scene.ply");
        write_tiny_ply(&ply);
        let out = dir.path().join("scene.sog");
        // `--emit-v5-tail` takes a GT PLY path; for this test we reuse
        // the input PLY since the hosted route only consumes the
        // request body (the GT PLY argument is forwarded by the
        // GLB-target path but ignored on the SOG hosted path).
        let assert = Command::cargo_bin("catetus")
            .unwrap()
            .args([
                "optimize",
                ply.to_str().unwrap(),
                "--preset",
                "web-mobile",
                "--target",
                "sog",
                "-o",
                out.to_str().unwrap(),
                "--emit-v5-tail",
                ply.to_str().unwrap(),
                "--api-url",
                &base,
            ])
            .env_remove("CATETUS_API_URL")
            .output()
            .expect("run catetus optimize --target sog --emit-v5-tail");
        assert!(
            assert.status.success(),
            "expected success, got stderr={}",
            String::from_utf8_lossy(&assert.stderr)
        );
        let sog_bytes = fs::read(&out).expect("output sog written");
        assert_eq!(sog_bytes, CANNED_SOG_BYTES);
        let sidecar_path = dir.path().join("scene.sog.v5tail");
        let sidecar_bytes = fs::read(&sidecar_path).expect("sidecar written");
        assert_eq!(sidecar_bytes, CANNED_SIDECAR_BYTES);
    });
}

#[test]
fn sog_emit_v5_tail_writes_sidecar_via_hosted_encode() {
    with_encode_stub(|base| {
        let dir = tempdir().unwrap();
        let gt = dir.path().join("gt.ply");
        write_tiny_ply(&gt);
        let sog = dir.path().join("existing.sog");
        // Pretend the user already has a SOG locally; the contents
        // don't matter — the hosted path discards them and uses the
        // GT PLY as the encode input.
        fs::write(&sog, b"placeholder existing sog").unwrap();
        let assert = Command::cargo_bin("catetus")
            .unwrap()
            .args([
                "sog-emit-v5-tail",
                sog.to_str().unwrap(),
                "--gt",
                gt.to_str().unwrap(),
                "--api-url",
                &base,
            ])
            .env_remove("CATETUS_API_URL")
            .output()
            .expect("run catetus sog-emit-v5-tail");
        assert!(
            assert.status.success(),
            "expected success, got stderr={}",
            String::from_utf8_lossy(&assert.stderr)
        );
        let sidecar_path = dir.path().join("existing.sog.v5tail");
        let bytes = fs::read(&sidecar_path).expect("sidecar written next to .sog");
        assert_eq!(bytes, CANNED_SIDECAR_BYTES);
    });
}

#[test]
fn sog_apply_v5_tail_round_trips_through_hosted_decode() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = rt.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/decode"))
            .and(query_param("source", "sog"))
            .and(query_param("v5tail", "true"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(CANNED_PLY_BYTES),
            )
            .mount(&server)
            .await;
        server
    });
    let base = server.uri();
    let dir = tempdir().unwrap();
    let sog = dir.path().join("scene.sog");
    let sidecar = dir.path().join("scene.sog.v5tail");
    let out = dir.path().join("scene.ply");
    fs::write(&sog, b"stub-sog-bytes").unwrap();
    fs::write(&sidecar, b"stub-sidecar-bytes").unwrap();
    let assert = Command::cargo_bin("catetus")
        .unwrap()
        .args([
            "sog-apply-v5-tail",
            sog.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--api-url",
            &base,
        ])
        .env_remove("CATETUS_API_URL")
        .output()
        .expect("run catetus sog-apply-v5-tail");
    assert!(
        assert.status.success(),
        "expected success, got stderr={}",
        String::from_utf8_lossy(&assert.stderr)
    );
    let bytes = fs::read(&out).expect("reconstructed PLY written");
    assert_eq!(bytes, CANNED_PLY_BYTES);
    drop(server);
    drop(rt);
}

#[test]
fn sog_apply_v5_tail_surfaces_clear_error_when_endpoint_missing() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // Empty wiremock — every request 404s. Confirms our client surfaces
    // a user-friendly "endpoint not yet implemented" message rather than
    // bubbling a raw reqwest error.
    let server = rt.block_on(MockServer::start());
    let base = server.uri();
    let dir = tempdir().unwrap();
    let sog = dir.path().join("scene.sog");
    let sidecar = dir.path().join("scene.sog.v5tail");
    let out = dir.path().join("scene.ply");
    fs::write(&sog, b"stub-sog").unwrap();
    fs::write(&sidecar, b"stub-sidecar").unwrap();
    let assert = Command::cargo_bin("catetus")
        .unwrap()
        .args([
            "sog-apply-v5-tail",
            sog.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--api-url",
            &base,
        ])
        .env_remove("CATETUS_API_URL")
        .output()
        .expect("run catetus sog-apply-v5-tail (missing endpoint)");
    assert!(
        !assert.status.success(),
        "expected non-zero exit when /v1/decode 404s"
    );
    let stderr = String::from_utf8_lossy(&assert.stderr);
    assert!(
        stderr.contains("not yet implemented") || stderr.contains("LAUNCH4_BLOCKER"),
        "stderr should mention the missing endpoint (got: {})",
        stderr
    );
}

#[test]
fn optimize_target_sog_rejects_non_ply_input() {
    // No stub needed — the client's own `looks_like_ply` magic check
    // refuses to send the request.
    let dir = tempdir().unwrap();
    let not_ply = dir.path().join("not-a-ply.bin");
    fs::write(&not_ply, b"definitely not a ply").unwrap();
    let out = dir.path().join("out.sog");
    let assert = Command::cargo_bin("catetus")
        .unwrap()
        .args([
            "optimize",
            not_ply.to_str().unwrap(),
            "--preset",
            "web-mobile",
            "--target",
            "sog",
            "-o",
            out.to_str().unwrap(),
            "--api-url",
            "http://127.0.0.1:1", // unused
        ])
        .env_remove("CATETUS_API_URL")
        .output()
        .expect("run catetus optimize --target sog with bad input");
    assert!(
        !assert.status.success(),
        "expected non-zero exit on non-PLY input"
    );
    let stderr = String::from_utf8_lossy(&assert.stderr);
    assert!(
        stderr.contains("does not look like a PLY"),
        "stderr should mention the PLY magic check (got: {})",
        stderr
    );
}

#[test]
fn api_url_flag_overrides_env_var() {
    with_encode_stub(|base| {
        let dir = tempdir().unwrap();
        let ply = dir.path().join("scene.ply");
        write_tiny_ply(&ply);
        let out = dir.path().join("scene.sog");
        // Set CATETUS_API_URL to a guaranteed-bad host. The flag must
        // win, so the run still succeeds.
        let assert = Command::cargo_bin("catetus")
            .unwrap()
            .args([
                "optimize",
                ply.to_str().unwrap(),
                "--preset",
                "web-mobile",
                "--target",
                "sog",
                "-o",
                out.to_str().unwrap(),
                "--api-url",
                &base,
            ])
            .env("CATETUS_API_URL", "http://127.0.0.1:1")
            .output()
            .expect("run catetus with flag overriding env");
        assert!(
            assert.status.success(),
            "flag should override env, but stderr={}",
            String::from_utf8_lossy(&assert.stderr)
        );
        let bytes = fs::read(&out).expect("output sog written");
        assert_eq!(bytes, CANNED_SOG_BYTES);
    });
}
