//! HTTP client for the hosted `/v1/encode` endpoint.
//!
//! The public CLI's SOG paths (`--target sog`, `--emit-v5-tail`,
//! `sog-apply-v5-tail`) live behind the open-core split — the actual
//! encoder is in the private `catetus-sog` crate, reached over HTTP at
//! `https://api.catetus.com/v1/encode`. This module is the blocking client
//! that talks the two-call protocol documented in
//! `splatforge-private/apps/api/src/routes/encode.rs`:
//!
//!   1. `POST /v1/encode?target={sog|glb}&v5tail={true|false}` with raw PLY
//!      bytes as the body. Returns 202 + `{ job_id, poll_url,
//!      poll_after_seconds }`.
//!   2. `GET <poll_url>` repeatedly. While Queued / Running the server
//!      returns 202 + a status JSON; on Done the same URL returns 200 +
//!      raw encoded bytes (`application/octet-stream` for SOG,
//!      `model/gltf-binary` for GLB).
//!   3. (Optional, when `v5tail=true` was requested) the status JSON on
//!      Done includes a `sidecar_url` relative to the API base; GET that
//!      to fetch the `.sog.v5tail` bytes.
//!
//! All calls share one blocking `reqwest::Client` per invocation; the
//! whole module is deliberately sync so the public CLI keeps its zero-
//! tokio surface (the existing `Submit` path already runs blocking
//! reqwest in the same crate).

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// Default API base when neither `--api-url` nor `CATETUS_API_URL` is set.
/// Matches the production deployment fronted by Cloudflare in front of
/// `catetus-api.fly.dev` (see `[[catetus-api-deploy]]`).
pub const DEFAULT_ENCODE_API_URL: &str = "https://api.catetus.com";

/// Inputs at or above this size use the presigned-upload flow
/// (`POST /v1/encode/upload` → streaming PUT → `POST /v1/encode` with
/// `{job_id, input_ref}`) instead of the inline-bytes body (LARGE-2
/// Part A). Below it, the simpler inline path is used. 100 MB is well
/// under the API's inline body cap (512 MiB) so small/medium scenes keep
/// the one-call path while multi-GB captures avoid buffering + the ~33%
/// base64 inflation that the inline dispatch envelope incurs.
pub const UPLOAD_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024;

/// Output container requested from the server. Mirrors `EncodeTarget`
/// in `routes/encode.rs` — kept in sync via the kebab-case URL params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeTarget {
    Sog,
    #[allow(dead_code)]
    Glb,
}

impl EncodeTarget {
    fn as_query_value(self) -> &'static str {
        match self {
            EncodeTarget::Sog => "sog",
            EncodeTarget::Glb => "glb",
        }
    }
}

/// What we want back from one call to `/v1/encode`. The caller decides
/// whether to additionally pull the V5.2 sidecar (only valid when
/// `target=sog`); when `v5tail` is `false` the sidecar slot in
/// `EncodeOutcome` is always `None`.
#[derive(Debug, Clone)]
pub struct EncodeRequest<'a> {
    pub api_url: &'a str,
    pub target: EncodeTarget,
    pub v5tail: bool,
    /// Raw PLY bytes — passed straight through to the server as the
    /// request body. Must start with the `ply\n` magic or the server's
    /// fast-path validator rejects it.
    pub ply_bytes: &'a [u8],
    /// Optional human-readable label echoed back in the status JSON.
    pub label: Option<&'a str>,
    /// Hard ceiling on total wall-clock time (POST + every poll). The
    /// real worker takes seconds to minutes; 5 minutes is the default
    /// the CLI ships with, callers can override.
    pub timeout: Duration,
    /// Lower bound on poll interval. The server's `poll_after_seconds`
    /// hint overrides this when larger.
    pub min_poll_interval: Duration,
}

/// What the caller gets back on success. `output_bytes` is the encoded
/// SOG / GLB binary; `sidecar_bytes` is the `.sog.v5tail` sidecar when
/// requested and the worker emitted one.
#[derive(Debug, Clone)]
pub struct EncodeOutcome {
    pub job_id: String,
    pub output_bytes: Vec<u8>,
    pub sidecar_bytes: Option<Vec<u8>>,
}

/// Parsed 202 POST response. Field names match `EncodeAccepted` in the
/// private API exactly.
#[derive(Debug, Clone, Deserialize)]
struct EncodeAccepted {
    job_id: String,
    poll_url: String,
    #[serde(default = "default_poll_after")]
    poll_after_seconds: u64,
}

fn default_poll_after() -> u64 {
    2
}

/// Parsed response from `POST /v1/encode/upload`. Field names match
/// `EncodeUploadTargetResponse` in the private API exactly.
#[derive(Debug, Clone, Deserialize)]
struct EncodeUploadTarget {
    job_id: String,
    upload_url: String,
    #[serde(default = "default_upload_method")]
    #[allow(dead_code)]
    upload_method: String,
    input_ref: String,
}

fn default_upload_method() -> String {
    "PUT".to_string()
}

/// Body for the by-ref `POST /v1/encode` call (large-file path). Mirrors
/// `EncodeByRefBody` in `routes/encode.rs`.
#[derive(Debug, Clone, Serialize)]
struct EncodeByRefBody<'a> {
    job_id: &'a str,
    input_ref: &'a str,
}

/// Parsed 202 / 503 / 422 status JSON returned by `GET /v1/encode/:id`
/// while the job is in flight or has failed. Done returns the raw
/// binary, not this envelope.
#[derive(Debug, Clone, Deserialize)]
struct EncodeJobView {
    #[serde(default)]
    #[allow(dead_code)]
    job_id: Option<String>,
    status: String,
    #[serde(default)]
    #[allow(dead_code)]
    sidecar_url: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

/// Error-envelope shape the server emits via `EncodeError::into_response`.
#[derive(Debug, Clone, Deserialize)]
struct EncodeErrorEnvelope {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

/// Build a normalized base URL (no trailing slash) the rest of the
/// module can append `/v1/encode...` paths to.
fn normalize_base(api_url: &str) -> String {
    api_url.trim_end_matches('/').to_string()
}

/// Resolve a `poll_url` (which the server returns as a relative path
/// like `/v1/encode/<uuid>`) to a full URL by joining with the base.
fn join_url(base: &str, path: &str) -> String {
    if path.starts_with("http://") || path.starts_with("https://") {
        path.to_string()
    } else if let Some(stripped) = path.strip_prefix('/') {
        format!("{base}/{stripped}")
    } else {
        format!("{base}/{path}")
    }
}

/// Round-trip one encode request against the hosted API.
///
/// Blocks the calling thread; the public CLI runs synchronously so this
/// is the right shape. All error paths surface a user-friendly anyhow
/// chain so `catetus: error: ...` prints something actionable.
pub fn encode_via_api(req: &EncodeRequest<'_>) -> Result<EncodeOutcome> {
    if req.v5tail && req.target != EncodeTarget::Sog {
        return Err(anyhow!(
            "--emit-v5-tail requires --target sog (the V5.2 sidecar only \
             rides on top of a SOG container)"
        ));
    }
    if req.ply_bytes.is_empty() {
        return Err(anyhow!("input is empty; expected raw PLY bytes"));
    }
    let head = &req.ply_bytes[..req.ply_bytes.len().min(8)];
    if !(head.starts_with(b"ply\n") || head.starts_with(b"ply\r\n")) {
        return Err(anyhow!(
            "input does not look like a PLY (expected 'ply\\n' magic in first \
             8 bytes; got {:?})",
            String::from_utf8_lossy(head)
        ));
    }

    let base = normalize_base(req.api_url);
    let client = reqwest::blocking::Client::builder()
        // The whole-request timeout is enforced by us via the poll loop.
        // Per-connection / per-request keeps us from hanging on a dead socket.
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(120))
        .build()
        .context("building HTTP client for the Catetus encode API")?;

    let started = Instant::now();

    // ---- 1. POST PLY, get the job_id + poll_url. ----
    let mut post_url = format!(
        "{base}/v1/encode?target={t}&v5tail={v}",
        base = base,
        t = req.target.as_query_value(),
        v = if req.v5tail { "true" } else { "false" },
    );
    if let Some(label) = req.label {
        // The label is passed straight through to the server; we
        // percent-encode just enough to keep `?` / `&` / `=` from
        // breaking the URL. Anything else is fair game.
        let encoded: String = label
            .bytes()
            .map(|b| match b {
                b'?' | b'&' | b'=' | b'#' | b' ' | b'+' | b'%' => format!("%{:02X}", b),
                _ => (b as char).to_string(),
            })
            .collect();
        post_url.push_str(&format!("&label={encoded}"));
    }
    let post_resp = client
        .post(&post_url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(req.ply_bytes.to_vec())
        .send()
        .with_context(|| {
            format!(
                "POST {post_url} failed — is the Catetus API reachable? \
                 (override with --api-url or CATETUS_API_URL)"
            )
        })?;
    let post_status = post_resp.status();
    if !(post_status == reqwest::StatusCode::ACCEPTED || post_status.is_success()) {
        let body = post_resp.text().unwrap_or_default();
        return Err(anyhow!(
            "encode submission rejected by {post_url}: HTTP {post_status}: {}",
            shorten(&body)
        ));
    }
    let accepted: EncodeAccepted = post_resp
        .json()
        .context("parsing 202 response from POST /v1/encode")?;
    let poll_url = join_url(&base, &accepted.poll_url);
    let min_interval = req
        .min_poll_interval
        .max(Duration::from_secs(accepted.poll_after_seconds.max(1)));

    // ---- 2. Poll until Done (200 + binary) / Error / timeout. ----
    // Shared with the presigned-upload path via `poll_until_done`.
    poll_until_done(
        &client,
        &base,
        &poll_url,
        &accepted.job_id,
        req.v5tail,
        started,
        req.timeout,
        min_interval,
    )
}

/// Round-trip one encode request against the hosted API using the
/// large-file presigned-upload path (LARGE-2 Part A).
///
/// Three calls instead of the inline path's one:
///   1. `POST /v1/encode/upload` → mint `{ job_id, upload_url, input_ref }`.
///   2. Streaming `PUT <upload_url>` of the PLY bytes (no base64, no
///      whole-file buffering beyond what reqwest's body needs).
///   3. `POST /v1/encode` with `{ job_id, input_ref }` to trigger the
///      encode, then poll exactly like the inline path.
///
/// `ply_path` is read with a streaming reqwest body so a multi-GB file
/// isn't loaded into a `Vec<u8>` for the PUT.
pub fn encode_via_api_upload(
    api_url: &str,
    target: EncodeTarget,
    v5tail: bool,
    ply_path: &Path,
    label: Option<&str>,
    timeout: Duration,
    min_poll_interval: Duration,
) -> Result<EncodeOutcome> {
    if v5tail && target != EncodeTarget::Sog {
        return Err(anyhow!(
            "--emit-v5-tail requires --target sog (the V5.2 sidecar only \
             rides on top of a SOG container)"
        ));
    }

    let base = normalize_base(api_url);
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // The PUT of a multi-GB file can take a while; allow the full
        // request budget rather than the inline path's 120 s ceiling.
        .timeout(timeout)
        .build()
        .context("building HTTP client for the Catetus encode API")?;

    let started = Instant::now();

    // ---- 1. Mint the upload target. ----
    let mut mint_url = format!(
        "{base}/v1/encode/upload?target={t}&v5tail={v}",
        base = base,
        t = target.as_query_value(),
        v = if v5tail { "true" } else { "false" },
    );
    if let Some(label) = label {
        let encoded: String = label
            .bytes()
            .map(|b| match b {
                b'?' | b'&' | b'=' | b'#' | b' ' | b'+' | b'%' => format!("%{:02X}", b),
                _ => (b as char).to_string(),
            })
            .collect();
        mint_url.push_str(&format!("&label={encoded}"));
    }
    let mint_resp = client
        .post(&mint_url)
        .send()
        .with_context(|| {
            format!(
                "POST {mint_url} failed — is the Catetus API reachable? \
                 (override with --api-url or CATETUS_API_URL)"
            )
        })?;
    let mint_status = mint_resp.status();
    if !mint_status.is_success() {
        let body = mint_resp.text().unwrap_or_default();
        let detail = parse_error_message(&body).unwrap_or_else(|| shorten(&body));
        return Err(anyhow!(
            "minting upload target rejected by {mint_url}: HTTP {mint_status}: {detail}"
        ));
    }
    let target_info: EncodeUploadTarget = mint_resp
        .json()
        .context("parsing response from POST /v1/encode/upload")?;
    let upload_url = join_url(&base, &target_info.upload_url);

    // ---- 2. Streaming PUT the PLY bytes. ----
    let file = std::fs::File::open(ply_path)
        .with_context(|| format!("opening input PLY {}", ply_path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("stat-ing input PLY {}", ply_path.display()))?
        .len();
    // `reqwest::blocking::Body::sized` streams the file through a reader
    // (no whole-file buffering) and sets Content-Length so the server's
    // body-limit layer + blob PUT see the size up-front.
    let body = reqwest::blocking::Body::sized(file, file_len);
    let put_resp = client
        .put(&upload_url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(body)
        .send()
        .with_context(|| format!("PUT {upload_url} failed while uploading the PLY"))?;
    let put_status = put_resp.status();
    if !put_status.is_success() {
        let body = put_resp.text().unwrap_or_default();
        return Err(anyhow!(
            "uploading the PLY to {upload_url} failed: HTTP {put_status}: {}",
            shorten(&body)
        ));
    }

    // ---- 3. Trigger the encode by reference. ----
    let by_ref = EncodeByRefBody {
        job_id: &target_info.job_id,
        input_ref: &target_info.input_ref,
    };
    let post_url = format!("{base}/v1/encode");
    let post_resp = client
        .post(&post_url)
        .json(&by_ref)
        .send()
        .with_context(|| format!("POST {post_url} (by-ref) failed"))?;
    let post_status = post_resp.status();
    if !(post_status == reqwest::StatusCode::ACCEPTED || post_status.is_success()) {
        let body = post_resp.text().unwrap_or_default();
        let detail = parse_error_message(&body).unwrap_or_else(|| shorten(&body));
        return Err(anyhow!(
            "by-ref encode submission rejected by {post_url}: HTTP {post_status}: {detail}"
        ));
    }
    let accepted: EncodeAccepted = post_resp
        .json()
        .context("parsing 202 response from by-ref POST /v1/encode")?;
    let poll_url = join_url(&base, &accepted.poll_url);
    let min_interval =
        min_poll_interval.max(Duration::from_secs(accepted.poll_after_seconds.max(1)));

    // ---- 4. Poll until Done / Error / timeout (same as inline path). ----
    poll_until_done(
        &client,
        &base,
        &poll_url,
        &accepted.job_id,
        v5tail,
        started,
        timeout,
        min_interval,
    )
}

/// Shared poll loop used by both the inline and presigned paths: GET the
/// poll URL until 200 (binary) / terminal error / timeout, then fetch the
/// v5tail sidecar when requested.
#[allow(clippy::too_many_arguments)]
fn poll_until_done(
    client: &reqwest::blocking::Client,
    base: &str,
    poll_url: &str,
    job_id: &str,
    v5tail: bool,
    started: Instant,
    timeout: Duration,
    min_interval: Duration,
) -> Result<EncodeOutcome> {
    loop {
        if started.elapsed() >= timeout {
            return Err(anyhow!(
                "encode job {job_id} did not finish within {}s (last poll at {poll_url}); \
                 the worker may be cold-starting — retry or raise the timeout",
                timeout.as_secs(),
            ));
        }
        std::thread::sleep(min_interval);

        let get_resp = client
            .get(poll_url)
            .send()
            .with_context(|| format!("GET {poll_url} failed while polling for job result"))?;
        let status = get_resp.status();
        if status == reqwest::StatusCode::OK {
            let output_bytes = get_resp
                .bytes()
                .context("reading encoded output bytes from API")?
                .to_vec();
            let sidecar_bytes = if v5tail {
                let sidecar_url = format!("{base}/v1/encode/{job_id}/sidecar");
                let sc = client.get(&sidecar_url).send().with_context(|| {
                    format!("GET {sidecar_url} failed while fetching v5tail sidecar")
                })?;
                let sc_status = sc.status();
                if sc_status == reqwest::StatusCode::OK {
                    Some(sc.bytes().context("reading sidecar bytes")?.to_vec())
                } else {
                    let body = sc.text().unwrap_or_default();
                    return Err(anyhow!(
                        "v5tail sidecar requested but {sidecar_url} returned HTTP {sc_status}: {}",
                        shorten(&body)
                    ));
                }
            } else {
                None
            };
            return Ok(EncodeOutcome {
                job_id: job_id.to_string(),
                output_bytes,
                sidecar_bytes,
            });
        }
        if status == reqwest::StatusCode::ACCEPTED {
            continue;
        }
        let body_text = get_resp.text().unwrap_or_default();
        let detail = parse_error_message(&body_text).unwrap_or_else(|| shorten(&body_text));
        return Err(anyhow!(
            "encode job {job_id} failed: HTTP {status} from {poll_url}: {detail}"
        ));
    }
}

/// Try to pull a useful error string out of either error envelope shape
/// (`{"error": "...", "code": "..."}` from `EncodeError::into_response`
/// or `EncodeJobView { error, status }` from the in-flight path).
fn parse_error_message(body: &str) -> Option<String> {
    if let Ok(env) = serde_json::from_str::<EncodeErrorEnvelope>(body) {
        if let Some(msg) = env.error {
            return Some(match env.code {
                Some(c) => format!("[{c}] {msg}"),
                None => msg,
            });
        }
    }
    if let Ok(view) = serde_json::from_str::<EncodeJobView>(body) {
        if let Some(msg) = view.error {
            return Some(format!("[status={}] {msg}", view.status));
        }
        return Some(format!("status={}", view.status));
    }
    None
}

fn shorten(s: &str) -> String {
    const MAX: usize = 512;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}… [truncated, {} bytes total]", &s[..MAX], s.len())
    }
}

// -------------------------------------------------------------------------
// Decode endpoint — placeholder client for `sog-apply-v5-tail`.
//
// The hosted `/v1/decode` route is not yet implemented in the private
// `apps/api` (see LAUNCH4_BLOCKER.md). This client is wired so that the
// moment the route lands, the CLI works without further changes; today
// it surfaces a clear "endpoint not implemented yet" error when the
// server returns 404 / 503.
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct DecodeRequestEnvelope<'a> {
    sog_b64: &'a str,
    sidecar_b64: &'a str,
}

/// Round-trip the apply-v5-tail call against the hosted `/v1/decode`
/// endpoint. Currently the server returns 404 — the client is here so
/// the CLI keeps a single source of truth for the endpoint shape, and
/// so that integration tests can pin the protocol against a wiremock
/// stub.
pub fn apply_v5tail_via_api(
    api_url: &str,
    sog_bytes: &[u8],
    sidecar_bytes: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>> {
    use base64::Engine;
    let base = normalize_base(api_url);
    let url = format!("{base}/v1/decode?source=sog&v5tail=true");
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(timeout)
        .build()
        .context("building HTTP client for the Catetus decode API")?;
    let payload = DecodeRequestEnvelope {
        sog_b64: &base64::engine::general_purpose::STANDARD.encode(sog_bytes),
        sidecar_b64: &base64::engine::general_purpose::STANDARD.encode(sidecar_bytes),
    };
    let resp = client
        .post(&url)
        .json(&payload)
        .send()
        .with_context(|| {
            format!(
                "POST {url} failed — is the Catetus decode endpoint reachable? \
                 (override with --api-url or CATETUS_API_URL)"
            )
        })?;
    let status = resp.status();
    if status == reqwest::StatusCode::OK {
        return Ok(resp
            .bytes()
            .context("reading decoded PLY bytes from /v1/decode")?
            .to_vec());
    }
    let body = resp.text().unwrap_or_default();
    let detail = parse_error_message(&body).unwrap_or_else(|| shorten(&body));
    if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
    {
        return Err(anyhow!(
            "POST {url} returned HTTP {status}: {detail}. The /v1/decode endpoint \
             is not yet implemented in the hosted API — see LAUNCH4_BLOCKER.md \
             for the open task. As a workaround, decode locally using a \
             Catetus-Pro build or wait for the next API deploy."
        ));
    }
    Err(anyhow!(
        "POST {url} returned HTTP {status}: {detail}"
    ))
}

/// Convenience wrapper for the `Optimize { --target sog }` path that
/// reads the input PLY from disk and writes the SOG (+ optional
/// sidecar) outputs to disk.
pub fn run_encode_to_disk(
    api_url: &str,
    target: EncodeTarget,
    v5tail: bool,
    input: &Path,
    output: &Path,
    sidecar_output: Option<&Path>,
    label: Option<&str>,
    timeout: Duration,
) -> Result<EncodeOutcome> {
    // Size-based dispatch: large inputs use the presigned-upload flow so
    // we never buffer a multi-GB PLY in memory or inflate it 33% through
    // base64; small inputs keep the one-call inline path (LARGE-2 Part A).
    let size = std::fs::metadata(input)
        .with_context(|| format!("stat-ing input PLY {}", input.display()))?
        .len();
    let outcome = if size >= UPLOAD_THRESHOLD_BYTES {
        encode_via_api_upload(
            api_url,
            target,
            v5tail,
            input,
            label,
            timeout,
            Duration::from_secs(2),
        )?
    } else {
        let ply_bytes = std::fs::read(input)
            .with_context(|| format!("reading input PLY {}", input.display()))?;
        encode_via_api(&EncodeRequest {
            api_url,
            target,
            v5tail,
            ply_bytes: &ply_bytes,
            label,
            timeout,
            min_poll_interval: Duration::from_secs(2),
        })?
    };
    std::fs::write(output, &outcome.output_bytes)
        .with_context(|| format!("writing encoded output to {}", output.display()))?;
    if let (true, Some(path), Some(bytes)) = (v5tail, sidecar_output, outcome.sidecar_bytes.as_ref())
    {
        std::fs::write(path, bytes)
            .with_context(|| format!("writing v5tail sidecar to {}", path.display()))?;
    }
    Ok(outcome)
}

/// Resolve the effective API base URL given (in priority order) the
/// `--api-url` flag, the `CATETUS_API_URL` env var, then the compile-
/// time default. Trims trailing slashes.
pub fn resolve_api_url(flag: Option<&str>) -> String {
    let raw = flag
        .map(|s| s.to_string())
        .or_else(|| std::env::var("CATETUS_API_URL").ok())
        .unwrap_or_else(|| DEFAULT_ENCODE_API_URL.to_string());
    normalize_base(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_trailing_slash() {
        assert_eq!(normalize_base("https://api.catetus.com/"), "https://api.catetus.com");
        assert_eq!(normalize_base("https://api.catetus.com"), "https://api.catetus.com");
        assert_eq!(normalize_base("http://127.0.0.1:1234//"), "http://127.0.0.1:1234");
    }

    #[test]
    fn join_url_handles_relative_and_absolute() {
        let b = "http://x";
        assert_eq!(join_url(b, "/v1/encode/abc"), "http://x/v1/encode/abc");
        assert_eq!(join_url(b, "v1/encode/abc"), "http://x/v1/encode/abc");
        assert_eq!(
            join_url(b, "https://other.example/abc"),
            "https://other.example/abc"
        );
    }

    #[test]
    fn resolve_api_url_prefers_flag_over_env() {
        let prev = std::env::var("CATETUS_API_URL").ok();
        // SAFETY: tests in the same crate may race. We restore at end.
        std::env::set_var("CATETUS_API_URL", "http://env-default");
        assert_eq!(resolve_api_url(Some("http://flag")), "http://flag");
        assert_eq!(resolve_api_url(None), "http://env-default");
        std::env::remove_var("CATETUS_API_URL");
        assert_eq!(resolve_api_url(None), DEFAULT_ENCODE_API_URL);
        if let Some(v) = prev {
            std::env::set_var("CATETUS_API_URL", v);
        }
    }

    #[test]
    fn parse_error_envelope_picks_up_code() {
        let body = r#"{"error":"bad ply","code":"bad_request"}"#;
        assert_eq!(
            parse_error_message(body).as_deref(),
            Some("[bad_request] bad ply")
        );
    }

    #[test]
    fn upload_threshold_is_100_mib() {
        // Guards the size-based dispatch in `run_encode_to_disk`: a
        // 99 MB file stays inline, a 101 MB file uses the upload flow.
        assert_eq!(UPLOAD_THRESHOLD_BYTES, 100 * 1024 * 1024);
        assert!(99 * 1024 * 1024 < UPLOAD_THRESHOLD_BYTES);
        assert!(101 * 1024 * 1024 >= UPLOAD_THRESHOLD_BYTES);
    }

    #[test]
    fn upload_target_deserializes_api_shape() {
        // Pins the CLI's parse against the private API's
        // EncodeUploadTargetResponse field names.
        let body = r#"{
            "job_id":"11111111-1111-1111-1111-111111111111",
            "upload_url":"http://api.test/v1/encode/blob/encode/abc.ply",
            "upload_method":"PUT",
            "input_ref":"encode/abc.ply"
        }"#;
        let t: EncodeUploadTarget = serde_json::from_str(body).unwrap();
        assert_eq!(t.job_id, "11111111-1111-1111-1111-111111111111");
        assert_eq!(t.input_ref, "encode/abc.ply");
        assert!(t.upload_url.ends_with("encode/abc.ply"));
    }

    #[test]
    fn parse_error_envelope_falls_back_to_job_view() {
        // EncodeErrorEnvelope shape (no `code` field) is also tried first;
        // a job-view body with an `error` field deserializes against it
        // because all fields are `#[serde(default)]`. Either shape's
        // message text is what gets bubbled up — assert on substring so
        // the impl can pick either path without breaking the test.
        let body = r#"{"status":"error","error":"worker exploded"}"#;
        let msg = parse_error_message(body).expect("some message");
        assert!(
            msg.contains("worker exploded"),
            "expected the error text to surface, got {msg:?}"
        );
    }
}
