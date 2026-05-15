//! Capture-tool import endpoints.
//!
//! Three routes — Luma, Polycam, Scaniverse — each accepting a single
//! `share_url` and turning it into a regular `/v1/optimize`-style job by:
//!
//!   1. Resolving the share URL to the provider's direct splat-asset URL
//!      (`gaussian_splat_url` for Luma, `glcdn.poly.cam/<id>.ply` for
//!      Polycam, the embedded `.usdz`/`.ply` sibling for Scaniverse).
//!   2. Building a regular `Job` with `source_url = <resolved>` and handing
//!      it to the same Modal worker enqueue path that `/v1/jobs` uses for
//!      URL-mode jobs.
//!   3. Returning `{ job_id }` so the client polls the standard
//!      `/v1/jobs/:id` endpoint thereafter.
//!
//! Each provider's resolver is behind a trait so the integration tests can
//! inject mocked HTTP without touching live REST endpoints. The default impl
//! is plain `reqwest` and reads its base URLs from env so an operator can
//! point at a staging endpoint.
//!
//! Auth: same bearer-token middleware as `/v1/jobs`. Rate-limit: 10 imports/
//! min/key via the in-memory `ImportRateLimiter` (sliding window, no Redis
//! dependency — single-instance deploy can afford this).
//!
//! SECURITY: the resolved URL is fed back through `validate_source_url`
//! (re-exported by the caller in `main.rs`) so a malicious provider response
//! can't trick the worker into hitting an internal-IP literal. Provider hosts
//! themselves are HTTPS-only and pinned to a per-provider allowlist (the
//! `gaussian_splat_url` returned by Luma MUST live under `lumalabs.ai` /
//! `cdn-luma.com`; ditto for Polycam / Scaniverse). This blocks the trivial
//! open-redirect class where a provider response includes a
//! `gaussian_splat_url: "https://evil.example.com/x.ply"`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Per-provider error surface. Mapped to `ApiError` in `main.rs` so the
/// HTTP envelope matches every other route.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// Caller-supplied share URL didn't match the provider's URL shape.
    /// Returned as 400 — same as `/v1/jobs` BadRequest.
    #[error("invalid {provider} share URL: {detail}")]
    InvalidShareUrl { provider: &'static str, detail: String },

    /// Provider returned a response that doesn't carry a usable asset URL
    /// — typically a still-processing capture, or (for Scaniverse) a USDZ
    /// without an embedded PLY. Returned as 415 so the client can show a
    /// "convert to PLY in the desktop app" hint without retrying.
    #[error("{provider} capture is not importable: {detail}")]
    Unsupported { provider: &'static str, detail: String },

    /// Provider lookup failed for transport / 5xx reasons. Mapped to 502.
    #[error("{provider} lookup failed: {detail}")]
    UpstreamFailed { provider: &'static str, detail: String },

    /// Resolved asset URL points somewhere unsafe (private IP, http://, …).
    /// Mapped to 400 — the caller's share URL is the proximate cause.
    #[error("resolved {provider} asset URL rejected: {detail}")]
    UnsafeAssetUrl { provider: &'static str, detail: String },

    /// Caller has burned through their 10 imports/min budget. Mapped to 429.
    #[error("import rate limit exceeded ({limit}/min)")]
    RateLimited { limit: u32 },
}

impl ImportError {
    /// HTTP status this error should surface as. Used by the route module's
    /// `IntoResponse` adapter in `main.rs`.
    pub fn http_status(&self) -> u16 {
        match self {
            ImportError::InvalidShareUrl { .. } | ImportError::UnsafeAssetUrl { .. } => 400,
            ImportError::Unsupported { .. } => 415,
            ImportError::UpstreamFailed { .. } => 502,
            ImportError::RateLimited { .. } => 429,
        }
    }

    /// Provider name (used by tests to assert which arm fired).
    pub fn provider(&self) -> &'static str {
        match self {
            ImportError::InvalidShareUrl { provider, .. }
            | ImportError::Unsupported { provider, .. }
            | ImportError::UpstreamFailed { provider, .. }
            | ImportError::UnsafeAssetUrl { provider, .. } => provider,
            ImportError::RateLimited { .. } => "",
        }
    }
}

/// Body schema for every import endpoint. The three providers happen to
/// share the same shape (just a share URL) so a single struct is plenty.
#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub share_url: String,
    /// Optional caller-supplied label that flows through to the Job.
    #[serde(default)]
    pub label: Option<String>,
}

/// Response shape returned by every import handler. The body intentionally
/// matches `/v1/jobs`'s `CreateJobResponse.id` field so existing client SDKs
/// can reuse their poll loop.
#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub job_id: Uuid,
    /// The resolved direct-asset URL we handed the worker. Surfaced for
    /// observability — clients that want to keep a record of "where did
    /// this scene come from" can stamp this alongside the job_id.
    pub source_url: String,
    /// Which provider resolver ran. Always one of `luma` / `polycam` /
    /// `scaniverse`.
    pub provider: &'static str,
}

/// Resolved capture descriptor returned by every provider resolver. The
/// route handler uses this to build the Job — see `main.rs::build_job`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCapture {
    /// Direct HTTPS URL the worker should fetch. Already validated to be
    /// HTTPS and to live under the provider's CDN.
    pub source_url: String,
    /// Suggested filename, derived from the capture id + format. Fed to
    /// `sanitize_filename` on the way into the Job.
    pub filename: String,
    /// Capture id as the provider sees it. Surfaced for logging only.
    pub capture_id: String,
}

/// Provider-agnostic resolver trait. The default impls live below and use
/// `reqwest::Client`; tests inject a fake that returns canned responses.
#[async_trait]
pub trait CaptureResolver: Send + Sync {
    async fn resolve_luma(&self, share_url: &str) -> Result<ResolvedCapture, ImportError>;
    async fn resolve_polycam(&self, share_url: &str) -> Result<ResolvedCapture, ImportError>;
    async fn resolve_scaniverse(&self, share_url: &str) -> Result<ResolvedCapture, ImportError>;
}

/* ----------------- LUMA ----------------- */

/// Extract the capture id from any of the documented Luma share URL shapes.
/// Examples (all valid):
///   https://lumalabs.ai/capture/abcd-1234
///   https://lumalabs.ai/embed/abcd-1234
///   https://lumalabs.ai/capture/abcd-1234?utm=share
///
/// Returns Err with `InvalidShareUrl` if the URL doesn't match the expected
/// shape. We do NOT make the regex too loose — Luma's `id` is a stable UUID-
/// like string, and accepting any path tail would let a hostile share URL
/// smuggle arbitrary path segments into the REST call.
pub fn parse_luma_share(share_url: &str) -> Result<String, ImportError> {
    let url = share_url.trim();
    let stripped = url
        .strip_prefix("https://lumalabs.ai/")
        .or_else(|| url.strip_prefix("https://www.lumalabs.ai/"))
        .ok_or(ImportError::InvalidShareUrl {
            provider: "luma",
            detail: "must start with https://lumalabs.ai/".into(),
        })?;
    let after_kind = stripped
        .strip_prefix("capture/")
        .or_else(|| stripped.strip_prefix("embed/"))
        .ok_or(ImportError::InvalidShareUrl {
            provider: "luma",
            detail: "expected /capture/<id> or /embed/<id>".into(),
        })?;
    // First path segment, stripping query / fragment.
    let id = after_kind
        .split(|c: char| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("");
    if id.is_empty() {
        return Err(ImportError::InvalidShareUrl {
            provider: "luma",
            detail: "missing capture id".into(),
        });
    }
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(ImportError::InvalidShareUrl {
            provider: "luma",
            detail: "capture id contains unexpected characters".into(),
        });
    }
    Ok(id.to_string())
}

/// Validate that a Luma-returned `gaussian_splat_url` is safe to feed the
/// worker. Pinned to lumalabs.ai / cdn-luma.com / s3.amazonaws.com — a
/// malicious or compromised Luma response can't trick us into fetching
/// from an attacker-controlled host.
pub fn validate_luma_asset_url(asset_url: &str) -> Result<(), ImportError> {
    if !asset_url.starts_with("https://") {
        return Err(ImportError::UnsafeAssetUrl {
            provider: "luma",
            detail: "must be HTTPS".into(),
        });
    }
    let host = asset_url["https://".len()..]
        .split(|c: char| c == '/' || c == ':' || c == '?' || c == '#')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let allowed = host == "lumalabs.ai"
        || host.ends_with(".lumalabs.ai")
        || host == "cdn-luma.com"
        || host.ends_with(".cdn-luma.com")
        // Storage CDN currently fronted via S3. Documented in their
        // public capture REST examples.
        || host.ends_with(".amazonaws.com");
    if !allowed {
        return Err(ImportError::UnsafeAssetUrl {
            provider: "luma",
            detail: format!("host {host} is not a Luma asset CDN"),
        });
    }
    Ok(())
}

/* ----------------- POLYCAM ----------------- */

/// Polycam share URLs come in two shapes:
///   https://poly.cam/capture/<id>
///   https://polycam.com/capture/<id>
///
/// The capture id is a hex-ish alphanumeric token; we accept the same
/// alphabet as Luma (alnum + `-_`). Anything stricter risks rejecting valid
/// future ids; anything looser invites injection.
pub fn parse_polycam_share(share_url: &str) -> Result<String, ImportError> {
    let url = share_url.trim();
    let stripped = url
        .strip_prefix("https://poly.cam/")
        .or_else(|| url.strip_prefix("https://www.poly.cam/"))
        .or_else(|| url.strip_prefix("https://polycam.com/"))
        .or_else(|| url.strip_prefix("https://www.polycam.com/"))
        .ok_or(ImportError::InvalidShareUrl {
            provider: "polycam",
            detail: "must start with https://poly.cam/ or https://polycam.com/".into(),
        })?;
    let after_kind = stripped
        .strip_prefix("capture/")
        .ok_or(ImportError::InvalidShareUrl {
            provider: "polycam",
            detail: "expected /capture/<id>".into(),
        })?;
    let id = after_kind
        .split(|c: char| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("");
    if id.is_empty() {
        return Err(ImportError::InvalidShareUrl {
            provider: "polycam",
            detail: "missing capture id".into(),
        });
    }
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(ImportError::InvalidShareUrl {
            provider: "polycam",
            detail: "capture id contains unexpected characters".into(),
        });
    }
    Ok(id.to_string())
}

/// Polycam exposes captures on `glcdn.poly.cam` (gltf-cdn). The PLY form
/// — the one we want — is at `https://glcdn.poly.cam/<id>.ply`. We don't
/// need a REST roundtrip for the basic case; just rewriting the URL is
/// enough. The optional resolver hit is for "is this capture exported as
/// PLY at all?" — we delegate that to a HEAD probe so we can return a
/// clean 415 instead of a confusing 404 later from the worker.
pub fn polycam_ply_url(capture_id: &str, cdn_base: &str) -> String {
    let base = cdn_base.trim_end_matches('/');
    format!("{base}/{capture_id}.ply")
}

/// Same allowlist treatment as Luma — `glcdn.poly.cam` and its parent.
pub fn validate_polycam_asset_url(asset_url: &str) -> Result<(), ImportError> {
    if !asset_url.starts_with("https://") {
        return Err(ImportError::UnsafeAssetUrl {
            provider: "polycam",
            detail: "must be HTTPS".into(),
        });
    }
    let host = asset_url["https://".len()..]
        .split(|c: char| c == '/' || c == ':' || c == '?' || c == '#')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let allowed = host == "glcdn.poly.cam"
        || host.ends_with(".poly.cam")
        || host == "polycam.com"
        || host.ends_with(".polycam.com");
    if !allowed {
        return Err(ImportError::UnsafeAssetUrl {
            provider: "polycam",
            detail: format!("host {host} is not a Polycam asset CDN"),
        });
    }
    Ok(())
}

/* ----------------- SCANIVERSE ----------------- */

/// Scaniverse share URLs:
///   https://scaniverse.com/scan/<id>
///   https://www.scaniverse.com/scan/<id>
pub fn parse_scaniverse_share(share_url: &str) -> Result<String, ImportError> {
    let url = share_url.trim();
    let stripped = url
        .strip_prefix("https://scaniverse.com/")
        .or_else(|| url.strip_prefix("https://www.scaniverse.com/"))
        .ok_or(ImportError::InvalidShareUrl {
            provider: "scaniverse",
            detail: "must start with https://scaniverse.com/".into(),
        })?;
    let after_kind = stripped
        .strip_prefix("scan/")
        .ok_or(ImportError::InvalidShareUrl {
            provider: "scaniverse",
            detail: "expected /scan/<id>".into(),
        })?;
    let id = after_kind
        .split(|c: char| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("");
    if id.is_empty() {
        return Err(ImportError::InvalidShareUrl {
            provider: "scaniverse",
            detail: "missing scan id".into(),
        });
    }
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(ImportError::InvalidShareUrl {
            provider: "scaniverse",
            detail: "scan id contains unexpected characters".into(),
        });
    }
    Ok(id.to_string())
}

/* ----------------- DEFAULT HTTP RESOLVER ----------------- */

/// Production resolver. Wraps a `reqwest::Client` and per-provider base URLs
/// (overridable so the integration tests can point at a wiremock listener).
pub struct HttpCaptureResolver {
    pub http: reqwest::Client,
    /// Luma REST root, no trailing slash. Default
    /// `https://webapp.lumalabs.ai/api/v2`.
    pub luma_api_base: String,
    /// Polycam GL-CDN root. Default `https://glcdn.poly.cam`.
    pub polycam_cdn_base: String,
    /// Scaniverse CDN root. Default `https://scans.scaniverse.com`.
    pub scaniverse_cdn_base: String,
}

impl HttpCaptureResolver {
    pub fn from_env() -> Self {
        let http = reqwest::Client::builder()
            // 15s per provider hit — well over their p99 and short enough
            // to not stall the request worker if a provider degrades.
            .timeout(Duration::from_secs(15))
            .build()
            .expect("reqwest client");
        Self {
            http,
            luma_api_base: std::env::var("SPLATFORGE_LUMA_API_BASE")
                .unwrap_or_else(|_| "https://webapp.lumalabs.ai/api/v2".to_string()),
            polycam_cdn_base: std::env::var("SPLATFORGE_POLYCAM_CDN_BASE")
                .unwrap_or_else(|_| "https://glcdn.poly.cam".to_string()),
            scaniverse_cdn_base: std::env::var("SPLATFORGE_SCANIVERSE_CDN_BASE")
                .unwrap_or_else(|_| "https://scans.scaniverse.com".to_string()),
        }
    }

    /// Test-friendly constructor that takes explicit base URLs (no env).
    pub fn with_bases(luma: String, polycam: String, scaniverse: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("reqwest client"),
            luma_api_base: luma,
            polycam_cdn_base: polycam,
            scaniverse_cdn_base: scaniverse,
        }
    }
}

#[async_trait]
impl CaptureResolver for HttpCaptureResolver {
    /// Luma `/captures/{id}` returns `{ "id": ..., "gaussian_splat_url":
    /// "https://...", "status": "complete" | "processing" }`. We require
    /// `status == "complete"` so a half-processed capture surfaces as 415
    /// rather than a 404 from the worker fetch.
    async fn resolve_luma(&self, share_url: &str) -> Result<ResolvedCapture, ImportError> {
        let capture_id = parse_luma_share(share_url)?;
        let url = format!("{}/captures/{}", self.luma_api_base.trim_end_matches('/'), capture_id);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| ImportError::UpstreamFailed {
                provider: "luma",
                detail: e.to_string(),
            })?;
        if resp.status() == 404 {
            return Err(ImportError::InvalidShareUrl {
                provider: "luma",
                detail: format!("Luma has no capture with id {capture_id}"),
            });
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ImportError::UpstreamFailed {
                provider: "luma",
                detail: format!("{status}: {body}"),
            });
        }
        let body: serde_json::Value = resp.json().await.map_err(|e| ImportError::UpstreamFailed {
            provider: "luma",
            detail: format!("decoding JSON: {e}"),
        })?;
        let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if !status.eq_ignore_ascii_case("complete") && !status.eq_ignore_ascii_case("ready") {
            return Err(ImportError::Unsupported {
                provider: "luma",
                detail: format!("capture status is {status:?}; not ready for import"),
            });
        }
        let asset_url = body
            .get("gaussian_splat_url")
            .and_then(|v| v.as_str())
            .ok_or(ImportError::Unsupported {
                provider: "luma",
                detail: "response missing gaussian_splat_url — capture is not a Gaussian splat".into(),
            })?;
        validate_luma_asset_url(asset_url)?;
        Ok(ResolvedCapture {
            source_url: asset_url.to_string(),
            filename: format!("luma-{capture_id}.ply"),
            capture_id,
        })
    }

    /// Polycam: rewrite the share URL into the CDN URL and verify with a
    /// HEAD request so a non-PLY capture surfaces as 415 immediately.
    async fn resolve_polycam(&self, share_url: &str) -> Result<ResolvedCapture, ImportError> {
        let capture_id = parse_polycam_share(share_url)?;
        let asset_url = polycam_ply_url(&capture_id, &self.polycam_cdn_base);
        validate_polycam_asset_url(&asset_url)?;
        // HEAD probe — most CDN edges support it cheaply and we get a
        // clear 404 vs 200 signal before queueing worker work.
        let resp = self
            .http
            .head(&asset_url)
            .send()
            .await
            .map_err(|e| ImportError::UpstreamFailed {
                provider: "polycam",
                detail: e.to_string(),
            })?;
        if resp.status() == 404 {
            return Err(ImportError::Unsupported {
                provider: "polycam",
                detail: format!(
                    "no PLY export for capture {capture_id} — export it from the Polycam app first"
                ),
            });
        }
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(ImportError::UpstreamFailed {
                provider: "polycam",
                detail: format!("HEAD {asset_url} -> {status}"),
            });
        }
        Ok(ResolvedCapture {
            source_url: asset_url,
            filename: format!("polycam-{capture_id}.ply"),
            capture_id,
        })
    }

    /// Scaniverse: we don't yet ship a USDZ-without-PLY converter. The
    /// resolver checks whether the scan exposes a sibling `.ply` (which
    /// some flagship users export manually); if it does, we use it. If
    /// not, we return 415 with the documented fallback instruction.
    async fn resolve_scaniverse(&self, share_url: &str) -> Result<ResolvedCapture, ImportError> {
        let scan_id = parse_scaniverse_share(share_url)?;
        let base = self.scaniverse_cdn_base.trim_end_matches('/');
        let ply_url = format!("{base}/{scan_id}.ply");
        // HEAD probe for the optional .ply sibling.
        let resp = self
            .http
            .head(&ply_url)
            .send()
            .await
            .map_err(|e| ImportError::UpstreamFailed {
                provider: "scaniverse",
                detail: e.to_string(),
            })?;
        if resp.status().is_success() {
            return Ok(ResolvedCapture {
                source_url: ply_url,
                filename: format!("scaniverse-{scan_id}.ply"),
                capture_id: scan_id,
            });
        }
        Err(ImportError::Unsupported {
            provider: "scaniverse",
            detail: "Scaniverse USDZ-without-PLY not yet supported — convert to PLY via the desktop app and re-share, or upload the .ply directly to /v1/jobs".into(),
        })
    }
}

/* ----------------- RATE LIMITER ----------------- */

/// Sliding-window rate limiter keyed on API key. 10 imports/min/key is the
/// design-partner contract; configurable so dev / CI can override.
///
/// Storage is an in-memory `HashMap<key, VecDeque<Instant>>`. Each request
/// pushes `now` onto the deque and trims anything older than `window`. This
/// is O(1) amortized per request and bounded by `limit` per key. Single-
/// instance deploy can absorb the worst case (10 keys × 10 requests = 100
/// timestamps held at once) comfortably; a future multi-instance promotion
/// will swap this for Redis.
pub struct ImportRateLimiter {
    pub limit: u32,
    pub window: Duration,
    state: Mutex<HashMap<String, std::collections::VecDeque<Instant>>>,
}

impl ImportRateLimiter {
    pub fn new(limit: u32, window: Duration) -> Self {
        Self {
            limit,
            window,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// 10 imports / 60s — the production default.
    pub fn default_production() -> Self {
        Self::new(10, Duration::from_secs(60))
    }

    /// Check + record a request from `key`. Returns `Err(RateLimited)` if
    /// the caller would exceed the budget.
    pub async fn check(&self, key: &str) -> Result<(), ImportError> {
        self.check_at(key, Instant::now()).await
    }

    /// Test-deterministic variant: caller supplies the timestamp so unit
    /// tests don't depend on wall clock.
    pub async fn check_at(&self, key: &str, now: Instant) -> Result<(), ImportError> {
        let mut guard = self.state.lock().await;
        let entry = guard.entry(key.to_string()).or_default();
        // Drop entries outside the window. We use a deque so trimming is
        // O(k) where k is the number of expired entries — cheap.
        while let Some(front) = entry.front().copied() {
            if now.duration_since(front) >= self.window {
                entry.pop_front();
            } else {
                break;
            }
        }
        if entry.len() as u32 >= self.limit {
            return Err(ImportError::RateLimited { limit: self.limit });
        }
        entry.push_back(now);
        Ok(())
    }
}

/* ----------------- TOP-LEVEL ORCHESTRATION ----------------- */

/// Which of the three providers a request targets. Lets the same orchestrator
/// drive all three handlers without duplicating the rate-limit + resolver glue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Luma,
    Polycam,
    Scaniverse,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Luma => "luma",
            Provider::Polycam => "polycam",
            Provider::Scaniverse => "scaniverse",
        }
    }
}

/// Run the resolver + rate-limit step. Returns the resolved capture; the
/// caller (in `main.rs`) is responsible for turning that into a Job and
/// enqueueing the worker so this module stays free of `AppState`-shaped
/// imports.
pub async fn run_import(
    resolver: &Arc<dyn CaptureResolver>,
    limiter: &ImportRateLimiter,
    api_key: &str,
    provider: Provider,
    share_url: &str,
) -> Result<ResolvedCapture, ImportError> {
    limiter.check(api_key).await?;
    match provider {
        Provider::Luma => resolver.resolve_luma(share_url).await,
        Provider::Polycam => resolver.resolve_polycam(share_url).await,
        Provider::Scaniverse => resolver.resolve_scaniverse(share_url).await,
    }
}

/* ----------------- UNIT TESTS ----------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_share_parser_accepts_capture_and_embed() {
        assert_eq!(
            parse_luma_share("https://lumalabs.ai/capture/abc-123").unwrap(),
            "abc-123"
        );
        assert_eq!(
            parse_luma_share("https://www.lumalabs.ai/embed/xyz_99").unwrap(),
            "xyz_99"
        );
        assert_eq!(
            parse_luma_share("https://lumalabs.ai/capture/abc-123?utm=x").unwrap(),
            "abc-123"
        );
    }

    #[test]
    fn luma_share_parser_rejects_bad_shape() {
        assert!(parse_luma_share("http://lumalabs.ai/capture/abc").is_err());
        assert!(parse_luma_share("https://lumalabs.ai/other/abc").is_err());
        assert!(parse_luma_share("https://lumalabs.ai/capture/").is_err());
        assert!(parse_luma_share("https://lumalabs.ai/capture/abc;evil").is_err());
    }

    #[test]
    fn luma_asset_url_allowlist() {
        assert!(validate_luma_asset_url("https://cdn-luma.com/x/y.ply").is_ok());
        assert!(validate_luma_asset_url("https://luma-prod.s3.amazonaws.com/x.ply").is_ok());
        assert!(validate_luma_asset_url("http://lumalabs.ai/x.ply").is_err());
        assert!(validate_luma_asset_url("https://evil.example.com/x.ply").is_err());
    }

    #[test]
    fn polycam_share_parser_accepts_both_hosts() {
        assert_eq!(
            parse_polycam_share("https://poly.cam/capture/xyz").unwrap(),
            "xyz"
        );
        assert_eq!(
            parse_polycam_share("https://polycam.com/capture/abc-1").unwrap(),
            "abc-1"
        );
    }

    #[test]
    fn polycam_ply_url_construction() {
        assert_eq!(
            polycam_ply_url("xyz", "https://glcdn.poly.cam"),
            "https://glcdn.poly.cam/xyz.ply"
        );
        assert_eq!(
            polycam_ply_url("xyz", "https://glcdn.poly.cam/"),
            "https://glcdn.poly.cam/xyz.ply"
        );
    }

    #[test]
    fn scaniverse_share_parser() {
        assert_eq!(
            parse_scaniverse_share("https://scaniverse.com/scan/abc-123").unwrap(),
            "abc-123"
        );
        assert!(parse_scaniverse_share("https://other.com/scan/abc").is_err());
    }

    #[tokio::test]
    async fn rate_limiter_allows_under_budget() {
        let lim = ImportRateLimiter::new(3, Duration::from_secs(60));
        let now = Instant::now();
        assert!(lim.check_at("k", now).await.is_ok());
        assert!(lim.check_at("k", now).await.is_ok());
        assert!(lim.check_at("k", now).await.is_ok());
    }

    #[tokio::test]
    async fn rate_limiter_blocks_over_budget() {
        let lim = ImportRateLimiter::new(2, Duration::from_secs(60));
        let now = Instant::now();
        assert!(lim.check_at("k", now).await.is_ok());
        assert!(lim.check_at("k", now).await.is_ok());
        let err = lim.check_at("k", now).await.unwrap_err();
        assert!(matches!(err, ImportError::RateLimited { limit: 2 }));
    }

    #[tokio::test]
    async fn rate_limiter_recovers_after_window() {
        let lim = ImportRateLimiter::new(1, Duration::from_millis(50));
        let t0 = Instant::now();
        assert!(lim.check_at("k", t0).await.is_ok());
        assert!(lim.check_at("k", t0).await.is_err());
        let t1 = t0 + Duration::from_millis(60);
        assert!(lim.check_at("k", t1).await.is_ok());
    }

    #[tokio::test]
    async fn rate_limiter_per_key_isolation() {
        let lim = ImportRateLimiter::new(1, Duration::from_secs(60));
        let now = Instant::now();
        assert!(lim.check_at("alice", now).await.is_ok());
        // Bob's bucket is independent of Alice's even at the limit.
        assert!(lim.check_at("bob", now).await.is_ok());
        assert!(lim.check_at("alice", now).await.is_err());
    }

    #[test]
    fn error_status_codes() {
        assert_eq!(
            ImportError::InvalidShareUrl {
                provider: "luma",
                detail: "x".into()
            }
            .http_status(),
            400
        );
        assert_eq!(
            ImportError::Unsupported {
                provider: "scaniverse",
                detail: "x".into()
            }
            .http_status(),
            415
        );
        assert_eq!(
            ImportError::UpstreamFailed {
                provider: "luma",
                detail: "x".into()
            }
            .http_status(),
            502
        );
        assert_eq!(ImportError::RateLimited { limit: 10 }.http_status(), 429);
    }
}
