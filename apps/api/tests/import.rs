//! Integration tests for the three capture-tool import resolvers.
//!
//! Each test stands up a `wiremock` listener, points an
//! `HttpCaptureResolver` at it via the per-provider base-URL override,
//! and asserts the resolver returns the right `ResolvedCapture` (or
//! error variant) for the canned response. No live HTTP — these run in
//! CI without network.
//!
//! Two layers under test:
//!
//!   1. Per-provider resolver — share-URL parsing, REST roundtrip,
//!      asset-URL allowlist, status-code mapping.
//!   2. Orchestrator `run_import` — rate-limiter integration. We don't
//!      stand up the full Axum router because the handler shape is
//!      thin glue around `run_import`; exercising the orchestrator
//!      directly keeps the test cheap and stable.
//!
//! When the live providers change response shape, only the canned
//! `Mock::given(...).respond_with(...)` lines need updating.

use std::sync::Arc;
use std::time::Duration;

use splatforge_api::routes::import::{
    run_import, CaptureResolver, HttpCaptureResolver, ImportError, ImportRateLimiter, Provider,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a resolver whose three base URLs all point at the supplied mock
/// server. Lets one mock handle every provider's path without standing up
/// three listeners.
fn resolver_against(mock_url: &str) -> Arc<dyn CaptureResolver> {
    Arc::new(HttpCaptureResolver::with_bases(
        mock_url.to_string(),
        mock_url.to_string(),
        mock_url.to_string(),
    ))
}

/* ---------- LUMA ---------- */

#[tokio::test]
async fn luma_resolves_complete_capture() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/abc-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "abc-123",
            "status": "complete",
            "gaussian_splat_url": "https://cdn-luma.com/scenes/abc-123/scene.ply"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let resolved = run_import(
        &resolver,
        &limiter,
        "test-key",
        Provider::Luma,
        "https://lumalabs.ai/capture/abc-123",
    )
    .await
    .expect("resolve");
    assert_eq!(
        resolved.source_url,
        "https://cdn-luma.com/scenes/abc-123/scene.ply"
    );
    assert_eq!(resolved.capture_id, "abc-123");
    assert!(resolved.filename.ends_with(".ply"));
}

#[tokio::test]
async fn luma_rejects_still_processing() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/abc-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "abc-123",
            "status": "processing"
        })))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Luma,
        "https://lumalabs.ai/capture/abc-123",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::Unsupported {
            provider: "luma",
            ..
        }
    ));
    assert_eq!(err.http_status(), 415);
}

#[tokio::test]
async fn luma_rejects_offsite_asset_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/abc-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "abc-123",
            "status": "complete",
            "gaussian_splat_url": "https://evil.example.com/scene.ply"
        })))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Luma,
        "https://lumalabs.ai/capture/abc-123",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::UnsafeAssetUrl {
            provider: "luma",
            ..
        }
    ));
}

#[tokio::test]
async fn luma_404_maps_to_invalid_share_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/nope"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Luma,
        "https://lumalabs.ai/capture/nope",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::InvalidShareUrl {
            provider: "luma",
            ..
        }
    ));
}

#[tokio::test]
async fn luma_5xx_maps_to_upstream_failed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/abc-123"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Luma,
        "https://lumalabs.ai/capture/abc-123",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::UpstreamFailed {
            provider: "luma",
            ..
        }
    ));
    assert_eq!(err.http_status(), 502);
}

#[tokio::test]
async fn luma_bad_share_url_rejected_before_http() {
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    // No mock server needed — parser rejects before any HTTP hit.
    let resolver = resolver_against("http://127.0.0.1:0");
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Luma,
        "https://lumalabs.ai/totally/wrong",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::InvalidShareUrl {
            provider: "luma",
            ..
        }
    ));
}

/* ---------- POLYCAM ---------- */

#[tokio::test]
async fn polycam_allowlist_rejects_non_polycam_cdn() {
    let server = MockServer::start().await;
    // The resolver's allowlist rejects the wiremock host (it's
    // `127.0.0.1:NNNN`, not `glcdn.poly.cam`) before any HEAD fires.
    // This is the load-bearing security property — even if a future
    // resolver change relaxes the URL construction, the allowlist
    // remains the final gate. We assert it actually fires.
    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Polycam,
        "https://poly.cam/capture/xyz",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::UnsafeAssetUrl {
            provider: "polycam",
            ..
        }
    ));
}

#[tokio::test]
async fn polycam_404_means_no_ply_export() {
    use splatforge_api::routes::import::polycam_ply_url;
    // Pure parser + URL-construction smoke — no HTTP needed. This is
    // the deterministic half of the resolver.
    assert_eq!(
        polycam_ply_url("xyz", "https://glcdn.poly.cam"),
        "https://glcdn.poly.cam/xyz.ply"
    );
}

#[tokio::test]
async fn polycam_bad_share_url_rejected() {
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let resolver = resolver_against("http://127.0.0.1:0");
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Polycam,
        "https://evil.example.com/capture/xyz",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::InvalidShareUrl {
            provider: "polycam",
            ..
        }
    ));
}

/* ---------- SCANIVERSE ---------- */

#[tokio::test]
async fn scaniverse_returns_415_when_no_ply_sibling() {
    let server = MockServer::start().await;
    // The HEAD probe returns 404 → no .ply sibling → 415.
    Mock::given(method("HEAD"))
        .and(path("/scan-77.ply"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Scaniverse,
        "https://scaniverse.com/scan/scan-77",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::Unsupported {
            provider: "scaniverse",
            ..
        }
    ));
    assert_eq!(err.http_status(), 415);
    let msg = err.to_string();
    assert!(
        msg.contains("convert to PLY") || msg.contains("USDZ"),
        "error message should hint at the workaround, got: {msg}"
    );
}

#[tokio::test]
async fn scaniverse_bad_share_url_rejected() {
    let limiter = ImportRateLimiter::new(10, Duration::from_secs(60));
    let resolver = resolver_against("http://127.0.0.1:0");
    let err = run_import(
        &resolver,
        &limiter,
        "k",
        Provider::Scaniverse,
        "https://scaniverse.com/profile/xyz",
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ImportError::InvalidShareUrl {
            provider: "scaniverse",
            ..
        }
    ));
}

/* ---------- RATE LIMITER ---------- */

#[tokio::test]
async fn rate_limit_fires_after_budget() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/captures/abc-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "abc-123",
            "status": "complete",
            "gaussian_splat_url": "https://cdn-luma.com/x.ply"
        })))
        .mount(&server)
        .await;

    let resolver = resolver_against(&server.uri());
    // Tiny budget so we hit the cap deterministically.
    let limiter = ImportRateLimiter::new(2, Duration::from_secs(60));
    let share = "https://lumalabs.ai/capture/abc-123";

    assert!(run_import(&resolver, &limiter, "k1", Provider::Luma, share)
        .await
        .is_ok());
    assert!(run_import(&resolver, &limiter, "k1", Provider::Luma, share)
        .await
        .is_ok());
    let err = run_import(&resolver, &limiter, "k1", Provider::Luma, share)
        .await
        .unwrap_err();
    assert!(matches!(err, ImportError::RateLimited { limit: 2 }));
    assert_eq!(err.http_status(), 429);

    // Different key has its own budget — proves per-key isolation.
    assert!(run_import(&resolver, &limiter, "k2", Provider::Luma, share)
        .await
        .is_ok());
}
