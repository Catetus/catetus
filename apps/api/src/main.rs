#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::manual_pattern_char_comparison,
    clippy::question_mark,
    clippy::too_many_arguments
)]
//! `splatforge-api` — hosted optimize endpoint.
//!
//! Public surface for the design-partner program. Responsibilities:
//!
//! 1. Create optimize jobs and hand the client a server-proxy upload URL.
//! 2. Proxy the client's splat bytes into Vercel Blob over HTTPS.
//! 3. Enqueue the Modal worker with the resulting blob URL + a callback URL.
//! 4. Accept the worker's callback and surface the final download URL.
//!
//! The actual splat work happens in `apps/worker` (Modal Python). This crate
//! stays HTTP-light so we can host it on any standard PaaS without rewriting
//! handlers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, Request, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, trace::TraceLayer};
use tracing::{info, instrument, warn};
use uuid::Uuid;

// All modules live in `lib.rs` so integration tests under `tests/` can
// reach them via the library crate name (`splatforge_api`). The bin
// crate is a thin wrapper that wires the handlers; no per-module
// re-instantiation happens here.
use splatforge_api::audit;
use splatforge_api::billing::{self, BillingClient, KeyCustomerMap};
use splatforge_api::checkout::{
    self, CheckoutConfig, CheckoutError, CreateSessionRequest, PendingClaimTokens, PendingKeyCache,
    RevealRequest, StripeCheckoutClient,
};
use splatforge_api::modal_client::{self, ModalClient};
use splatforge_api::ratelimit::{self, Decision, Limiter, Limits, RouteClass};
use splatforge_api::ratings::{respondent_hash, validate_rating, RATING_RATE_LIMIT_PER_HOUR};
use splatforge_api::routes::import::{
    self as import_route, CaptureResolver, HttpCaptureResolver, ImportError, ImportRateLimiter,
    ImportRequest, ImportResponse, Provider,
};
use splatforge_api::store::{
    self, AuditEvent, DynJobStore, Job, JobStatus, RatingSummaryRow, Tier,
};

/// Top-level app state shared with every handler.
#[derive(Clone)]
pub struct AppState {
    /// Trait-object handle to whichever backend `DATABASE_URL`'s scheme
    /// selected at startup (SQLite for the single-instance Fly deploy,
    /// Postgres for multi-instance promotion — see v2 plan §3b and
    /// `apps/api/STORE-BACKENDS.md`). Every handler talks `JobStoreApi`;
    /// no production code touches the concrete impl.
    pub jobs: DynJobStore,
    /// Modal worker client.
    pub modal: Arc<ModalClient>,
    /// Blob storage adapter (Vercel Blob today; R2/S3 later).
    pub blob: Arc<store::BlobBackend>,
    /// Publicly addressable base URL for this API (no trailing slash).
    /// Used to build the worker's callback URL so it can POST results back.
    pub public_base_url: String,
    /// Accepted bearer tokens. Empty means "auth disabled" (dev mode);
    /// non-empty means every paid route must present one of these.
    pub api_keys: Arc<HashSet<String>>,
    /// Bearer tokens accepted on the paid `/repack` route. Must be a subset
    /// of (or disjoint from) `api_keys` — both are checked, so a paid key
    /// also needs to pass the free-tier gate. Empty disables paid gating.
    pub paid_api_keys: Arc<HashSet<String>>,
    /// Outbound HTTP client used for user-supplied webhook callbacks.
    /// Separate from the Modal/blob clients so a slow subscriber can't
    /// starve those connection pools.
    pub webhook_http: Arc<reqwest::Client>,
    /// Stripe billing client. Always present — falls back to dry-run mode
    /// when STRIPE_SECRET_KEY is unset (local dev). The paid `/repack`
    /// handler and the Modal callback both fire `record_repack_job`.
    pub billing: Arc<BillingClient>,
    /// API-key → Stripe customer id resolver. Populated from
    /// `SPLATFORGE_KEY_CUSTOMERS`. Unknown keys map to `None` and emit
    /// no billing events (paid pipeline still runs — operator decision
    /// whether to refuse those at the gate).
    pub key_customers: Arc<KeyCustomerMap>,
    /// HMAC secret for verifying `/v1/stripe/webhook` signatures.
    /// `None` disables the webhook handler entirely (returns 503).
    pub stripe_webhook_secret: Option<Arc<String>>,
    /// Self-serve Team-tier signup config + Stripe client. The client
    /// is `None` when `STRIPE_SECRET_KEY` is unset (dev / CI); the
    /// `/v1/checkout/create-session` route returns 503 in that case
    /// and the welcome page renders a stub.
    pub checkout_config: Arc<CheckoutConfig>,
    pub checkout_client: Option<Arc<StripeCheckoutClient>>,
    /// In-memory plaintext-key cache populated by the webhook
    /// (`provision_from_session`) and drained exactly once by
    /// `/v1/checkout/reveal`. Plaintexts NEVER hit disk and NEVER hit
    /// the log pipeline.
    pub pending_keys: Arc<PendingKeyCache>,
    /// In-memory map of session id -> claim_token, populated by
    /// `/v1/checkout/create-session` and consumed by the webhook.
    /// See checkout.rs::provision_from_session for the failover
    /// semantics on a process restart between the two halves.
    pub pending_claim_tokens: Arc<PendingClaimTokens>,
    /// HMAC secret for the dedicated `/v1/checkout/webhook` route.
    /// May be the same as `stripe_webhook_secret` (one Stripe webhook
    /// endpoint configured for both event categories) or different
    /// (separate endpoint per event category). Operator decision.
    pub checkout_webhook_secret: Option<Arc<String>>,
    /// Per-API-key token-bucket rate limiter. Wrapped in `Arc` so the
    /// `AppState` clone stays cheap; the limiter itself holds an
    /// internal mutex for the bucket map.
    pub limiter: Arc<Limiter>,
    /// Bearer tokens permitted to read the audit log via
    /// `GET /v1/admin/audit`. Separate from `api_keys` so a leaked
    /// customer key cannot read other customers' activity. Empty
    /// means the admin endpoint is fully disabled (returns 503).
    pub admin_api_keys: Arc<HashSet<String>>,
    /// Pro on-prem license issuer. `None` disables the `/v1/license/*`
    /// endpoints (dev / OSS-only deploys). See `license::LicenseIssuer`.
    pub license_issuer: Option<Arc<splatforge_api::license::LicenseIssuer>>,
    /// Capture-tool import resolver.
    pub capture_resolver: Arc<dyn CaptureResolver>,
    /// In-memory rate limiter for the three `/v1/import/*` endpoints.
    pub import_limiter: Arc<ImportRateLimiter>,
}

impl axum::extract::FromRef<AppState>
    for Option<std::sync::Arc<splatforge_api::license::LicenseIssuer>>
{
    fn from_ref(input: &AppState) -> Self {
        input.license_issuer.clone()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "splatforge_api=info,tower_http=info".into()),
        )
        .init();

    let bind = std::env::var("SPLATFORGE_API_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let modal_url = std::env::var("SPLATFORGE_MODAL_URL").ok();
    let modal_repack_url = std::env::var("SPLATFORGE_MODAL_REPACK_URL").ok();
    let blob_token = std::env::var("BLOB_READ_WRITE_TOKEN").ok();
    // Persisted job state. Default to `./data/jobs.db` so a vanilla `cargo
    // run` works without ceremony; production sets DATABASE_URL to a
    // mounted-volume path.
    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite://data/jobs.db".to_string());
    if let Some(path) = database_url
        .strip_prefix("sqlite://")
        .or_else(|| database_url.strip_prefix("sqlite:"))
    {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating sqlite parent dir {}", parent.display()))?;
            }
        }
    }
    // Default to the droplet's well-known address; override in env for
    // production behind a proper hostname.
    let public_base_url = std::env::var("SPLATFORGE_PUBLIC_BASE_URL")
        .unwrap_or_else(|_| "http://splatforge-api.fly.dev:8080".to_string());
    // Comma-separated list of accepted bearer tokens. Empty = auth disabled
    // (only acceptable in local dev; the deployed binary should always have
    // at least one key set).
    let api_keys: HashSet<String> = parse_keys(std::env::var("SPLATFORGE_API_KEYS").ok());
    let paid_api_keys: HashSet<String> = parse_keys(std::env::var("SPLATFORGE_PAID_API_KEYS").ok());
    let admin_api_keys: HashSet<String> =
        parse_keys(std::env::var("SPLATFORGE_ADMIN_API_KEYS").ok());
    if admin_api_keys.is_empty() {
        warn!(
            "SPLATFORGE_ADMIN_API_KEYS is empty — /v1/admin/audit is disabled (503). \
             Set this to enable the audit-log read endpoint."
        );
    } else {
        info!(
            n_admin_keys = admin_api_keys.len(),
            "admin audit endpoint enabled"
        );
    }
    let rate_limits = Limits::from_env(std::env::var("SPLATFORGE_RATE_LIMITS").ok().as_deref());
    info!(
        create_free = rate_limits.create_free.capacity,
        create_paid = rate_limits.create_paid.capacity,
        upload_free = rate_limits.upload_free.capacity,
        upload_paid = rate_limits.upload_paid.capacity,
        repack = rate_limits.repack.capacity,
        get_job = rate_limits.get_job.capacity,
        batch_paid = rate_limits.batch_paid.capacity,
        "rate limits (per hour, per API key)"
    );
    if api_keys.is_empty() {
        warn!(
            "SPLATFORGE_API_KEYS is empty — running with NO authentication. \
             Set this in production to enable bearer-token gating on /v1/jobs."
        );
    } else {
        info!(n_keys = api_keys.len(), "bearer auth enabled");
    }
    if paid_api_keys.is_empty() {
        warn!(
            "SPLATFORGE_PAID_API_KEYS is empty — /v1/jobs/:id/repack will accept \
             any key that passes the free-tier gate. Set this to restrict the \
             A100 paid path to billing-attached customers."
        );
    } else {
        info!(
            n_paid_keys = paid_api_keys.len(),
            "paid-tier bearer auth enabled"
        );
    }

    // Dedicated client for outbound webhook firing. Short timeout so a
    // misbehaving subscriber doesn't stall the result-callback handler.
    let webhook_http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");

    // `store::connect` is the only place in the binary that knows which
    // backend is in play. SQLite vs Postgres is selected by URL scheme;
    // see `store/mod.rs::connect` for the dispatch rule.
    let jobs: DynJobStore = store::connect(&database_url)
        .await
        .with_context(|| format!("opening jobs db at {database_url}"))?;
    info!(%database_url, "job store ready");

    // Billing setup. Mode is one of "live" / "test" / "dry-run" per
    // BillingClient::from_env. The dry-run fallback keeps `cargo run`
    // working in checkouts without Stripe credentials.
    let (billing, billing_mode) = BillingClient::from_env(jobs.clone());
    info!(mode = billing_mode, "billing client initialized");

    // SPLATFORGE_KEY_CUSTOMERS — `key1:cus_xxx,key2:cus_yyy`. Empty means
    // every paid call falls through to the no-customer code path, which
    // logs but doesn't bill. Useful for closed beta where the operator
    // is invoicing manually.
    let key_customers = KeyCustomerMap::parse(std::env::var("SPLATFORGE_KEY_CUSTOMERS").ok());
    if key_customers.is_empty() {
        warn!(
            "SPLATFORGE_KEY_CUSTOMERS is empty — paid jobs will not be billed. \
             Set this to enable usage-based charges."
        );
    } else {
        info!(n_customers = key_customers.len(), "key→customer map loaded");
    }

    let stripe_webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET")
        .ok()
        .filter(|s| !s.is_empty())
        .map(Arc::new);
    if stripe_webhook_secret.is_none() {
        warn!(
            "STRIPE_WEBHOOK_SECRET is unset — /v1/stripe/webhook will reject all requests \
             with 503. Set this to the `whsec_...` value from `stripe listen` (dev) or \
             your endpoint config (prod)."
        );
    }

    // Self-serve Team-tier signup. Public site URL is used to build
    // the success_url Stripe redirects to after payment; default to
    // splatforge.dev which is where pricing.astro / welcome.astro
    // ship.
    let public_site_url = std::env::var("SPLATFORGE_PUBLIC_SITE_URL")
        .unwrap_or_else(|_| "https://splatforge.dev".to_string());
    let checkout_config = CheckoutConfig::from_env(public_site_url);
    let checkout_client = checkout_config
        .stripe_secret
        .as_deref()
        .filter(|s| !s.starts_with("sk_live_") || checkout_config.live_mode)
        .map(|secret| {
            Arc::new(StripeCheckoutClient::new(
                secret.to_string(),
                checkout_config.stripe_base_url.clone(),
            ))
        });
    if checkout_client.is_none() {
        warn!(
            "checkout: Stripe not configured (or sk_live_ key without STRIPE_LIVE_MODE=true). \
             /v1/checkout/create-session will return 503 and /pricing's Team CTA will fall back \
             to the mailto: enterprise path."
        );
    } else {
        match checkout_config.team_price_id.as_deref() {
            Some(p) => info!(team_price_id = p, live_mode = checkout_config.live_mode, "checkout enabled"),
            None => warn!(
                "STRIPE_TEAM_PRICE_ID is unset — /v1/checkout/create-session will return 503. \
                 Create the $99/seat/mo recurring price in the Stripe dashboard and set this env var."
            ),
        }
    }
    // Reuse the same webhook secret for the checkout endpoint by
    // default; allow a distinct STRIPE_CHECKOUT_WEBHOOK_SECRET when
    // the operator wants a per-event-category endpoint config (which
    // Stripe supports natively).
    let checkout_webhook_secret = std::env::var("STRIPE_CHECKOUT_WEBHOOK_SECRET")
        .ok()
        .filter(|s| !s.is_empty())
        .map(Arc::new)
        .or_else(|| stripe_webhook_secret.clone());

    let license_issuer = splatforge_api::license::LicenseIssuer::from_env()
        .with_context(|| "loading LICENSE_PRIVATE_KEY")?;
    let state = AppState {
        jobs,
        modal: Arc::new(ModalClient::new(modal_url, modal_repack_url)),
        blob: Arc::new(store::BlobBackend::new(blob_token)),
        public_base_url: public_base_url.trim_end_matches('/').to_string(),
        api_keys: Arc::new(api_keys),
        paid_api_keys: Arc::new(paid_api_keys),
        webhook_http: Arc::new(webhook_http),
        billing: Arc::new(billing),
        key_customers: Arc::new(key_customers),
        stripe_webhook_secret,
        checkout_config: Arc::new(checkout_config),
        checkout_client,
        pending_keys: Arc::new(PendingKeyCache::new()),
        pending_claim_tokens: Arc::new(PendingClaimTokens::new()),
        checkout_webhook_secret,
        limiter: Arc::new(Limiter::new(rate_limits)),
        admin_api_keys: Arc::new(admin_api_keys),
        license_issuer: license_issuer.map(Arc::new),
        capture_resolver: Arc::new(HttpCaptureResolver::from_env()),
        import_limiter: Arc::new(ImportRateLimiter::default_production()),
    };

    // Background sweep: drop any pending plaintext / claim_token entry
    // older than PENDING_KEY_TTL (10 min). Without this, a buyer who
    // never lands on /welcome would leave their plaintext sitting in
    // RAM until process restart. The sweep is fire-and-forget; cancel
    // is implicit on shutdown.
    {
        let pk = state.pending_keys.clone();
        let pt = state.pending_claim_tokens.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(60));
            // Skip the first immediate tick — nothing to sweep at boot.
            tick.tick().await;
            loop {
                tick.tick().await;
                pk.sweep_expired().await;
                pt.sweep_expired().await;
            }
        });
    }

    // Router layout:
    //   - `open`  — always public (healthz, worker callback, stripe webhook,
    //               openapi spec, docs UI). Worker callback is protected by
    //               the per-job UUID, the webhook by HMAC, the spec/docs are
    //               static. No rate limiting or audit (operator-targeted).
    //   - `paid`  — gated on the bearer token when SPLATFORGE_API_KEYS is set.
    //               Rate-limited per-key per-class, audited on completion.
    //               Job creation + GET (clients poll their own job state).
    //   - `repack`— additionally requires a paid-tier key.
    //   - `upload`— same auth as `paid` but with a 2 GB body cap.
    //   - `admin` — gated on SPLATFORGE_ADMIN_API_KEYS; serves the audit log.
    let auth_layer = middleware::from_fn_with_state(state.clone(), require_api_key);
    let rate_audit_layer = middleware::from_fn_with_state(state.clone(), rate_limit_and_audit);
    let admin_layer = middleware::from_fn_with_state(state.clone(), require_admin_api_key);

    let open = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/jobs/:id/result", post(job_result))
        // Stripe webhook is "open" because it has its own HMAC-SHA256
        // signature gate (`STRIPE_WEBHOOK_SECRET`), not a bearer token.
        // 1 MB is well over the largest event payload Stripe sends.
        .route("/v1/stripe/webhook", post(stripe_webhook))
        // Self-serve Team-tier signup. All three routes are "open"
        // because they each have their own gate:
        //   - create-session: no auth needed (anyone can pay)
        //   - checkout/webhook: HMAC-SHA256 against the raw body
        //   - reveal: claim_token in the URL query (entropy match)
        .route("/v1/checkout/create-session", post(create_session_route))
        .route("/v1/checkout/webhook", post(checkout_webhook))
        .route("/v1/checkout/reveal", post(reveal_route))
        // fidelity-ml v0.4 — human pairwise rating collection. No bearer
        // token: anyone visiting splatforge.com/rate can submit. The
        // post_rating handler computes a SHA-256(IP || "|" || UA) hash
        // from request headers (never persisted in plaintext) and uses
        // it for a 100-ratings/hour cap so a single browser tab can't
        // flood the corpus.
        .route("/v1/ratings", post(post_rating))
        .route("/v1/ratings/summary", get(ratings_summary))
        // Self-served OpenAPI spec + Swagger UI. Both static, no auth.
        .route("/openapi.yaml", get(openapi_yaml))
        .route("/openapi.json", get(openapi_json_passthrough))
        .route("/docs", get(docs_ui))
        // Pro on-prem license framework. issue is admin-only (bearer
        // check inside the handler); refresh + heartbeat are customer-
        // facing — refresh is gated by the inbound license's signature,
        // heartbeat is best-effort telemetry.
        .route(
            "/v1/license/issue",
            post(splatforge_api::license::issue_route),
        )
        .route(
            "/v1/license/refresh",
            post(splatforge_api::license::refresh_route),
        )
        .route(
            "/v1/license/heartbeat",
            post(splatforge_api::license::heartbeat_route),
        )
        // Per-job pricing preview — pure-compute quote, no auth, no DB.
        // The customer-facing `/pricing` calculator on the marketing
        // site posts here so the displayed cents match the meter-emitted
        // cents to the cent.
        .route(
            "/v1/pricing/preview",
            post(splatforge_api::routes::pricing::preview),
        )
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024));

    // Self-serve SDK licensing surface. State is the operator-gated
    // signing material from env — entirely disjoint from the main
    // AppState so a future rev that moves SDK licensing into its own
    // service can lift this Router out cleanly.
    let sdk_state = splatforge_api::routes::sdk_license::SdkLicenseState::from_env();
    let sdk = Router::new()
        .route(
            "/v1/sdk/license",
            post(splatforge_api::routes::sdk_license::issue_license),
        )
        .route(
            "/v1/sdk/license/verify",
            post(splatforge_api::routes::sdk_license::verify_license_route),
        )
        .route(
            "/v1/sdk/beacon",
            post(splatforge_api::routes::sdk_license::beacon),
        )
        .route(
            "/v1/sdk/pricing/preview",
            post(splatforge_api::routes::sdk_license::sdk_preview),
        )
        .layer(RequestBodyLimitLayer::new(64 * 1024))
        .with_state(sdk_state);

    let paid_layer = middleware::from_fn_with_state(state.clone(), require_paid_api_key);

    let paid = Router::new()
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/batch", post(create_jobs_batch))
        .route("/v1/jobs/:id", get(get_job))
        // Capture-tool imports. Same bearer auth as `/v1/jobs` (so the
        // import surface is paid-customer-only); rate-limited at
        // 10/min/key inside the handler via `state.import_limiter`. The
        // body cap is tiny — the request only carries a share URL.
        .route("/v1/import/luma", post(import_luma))
        .route("/v1/import/polycam", post(import_polycam))
        .route("/v1/import/scaniverse", post(import_scaniverse))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024))
        // Order matters: layers apply outermost-first. Auth must run
        // before the rate-limit layer so the limiter has a verified key
        // to bucket by.
        .layer(rate_audit_layer.clone())
        .layer(auth_layer.clone());

    let repack = Router::new()
        .route("/v1/jobs/:id/repack", post(repack_job))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024))
        .layer(rate_audit_layer.clone())
        // Free gate runs first (rejects unauthenticated requests early),
        // paid gate runs second so it only sees pre-authenticated calls.
        .layer(paid_layer)
        .layer(auth_layer.clone());

    let upload = Router::new()
        .route("/v1/jobs/:id/upload", post(upload_job))
        // 2 GB — covers bicycle (855 MB), bonsai (274 MB), and the Sweet
        // Corals reef tiles (700-950 MB each) which were all over the prior
        // 250 MB cap. The body is streamed through to Vercel Blob; we never
        // buffer it in memory. Users with cloud-hosted data should still
        // prefer the source_url form which skips the proxy entirely.
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024 * 1024))
        .layer(rate_audit_layer)
        .layer(auth_layer);

    let admin = Router::new()
        .route("/v1/admin/audit", get(admin_audit))
        .layer(admin_layer);

    let app = Router::new()
        .merge(open)
        .merge(paid)
        .merge(repack)
        .merge(upload)
        .merge(admin)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
        // SDK router merges last because it carries its own state
        // (signing secret + operator key) and must not be re-stated by
        // `.with_state(state)` above.
        .merge(sdk);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    info!(%bind, "splatforge-api listening");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Parse a comma-separated env var into a deduped, trimmed token set.
fn parse_keys(raw: Option<String>) -> HashSet<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Paid-tier gate. Free-tier `require_api_key` runs first; this layer
/// additionally requires the presented key to be in `paid_api_keys`. When
/// `paid_api_keys` is empty the gate is a no-op (every authenticated user
/// can hit /repack — useful in dev, never in prod).
async fn require_paid_api_key(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    if state.paid_api_keys.is_empty() {
        return Ok(next.run(req).await);
    }
    let auth = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let presented = auth.strip_prefix("Bearer ").unwrap_or_default().trim();
    if !state.paid_api_keys.contains(presented) {
        return Err(ApiError::Unauthorized(
            "this key is not enabled for paid-tier endpoints".to_string(),
        ));
    }
    Ok(next.run(req).await)
}

/// The bearer key the request was authenticated with, if any. Stamped into
/// the request extensions by `require_api_key` so downstream handlers
/// (notably `create_job` → `build_job`) can look up the customer mapping
/// without re-parsing the header. Wrapped in a newtype so it doesn't
/// collide with any other `String` extension.
#[derive(Clone)]
struct AuthenticatedKey(String);

/// Bearer-token middleware. When `state.api_keys` is non-empty, every
/// request to a route under this layer must present
/// `Authorization: Bearer <key>` matching one of the configured keys.
/// Returns 401 with the canonical error envelope otherwise.
///
/// On success, stamps the verified key into the request extensions as
/// `AuthenticatedKey` so handlers can map it to a Stripe customer.
async fn require_api_key(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    if state.api_keys.is_empty() {
        // Auth disabled — dev mode. Logged once at startup; don't log per
        // request to avoid spam. No AuthenticatedKey is stamped.
        return Ok(next.run(req).await);
    }
    // Extract the bearer token into an owned String so we can drop the
    // immutable header borrow before taking the mutable extensions borrow.
    let presented: String = {
        let auth = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        auth.strip_prefix("Bearer ")
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    if presented.is_empty() {
        return Err(ApiError::Unauthorized(
            "missing Authorization: Bearer <key>".to_string(),
        ));
    }
    if !state.api_keys.contains(&presented) {
        return Err(ApiError::Unauthorized("invalid API key".to_string()));
    }
    req.extensions_mut().insert(AuthenticatedKey(presented));
    Ok(next.run(req).await)
}

/// Admin-tier auth for `/v1/admin/*`. Independent set of bearer tokens
/// from `SPLATFORGE_API_KEYS` — leaking a customer key must not give
/// access to the audit log. An empty admin-key set disables the
/// endpoint entirely (returns 503), which is the safe default for
/// fresh deployments that haven't provisioned an operator key yet.
async fn require_admin_api_key(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    if state.admin_api_keys.is_empty() {
        return Err(ApiError::Unauthorized(
            "admin endpoint disabled (SPLATFORGE_ADMIN_API_KEYS unset)".to_string(),
        ));
    }
    let auth = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let presented = auth.strip_prefix("Bearer ").unwrap_or_default().trim();
    if presented.is_empty() {
        return Err(ApiError::Unauthorized(
            "missing Authorization: Bearer <admin-key>".to_string(),
        ));
    }
    if !state.admin_api_keys.contains(presented) {
        return Err(ApiError::Unauthorized("invalid admin API key".to_string()));
    }
    Ok(next.run(req).await)
}

/// Pick the rate-limit class for an inbound request based on
/// (method, path). Returns `None` for routes that aren't rate-limited
/// — the middleware falls through to `next` in that case. Path
/// matching is structural so `/v1/jobs/<uuid>/upload` correctly maps
/// to `RouteClass::Upload`.
fn classify_for_ratelimit(method: &axum::http::Method, path: &str) -> Option<RouteClass> {
    let method_str = method.as_str();
    match method_str {
        "POST" => {
            if path == "/v1/jobs" {
                return Some(RouteClass::CreateJob);
            }
            if path == "/v1/jobs/batch" {
                return Some(RouteClass::CreateBatch);
            }
            let rest = path.strip_prefix("/v1/jobs/")?;
            let (_id, tail) = rest.split_once('/')?;
            match tail {
                "upload" => Some(RouteClass::Upload),
                "repack" => Some(RouteClass::Repack),
                _ => None,
            }
        }
        "GET" => {
            let rest = path.strip_prefix("/v1/jobs/")?;
            // /v1/jobs/<id> with no trailing segment.
            if !rest.is_empty() && !rest.contains('/') {
                return Some(RouteClass::GetJob);
            }
            None
        }
        _ => None,
    }
}

/// Combined rate-limit + audit middleware. Two responsibilities folded
/// into one pass because they both want the same data (key prefix,
/// route template, request status, duration) and pulling the request
/// apart twice would double the path-matching cost on the hot path.
///
/// Order of operations:
///   1. Read the authenticated key from the request extensions.
///      Unauthenticated requests bypass the limiter (they only reach
///      open routes; this layer never wraps `/healthz`).
///   2. Classify the (method, path) into a `RouteClass`. Unknown
///      routes pass straight through to `next` without limiting or
///      auditing.
///   3. Decide tier from the paid-key set. Limiter caps differ.
///   4. `Limiter::take` — returns Allow (with remaining) or Deny
///      (with retry_after). Deny → 429 carrying both headers, and
///      we still write the audit row (operator wants to see
///      throttled traffic).
///   5. Run the inner handler; capture status + duration.
///   6. Best-effort audit insert. DB failures here are logged but
///      MUST NOT propagate to the response.
async fn rate_limit_and_audit(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let class = classify_for_ratelimit(&method, &path);

    // Read the authenticated key out of extensions. If auth is disabled
    // (dev mode), there's no key — we use a sentinel "anon" bucket so
    // a developer poking the API locally still sees the limiter behave
    // sensibly. In prod this branch never runs because the auth layer
    // is mandatory.
    let key = req
        .extensions()
        .get::<AuthenticatedKey>()
        .map(|k| k.0.clone())
        .unwrap_or_else(|| "anon".to_string());

    let tier = if state.paid_api_keys.contains(&key) {
        ratelimit::Tier::Paid
    } else {
        ratelimit::Tier::Free
    };
    let key_prefix = ratelimit::key_prefix(&key);

    // Free-tier callers can't use the batch endpoint at all — return a
    // structured 403 instead of pretending they have a 0-cap bucket.
    if matches!(class, Some(RouteClass::CreateBatch)) && tier == ratelimit::Tier::Free {
        let started = Instant::now();
        let err_msg = "batch endpoint requires a paid-tier API key";
        // Audit the rejection so the operator can see "free tier trying
        // to batch" patterns and reach out about upgrades.
        let route_tmpl = audit::route_template(method.as_str(), &path).unwrap_or("/v1/jobs/batch");
        let elapsed = started.elapsed().as_millis() as u64;
        audit::record(
            &state.jobs,
            &key_prefix,
            route_tmpl,
            method.as_str(),
            403,
            0,
            elapsed,
            Some("free-tier-batch-forbidden"),
        )
        .await;
        return Err(ApiError::Forbidden(err_msg.to_string()));
    }

    // Rate-limit decision.
    let decision = if let Some(class) = class {
        Some(state.limiter.take(&key, class, tier))
    } else {
        None
    };

    let started = Instant::now();
    let (response, body_size, status_code, error_note) = match decision {
        Some(Decision::Deny {
            retry_after_s,
            remaining,
        }) => {
            // Build a structured 429 with operator-canonical headers.
            let body = serde_json::json!({
                "error": "rate limit exceeded",
                "retry_after_s": retry_after_s,
            });
            let body_bytes = serde_json::to_vec(&body).expect("json");
            let body_size = body_bytes.len() as u64;
            let mut response = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
            response.headers_mut().insert(
                "Retry-After",
                HeaderValue::from_str(&retry_after_s.to_string())
                    .unwrap_or(HeaderValue::from_static("1")),
            );
            response.headers_mut().insert(
                "X-RateLimit-Remaining",
                HeaderValue::from_str(&remaining.to_string())
                    .unwrap_or(HeaderValue::from_static("0")),
            );
            (
                response,
                body_size,
                429u16,
                Some("rate-limited".to_string()),
            )
        }
        _ => {
            // Allowed (or unclassified route): proceed and capture
            // status + (approximate) body size from the inner response.
            let remaining_hdr = match decision {
                Some(Decision::Allow { remaining }) => Some(remaining),
                _ => None,
            };
            let mut response = next.run(req).await;
            let status = response.status().as_u16();
            // Body size is opportunistic — axum's body is a Stream; we
            // don't drain it here (that would buffer the entire upload
            // response in RAM). The audit row's body_size is therefore
            // 0 for non-429 paths today. Documented in the audit
            // schema comment as a known limitation.
            let body_size = 0u64;
            if let Some(r) = remaining_hdr {
                let _ = response.headers_mut().insert(
                    "X-RateLimit-Remaining",
                    HeaderValue::from_str(&r.to_string()).unwrap_or(HeaderValue::from_static("0")),
                );
            }
            let err_note = if status >= 400 {
                Some(format!("status-{status}"))
            } else {
                None
            };
            (response, body_size, status, err_note)
        }
    };

    // Audit on completion. The spec calls for mutating-only routes;
    // GET /v1/jobs/:id passes through here too (it has its own bucket)
    // but `audit::is_audited` returns false for it, so we skip.
    if audit::is_audited(method.as_str(), &path) {
        let route_tmpl = audit::route_template(method.as_str(), &path).unwrap_or("/v1/unknown");
        let elapsed = started.elapsed().as_millis() as u64;
        audit::record(
            &state.jobs,
            &key_prefix,
            route_tmpl,
            method.as_str(),
            status_code,
            body_size,
            elapsed,
            error_note.as_deref(),
        )
        .await;
    }

    Ok(response)
}

/// Resolve the Stripe customer id for the request's authenticated key.
/// Returns `None` when:
///   * auth is disabled (no AuthenticatedKey extension)
///   * the key isn't in SPLATFORGE_KEY_CUSTOMERS
/// `None` means "do not bill" — the paid pipeline still runs, but no
/// meter event is emitted. This is intentional for the closed-beta
/// stage where the operator may be invoicing manually.
fn resolve_customer(state: &AppState, extensions: &axum::http::Extensions) -> Option<String> {
    let key = extensions.get::<AuthenticatedKey>()?;
    state.key_customers.lookup(&key.0).map(str::to_owned)
}

#[instrument(skip(_state))]
async fn healthz(State(_state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "service": "splatforge-api",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Payload for `POST /v1/jobs`. Two mutually exclusive modes:
///
/// 1. **Proxy upload** (designer-friendly): caller sets `filename` +
///    `size_bytes` and gets back an `upload_url` they PUT the bytes to.
///    Cap is 2 GB to cover the largest scenes in SplatBench v0 (bicycle 855
///    MB) and Sweet Corals tiles (700-950 MB each).
///
/// 2. **Source URL** (enterprise-friendly): caller sets `source_url` to a
///    publicly-fetchable HTTPS URL (HuggingFace, S3, GCS, R2, Cloudflare
///    R2, etc.). The worker fetches the bytes directly server-side, so the
///    client never uploads anything. Skips the `AwaitingUpload` /
///    `Uploading` states and the job lands in `Queued` immediately.
///
/// Both modes accept `webhook_url`, an HTTPS endpoint we POST the final
/// Job JSON to when the job hits a terminal state (Done / Error).
#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    /// One of `lossless-repack` / `web-mobile` / `size-min` / etc.
    pub preset: String,
    /// Proxy-upload mode: suggested filename (used to derive the blob key).
    #[serde(default)]
    pub filename: Option<String>,
    /// Proxy-upload mode: size in bytes (used for early size-cap rejection
    /// before the bytes start streaming).
    #[serde(default)]
    pub size_bytes: Option<u64>,
    /// URL-mode: HTTPS URL the worker fetches the input from directly.
    /// Mutually exclusive with `filename` / `size_bytes`.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Optional caller-supplied label for the job (e.g. `acme-q3-walkthrough`).
    #[serde(default)]
    pub label: Option<String>,
    /// HTTPS endpoint to POST the Job JSON to on terminal state.
    #[serde(default)]
    pub webhook_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateJobResponse {
    pub id: Uuid,
    pub status: JobStatus,
    /// Where to PUT the bytes when in proxy-upload mode. Absent in URL mode
    /// because the worker already has everything it needs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_url: Option<String>,
    /// Always `PUT` in proxy-upload mode; absent in URL mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_method: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Payload for `POST /v1/jobs/batch`. Each entry is a regular
/// `CreateJobRequest`; the response is a list of `CreateJobResponse` plus
/// a shared `batch_id` that's stamped on every Job in the batch.
#[derive(Debug, Deserialize)]
pub struct BatchCreateRequest {
    /// Max 100 jobs per batch — covers the largest tiled scene we know of
    /// (Sweet Corals has 40 tiles) with plenty of headroom.
    pub jobs: Vec<CreateJobRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchCreateResponse {
    pub batch_id: Uuid,
    pub jobs: Vec<CreateJobResponse>,
}

/// Maximum input size accepted by the optimizer. Bicycle (3.6M splats) is
/// ~860 MB raw PLY; Sweet Corals tiles top out near 950 MB. 3 GB gives
/// headroom for future larger captures without inviting 50 GB uploads
/// that would blow Modal's budget.
const MAX_INPUT_BYTES: u64 = 3 * 1024 * 1024 * 1024;

#[instrument(skip(state, extensions, req), fields(preset = %req.preset))]
async fn create_job(
    State(state): State<AppState>,
    extensions: axum::http::Extensions,
    Json(req): Json<CreateJobRequest>,
) -> Result<Json<CreateJobResponse>, ApiError> {
    let customer_id = resolve_customer(&state, &extensions);
    let job = build_job(&state, req, None, customer_id).await?;
    let response = job_creation_response(&job, &state)?;
    state.jobs.insert(&job).await?;
    // URL-mode jobs are immediately enqueueable — kick the worker now so
    // the caller doesn't have to do a follow-up call. Proxy-upload mode
    // jobs wait on `/upload` to flip them to Queued.
    if job.source_url.is_some() {
        enqueue_url_job(&state, &job).await?;
    }
    Ok(Json(response))
}

/// `POST /v1/jobs/batch` — create N jobs atomically. All-or-nothing
/// validation: if any single entry is malformed the whole batch is
/// rejected with a 400 and no jobs are inserted. On success every job
/// in the batch carries the same `batch_id` for downstream correlation
/// (e.g. a 40-tile dataset).
#[instrument(skip(state, extensions, req), fields(n_jobs = req.jobs.len()))]
async fn create_jobs_batch(
    State(state): State<AppState>,
    extensions: axum::http::Extensions,
    Json(req): Json<BatchCreateRequest>,
) -> Result<Json<BatchCreateResponse>, ApiError> {
    let customer_id = resolve_customer(&state, &extensions);
    if req.jobs.is_empty() {
        return Err(ApiError::BadRequest(
            "batch must contain at least one job".to_string(),
        ));
    }
    const MAX_BATCH: usize = 100;
    if req.jobs.len() > MAX_BATCH {
        return Err(ApiError::BadRequest(format!(
            "batch contains {} jobs; cap is {MAX_BATCH}",
            req.jobs.len()
        )));
    }

    let batch_id = Uuid::new_v4();
    // Validate-then-build every job before any state mutation, so a bad
    // entry doesn't leave half a batch persisted.
    let mut built: Vec<Job> = Vec::with_capacity(req.jobs.len());
    for entry in req.jobs {
        built.push(build_job(&state, entry, Some(batch_id), customer_id.clone()).await?);
    }
    let responses: Vec<CreateJobResponse> = built
        .iter()
        .map(|j| job_creation_response(j, &state))
        .collect::<Result<_, _>>()?;

    // Persist + enqueue URL-mode jobs. Proxy-upload jobs wait on /upload.
    for job in &built {
        state.jobs.insert(job).await?;
    }
    for job in &built {
        if job.source_url.is_some() {
            if let Err(e) = enqueue_url_job(&state, job).await {
                warn!(job_id = %job.id, error = %e, "batch member enqueue failed");
                // Mark the individual job as Error rather than failing the
                // whole batch — the caller can re-issue just the broken
                // ones via the per-id endpoint.
                let mut bad = job.clone();
                bad.status = JobStatus::Error;
                bad.error = Some(format!("enqueue failed: {e}"));
                let _ = state.jobs.update(&bad).await;
            }
        }
    }

    Ok(Json(BatchCreateResponse {
        batch_id,
        jobs: responses,
    }))
}

/// Build a `Job` from a `CreateJobRequest`, dispatching on which input
/// mode the caller chose. Validates the input shape but does not mutate
/// `state.jobs`, so the caller can decide when to persist (relevant for
/// the batch endpoint which validates-then-commits).
async fn build_job(
    state: &AppState,
    req: CreateJobRequest,
    batch_id: Option<Uuid>,
    customer_id: Option<String>,
) -> Result<Job, ApiError> {
    // Enforce input-mode XOR up front. The schema lets the caller send
    // both upload + URL fields by accident; reject explicitly so we
    // never silently prefer one over the other.
    let has_upload = req.filename.is_some() || req.size_bytes.is_some();
    let has_url = req.source_url.is_some();
    if has_upload && has_url {
        return Err(ApiError::BadRequest(
            "request must specify exactly one of (filename + size_bytes) or source_url".to_string(),
        ));
    }
    if !has_upload && !has_url {
        return Err(ApiError::BadRequest(
            "request must specify either (filename + size_bytes) for proxy upload, \
             or source_url for direct fetch"
                .to_string(),
        ));
    }

    if let Some(url) = req.webhook_url.as_deref() {
        validate_webhook_url(url)?;
    }

    let id = Uuid::new_v4();
    let created_at = Utc::now();

    if has_url {
        let url = req.source_url.unwrap();
        validate_source_url(&url)?;
        // Derive a safe filename from the URL path. Strip the query string
        // first (presigned URLs like R2/S3 carry kilobytes of signature
        // params), then take the last path segment. The result is fed to
        // sanitize_filename and capped at 200 bytes so we stay well under
        // ext4's 255-byte filename limit on the worker side.
        let path = url.split('?').next().unwrap_or(&url);
        let last_seg = path
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or("scene.bin");
        let mut filename = sanitize_filename(last_seg);
        if filename.is_empty() {
            filename = "scene.bin".to_string();
        }
        if filename.len() > 200 {
            // Preserve the trailing extension when truncating.
            let ext = std::path::Path::new(&filename)
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| format!(".{e}"))
                .unwrap_or_default();
            let mut keep = 200usize.saturating_sub(ext.len());
            // Step back to the nearest char boundary so we don't slice mid-codepoint.
            while keep > 0 && !filename.is_char_boundary(keep) {
                keep -= 1;
            }
            filename = format!("{}{ext}", &filename[..keep]);
        }
        let blob_key = format!("jobs/{id}/{filename}");
        Ok(Job {
            id,
            preset: req.preset,
            filename,
            size_bytes: 0, // unknown until worker fetches
            label: req.label,
            status: JobStatus::Queued,
            blob_key,
            blob_url: Some(url.clone()),
            source_url: Some(url),
            upload_size_bytes: None,
            output_url: None,
            preview_url: None,
            phase: None,
            percent: None,
            webhook_url: req.webhook_url,
            batch_id,
            tier: Tier::Free,
            customer_id,
            created_at,
            error: None,
        })
    } else {
        let filename = req.filename.unwrap();
        let size_bytes = req.size_bytes.unwrap();
        if size_bytes == 0 || size_bytes > MAX_INPUT_BYTES {
            return Err(ApiError::BadRequest(format!(
                "size_bytes must be in (0, {MAX_INPUT_BYTES}); got {size_bytes}",
            )));
        }
        let blob_key = format!("jobs/{id}/{}", sanitize_filename(&filename));
        // Presign the upload URL here so the caller can immediately PUT bytes.
        // The blob backend may return a server-proxy URL if it can't issue a
        // direct presign — both forms route through the same upload handler.
        let _ = state
            .blob
            .presign_upload(&blob_key, std::time::Duration::from_secs(900))
            .await?;
        Ok(Job {
            id,
            preset: req.preset,
            filename,
            size_bytes,
            label: req.label,
            status: JobStatus::AwaitingUpload,
            blob_key,
            blob_url: None,
            source_url: None,
            upload_size_bytes: None,
            output_url: None,
            preview_url: None,
            phase: None,
            percent: None,
            webhook_url: req.webhook_url,
            batch_id,
            tier: Tier::Free,
            customer_id,
            created_at,
            error: None,
        })
    }
}

fn job_creation_response(job: &Job, _state: &AppState) -> Result<CreateJobResponse, ApiError> {
    let (upload_url, upload_method) = match job.status {
        JobStatus::AwaitingUpload => (
            Some(format!("blob://stub/{}", job.blob_key)),
            Some("PUT".to_string()),
        ),
        _ => (None, None),
    };
    Ok(CreateJobResponse {
        id: job.id,
        status: job.status,
        upload_url,
        upload_method,
        created_at: job.created_at,
    })
}

/// Hand a URL-mode job off to the Modal worker. Idempotent — safe to call
/// multiple times; the worker's job_id is the dedupe key on its side.
async fn enqueue_url_job(state: &AppState, job: &Job) -> Result<(), ApiError> {
    let Some(url) = job.source_url.as_deref() else {
        return Ok(());
    };
    let callback_url = format!("{}/v1/jobs/{}/result", state.public_base_url, job.id);
    state.modal.enqueue(job, url, &callback_url).await?;
    Ok(())
}

/// Allowlist + safety check for user-supplied source URLs. Rejects:
///   - non-HTTPS schemes (HTTP is plaintext; file:// and others are obvious SSRF)
///   - hosts that resolve to private / link-local / loopback IP literals
///     (basic SSRF guard — doesn't catch DNS rebinding, but blocks the
///     trivial `http://169.254.169.254/` / `http://127.0.0.1/` cases)
fn validate_source_url(url: &str) -> Result<(), ApiError> {
    if !url.starts_with("https://") {
        return Err(ApiError::BadRequest(
            "source_url must be an HTTPS URL".to_string(),
        ));
    }
    // Reject the obvious private-IP-literal shapes. We don't do DNS lookups
    // here (network call inside a sync validator), so DNS-rebind attacks are
    // out of scope for now — the worker side has its own size + content-type
    // sanity check.
    let after_scheme = &url["https://".len()..];
    let host = after_scheme
        .split(['/', ':', '?', '#'])
        .next()
        .unwrap_or("");
    if host.is_empty() {
        return Err(ApiError::BadRequest("source_url missing host".to_string()));
    }
    let host_lower = host.to_ascii_lowercase();
    const FORBIDDEN_HOST_PREFIXES: &[&str] = &[
        "localhost",
        "127.",
        "10.",
        "192.168.",
        "169.254.",
        "0.",
        "[::1]",
        "[fc",
        "[fd",
    ];
    if FORBIDDEN_HOST_PREFIXES
        .iter()
        .any(|p| host_lower.starts_with(p))
    {
        return Err(ApiError::BadRequest(format!(
            "source_url host {host} is in a private / loopback range"
        )));
    }
    // 172.16.0.0/12: 172.16. through 172.31.
    if let Some(rest) = host_lower.strip_prefix("172.") {
        if let Some(second_octet) = rest.split('.').next() {
            if let Ok(n) = second_octet.parse::<u8>() {
                if (16..=31).contains(&n) {
                    return Err(ApiError::BadRequest(format!(
                        "source_url host {host} is in private range 172.16.0.0/12"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Webhook URLs only need the HTTPS check — the worker doesn't fetch
/// from them so SSRF isn't the threat model. Sending to a private IP is
/// still pointless (we couldn't reach it from production), so we apply
/// the same allowlist.
fn validate_webhook_url(url: &str) -> Result<(), ApiError> {
    validate_source_url(url).map_err(|e| match e {
        ApiError::BadRequest(msg) => ApiError::BadRequest(msg.replace("source_url", "webhook_url")),
        other => other,
    })
}

/// Fire-and-forget POST to the user's configured webhook with the latest
/// Job JSON. Logs but never errors — webhook delivery is best-effort.
/// Caller should already have persisted the job before invoking this.
async fn fire_webhook(state: &AppState, job: &Job) {
    let Some(url) = job.webhook_url.as_deref() else {
        return;
    };
    let payload = match serde_json::to_value(job) {
        Ok(v) => v,
        Err(e) => {
            warn!(job_id = %job.id, error = %e, "webhook payload serialization failed");
            return;
        }
    };
    let resp = state.webhook_http.post(url).json(&payload).send().await;
    match resp {
        Ok(r) if r.status().is_success() => {
            info!(job_id = %job.id, status = %r.status(), "webhook delivered");
        }
        Ok(r) => {
            warn!(
                job_id = %job.id,
                status = %r.status(),
                "webhook subscriber returned non-2xx"
            );
        }
        Err(e) => {
            warn!(job_id = %job.id, error = %e, "webhook transport failed");
        }
    }
}

#[instrument(skip(state))]
async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Job>, ApiError> {
    state
        .jobs
        .get(&id)
        .await?
        .map(Json)
        .ok_or(ApiError::NotFound)
}

/// `POST /v1/jobs/:id/upload`
///
/// Streams the request body through to Vercel Blob, updates the job with
/// the canonical public URL, and enqueues the Modal worker with a
/// callback URL so the worker can POST the result when it's done.
#[instrument(skip(state, headers, body), fields(job_id = %id))]
async fn upload_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Job>, ApiError> {
    let Some(mut job) = state.jobs.get(&id).await? else {
        return Err(ApiError::NotFound);
    };
    if !matches!(job.status, JobStatus::AwaitingUpload) {
        return Err(ApiError::BadRequest(format!(
            "job {id} is {:?}; cannot re-upload",
            job.status
        )));
    }

    // Flip to Uploading before we start streaming so the client polling
    // `/v1/jobs/:id` sees the transition.
    job.status = JobStatus::Uploading;
    state.jobs.update(&job).await?;

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();

    // Stream the axum body directly into reqwest so we never buffer the
    // whole splat in memory. axum's body is a Stream<Item = Result<Bytes, _>>;
    // reqwest::Body has `wrap_stream` for exactly this case.
    let stream = body.into_data_stream().map_err(std::io::Error::other);
    let reqwest_body = reqwest::Body::wrap_stream(stream);

    let blob_url = match state
        .blob
        .put_bytes(&job.blob_key, reqwest_body, &content_type)
        .await
    {
        Ok(url) => url,
        Err(e) => {
            warn!(error = %e, "blob upload failed");
            job.status = JobStatus::Error;
            job.error = Some(format!("blob upload failed: {e}"));
            let _ = state.jobs.update(&job).await;
            return Err(ApiError::Storage(e));
        }
    };

    job.blob_url = Some(blob_url.clone());
    // We don't have a clean way to recover the streamed byte count without
    // a counting middleware; fall back to the client-supplied size_bytes
    // (already validated to be in range) so the field is at least populated.
    job.upload_size_bytes = Some(job.size_bytes);
    job.status = JobStatus::Queued;
    state.jobs.update(&job).await?;

    let callback_url = format!("{}/v1/jobs/{}/result", state.public_base_url, id);
    match state.modal.enqueue(&job, &blob_url, &callback_url).await {
        Ok(ack) => {
            if let Some(msg) = ack.error.as_deref() {
                warn!(error = msg, "modal enqueue warning");
            }
        }
        Err(e) => {
            warn!(error = %e, "modal enqueue failed");
            job.status = JobStatus::Error;
            job.error = Some(format!("modal enqueue failed: {e}"));
            let _ = state.jobs.update(&job).await;
            return Err(ApiError::Modal(e));
        }
    }

    Ok(Json(job))
}

/// Payload for `POST /v1/jobs/:id/repack`. Dispatches the (already
/// uploaded) splat into the differentiable-repack A100 worker. Iteration
/// count and target byte budget are the two knobs that meaningfully change
/// cost; everything else lives on the Modal side.
#[derive(Debug, Deserialize)]
pub struct RepackRequest {
    /// Hard ceiling on the repacked output size. The worker stops compressing
    /// when it hits this, even if quality could still be traded down. The
    /// bonsai reference (143 MB at 50% of 287 MB baseline → +6.4 dB) lives
    /// at `target_bytes ≈ size_bytes / 2`.
    pub target_bytes: u64,
    /// Adam iterations. Bonsai converges in 1000 (~18s on A100); raising
    /// this past 2000 hits diminishing returns and inflates cost.
    #[serde(default = "default_iterations")]
    pub iterations: u32,
}

fn default_iterations() -> u32 {
    1000
}

/// `POST /v1/jobs/:id/repack`
///
/// Paid-tier endpoint. The job must already be in `Done` state (i.e. it has
/// been through the free pipeline at least once) so we have a known-good
/// baseline render to optimize against. The worker fetches the original
/// input via `source_url` or `blob_url`, runs gsplat-on-A100 with the
/// supplied params, and POSTs the result back through the same callback
/// shape as the free pipeline. The job is re-marked `Running` while the
/// repack runs and lands back at `Done` with a new `output_url`.
#[instrument(skip(state, extensions, body), fields(job_id = %id, target = body.target_bytes))]
async fn repack_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    extensions: axum::http::Extensions,
    Json(body): Json<RepackRequest>,
) -> Result<Json<Job>, ApiError> {
    let Some(mut job) = state.jobs.get(&id).await? else {
        return Err(ApiError::NotFound);
    };
    // Stamp the customer onto the job at repack time. The original
    // (free) job may have been created before the operator mapped the
    // key, or with a different key entirely. The repack call is the
    // billing event, so the repack key wins.
    let repack_customer = resolve_customer(&state, &extensions);
    if repack_customer.is_some() {
        job.customer_id = repack_customer.clone();
    }
    if body.target_bytes == 0 || body.target_bytes > MAX_INPUT_BYTES {
        return Err(ApiError::BadRequest(format!(
            "target_bytes must be in (0, {MAX_INPUT_BYTES}); got {}",
            body.target_bytes
        )));
    }
    if body.iterations == 0 || body.iterations > 5000 {
        return Err(ApiError::BadRequest(format!(
            "iterations must be in (0, 5000]; got {}",
            body.iterations
        )));
    }
    // Refuse to repack jobs that never produced a baseline. Repack quality is
    // measured against the previous render; without a `Done` ancestor we
    // can't validate the result, and the worker would just be running the
    // free pipeline twice.
    if !matches!(job.status, JobStatus::Done) {
        return Err(ApiError::BadRequest(format!(
            "job {id} status is {:?}; repack requires the job to be Done first",
            job.status
        )));
    }
    let input_url = job
        .source_url
        .clone()
        .or_else(|| job.blob_url.clone())
        .ok_or_else(|| {
            ApiError::BadRequest("job has no source_url or blob_url to repack".to_string())
        })?;

    job.tier = Tier::Paid;
    job.status = JobStatus::Running;
    job.phase = Some("repack-enqueue".to_string());
    job.percent = Some(0.0);
    job.error = None;
    state.jobs.update(&job).await?;

    let callback_url = format!("{}/v1/jobs/{}/result", state.public_base_url, id);
    let params = serde_json::json!({
        "target_bytes": body.target_bytes,
        "iterations": body.iterations,
    });
    if let Err(e) = state
        .modal
        .enqueue_repack(&job, &input_url, &callback_url, params)
        .await
    {
        warn!(error = %e, "repack enqueue failed");
        job.status = JobStatus::Error;
        job.error = Some(format!("repack enqueue failed: {e}"));
        let _ = state.jobs.update(&job).await;
        return Err(ApiError::Modal(e));
    }
    // Bill the per-run flat fee on successful dispatch. Seconds are
    // billed when the worker callback lands with elapsed time. The
    // ledger UNIQUE(job_id, sku) constraint makes both paths idempotent
    // — a duplicate dispatch (e.g. user double-clicks the button) only
    // produces one charge.
    if let Err(e) = state
        .billing
        .record_repack_job(
            job.id,
            job.customer_id.as_deref(),
            job.size_bytes,
            body.iterations,
            None,
        )
        .await
    {
        // Billing failure is logged but does not roll back the run. The
        // alternative — refusing to start the job because Stripe is
        // down — punishes the customer for our outage. The ledger row
        // is already claimed, so a backfill script can re-emit later.
        warn!(error = %e, job_id = %job.id, "billing record_repack_job failed; continuing");
    }
    Ok(Json(job))
}

/// `POST /v1/jobs/:id/result`
///
/// Worker callback. Payload:
/// ```json
/// { "status": "done" | "error", "output_url": "https://...",
///   "preview_url": "https://...", "error": "..." }
/// ```
/// `preview_url` is optional; when present it points to a .gltf JSON manifest
/// (with absolute buffer URIs) for in-browser preview, while `output_url`
/// points to the self-contained .glb users actually download.
#[derive(Debug, Deserialize)]
pub struct ResultPayload {
    pub status: String,
    #[serde(default)]
    pub output_url: Option<String>,
    #[serde(default)]
    pub preview_url: Option<String>,
    /// Optional phase string ("fetching" | "optimizing" | "packaging") sent
    /// during `status=running` so the UI can show what step is happening.
    #[serde(default)]
    pub phase: Option<String>,
    /// Optional fractional progress in [0, 1] alongside `phase`. Workers
    /// forwarding splatforge CLI `--progress` output include this so the
    /// UI can render a determinate bar instead of an indeterminate slide.
    #[serde(default)]
    pub percent: Option<f32>,
    /// Wall-clock compute seconds the worker burned on this job. Reported
    /// by the worker on the terminal `done` callback so the billing path
    /// can emit the `splatforge_repack_seconds` meter event. Free-tier
    /// jobs may also send this; it's ignored because no customer_id is
    /// attached.
    #[serde(default)]
    pub compute_seconds: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[instrument(skip(state, body), fields(job_id = %id, status = %body.status))]
async fn job_result(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<ResultPayload>,
) -> Result<Json<Job>, ApiError> {
    let Some(mut job) = state.jobs.get(&id).await? else {
        return Err(ApiError::NotFound);
    };
    let mut terminal = false;
    match body.status.as_str() {
        "done" | "succeeded" => {
            let Some(url) = body.output_url else {
                return Err(ApiError::BadRequest(
                    "status=done requires output_url".to_string(),
                ));
            };
            job.status = JobStatus::Done;
            job.output_url = Some(url);
            if let Some(preview) = body.preview_url {
                job.preview_url = Some(preview);
            }
            job.error = None;
            terminal = true;
        }
        "error" | "failed" => {
            job.status = JobStatus::Error;
            job.error = Some(body.error.unwrap_or_else(|| "unknown worker error".into()));
            terminal = true;
        }
        "running" => {
            job.status = JobStatus::Running;
            if let Some(phase) = body.phase {
                job.phase = Some(phase);
            }
            if let Some(pct) = body.percent {
                job.percent = Some(pct.clamp(0.0, 1.0));
            }
        }
        other => {
            return Err(ApiError::BadRequest(format!("unknown status: {other}")));
        }
    }
    state.jobs.update(&job).await?;
    // Billing on terminal `done` of a paid job, when the worker
    // reported compute seconds. The ledger UNIQUE constraint on
    // (job_id, sku) makes this safe to double-fire: a flaky callback
    // that retries will see the seconds row already claimed and skip.
    // This is the load-bearing invariant — see BILLING.md "double-fire"
    // section.
    if terminal && matches!(job.status, JobStatus::Done) && job.tier == Tier::Paid {
        if let Err(e) = state
            .billing
            .record_repack_job(
                job.id,
                job.customer_id.as_deref(),
                job.size_bytes,
                0, // iterations unknown at callback time; not used downstream
                body.compute_seconds,
            )
            .await
        {
            warn!(error = %e, job_id = %job.id, "billing on callback failed; continuing");
        }
    }
    // Only fire webhooks on terminal states so batches of 40 don't
    // generate 80+ wakeups for each subscriber.
    if terminal {
        fire_webhook(&state, &job).await;
    }
    Ok(Json(job))
}

/// `POST /v1/stripe/webhook` — Stripe webhook receiver.
///
/// Verifies the `Stripe-Signature` header against the raw body via
/// HMAC-SHA256 (see `billing::verify_webhook`), then dispatches on
/// `event.type`. Only the events listed below are handled; everything
/// else is ack'd with 200 so Stripe stops retrying.
///
/// Events handled:
///   * `customer.subscription.created` / `updated`  — log status; this is
///     where tier upgrades land. We log the customer + status so the
///     operator can reconcile against the static key→customer map.
///   * `customer.subscription.deleted`              — log; downgrade target.
///   * `invoice.payment_failed`                     — warn; downgrade target.
///   * `invoice.payment_succeeded`                  — log for observability.
///
/// Automatic tier flipping is deliberately *not* wired up here: the
/// key→customer map is a static env var, and we don't want a webhook to
/// silently revoke a key. The handler emits structured logs the operator
/// (or a future control-plane DB) can reconcile against.
#[instrument(skip(state, body))]
async fn stripe_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(secret) = state.stripe_webhook_secret.as_ref() else {
        // No secret configured → reject everything. Better than silently
        // accepting unsigned events. 401 (not 503) so this matches the
        // signature-failure path and Stripe's retry budget can chew it.
        return Err(ApiError::Unauthorized(
            "stripe webhook secret not configured".to_string(),
        ));
    };
    let sig = headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok());
    let now = chrono::Utc::now().timestamp();
    let event = billing::verify_webhook(
        &body,
        sig,
        secret.as_str(),
        now,
        billing::WEBHOOK_DEFAULT_TOLERANCE_SECS,
    )
    .map_err(|e| ApiError::Unauthorized(format!("webhook signature: {e}")))?;

    let event_type = event
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("(unknown)");
    let event_id = event
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("(no-id)");
    let customer = event
        .pointer("/data/object/customer")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match event_type {
        "customer.subscription.updated" | "customer.subscription.created" => {
            let status = event
                .pointer("/data/object/status")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            info!(
                %event_id, event_type, customer, status,
                "stripe webhook: subscription state change (manual tier reconciliation)"
            );
        }
        "customer.subscription.deleted" => {
            info!(
                %event_id, event_type, customer,
                "stripe webhook: subscription cancelled (downgrade target)"
            );
        }
        "invoice.payment_failed" => {
            warn!(
                %event_id, event_type, customer,
                "stripe webhook: invoice payment failed — downgrade target"
            );
        }
        "invoice.payment_succeeded" => {
            info!(
                %event_id, event_type, customer,
                "stripe webhook: payment ok"
            );
        }
        other => {
            info!(%event_id, event_type = other, "stripe webhook: ignored");
        }
    }
    Ok(Json(serde_json::json!({ "received": true })))
}

/* ---------- self-serve Team-tier signup ---------- */

/// `POST /v1/checkout/create-session`
///
/// Body: `{ "email": "alice@acme.com", "nonce"?: "…" }`. Returns
/// `{ "url": "https://checkout.stripe.com/c/pay/…", "session_id": "cs_test_…" }`.
/// The frontend redirects the browser to `url` immediately. Stripe
/// hosts the payment UI; on success it redirects to
/// `<public_site_url>/welcome?session_id=<cs_…>&token=<claim_token>`.
///
/// Returns 503 when Stripe isn't configured (dev / CI). The frontend
/// surfaces this as "self-serve unavailable, contact sales" — falls
/// back to the Enterprise mailto: gracefully.
#[instrument(skip(state, req), fields(email = %req.email))]
async fn create_session_route(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(client) = state.checkout_client.as_ref() else {
        return Err(ApiError::ServiceUnavailable(
            "self-serve checkout not configured on this deployment".to_string(),
        ));
    };
    let resp = checkout::create_session_and_register(
        state.checkout_config.as_ref(),
        client.as_ref(),
        state.pending_claim_tokens.as_ref(),
        req,
    )
    .await?;
    Ok(Json(serde_json::json!({
        "url": resp.url,
        "session_id": resp.session_id,
    })))
}

/// `POST /v1/checkout/webhook` — Stripe `checkout.session.completed`.
///
/// Verifies the `Stripe-Signature` header against the raw body (same
/// HMAC-SHA256 path the billing webhook uses) and provisions a fresh
/// `sf_live_…` key on a verified `checkout.session.completed`. All
/// other event types are ack'd with 200 so Stripe stops retrying.
///
/// IDEMPOTENCY: `provision_from_session` uses `INSERT … ON CONFLICT
/// DO NOTHING` against `team_signups.stripe_session_id`. A retry that
/// lands a second time after a transport error sees the row already
/// present, returns `Ok(())`, and we 200 back — no second key minted.
#[instrument(skip(state, body))]
async fn checkout_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(secret) = state.checkout_webhook_secret.as_ref() else {
        return Err(ApiError::Unauthorized(
            "checkout webhook secret not configured".to_string(),
        ));
    };
    let sig = headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok());
    let now = chrono::Utc::now().timestamp();
    let event = billing::verify_webhook(
        &body,
        sig,
        secret.as_str(),
        now,
        billing::WEBHOOK_DEFAULT_TOLERANCE_SECS,
    )
    .map_err(|e| ApiError::Unauthorized(format!("checkout webhook signature: {e}")))?;

    let event_type = event
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("(unknown)");
    let event_id = event
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("(no-id)");

    match event_type {
        "checkout.session.completed" => {
            // The load-bearing path. Errors here roll up to ApiError
            // and we return non-2xx so Stripe retries — except for
            // BadRequest (malformed event) which is fatal and we
            // 400 to break the retry loop.
            checkout::provision_from_session(
                state.jobs.as_ref(),
                state.pending_keys.as_ref(),
                state.pending_claim_tokens.as_ref(),
                &event,
            )
            .await?;
            info!(%event_id, "checkout.session.completed provisioned");
        }
        other => {
            info!(%event_id, event_type = other, "checkout webhook: ignored");
        }
    }
    Ok(Json(serde_json::json!({ "received": true })))
}

/// `POST /v1/checkout/reveal` — one-time plaintext-key fetch for the
/// welcome page.
///
/// Body: `{ "session_id": "cs_…", "token": "<claim_token>" }`.
/// Response: `{ "api_key": "sf_live_…", "key_prefix": "sf_live_XXXX",
///              "authorization_header": "Bearer sf_live_…", "email": "…" }`.
///
/// EXACTLY-ONCE INVARIANT: see `checkout::reveal_key`. Three gates:
/// constant-time claim_token compare, atomic
/// `mark_team_signup_revealed`, and the in-memory cache `take`. The
/// second call returns 410 Gone.
#[instrument(skip(state, req))]
async fn reveal_route(
    State(state): State<AppState>,
    Json(req): Json<RevealRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let resp = checkout::reveal_key(state.jobs.as_ref(), state.pending_keys.as_ref(), req).await?;
    Ok(Json(serde_json::json!({
        "api_key": resp.api_key,
        "key_prefix": resp.key_prefix,
        "email": resp.email,
        "authorization_header": resp.authorization_header,
    })))
}

/// Strip path separators + control chars so the blob key stays inside the
/// `jobs/<uuid>/` prefix and can't be used to escape into other tenants.
fn sanitize_filename(name: &str) -> String {
    let trimmed: String = name
        .chars()
        .filter(|c| !c.is_control() && *c != '/' && *c != '\\')
        .collect();
    if trimmed.is_empty() {
        "splat.bin".to_string()
    } else {
        trimmed
    }
}

/* ---------- ratings (fidelity-ml v0.4 collection) ---------- */
//
// Validation, hashing, and the rate-limit constant all live in
// `splatforge_api::ratings` so the integration tests under
// `tests/ratings.rs` can exercise them without spinning up the full
// Axum app. The handlers below are thin glue: parse JSON, call the
// pure helpers, hit the store, return JSON.

#[derive(Debug, Deserialize)]
pub struct PostRatingRequest {
    pub scene_id: String,
    pub left_preset: String,
    pub right_preset: String,
    pub winner: String,
}

#[derive(Debug, Serialize)]
pub struct PostRatingResponse {
    pub id: i64,
    /// Ratings remaining in this respondent's rolling-hour window after
    /// this submission. The page surfaces this so a heavy rater knows
    /// they're approaching the cap (and can plausibly suspect rate
    /// limiting if their next POSTs start 429-ing).
    pub remaining: i64,
}

#[instrument(skip(state, headers, req), fields(scene = %req.scene_id))]
async fn post_rating(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PostRatingRequest>,
) -> Result<Json<PostRatingResponse>, ApiError> {
    validate_rating(
        &req.scene_id,
        &req.left_preset,
        &req.right_preset,
        &req.winner,
    )
    .map_err(ApiError::BadRequest)?;

    let respondent = respondent_hash(&headers);

    // Cheap rate-limit gate: count rows from this hash in the trailing
    // hour. SQLite + indexed column + a small table — this scales to
    // millions of rows before the count starts to hurt.
    let recent = state
        .jobs
        .count_recent_ratings(&respondent, chrono::Duration::hours(1))
        .await?;
    if recent >= RATING_RATE_LIMIT_PER_HOUR {
        return Err(ApiError::TooManyRequests(format!(
            "respondent has submitted {recent} ratings in the last hour; cap is {RATING_RATE_LIMIT_PER_HOUR}"
        )));
    }

    let id = state
        .jobs
        .insert_rating(
            &req.scene_id,
            &req.left_preset,
            &req.right_preset,
            &req.winner,
            &respondent,
        )
        .await?;
    let remaining = (RATING_RATE_LIMIT_PER_HOUR - recent - 1).max(0);
    Ok(Json(PostRatingResponse { id, remaining }))
}

#[derive(Debug, Serialize)]
pub struct RatingsSummaryResponse {
    pub pairs: Vec<RatingSummaryRow>,
    pub total_ratings: i64,
}

#[instrument(skip(state))]
async fn ratings_summary(
    State(state): State<AppState>,
) -> Result<Json<RatingsSummaryResponse>, ApiError> {
    let pairs = state.jobs.summarize_ratings().await?;
    let total: i64 = pairs.iter().map(|r| r.total).sum();
    Ok(Json(RatingsSummaryResponse {
        pairs,
        total_ratings: total,
    }))
}

/* ---------- capture-tool imports ---------- */

/// Shared body for the three `/v1/import/*` handlers. The handlers are
/// trivial wrappers that pick a `Provider` and delegate; keeping the
/// orchestration in one function makes it easier to reason about the
/// rate-limit + resolver + Job-build sequence (and easier to keep the
/// integration test honest — `tests/import.rs` exercises the same path
/// with a wiremock-backed resolver).
#[instrument(skip(state, headers, req), fields(provider = provider.as_str()))]
async fn do_import(
    state: AppState,
    headers: HeaderMap,
    extensions: axum::http::Extensions,
    provider: Provider,
    req: ImportRequest,
) -> Result<Json<ImportResponse>, ApiError> {
    // Rate-limit key — prefer the authenticated bearer (already validated
    // by `require_api_key`), fall back to the raw header for the
    // auth-disabled dev mode. Using the API key keeps the bucket scoped
    // to one customer; falling back to the raw header value keeps dev
    // mode usable without inventing a "no-auth" key.
    let rate_key = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_start_matches("Bearer ").trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "anonymous".to_string());

    let resolved = import_route::run_import(
        &state.capture_resolver,
        state.import_limiter.as_ref(),
        &rate_key,
        provider,
        &req.share_url,
    )
    .await?;

    // Defense in depth: the per-provider resolver already validated the
    // returned host against its allowlist, but `validate_source_url`
    // additionally blocks the private-IP-literal class for any future
    // provider we add that forgets the allowlist.
    validate_source_url(&resolved.source_url)?;

    let id = Uuid::new_v4();
    let filename = sanitize_filename(&resolved.filename);
    let blob_key = format!("jobs/{id}/{filename}");
    let customer_id = resolve_customer(&state, &extensions);
    let job = Job {
        id,
        // Same default preset as the proxy-upload flow used to land on.
        // Operators tweak this per deploy via the SDK; for now web-mobile
        // is the universally-supported one.
        preset: "web-mobile".to_string(),
        filename,
        size_bytes: 0,
        label: req
            .label
            .or_else(|| Some(format!("{}:{}", provider.as_str(), resolved.capture_id))),
        status: JobStatus::Queued,
        blob_key,
        blob_url: Some(resolved.source_url.clone()),
        source_url: Some(resolved.source_url.clone()),
        upload_size_bytes: None,
        output_url: None,
        preview_url: None,
        phase: None,
        percent: None,
        webhook_url: None,
        batch_id: None,
        tier: Tier::Free,
        customer_id,
        created_at: Utc::now(),
        error: None,
    };
    state.jobs.insert(&job).await?;
    if let Err(e) = enqueue_url_job(&state, &job).await {
        // Bubble the worker enqueue failure as 502 — same shape as a
        // `/v1/jobs` URL-mode failure. The job row stays in `Queued`
        // until a retry; the operator can re-run via the worker's
        // resume path.
        warn!(job_id = %job.id, error = %e, "capture import enqueue failed");
        return Err(e);
    }
    info!(
        job_id = %job.id,
        provider = provider.as_str(),
        capture_id = resolved.capture_id,
        "capture imported"
    );
    Ok(Json(ImportResponse {
        job_id: job.id,
        source_url: resolved.source_url,
        provider: provider.as_str(),
    }))
}

async fn import_luma(
    State(state): State<AppState>,
    headers: HeaderMap,
    extensions: axum::http::Extensions,
    Json(req): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, ApiError> {
    do_import(state, headers, extensions, Provider::Luma, req).await
}

async fn import_polycam(
    State(state): State<AppState>,
    headers: HeaderMap,
    extensions: axum::http::Extensions,
    Json(req): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, ApiError> {
    do_import(state, headers, extensions, Provider::Polycam, req).await
}

async fn import_scaniverse(
    State(state): State<AppState>,
    headers: HeaderMap,
    extensions: axum::http::Extensions,
    Json(req): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, ApiError> {
    do_import(state, headers, extensions, Provider::Scaniverse, req).await
}

/* ---------- errors ---------- */

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found")]
    NotFound,
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("storage error: {0}")]
    Storage(#[from] store::BlobError),
    #[error("modal error: {0}")]
    Modal(#[from] modal_client::ModalError),
    #[error("internal: {0}")]
    Internal(#[from] store::StoreError),
    #[error("service unavailable: {0}")]
    ServiceUnavailable(String),
    #[error("gone")]
    Gone,
    #[error("bad gateway: stripe: {0}")]
    Stripe(String),
    #[error("too many requests: {0}")]
    TooManyRequests(String),
    /// 415 — surfaced by the `/v1/import/*` routes when the provider says
    /// "yes, this capture exists, but it's not in a format we can ingest"
    /// (still-processing Luma capture, Scaniverse USDZ without PLY, …).
    #[error("unsupported media: {0}")]
    Unsupported(String),
    /// 429 — burnt through the per-key import rate budget.
    #[error("rate limited: {0}")]
    RateLimited(String),
    /// 502 — wraps a non-Stripe upstream provider failure (Luma/Polycam/
    /// Scaniverse timed out, returned 5xx, etc.). Distinct from `Stripe`
    /// so the IntoResponse arm can read the same way.
    #[error("bad gateway: {0}")]
    BadGateway(String),
}

impl From<checkout::CheckoutError> for ApiError {
    fn from(e: CheckoutError) -> Self {
        match e {
            CheckoutError::NotConfigured | CheckoutError::PriceNotConfigured => {
                ApiError::ServiceUnavailable(e.to_string())
            }
            CheckoutError::BadRequest(msg) => ApiError::BadRequest(msg),
            CheckoutError::NotFound => ApiError::NotFound,
            CheckoutError::Gone => ApiError::Gone,
            CheckoutError::Forbidden => ApiError::Forbidden("checkout".to_string()),
            CheckoutError::Stripe { status: _, body } => ApiError::Stripe(body),
            CheckoutError::Transport(s) | CheckoutError::BadResponse(s) => ApiError::Stripe(s),
            CheckoutError::Store(s) => ApiError::Internal(s),
        }
    }
}

impl From<ImportError> for ApiError {
    fn from(e: ImportError) -> Self {
        let msg = e.to_string();
        match e {
            ImportError::InvalidShareUrl { .. } | ImportError::UnsafeAssetUrl { .. } => {
                ApiError::BadRequest(msg)
            }
            ImportError::Unsupported { .. } => ApiError::Unsupported(msg),
            ImportError::UpstreamFailed { .. } => ApiError::BadGateway(msg),
            ImportError::RateLimited { .. } => ApiError::RateLimited(msg),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match &self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Forbidden(_) => (StatusCode::FORBIDDEN, self.to_string()),
            ApiError::Gone => (StatusCode::GONE, self.to_string()),
            ApiError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ApiError::Storage(_)
            | ApiError::Modal(_)
            | ApiError::Stripe(_)
            | ApiError::BadGateway(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ApiError::Unsupported(_) => (StatusCode::UNSUPPORTED_MEDIA_TYPE, self.to_string()),
            ApiError::RateLimited(_) => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::TooManyRequests(_) => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}

/* ---------- openapi spec + docs ---------- */

/// The OpenAPI spec is shipped as a static asset baked into the binary
/// so the deploy is single-artifact: no sidecar file path to mount, no
/// CDN dependency. `include_str!` resolves at compile time, which means
/// `cargo build` will refuse to compile if the spec is missing — that's
/// a feature, not a bug, since shipping a binary with a stale spec is
/// strictly worse than a build failure.
const OPENAPI_YAML: &str = include_str!("../openapi.yaml");

#[instrument]
async fn openapi_yaml() -> impl IntoResponse {
    (
        [("content-type", "application/yaml; charset=utf-8")],
        OPENAPI_YAML,
    )
}

/// Some tooling (Stripe Workbench, Postman) prefers JSON OpenAPI. We
/// ship YAML and let clients use any standard yaml→json converter;
/// returning 406 here is the least-surprise behavior.
#[instrument]
async fn openapi_json_passthrough() -> impl IntoResponse {
    (
        StatusCode::NOT_ACCEPTABLE,
        Json(serde_json::json!({
            "error": "JSON OpenAPI not served. Convert /openapi.yaml client-side.",
            "yaml_url": "/openapi.yaml"
        })),
    )
}

/// Swagger UI served from CDN — single HTML page, no build step, no
/// node_modules. The UI is a thin pointer at `/openapi.yaml`. Pinned
/// to a specific Swagger UI version so the docs UI doesn't silently
/// shift behavior when the CDN updates.
const SWAGGER_UI_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SplatForge API — docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css" />
  <style>body{margin:0;background:#fafafa;}</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js" crossorigin></script>
  <script>
    window.addEventListener("load", () => {
      window.ui = SwaggerUIBundle({
        url: "/openapi.yaml",
        dom_id: "#swagger-ui",
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
      });
    });
  </script>
</body>
</html>"##;

#[instrument]
async fn docs_ui() -> impl IntoResponse {
    (
        [("content-type", "text/html; charset=utf-8")],
        SWAGGER_UI_HTML,
    )
}

/* ---------- admin audit endpoint ---------- */

#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    /// How many rows to return, newest-first. Capped at
    /// `ADMIN_AUDIT_DEFAULT_LIMIT` (1000) — see `audit.rs`.
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct AuditListResponse {
    pub events: Vec<AuditEvent>,
}

#[instrument(skip(state))]
async fn admin_audit(
    State(state): State<AppState>,
    Query(q): Query<AuditQuery>,
) -> Result<Json<AuditListResponse>, ApiError> {
    let limit = q
        .limit
        .unwrap_or(audit::ADMIN_AUDIT_DEFAULT_LIMIT)
        .min(audit::ADMIN_AUDIT_DEFAULT_LIMIT);
    let events = state.jobs.list_audit_events(limit).await?;
    Ok(Json(AuditListResponse { events }))
}
