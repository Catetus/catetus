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
use std::time::Duration;

use anyhow::Context;
use axum::{
    body::{Body, Bytes},
    extract::{Path, State},
    http::{HeaderMap, Request, StatusCode},
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
use splatforge_api::billing::{self, BillingClient, KeyCustomerMap};
use splatforge_api::modal_client::{self, ModalClient};
use splatforge_api::store::{self, Job, JobStatus, JobStore, Tier};

/// Top-level app state shared with every handler.
#[derive(Clone)]
pub struct AppState {
    /// In-memory job store. Swapped for Postgres once we have multi-instance
    /// deployment; everything below treats it as an opaque trait object.
    pub jobs: Arc<JobStore>,
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
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://data/jobs.db".to_string());
    if let Some(path) = database_url
        .strip_prefix("sqlite://")
        .or_else(|| database_url.strip_prefix("sqlite:"))
    {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating sqlite parent dir {}", parent.display())
                })?;
            }
        }
    }
    // Default to the droplet's well-known address; override in env for
    // production behind a proper hostname.
    let public_base_url = std::env::var("SPLATFORGE_PUBLIC_BASE_URL")
        .unwrap_or_else(|_| "http://167.99.231.209:8080".to_string());
    // Comma-separated list of accepted bearer tokens. Empty = auth disabled
    // (only acceptable in local dev; the deployed binary should always have
    // at least one key set).
    let api_keys: HashSet<String> = parse_keys(std::env::var("SPLATFORGE_API_KEYS").ok());
    let paid_api_keys: HashSet<String> = parse_keys(std::env::var("SPLATFORGE_PAID_API_KEYS").ok());
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
        info!(n_paid_keys = paid_api_keys.len(), "paid-tier bearer auth enabled");
    }

    // Dedicated client for outbound webhook firing. Short timeout so a
    // misbehaving subscriber doesn't stall the result-callback handler.
    let webhook_http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");

    let jobs = JobStore::connect(&database_url)
        .await
        .with_context(|| format!("opening jobs db at {database_url}"))?;
    info!(%database_url, "job store ready");
    let jobs = Arc::new(jobs);

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
    };

    // Three routers:
    //   - `open`  — always public (healthz, worker callback). The worker
    //               callback is protected by the per-job UUID, not the
    //               bearer token, so a worker doesn't need an API key.
    //   - `paid`  — gated on the bearer token when SPLATFORGE_API_KEYS is set.
    //               Job creation + GET (clients poll their own job state).
    //   - `upload`— same auth as `paid` but with a 250 MB body cap.
    let auth_layer = middleware::from_fn_with_state(state.clone(), require_api_key);

    let open = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/jobs/:id/result", post(job_result))
        // Stripe webhook is "open" because it has its own HMAC-SHA256
        // signature gate (`STRIPE_WEBHOOK_SECRET`), not a bearer token.
        // 1 MB is well over the largest event payload Stripe sends.
        .route("/v1/stripe/webhook", post(stripe_webhook))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024));

    let paid_layer =
        middleware::from_fn_with_state(state.clone(), require_paid_api_key);

    let paid = Router::new()
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/batch", post(create_jobs_batch))
        .route("/v1/jobs/:id", get(get_job))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024))
        .layer(auth_layer.clone());

    let repack = Router::new()
        .route("/v1/jobs/:id/repack", post(repack_job))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024))
        // Order matters: tower layers apply outermost-first. Free gate runs
        // first (rejects unauthenticated requests early), paid gate runs
        // second so it only sees pre-authenticated calls.
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
        .layer(auth_layer);

    let app = Router::new()
        .merge(open)
        .merge(paid)
        .merge(repack)
        .merge(upload)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

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
        auth.strip_prefix("Bearer ").unwrap_or_default().trim().to_owned()
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

fn job_creation_response(
    job: &Job,
    _state: &AppState,
) -> Result<CreateJobResponse, ApiError> {
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
        .split(|c: char| c == '/' || c == ':' || c == '?' || c == '#')
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
    if FORBIDDEN_HOST_PREFIXES.iter().any(|p| host_lower.starts_with(p)) {
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
    let Some(url) = job.webhook_url.as_deref() else { return };
    let payload = match serde_json::to_value(job) {
        Ok(v) => v,
        Err(e) => {
            warn!(job_id = %job.id, error = %e, "webhook payload serialization failed");
            return;
        }
    };
    let resp = state
        .webhook_http
        .post(url)
        .json(&payload)
        .send()
        .await;
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
    if terminal
        && matches!(job.status, JobStatus::Done)
        && job.tier == Tier::Paid
    {
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
    let sig = headers.get("stripe-signature").and_then(|v| v.to_str().ok());
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

/* ---------- errors ---------- */

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found")]
    NotFound,
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("storage error: {0}")]
    Storage(#[from] store::BlobError),
    #[error("modal error: {0}")]
    Modal(#[from] modal_client::ModalError),
    #[error("internal: {0}")]
    Internal(#[from] store::StoreError),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match &self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Storage(_) | ApiError::Modal(_) => {
                (StatusCode::BAD_GATEWAY, self.to_string())
            }
            ApiError::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}
