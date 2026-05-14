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

use std::sync::Arc;

use anyhow::Context;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
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

mod modal_client;
mod store;

use modal_client::ModalClient;
use store::{Job, JobStatus, JobStore};

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
    let blob_token = std::env::var("BLOB_READ_WRITE_TOKEN").ok();
    // Default to the droplet's well-known address; override in env for
    // production behind a proper hostname.
    let public_base_url = std::env::var("SPLATFORGE_PUBLIC_BASE_URL")
        .unwrap_or_else(|_| "http://167.99.231.209:8080".to_string());

    let state = AppState {
        jobs: Arc::new(JobStore::default()),
        modal: Arc::new(ModalClient::new(modal_url)),
        blob: Arc::new(store::BlobBackend::new(blob_token)),
        public_base_url: public_base_url.trim_end_matches('/').to_string(),
    };

    // Two routers so the upload path can run with a 250 MB body cap while
    // the small JSON routes keep the strict 50 MB default.
    let small = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/:id", get(get_job))
        .route("/v1/jobs/:id/result", post(job_result))
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024));

    let upload = Router::new()
        .route("/v1/jobs/:id/upload", post(upload_job))
        // 250 MB — covers the SplatBench indoor proxy (50 MB) with plenty of
        // headroom for typical splats. The body is streamed through to Vercel
        // Blob; we never buffer it in memory.
        .layer(RequestBodyLimitLayer::new(250 * 1024 * 1024));

    let app = Router::new()
        .merge(small)
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

#[instrument(skip(_state))]
async fn healthz(State(_state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "service": "splatforge-api",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Payload for `POST /v1/jobs`. The client describes the optimize request
/// upfront; the response carries a server-proxy URL for the byte upload.
#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    /// One of `lossless-repack` / `web-mobile` / `size-min` / etc.
    pub preset: String,
    /// Suggested filename (used to derive the blob key).
    pub filename: String,
    /// Size in bytes — clients should set this so the server can reject
    /// anything obviously too large before issuing a presign.
    pub size_bytes: u64,
    /// Optional caller-supplied label for the job (e.g. `acme-q3-walkthrough`).
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateJobResponse {
    pub id: Uuid,
    pub status: JobStatus,
    pub upload_url: String,
    pub upload_method: String,
    pub created_at: DateTime<Utc>,
}

#[instrument(skip(state, req), fields(preset = %req.preset, size = req.size_bytes))]
async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> Result<Json<CreateJobResponse>, ApiError> {
    // Cap at 1.5 GB so we don't accidentally accept a 50 GB splat that would
    // blow Modal's worker budget. Bicycle (3.6M splats) is ~860 MB raw PLY.
    const MAX_BYTES: u64 = 1_500_000_000;
    if req.size_bytes == 0 || req.size_bytes > MAX_BYTES {
        return Err(ApiError::BadRequest(format!(
            "size_bytes must be in (0, {MAX_BYTES}); got {}",
            req.size_bytes
        )));
    }

    let id = Uuid::new_v4();
    let blob_key = format!("jobs/{id}/{}", sanitize_filename(&req.filename));
    let upload_url = state
        .blob
        .presign_upload(&blob_key, std::time::Duration::from_secs(900))
        .await?;

    let job = Job {
        id,
        preset: req.preset.clone(),
        filename: req.filename.clone(),
        size_bytes: req.size_bytes,
        label: req.label.clone(),
        status: JobStatus::AwaitingUpload,
        blob_key: blob_key.clone(),
        blob_url: None,
        upload_size_bytes: None,
        output_url: None,
        created_at: Utc::now(),
        error: None,
    };
    state.jobs.insert(job.clone()).await;

    Ok(Json(CreateJobResponse {
        id,
        status: job.status,
        upload_url,
        upload_method: "PUT".to_string(),
        created_at: job.created_at,
    }))
}

#[instrument(skip(state))]
async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Job>, ApiError> {
    state
        .jobs
        .get(&id)
        .await
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
    let Some(mut job) = state.jobs.get(&id).await else {
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
    state.jobs.update(job.clone()).await;

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
            state.jobs.update(job.clone()).await;
            return Err(ApiError::Storage(e));
        }
    };

    job.blob_url = Some(blob_url.clone());
    // We don't have a clean way to recover the streamed byte count without
    // a counting middleware; fall back to the client-supplied size_bytes
    // (already validated to be in range) so the field is at least populated.
    job.upload_size_bytes = Some(job.size_bytes);
    job.status = JobStatus::Queued;
    state.jobs.update(job.clone()).await;

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
            state.jobs.update(job.clone()).await;
            return Err(ApiError::Modal(e));
        }
    }

    Ok(Json(job))
}

/// `POST /v1/jobs/:id/result`
///
/// Worker callback. Payload:
/// ```json
/// { "status": "done" | "error", "output_url": "https://...", "error": "..." }
/// ```
#[derive(Debug, Deserialize)]
pub struct ResultPayload {
    pub status: String,
    #[serde(default)]
    pub output_url: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[instrument(skip(state, body), fields(job_id = %id, status = %body.status))]
async fn job_result(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<ResultPayload>,
) -> Result<Json<Job>, ApiError> {
    let Some(mut job) = state.jobs.get(&id).await else {
        return Err(ApiError::NotFound);
    };
    match body.status.as_str() {
        "done" | "succeeded" => {
            let Some(url) = body.output_url else {
                return Err(ApiError::BadRequest(
                    "status=done requires output_url".to_string(),
                ));
            };
            job.status = JobStatus::Done;
            job.output_url = Some(url);
            job.error = None;
        }
        "error" | "failed" => {
            job.status = JobStatus::Error;
            job.error = Some(body.error.unwrap_or_else(|| "unknown worker error".into()));
        }
        "running" => {
            job.status = JobStatus::Running;
        }
        other => {
            return Err(ApiError::BadRequest(format!("unknown status: {other}")));
        }
    }
    state.jobs.update(job.clone()).await;
    Ok(Json(job))
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
    #[error("storage error: {0}")]
    Storage(#[from] store::BlobError),
    #[error("modal error: {0}")]
    Modal(#[from] modal_client::ModalError),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match &self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Storage(_) | ApiError::Modal(_) => {
                (StatusCode::BAD_GATEWAY, self.to_string())
            }
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}
