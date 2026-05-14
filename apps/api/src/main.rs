//! `splatforge-api` — hosted optimize endpoint.
//!
//! Public surface for the design-partner program. Three responsibilities:
//!
//! 1. Issue presigned upload URLs into Vercel Blob (or compatible store).
//! 2. Enqueue optimize jobs against the Modal worker.
//! 3. Serve job status + result download URLs.
//!
//! The actual splat work happens in `apps/worker` (Modal Python). This crate
//! stays HTTP-light so we can host it on either Modal `web_endpoint` or any
//! standard PaaS without rewriting handlers.

use std::sync::Arc;

use anyhow::Context;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, trace::TraceLayer};
use tracing::{info, instrument};
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

    let state = AppState {
        jobs: Arc::new(JobStore::default()),
        modal: Arc::new(ModalClient::new(modal_url)),
        blob: Arc::new(store::BlobBackend::new(blob_token)),
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/:id", get(get_job))
        .route("/v1/jobs/:id/upload", post(presign_upload))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        // 50 MB request body cap — splats upload directly to Blob via
        // presigned URL, so the API request payload itself stays small.
        .layer(RequestBodyLimitLayer::new(50 * 1024 * 1024))
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
/// upfront; the response carries a presign URL the client uploads the splat to.
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
        created_at: Utc::now(),
        result: None,
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

/// POST /v1/jobs/:id/upload — the client calls this AFTER they've finished
/// PUTting the splat to the presigned URL. We then enqueue the actual optimize.
#[instrument(skip(state))]
async fn presign_upload(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Job>, ApiError> {
    let Some(mut job) = state.jobs.get(&id).await else {
        return Err(ApiError::NotFound);
    };
    if !matches!(job.status, JobStatus::AwaitingUpload) {
        return Err(ApiError::BadRequest(format!(
            "job {id} is already {:?}; cannot re-enqueue",
            job.status
        )));
    }

    // Hand off to Modal — it will pull the blob, run splatforge optimize,
    // and write the result back via the worker's status webhook (next iter).
    let blob_url = state.blob.public_url(&job.blob_key);
    let enqueued = state.modal.enqueue(&job, &blob_url).await?;

    job.status = JobStatus::Queued;
    job.error = enqueued.error;
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

// Silence unused-import warning on the default in-memory store.
#[allow(dead_code)]
fn _hashmap_witness() -> HashMap<Uuid, RwLock<Job>> {
    HashMap::new()
}
