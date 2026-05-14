//! Job + blob storage. Both are stubbed against in-memory + Vercel Blob today;
//! the trait surface is meant to swap for Postgres + R2 once we outgrow the
//! single-instance deploy.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Lifecycle of an optimize job.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum JobStatus {
    /// Job was created; we issued a presign and are waiting for the client
    /// to actually upload the splat to Blob.
    AwaitingUpload,
    /// Upload confirmed; Modal worker queue is processing it.
    Queued,
    /// Modal worker is actively running splatforge optimize.
    Running,
    /// Optimize succeeded; `result` is populated with the SPZ + report URLs.
    Succeeded,
    /// Optimize failed or worker timed out; `error` is populated.
    Failed,
}

/// One optimize job — created by `POST /v1/jobs`, polled via `GET /v1/jobs/:id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub preset: String,
    pub filename: String,
    pub size_bytes: u64,
    pub label: Option<String>,
    pub status: JobStatus,
    /// Storage backend key (e.g. `jobs/<uuid>/scene.ply`). The full URL is
    /// derived via `BlobBackend::public_url`.
    pub blob_key: String,
    pub created_at: DateTime<Utc>,
    /// Populated by the Modal worker via the status webhook on success.
    pub result: Option<JobResult>,
    /// Populated on failure with a short, user-safe message.
    pub error: Option<String>,
}

/// What the client gets back after a successful optimize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub spz_url: String,
    pub gltf_url: String,
    pub report_url: String,
    /// Compression ratio computed by the worker (bytes_in / bytes_out_spz).
    pub ratio: f32,
    /// Wall-clock time spent inside the optimize binary (ms).
    pub optimize_ms: u64,
}

/// In-memory job store. Swapped for Postgres in a later iteration; the
/// surface is intentionally simple so the swap is a drop-in.
#[derive(Default)]
pub struct JobStore {
    inner: RwLock<HashMap<Uuid, Job>>,
}

impl JobStore {
    pub async fn insert(&self, job: Job) {
        self.inner.write().await.insert(job.id, job);
    }

    pub async fn update(&self, job: Job) {
        self.inner.write().await.insert(job.id, job);
    }

    pub async fn get(&self, id: &Uuid) -> Option<Job> {
        self.inner.read().await.get(id).cloned()
    }
}

/* ---------- Blob backend ---------- */

#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    #[error("blob storage not configured (set BLOB_READ_WRITE_TOKEN)")]
    NotConfigured,
    #[error("blob api: {0}")]
    Api(String),
}

/// Adapter around a Vercel Blob (or compatible) storage backend.
///
/// The actual presigning protocol is encapsulated behind two methods so we
/// can swap to R2 / S3 without touching call sites.
pub struct BlobBackend {
    token: Option<String>,
}

impl BlobBackend {
    pub fn new(token: Option<String>) -> Self {
        Self { token }
    }

    /// Issue a write-only URL valid for `ttl`. Today this returns a
    /// `vercel-blob://<key>` placeholder; the real Vercel Blob presigning
    /// call lives behind the `BLOB_READ_WRITE_TOKEN` env var and is wired
    /// in once apps/worker is provisioned.
    pub async fn presign_upload(&self, key: &str, _ttl: Duration) -> Result<String, BlobError> {
        // Vercel Blob doesn't expose a stable HTTPS presign API; the JS
        // `@vercel/blob/client.generateClientTokenFromReadWriteToken` is the
        // canonical path and isn't trivially portable to Rust. For v0.1 we
        // return a sentinel URL so the contract still resolves; the actual
        // upload flow goes through `POST /v1/jobs/:id/upload` (the API proxies
        // bytes to Vercel Blob server-side using the bearer token).
        let suffix = if self.token.is_some() {
            "?ttl=900&mode=server-proxy"
        } else {
            "?ttl=900&mode=stub"
        };
        Ok(format!("blob://stub/{key}{suffix}"))
    }

    /// Return the publicly fetchable URL the worker should pull from. Stub
    /// matches `presign_upload` until the real client is wired in.
    pub fn public_url(&self, key: &str) -> String {
        format!("blob://stub/{key}")
    }
}
