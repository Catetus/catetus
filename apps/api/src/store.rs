//! Job + blob storage. Both are stubbed against in-memory + Vercel Blob today;
//! the trait surface is meant to swap for Postgres + R2 once we outgrow the
//! single-instance deploy.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Body;
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
    /// Bytes are streaming through the API into Vercel Blob right now.
    Uploading,
    /// Upload finished; Modal worker queue is processing it.
    Queued,
    /// Modal worker is actively running splatforge optimize.
    Running,
    /// Optimize finished; `output_url` is a downloadable Vercel Blob URL.
    Done,
    /// Optimize failed or worker timed out; `error` is populated.
    Error,
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
    /// derived via `BlobBackend::public_url` once the bytes have landed.
    pub blob_key: String,
    /// Public Vercel Blob URL the worker should pull from; only populated
    /// after the upload proxy finishes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_url: Option<String>,
    /// User-supplied source URL the worker fetches from directly. Mutually
    /// exclusive with the upload path: when set, the job skips `AwaitingUpload`
    /// + `Uploading` and lands in `Queued` immediately. Used for inputs that
    /// already live on the internet (HuggingFace, S3, GCS, R2, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// Actual bytes received by the upload proxy (may differ from
    /// the `size_bytes` hint the client gave us at job creation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_size_bytes: Option<u64>,
    /// Downloadable URL for the optimized artifact, set by the worker callback.
    /// Points to a self-contained `.glb` (binary glTF) that bundles the splat
    /// manifest and buffer data in a single file — drag into any viewer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_url: Option<String>,
    /// Companion URL for in-browser preview using the splatforge viewer. Points
    /// to a `.gltf` JSON manifest whose buffer URIs have been rewritten to
    /// absolute Vercel Blob URLs, so the viewer can lazy-stream chunks. Set
    /// alongside `output_url` by the worker callback when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
    /// Worker-emitted step name during the `Running` phase. The worker
    /// posts intermediate updates with `phase` like "fetching", "optimizing",
    /// "packaging" so the UI can show what's happening instead of a single
    /// opaque "running" badge for the entire job duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Webhook the API fires when this job reaches a terminal state (Done
    /// or Error). Lets callers run a 40-tile batch without polling 40
    /// endpoints. POST body is the Job JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_url: Option<String>,
    /// Group identifier when this job was created via `POST /v1/jobs/batch`.
    /// All jobs in the same `/batch` call share a batch_id so clients can
    /// reassemble results, and webhooks can identify which batch fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    /// Populated on failure with a short, user-safe message.
    pub error: Option<String>,
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
    #[error("blob transport: {0}")]
    Transport(String),
}

/// Adapter around a Vercel Blob (or compatible) storage backend.
///
/// The Vercel Blob HTTPS protocol is dead-simple: PUT to
/// `https://blob.vercel-storage.com/<pathname>?addRandomSuffix=0` with the
/// bearer token and content-type headers. The 200 response carries the
/// canonical public URL.
pub struct BlobBackend {
    token: Option<String>,
    http: reqwest::Client,
}

const BLOB_HOST: &str = "https://blob.vercel-storage.com";
const BLOB_API_VERSION: &str = "7";

impl BlobBackend {
    pub fn new(token: Option<String>) -> Self {
        Self {
            token,
            http: reqwest::Client::builder()
                // Splat uploads can be hundreds of MB on slow links; keep the
                // overall PUT generous but require steady progress via the
                // tcp keepalive defaults baked into reqwest.
                .timeout(Duration::from_secs(600))
                .pool_max_idle_per_host(4)
                .build()
                .expect("reqwest client"),
        }
    }

    /// Issue a write-only URL valid for `ttl`. We don't have a real presign
    /// API (Vercel Blob doesn't expose stable HTTPS presigning for Rust), so
    /// the contract here returns a sentinel pointing back at this same API:
    /// the client PUTs to `/v1/jobs/:id/upload` and we proxy the bytes to
    /// Vercel Blob server-side using the bearer token.
    pub async fn presign_upload(&self, key: &str, _ttl: Duration) -> Result<String, BlobError> {
        let suffix = if self.token.is_some() {
            "?ttl=900&mode=server-proxy"
        } else {
            "?ttl=900&mode=stub"
        };
        Ok(format!("blob://stub/{key}{suffix}"))
    }

    /// Best-effort guess at the public URL for a key, used only when the
    /// server-side PUT hasn't run yet (e.g. for fallback rendering). The
    /// real, canonical URL is the one Vercel returns from `put`.
    pub fn public_url(&self, key: &str) -> String {
        format!("blob://stub/{key}")
    }

    /// Stream raw bytes to Vercel Blob and return the public URL Vercel
    /// hands back. `content_type` is forwarded as the resulting object's
    /// content type; pass `application/octet-stream` for splats.
    pub async fn put_bytes(
        &self,
        key: &str,
        body: Body,
        content_type: &str,
    ) -> Result<String, BlobError> {
        let token = self
            .token
            .as_ref()
            .ok_or(BlobError::NotConfigured)?
            .clone();
        let url = format!("{BLOB_HOST}/{}?addRandomSuffix=0", key.trim_start_matches('/'));
        let resp = self
            .http
            .put(&url)
            .header("authorization", format!("Bearer {token}"))
            .header("x-content-type", content_type)
            .header("x-api-version", BLOB_API_VERSION)
            .body(body)
            .send()
            .await
            .map_err(|e| BlobError::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(BlobError::Api(format!("{status}: {text}")));
        }
        // Response envelope: `{ "url": "...", "pathname": "...", ... }`.
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BlobError::Api(format!("decoding blob response: {e}")))?;
        body.get("url")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| BlobError::Api(format!("blob response missing url field: {body}")))
    }
}
