//! Thin client around the Modal worker's webhook endpoint.
//!
//! The worker exposes a single `POST /enqueue` route that accepts a job
//! descriptor + blob URL and returns immediately after the Modal Function is
//! spawned. Result updates flow back through `callback_url` (the API's
//! `/v1/jobs/:id/result` route), not by polling Modal directly.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::store::Job;

#[derive(Debug, thiserror::Error)]
pub enum ModalError {
    #[error("modal endpoint not configured (set SPLATFORGE_MODAL_URL)")]
    NotConfigured,
    #[error("modal request failed: {0}")]
    Request(String),
    #[error("modal rejected enqueue: {0}")]
    Rejected(String),
}

/// HTTP client targeted at the Modal worker's web endpoint(s). The free
/// pipeline (CPU optimize) and the paid pipeline (A100 differentiable
/// repack) are deployed as separate Modal apps so spot-pricing decisions,
/// concurrency, and timeouts can differ. This client keeps both URLs side
/// by side so callers don't have to plumb the choice through the request.
pub struct ModalClient {
    base_url: Option<String>,
    repack_url: Option<String>,
    http: reqwest::Client,
}

impl ModalClient {
    pub fn new(base_url: Option<String>, repack_url: Option<String>) -> Self {
        Self {
            base_url,
            repack_url,
            // 30s for the synchronous /enqueue handshake. The actual work
            // happens asynchronously on Modal — the worker calls back into
            // /v1/jobs/:id/result when done, so the handshake just needs to
            // confirm the function was spawned.
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
        }
    }

    /// POST the job to the Modal worker. `callback_url` is the absolute URL
    /// the worker should POST the final result to.
    pub async fn enqueue(
        &self,
        job: &Job,
        blob_url: &str,
        callback_url: &str,
    ) -> Result<EnqueueAck, ModalError> {
        let Some(base) = &self.base_url else {
            return Ok(EnqueueAck {
                queued: false,
                error: Some("modal worker not configured; job remains AwaitingUpload".to_string()),
            });
        };
        let url = resolve_endpoint(base);
        let payload = EnqueuePayload {
            job_id: job.id,
            preset: &job.preset,
            blob_url,
            filename: &job.filename,
            size_bytes: job.size_bytes,
            label: job.label.as_deref(),
            callback_url,
            params: None,
        };
        self.post(&url, &payload).await
    }

    /// Dispatch a differentiable-repack run to the A100 worker. `params`
    /// carries the per-job knobs (target byte budget, iteration count); the
    /// rest of the envelope matches `/enqueue` so the worker can reuse the
    /// same input fetch + callback plumbing.
    pub async fn enqueue_repack(
        &self,
        job: &Job,
        blob_url: &str,
        callback_url: &str,
        params: serde_json::Value,
    ) -> Result<EnqueueAck, ModalError> {
        let Some(base) = &self.repack_url else {
            return Err(ModalError::NotConfigured);
        };
        let url = resolve_endpoint(base);
        let payload = EnqueuePayload {
            job_id: job.id,
            preset: &job.preset,
            blob_url,
            filename: &job.filename,
            size_bytes: job.size_bytes,
            label: job.label.as_deref(),
            callback_url,
            params: Some(params),
        };
        self.post(&url, &payload).await
    }

    async fn post(&self, url: &str, payload: &EnqueuePayload<'_>) -> Result<EnqueueAck, ModalError> {
        let resp = self
            .http
            .post(url)
            .json(payload)
            .send()
            .await
            .map_err(|e| ModalError::Request(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ModalError::Rejected(format!("{status}: {body}")));
        }
        let ack: EnqueueAck = resp
            .json()
            .await
            .map_err(|e| ModalError::Request(e.to_string()))?;
        Ok(ack)
    }
}

/// Modal publishes one URL per `fastapi_endpoint`. Accept either the fully
/// qualified `https://.../enqueue` form or the bare host form for back-compat.
fn resolve_endpoint(base: &str) -> String {
    if base.contains("enqueue") {
        base.to_string()
    } else {
        format!("{}/enqueue", base.trim_end_matches('/'))
    }
}

#[derive(Debug, Serialize)]
struct EnqueuePayload<'a> {
    job_id: uuid::Uuid,
    preset: &'a str,
    blob_url: &'a str,
    filename: &'a str,
    size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<&'a str>,
    callback_url: &'a str,
    /// Free-form worker-specific knobs. Repack uses
    /// `{ "target_bytes": <u64>, "iterations": <u32> }`.
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

/// Worker's reply to `/enqueue`. `queued=true` means the Modal Function has
/// been spawned; a `None` error implies no warnings. Both `false`+error and
/// `true`+error are valid combinations (e.g. queued but degraded).
#[derive(Debug, Clone, Deserialize)]
pub struct EnqueueAck {
    pub queued: bool,
    pub error: Option<String>,
}
