//! Thin client around the Modal worker's webhook endpoint.
//!
//! The worker exposes a single `POST /enqueue` route that accepts a job
//! descriptor + blob URL and returns immediately after the Modal Function is
//! spawned. Result updates flow back through the API's status webhook
//! (`POST /v1/jobs/:id/status`), not by polling Modal directly.

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

/// HTTP client targeted at the Modal worker's web endpoint.
pub struct ModalClient {
    base_url: Option<String>,
    http: reqwest::Client,
}

impl ModalClient {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("reqwest client"),
        }
    }

    /// POST the job to the Modal worker. Returns an `EnqueueAck` that may
    /// carry a worker-side warning (e.g. "Modal cold-start in flight, expect
    /// ~5 s delay before the job starts").
    pub async fn enqueue(&self, job: &Job, blob_url: &str) -> Result<EnqueueAck, ModalError> {
        let Some(base) = &self.base_url else {
            // No worker configured — degrade gracefully so the API can still
            // function in dev environments without Modal access.
            return Ok(EnqueueAck {
                queued: false,
                error: Some("modal worker not configured; job remains AwaitingUpload".to_string()),
            });
        };
        let url = format!("{}/enqueue", base.trim_end_matches('/'));
        let payload = EnqueuePayload {
            job_id: job.id,
            preset: &job.preset,
            blob_url,
            filename: &job.filename,
            size_bytes: job.size_bytes,
            label: job.label.as_deref(),
        };
        let resp = self
            .http
            .post(&url)
            .json(&payload)
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

#[derive(Debug, Serialize)]
struct EnqueuePayload<'a> {
    job_id: uuid::Uuid,
    preset: &'a str,
    blob_url: &'a str,
    filename: &'a str,
    size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<&'a str>,
}

/// Worker's reply to `/enqueue`. `queued=true` means the Modal Function has
/// been spawned; a `None` error implies no warnings. Both `false`+error and
/// `true`+error are valid combinations (e.g. queued but degraded).
#[derive(Debug, Clone, Deserialize)]
pub struct EnqueueAck {
    pub queued: bool,
    pub error: Option<String>,
}
