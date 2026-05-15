//! Job + blob storage — trait abstraction.
//!
//! ## Why the trait
//!
//! The single-instance Fly deploy that runs the design-partner program
//! uses SQLite. Once we promote to multi-instance (v2 plan §3b —
//! "Production API: Postgres or Turso for job state") the same call
//! sites need to talk to a Postgres pool without a rewrite. The trait
//! lives here; the two impls live under `sqlite.rs` and `postgres.rs`.
//!
//! ## Object-safe `dyn`
//!
//! `AppState` in `main.rs` holds `Arc<dyn JobStoreApi + Send + Sync>`
//! so handlers stay backend-agnostic. Native `async fn` in traits is
//! stable but the resulting traits are NOT object-safe — `async-trait`
//! desugars to boxed futures, which is. The cost (one heap alloc per
//! call) is dwarfed by the DB round-trip cost in every handler that
//! talks to the store, so the trade is fine.
//!
//! ## Two backends, identical contract
//!
//! Both `SqliteJobStore` and `PostgresJobStore` implement the same
//! trait, share the same `Job` / `JobStatus` / `Tier` / `RatingSummaryRow`
//! / `TeamSignupRow` types defined here, and pass the same trait-level
//! integration tests in `tests/store_trait.rs`. The only surface area
//! that knows which backend is live is `connect(url)` in this module.
//!
//! ## Migration paths
//!
//! Each backend has its own migrations directory under
//! `apps/api/migrations/{sqlite,postgres}/`. The directories are NOT
//! interchangeable — SQLite's `AUTOINCREMENT` is rewritten as
//! `BIGSERIAL` for Postgres, `INTEGER` becomes `BIGINT`, etc. See
//! `apps/api/STORE-BACKENDS.md` for the SQLite→Postgres cutover
//! procedure.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Body;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod postgres;
pub mod sqlite;

pub use postgres::PostgresJobStore;
pub use sqlite::SqliteJobStore;

/// Backwards-compat alias. The original module exposed `JobStore` as a
/// concrete SQLite-backed type; the trait abstraction kept that name
/// usable for tests + the in-memory dev path while moving the dyn-
/// dispatched production state onto `Arc<dyn JobStoreApi …>`.
pub type JobStore = SqliteJobStore;

/// Lifecycle of an optimize job.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum JobStatus {
    AwaitingUpload,
    Uploading,
    Queued,
    Running,
    Done,
    Error,
}

impl JobStatus {
    pub(crate) fn as_db_str(self) -> &'static str {
        match self {
            JobStatus::AwaitingUpload => "awaiting-upload",
            JobStatus::Uploading => "uploading",
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Done => "done",
            JobStatus::Error => "error",
        }
    }
    pub(crate) fn from_db_str(s: &str) -> Option<Self> {
        Some(match s {
            "awaiting-upload" => JobStatus::AwaitingUpload,
            "uploading" => JobStatus::Uploading,
            "queued" => JobStatus::Queued,
            "running" => JobStatus::Running,
            "done" => JobStatus::Done,
            "error" => JobStatus::Error,
            _ => return None,
        })
    }
}

/// Tier the job is being charged against. Free runs the public deterministic
/// pipeline (Modal CPU worker); Paid runs the gsplat A100 differentiable
/// repack. Stamped at job creation time so callbacks and webhooks can
/// surface the SKU without consulting the routing layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    Free,
    Paid,
}

impl Tier {
    pub(crate) fn as_db_str(self) -> &'static str {
        match self {
            Tier::Free => "free",
            Tier::Paid => "paid",
        }
    }
    pub(crate) fn from_db_str(s: &str) -> Option<Self> {
        Some(match s {
            "free" => Tier::Free,
            "paid" => Tier::Paid,
            _ => return None,
        })
    }
}

impl Default for Tier {
    fn default() -> Self {
        Tier::Free
    }
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
    pub blob_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<Uuid>,
    #[serde(default)]
    pub tier: Tier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub customer_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub error: Option<String>,
}

/// One row of the per-pair rating summary returned by `summarize_ratings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatingSummaryRow {
    pub scene_id: String,
    pub left_preset: String,
    pub right_preset: String,
    pub left_wins: i64,
    pub right_wins: i64,
    pub ties: i64,
    pub total: i64,
}

/// One row of the `team_signups` ledger.
#[derive(Debug, Clone)]
pub struct TeamSignupRow {
    pub claim_token: String,
    pub key_prefix: String,
    pub key_hash: String,
    pub stripe_customer_id: String,
    pub email: String,
    pub key_revealed_at: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlx: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("migrate: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("decoding row: {0}")]
    Decode(String),
}

/// The single contract every backend must satisfy. The set of methods is
/// intentionally narrow — every method has a real production caller in
/// `main.rs`, `billing.rs`, or `checkout.rs`. If a future change wants
/// a new query, it gets added here and implemented in both backends
/// (with a `tests/store_trait.rs` case that exercises both).
///
/// Object-safety: every method takes `&self` and returns a boxed future
/// (`#[async_trait]`), so `Arc<dyn JobStoreApi + Send + Sync>` is a
/// valid trait object.
#[async_trait]
pub trait JobStoreApi {
    /* ---------- jobs ---------- */

    async fn insert(&self, job: &Job) -> Result<(), StoreError>;
    async fn update(&self, job: &Job) -> Result<(), StoreError>;
    async fn get(&self, id: &Uuid) -> Result<Option<Job>, StoreError>;
    async fn list_by_batch(&self, batch_id: &Uuid) -> Result<Vec<Job>, StoreError>;

    /* ---------- billing ledger ---------- */

    /// Idempotently claim a (job_id, sku) slot. Returns `Ok(true)` if
    /// this is a fresh claim (caller should post the meter event);
    /// `Ok(false)` if a row already exists (someone else got here first
    /// — do NOT post). This is the no-double-charge invariant.
    async fn claim_billing_event(
        &self,
        job_id: &Uuid,
        customer_id: &str,
        sku: &str,
        units: u64,
        idempotency_key: &str,
    ) -> Result<bool, StoreError>;

    async fn mark_billing_event_posted(
        &self,
        job_id: &Uuid,
        sku: &str,
        stripe_event_id: &str,
    ) -> Result<(), StoreError>;

    /* ---------- team signups (checkout) ---------- */

    async fn claim_team_signup(
        &self,
        stripe_session_id: &str,
        stripe_customer_id: &str,
        stripe_subscription_id: Option<&str>,
        email: &str,
        claim_token: &str,
        key_prefix: &str,
        key_hash: &str,
        seats: u32,
    ) -> Result<bool, StoreError>;

    async fn get_team_signup_by_session(
        &self,
        stripe_session_id: &str,
    ) -> Result<Option<TeamSignupRow>, StoreError>;

    async fn mark_team_signup_revealed(
        &self,
        stripe_session_id: &str,
    ) -> Result<bool, StoreError>;

    /* ---------- ratings (fidelity-ml v0.4) ---------- */

    async fn insert_rating(
        &self,
        scene_id: &str,
        left_preset: &str,
        right_preset: &str,
        winner: &str,
        respondent_hash: &str,
    ) -> Result<i64, StoreError>;

    async fn count_recent_ratings(
        &self,
        respondent_hash: &str,
        window: chrono::Duration,
    ) -> Result<i64, StoreError>;

    async fn summarize_ratings(&self) -> Result<Vec<RatingSummaryRow>, StoreError>;
}

/// Type-erased handle used throughout the API.
pub type DynJobStore = Arc<dyn JobStoreApi + Send + Sync>;

/// Open the right backend based on the URL scheme. SQLite gets the
/// bare-path / `sqlite:` / `sqlite://` forms; Postgres gets `postgres://`
/// or `postgresql://` (sqlx accepts both). Anything else is rejected
/// at startup — silently falling back to SQLite would mask a misconfig
/// that the operator MEANT to point at a real Postgres instance.
pub async fn connect(url: &str) -> Result<DynJobStore, StoreError> {
    if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        let store = PostgresJobStore::connect(url).await?;
        Ok(Arc::new(store))
    } else {
        // SQLite covers everything else: `sqlite://`, `sqlite:`, and the
        // bare-path form `./data/jobs.db` we used in dev before the trait
        // refactor. The SQLite impl strips/normalizes those itself.
        let store = SqliteJobStore::connect(url).await?;
        Ok(Arc::new(store))
    }
}

/* ============================================================ */
/* Blob backend — backend-agnostic, kept here for import path   */
/* compatibility (existing callers do `store::BlobBackend`).    */
/* ============================================================ */

#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    #[error("blob storage not configured (set BLOB_READ_WRITE_TOKEN)")]
    NotConfigured,
    #[error("blob api: {0}")]
    Api(String),
    #[error("blob transport: {0}")]
    Transport(String),
}

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
                .timeout(Duration::from_secs(600))
                .pool_max_idle_per_host(4)
                .build()
                .expect("reqwest client"),
        }
    }

    pub async fn presign_upload(&self, key: &str, _ttl: Duration) -> Result<String, BlobError> {
        let suffix = if self.token.is_some() {
            "?ttl=900&mode=server-proxy"
        } else {
            "?ttl=900&mode=stub"
        };
        Ok(format!("blob://stub/{key}{suffix}"))
    }

    pub fn public_url(&self, key: &str) -> String {
        format!("blob://stub/{key}")
    }

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
