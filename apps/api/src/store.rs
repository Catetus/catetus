//! Job + blob storage. Jobs persist to SQLite via sqlx; blob bytes proxy
//! through Vercel Blob. SQLite is enough for the single-instance droplet
//! deploy — swap to Postgres by changing the `Pool<Sqlite>` to a generic
//! `Pool<Any>` once we outgrow it; the call sites here are the only place
//! that touches the concrete backend.

use std::str::FromStr;
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Body;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, Sqlite, SqlitePool};
use uuid::Uuid;

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
    fn as_db_str(self) -> &'static str {
        match self {
            JobStatus::AwaitingUpload => "awaiting-upload",
            JobStatus::Uploading => "uploading",
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Done => "done",
            JobStatus::Error => "error",
        }
    }
    fn from_db_str(s: &str) -> Option<Self> {
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
    fn as_db_str(self) -> &'static str {
        match self {
            Tier::Free => "free",
            Tier::Paid => "paid",
        }
    }
    fn from_db_str(s: &str) -> Option<Self> {
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
    /// Stripe customer (`cus_xxx`) that owns this job, if the bearer key
    /// was associated with one at creation time. `None` means "free /
    /// untracked" — the billing module short-circuits on these and never
    /// emits meter events. Stored as a string because Stripe customer IDs
    /// aren't UUIDs and we never parse them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub customer_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub error: Option<String>,
}

/// SQLite-backed job store.
#[derive(Clone)]
pub struct JobStore {
    pool: SqlitePool,
}

/// One row of the per-pair rating summary returned by
/// `JobStore::summarize_ratings`. Surfaced verbatim through
/// `GET /v1/ratings/summary` so the v0.4 training pipeline can pull
/// numbers without re-running the aggregation.
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

/// One row of the `team_signups` ledger. Returned by
/// `JobStore::get_team_signup_by_session` so the checkout module can
/// gate `/v1/checkout/reveal` on the `claim_token` + `key_revealed_at`
/// pair without re-running the query.
#[derive(Debug, Clone)]
pub struct TeamSignupRow {
    pub claim_token: String,
    pub key_prefix: String,
    pub key_hash: String,
    pub stripe_customer_id: String,
    pub email: String,
    /// `Some(rfc3339)` if the plaintext has already been revealed —
    /// the `/reveal` endpoint MUST refuse the second hit.
    pub key_revealed_at: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("migrate: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("decoding row: {0}")]
    Decode(String),
}

impl JobStore {
    /// Connect to (and migrate) the SQLite database at `url`. Accepts the
    /// usual `sqlite:` URLs plus the bare path form `./data/jobs.db` for
    /// convenience in dev. Creates the file if missing.
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let opts = if let Some(rest) = url.strip_prefix("sqlite://") {
            SqliteConnectOptions::from_str(&format!("sqlite://{rest}"))?
        } else if let Some(rest) = url.strip_prefix("sqlite:") {
            SqliteConnectOptions::from_str(&format!("sqlite:{rest}"))?
        } else {
            SqliteConnectOptions::new().filename(url)
        }
        .create_if_missing(true)
        // WAL keeps reads non-blocking against writers, which matters once
        // the polling client + the worker callback hit the same row. NORMAL
        // sync is the standard recommendation for WAL and survives crash.
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(Duration::from_secs(5));

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .acquire_timeout(Duration::from_secs(5))
            .connect_with(opts)
            .await?;

        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    /// In-memory database. Used by unit + integration tests; also handy
    /// for ad-hoc dry-runs that don't want to touch disk. The same
    /// `migrations/` directory is replayed so callers get the full
    /// schema (including the billing_events ledger). Not gated on
    /// `#[cfg(test)]` so integration tests under `tests/` can reach it.
    pub async fn in_memory() -> Result<Self, StoreError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::from_str("sqlite::memory:")?
                    .create_if_missing(true),
            )
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    /// Direct pool accessor for read-only listing endpoints (e.g. `/v1/jobs`
    /// admin view). Kept public so we don't grow a new method per query.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn insert(&self, job: &Job) -> Result<(), StoreError> {
        let now = Utc::now();
        sqlx::query(
            r#"
            INSERT INTO jobs (
                id, preset, filename, size_bytes, label, status,
                blob_key, blob_url, source_url, upload_size_bytes,
                output_url, preview_url, phase, percent, webhook_url,
                batch_id, tier, customer_id, created_at, updated_at, error
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                      ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
            "#,
        )
        .bind(job.id.to_string())
        .bind(&job.preset)
        .bind(&job.filename)
        .bind(job.size_bytes as i64)
        .bind(&job.label)
        .bind(job.status.as_db_str())
        .bind(&job.blob_key)
        .bind(&job.blob_url)
        .bind(&job.source_url)
        .bind(job.upload_size_bytes.map(|v| v as i64))
        .bind(&job.output_url)
        .bind(&job.preview_url)
        .bind(&job.phase)
        .bind(job.percent)
        .bind(&job.webhook_url)
        .bind(job.batch_id.map(|b| b.to_string()))
        .bind(job.tier.as_db_str())
        .bind(&job.customer_id)
        .bind(job.created_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(&job.error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update(&self, job: &Job) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE jobs SET
                preset = ?2, filename = ?3, size_bytes = ?4, label = ?5,
                status = ?6, blob_key = ?7, blob_url = ?8, source_url = ?9,
                upload_size_bytes = ?10, output_url = ?11, preview_url = ?12,
                phase = ?13, percent = ?14, webhook_url = ?15,
                batch_id = ?16, tier = ?17, customer_id = ?18,
                updated_at = ?19, error = ?20
            WHERE id = ?1
            "#,
        )
        .bind(job.id.to_string())
        .bind(&job.preset)
        .bind(&job.filename)
        .bind(job.size_bytes as i64)
        .bind(&job.label)
        .bind(job.status.as_db_str())
        .bind(&job.blob_key)
        .bind(&job.blob_url)
        .bind(&job.source_url)
        .bind(job.upload_size_bytes.map(|v| v as i64))
        .bind(&job.output_url)
        .bind(&job.preview_url)
        .bind(&job.phase)
        .bind(job.percent)
        .bind(&job.webhook_url)
        .bind(job.batch_id.map(|b| b.to_string()))
        .bind(job.tier.as_db_str())
        .bind(&job.customer_id)
        .bind(Utc::now().to_rfc3339())
        .bind(&job.error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get(&self, id: &Uuid) -> Result<Option<Job>, StoreError> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id = ?1")
            .bind(id.to_string())
            .fetch_optional(&self.pool)
            .await?;
        row.map(row_to_job).transpose()
    }

    /// Idempotently claim a (job_id, sku) slot in the billing ledger. Returns
    /// `Ok(true)` if this is a fresh claim (the caller should post the meter
    /// event to Stripe); `Ok(false)` if a row already exists (someone else got
    /// here first — *do not* post another event). Backed by SQLite's UNIQUE
    /// constraint on (job_id, sku), so concurrent callers serialize cleanly.
    ///
    /// This is the no-double-charge invariant. Even if the Modal callback
    /// fires twice for the same job (flaky webhooks, retries), only one
    /// caller's INSERT succeeds.
    pub async fn claim_billing_event(
        &self,
        job_id: &Uuid,
        customer_id: &str,
        sku: &str,
        units: u64,
        idempotency_key: &str,
    ) -> Result<bool, StoreError> {
        let row_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            INSERT INTO billing_events
                (id, job_id, customer_id, sku, units, idempotency_key, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(job_id, sku) DO NOTHING
            "#,
        )
        .bind(row_id)
        .bind(job_id.to_string())
        .bind(customer_id)
        .bind(sku)
        .bind(units as i64)
        .bind(idempotency_key)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    /// Stamp the Stripe-side event id onto a previously-claimed billing row.
    /// Best-effort: a missing row (e.g. if the ledger was wiped) is a no-op.
    pub async fn mark_billing_event_posted(
        &self,
        job_id: &Uuid,
        sku: &str,
        stripe_event_id: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE billing_events SET stripe_event_id = ?3 WHERE job_id = ?1 AND sku = ?2",
        )
        .bind(job_id.to_string())
        .bind(sku)
        .bind(stripe_event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Idempotently record a Team-tier Stripe Checkout completion.
    ///
    /// `INSERT … ON CONFLICT(stripe_session_id) DO NOTHING` is the
    /// no-double-issuance gate: a Stripe webhook retry that lands a
    /// second time finds the row already present, gets `Ok(false)`, and
    /// skips the (one-time-only) plaintext-key minting. The first call
    /// owns the (`key_prefix`, `key_hash`) pair and the plaintext that
    /// will be revealed exactly once at `/welcome`.
    ///
    /// Returns `Ok(true)` if this row is freshly inserted (the caller
    /// should also cache the plaintext for the reveal endpoint);
    /// `Ok(false)` if a row already existed for this session.
    pub async fn claim_team_signup(
        &self,
        stripe_session_id: &str,
        stripe_customer_id: &str,
        stripe_subscription_id: Option<&str>,
        email: &str,
        claim_token: &str,
        key_prefix: &str,
        key_hash: &str,
        seats: u32,
    ) -> Result<bool, StoreError> {
        let row_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            INSERT INTO team_signups
                (id, stripe_session_id, stripe_customer_id, stripe_subscription_id,
                 email, claim_token, key_prefix, key_hash, seats, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(stripe_session_id) DO NOTHING
            "#,
        )
        .bind(row_id)
        .bind(stripe_session_id)
        .bind(stripe_customer_id)
        .bind(stripe_subscription_id)
        .bind(email)
        .bind(claim_token)
        .bind(key_prefix)
        .bind(key_hash)
        .bind(seats as i64)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    /// Look up a team signup by Stripe Checkout Session id. Returns
    /// `(claim_token, key_prefix, key_hash, customer_id, key_revealed_at, email)`.
    /// The full row is exposed via a tuple instead of a struct because
    /// only the checkout module touches it; the public surface is
    /// `/v1/checkout/reveal`'s response shape.
    pub async fn get_team_signup_by_session(
        &self,
        stripe_session_id: &str,
    ) -> Result<Option<TeamSignupRow>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT claim_token, key_prefix, key_hash, stripe_customer_id, email, key_revealed_at
            FROM team_signups
            WHERE stripe_session_id = ?1
            "#,
        )
        .bind(stripe_session_id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else { return Ok(None) };
        Ok(Some(TeamSignupRow {
            claim_token: row.try_get("claim_token")?,
            key_prefix: row.try_get("key_prefix")?,
            key_hash: row.try_get("key_hash")?,
            stripe_customer_id: row.try_get("stripe_customer_id")?,
            email: row.try_get("email")?,
            key_revealed_at: row.try_get("key_revealed_at")?,
        }))
    }

    /// Mark the plaintext as revealed. Returns `Ok(true)` if this call
    /// flipped the column (the caller is the legitimate first reveal);
    /// `Ok(false)` if the row was already revealed (the caller MUST
    /// refuse to return the plaintext — this is the "plaintext shown
    /// exactly once" invariant the deliverable names).
    pub async fn mark_team_signup_revealed(
        &self,
        stripe_session_id: &str,
    ) -> Result<bool, StoreError> {
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            UPDATE team_signups
            SET key_revealed_at = ?2
            WHERE stripe_session_id = ?1 AND key_revealed_at IS NULL
            "#,
        )
        .bind(stripe_session_id)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    /// Insert one human pairwise rating from `splatforge.com/rate`. The
    /// caller is expected to have already enforced the 100/hour cap via
    /// `count_recent_ratings`; this method just writes. Returns the
    /// freshly-assigned auto-increment id so the handler can echo it
    /// back to the page (useful for the "your last vote is in" toast).
    pub async fn insert_rating(
        &self,
        scene_id: &str,
        left_preset: &str,
        right_preset: &str,
        winner: &str,
        respondent_hash: &str,
    ) -> Result<i64, StoreError> {
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            INSERT INTO ratings (scene_id, left_preset, right_preset, winner, respondent_hash, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
        )
        .bind(scene_id)
        .bind(left_preset)
        .bind(right_preset)
        .bind(winner)
        .bind(respondent_hash)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    /// Count ratings posted by this respondent in the trailing `window`.
    /// Used to enforce the 100/hour rate limit. Stored as RFC3339 strings,
    /// so the threshold is computed in Rust and compared lexicographically
    /// — that ordering matches numeric ordering for RFC3339 dates as long
    /// as the format is consistent (which it is, because we generate all
    /// rows with `Utc::now().to_rfc3339()`).
    pub async fn count_recent_ratings(
        &self,
        respondent_hash: &str,
        window: chrono::Duration,
    ) -> Result<i64, StoreError> {
        let threshold = (Utc::now() - window).to_rfc3339();
        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ratings WHERE respondent_hash = ?1 AND created_at >= ?2",
        )
        .bind(respondent_hash)
        .bind(threshold)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.0)
    }

    /// Aggregated rating counts for the v0.4 training pipeline. One row
    /// per (scene, left_preset, right_preset) pair with the winner
    /// breakdown. The aggregator deliberately preserves left/right
    /// ordering (rather than normalizing to a canonical pair) so
    /// downstream analysis can detect side-bias before the
    /// Bradley-Terry fit ingests these.
    pub async fn summarize_ratings(&self) -> Result<Vec<RatingSummaryRow>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT scene_id, left_preset, right_preset,
                   SUM(CASE WHEN winner = 'left'  THEN 1 ELSE 0 END) AS left_wins,
                   SUM(CASE WHEN winner = 'right' THEN 1 ELSE 0 END) AS right_wins,
                   SUM(CASE WHEN winner = 'tie'   THEN 1 ELSE 0 END) AS ties,
                   COUNT(*) AS total
            FROM ratings
            GROUP BY scene_id, left_preset, right_preset
            ORDER BY scene_id, left_preset, right_preset
            "#,
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(RatingSummaryRow {
                scene_id: row.try_get("scene_id")?,
                left_preset: row.try_get("left_preset")?,
                right_preset: row.try_get("right_preset")?,
                left_wins: row.try_get::<i64, _>("left_wins")?,
                right_wins: row.try_get::<i64, _>("right_wins")?,
                ties: row.try_get::<i64, _>("ties")?,
                total: row.try_get::<i64, _>("total")?,
            });
        }
        Ok(out)
    }

    /// All jobs in a batch, ordered by insertion. Used by the batch-status
    /// endpoint so a 40-tile client doesn't have to poll 40 IDs.
    pub async fn list_by_batch(&self, batch_id: &Uuid) -> Result<Vec<Job>, StoreError> {
        let rows = sqlx::query("SELECT * FROM jobs WHERE batch_id = ?1 ORDER BY created_at ASC")
            .bind(batch_id.to_string())
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter().map(row_to_job).collect()
    }
}

fn row_to_job(row: sqlx::sqlite::SqliteRow) -> Result<Job, StoreError> {
    let id_str: String = row.try_get("id")?;
    let id = Uuid::parse_str(&id_str).map_err(|e| StoreError::Decode(e.to_string()))?;

    let status_str: String = row.try_get("status")?;
    let status = JobStatus::from_db_str(&status_str)
        .ok_or_else(|| StoreError::Decode(format!("unknown status: {status_str}")))?;

    let tier_str: String = row.try_get("tier")?;
    let tier = Tier::from_db_str(&tier_str)
        .ok_or_else(|| StoreError::Decode(format!("unknown tier: {tier_str}")))?;

    let batch_id_opt: Option<String> = row.try_get("batch_id")?;
    let batch_id = batch_id_opt
        .map(|s| Uuid::parse_str(&s).map_err(|e| StoreError::Decode(e.to_string())))
        .transpose()?;

    let created_at_str: String = row.try_get("created_at")?;
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map_err(|e| StoreError::Decode(e.to_string()))?
        .with_timezone(&Utc);

    let size_bytes: i64 = row.try_get("size_bytes")?;
    let upload_size_bytes: Option<i64> = row.try_get("upload_size_bytes")?;

    Ok(Job {
        id,
        preset: row.try_get("preset")?,
        filename: row.try_get("filename")?,
        size_bytes: size_bytes as u64,
        label: row.try_get("label")?,
        status,
        blob_key: row.try_get("blob_key")?,
        blob_url: row.try_get("blob_url")?,
        source_url: row.try_get("source_url")?,
        upload_size_bytes: upload_size_bytes.map(|v| v as u64),
        output_url: row.try_get("output_url")?,
        preview_url: row.try_get("preview_url")?,
        phase: row.try_get("phase")?,
        percent: row.try_get("percent")?,
        webhook_url: row.try_get("webhook_url")?,
        batch_id,
        tier,
        customer_id: row.try_get("customer_id")?,
        created_at,
        error: row.try_get("error")?,
    })
}

// Avoid "unused" lint on Sqlite alias when only used in path positions.
const _: Option<Sqlite> = None;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_job() -> Job {
        Job {
            id: Uuid::new_v4(),
            preset: "web-mobile".into(),
            filename: "scene.ply".into(),
            size_bytes: 1024,
            label: Some("smoke".into()),
            status: JobStatus::AwaitingUpload,
            blob_key: "jobs/x/scene.ply".into(),
            blob_url: None,
            source_url: None,
            upload_size_bytes: None,
            output_url: None,
            preview_url: None,
            phase: None,
            percent: None,
            webhook_url: None,
            batch_id: None,
            tier: Tier::Free,
            customer_id: None,
            created_at: Utc::now(),
            error: None,
        }
    }

    #[tokio::test]
    async fn roundtrip_insert_get_update() {
        let store = JobStore::in_memory().await.expect("store");
        let mut job = sample_job();
        store.insert(&job).await.expect("insert");
        let got = store.get(&job.id).await.expect("get").expect("present");
        assert_eq!(got.id, job.id);
        assert_eq!(got.preset, "web-mobile");
        assert_eq!(got.tier, Tier::Free);
        assert_eq!(got.status, JobStatus::AwaitingUpload);

        job.status = JobStatus::Done;
        job.output_url = Some("https://example.com/out.glb".into());
        job.tier = Tier::Paid;
        store.update(&job).await.expect("update");
        let got = store.get(&job.id).await.expect("get").expect("present");
        assert_eq!(got.status, JobStatus::Done);
        assert_eq!(got.output_url.as_deref(), Some("https://example.com/out.glb"));
        assert_eq!(got.tier, Tier::Paid);
    }

    #[tokio::test]
    async fn list_by_batch_returns_grouped_jobs() {
        let store = JobStore::in_memory().await.expect("store");
        let batch_id = Uuid::new_v4();
        for i in 0..3 {
            let mut j = sample_job();
            j.batch_id = Some(batch_id);
            j.filename = format!("tile-{i}.ply");
            store.insert(&j).await.expect("insert");
        }
        // One job NOT in the batch — must not be returned.
        let other = sample_job();
        store.insert(&other).await.expect("insert");

        let batch = store.list_by_batch(&batch_id).await.expect("list");
        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|j| j.batch_id == Some(batch_id)));
    }

    #[tokio::test]
    async fn missing_id_returns_none() {
        let store = JobStore::in_memory().await.expect("store");
        let got = store.get(&Uuid::new_v4()).await.expect("get");
        assert!(got.is_none());
    }
}
