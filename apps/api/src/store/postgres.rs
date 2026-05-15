//! Postgres-backed `JobStoreApi` implementation.
//!
//! Mirror of `store/sqlite.rs` against a Postgres pool. Same `Job` shape,
//! same trait surface, same `INSERT … ON CONFLICT … DO NOTHING` idiom for
//! the no-double-charge invariant. The differences are mechanical:
//!
//!   * Positional parameters are `$1, $2, …` instead of `?1, ?2, …`.
//!   * `last_insert_rowid()` doesn't exist on Postgres — `insert_rating`
//!     uses `INSERT … RETURNING id` instead.
//!   * `BIGINT` is the canonical sqlx `i64` mapping (SQLite's INTEGER is
//!     a width-flexible alias for the same Rust type, which is why the
//!     application code carries no `as i64` casts beyond what's already
//!     there).
//!
//! See `apps/api/STORE-BACKENDS.md` for the operator-side cutover playbook
//! (the `pg_dump` shape, env vars, etc).

use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use super::{
    DateTime, Job, JobStatus, JobStoreApi, RatingSummaryRow, StoreError, TeamSignupRow, Tier,
};

/// Postgres-backed job store. Accepts `postgres://` and `postgresql://`
/// URLs.
#[derive(Clone)]
pub struct PostgresJobStore {
    pool: PgPool,
}

impl PostgresJobStore {
    /// Connect to (and migrate) the Postgres database at `url`. The
    /// `migrations/postgres/` directory is replayed on every startup;
    /// sqlx tracks applied migrations in a `_sqlx_migrations` table so
    /// re-running is a no-op.
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new()
            .max_connections(16)
            .acquire_timeout(Duration::from_secs(10))
            .connect(url)
            .await?;
        sqlx::migrate!("./migrations/postgres").run(&pool).await?;
        Ok(Self { pool })
    }

    /// Direct pool accessor — kept inherent (not on the trait) for the
    /// same reason `SqliteJobStore::pool` is: rarely-needed raw-SQL
    /// inspection at the test layer. Production code MUST go through
    /// the trait.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[async_trait]
impl JobStoreApi for PostgresJobStore {
    async fn insert(&self, job: &Job) -> Result<(), StoreError> {
        let now = Utc::now();
        sqlx::query(
            r#"
            INSERT INTO jobs (
                id, preset, filename, size_bytes, label, status,
                blob_key, blob_url, source_url, upload_size_bytes,
                output_url, preview_url, phase, percent, webhook_url,
                batch_id, tier, customer_id, created_at, updated_at, error
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                      $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
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
        // percent maps to DOUBLE PRECISION; cast f32 -> f64 here so the
        // bound type matches the column type and sqlx doesn't try to
        // narrow on the wire.
        .bind(job.percent.map(|p| p as f64))
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

    async fn update(&self, job: &Job) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            UPDATE jobs SET
                preset = $2, filename = $3, size_bytes = $4, label = $5,
                status = $6, blob_key = $7, blob_url = $8, source_url = $9,
                upload_size_bytes = $10, output_url = $11, preview_url = $12,
                phase = $13, percent = $14, webhook_url = $15,
                batch_id = $16, tier = $17, customer_id = $18,
                updated_at = $19, error = $20
            WHERE id = $1
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
        .bind(job.percent.map(|p| p as f64))
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

    async fn get(&self, id: &Uuid) -> Result<Option<Job>, StoreError> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id = $1")
            .bind(id.to_string())
            .fetch_optional(&self.pool)
            .await?;
        row.map(row_to_job).transpose()
    }

    async fn list_by_batch(&self, batch_id: &Uuid) -> Result<Vec<Job>, StoreError> {
        let rows = sqlx::query("SELECT * FROM jobs WHERE batch_id = $1 ORDER BY created_at ASC")
            .bind(batch_id.to_string())
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter().map(row_to_job).collect()
    }

    async fn claim_billing_event(
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
            VALUES ($1, $2, $3, $4, $5, $6, $7)
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
        // Postgres's `rows_affected()` for ON CONFLICT DO NOTHING returns
        // 0 when the conflict path was taken — same semantic as the
        // SQLite impl. This is what makes the no-double-charge invariant
        // portable across backends without an explicit SELECT-then-INSERT
        // dance.
        Ok(res.rows_affected() == 1)
    }

    async fn mark_billing_event_posted(
        &self,
        job_id: &Uuid,
        sku: &str,
        stripe_event_id: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE billing_events SET stripe_event_id = $3 WHERE job_id = $1 AND sku = $2",
        )
        .bind(job_id.to_string())
        .bind(sku)
        .bind(stripe_event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

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
    ) -> Result<bool, StoreError> {
        let row_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            INSERT INTO team_signups
                (id, stripe_session_id, stripe_customer_id, stripe_subscription_id,
                 email, claim_token, key_prefix, key_hash, seats, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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

    async fn get_team_signup_by_session(
        &self,
        stripe_session_id: &str,
    ) -> Result<Option<TeamSignupRow>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT claim_token, key_prefix, key_hash, stripe_customer_id, email, key_revealed_at
            FROM team_signups
            WHERE stripe_session_id = $1
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

    async fn mark_team_signup_revealed(
        &self,
        stripe_session_id: &str,
    ) -> Result<bool, StoreError> {
        let now = Utc::now().to_rfc3339();
        let res = sqlx::query(
            r#"
            UPDATE team_signups
            SET key_revealed_at = $2
            WHERE stripe_session_id = $1 AND key_revealed_at IS NULL
            "#,
        )
        .bind(stripe_session_id)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    async fn insert_rating(
        &self,
        scene_id: &str,
        left_preset: &str,
        right_preset: &str,
        winner: &str,
        respondent_hash: &str,
    ) -> Result<i64, StoreError> {
        // Postgres has no session-level last-insert-id. Use RETURNING to
        // pull the autogen BIGSERIAL out of the INSERT directly. The
        // SQLite impl uses last_insert_rowid() — same end value (i64),
        // different mechanism.
        let now = Utc::now().to_rfc3339();
        let row: (i64,) = sqlx::query_as(
            r#"
            INSERT INTO ratings (scene_id, left_preset, right_preset, winner, respondent_hash, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            "#,
        )
        .bind(scene_id)
        .bind(left_preset)
        .bind(right_preset)
        .bind(winner)
        .bind(respondent_hash)
        .bind(now)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.0)
    }

    async fn count_recent_ratings(
        &self,
        respondent_hash: &str,
        window: chrono::Duration,
    ) -> Result<i64, StoreError> {
        let threshold = (Utc::now() - window).to_rfc3339();
        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM ratings WHERE respondent_hash = $1 AND created_at >= $2",
        )
        .bind(respondent_hash)
        .bind(threshold)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.0)
    }

    async fn summarize_ratings(&self) -> Result<Vec<RatingSummaryRow>, StoreError> {
        // SUM(CASE WHEN …) returns a NUMERIC (i.e. arbitrary precision)
        // in Postgres, which sqlx does NOT auto-decode to i64. Wrap each
        // sum in COALESCE+CAST so the column comes back as BIGINT, which
        // does decode cleanly. The SQLite version doesn't need this
        // because SQLite types are dynamic and SUM-of-INT stays INT.
        let rows = sqlx::query(
            r#"
            SELECT scene_id, left_preset, right_preset,
                   COALESCE(SUM(CASE WHEN winner = 'left'  THEN 1 ELSE 0 END), 0)::BIGINT AS left_wins,
                   COALESCE(SUM(CASE WHEN winner = 'right' THEN 1 ELSE 0 END), 0)::BIGINT AS right_wins,
                   COALESCE(SUM(CASE WHEN winner = 'tie'   THEN 1 ELSE 0 END), 0)::BIGINT AS ties,
                   COUNT(*)::BIGINT AS total
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
}

fn row_to_job(row: PgRow) -> Result<Job, StoreError> {
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
    // Postgres DOUBLE PRECISION decodes to f64; narrow to f32 to keep
    // the shared Job type unchanged across backends.
    let percent_f64: Option<f64> = row.try_get("percent")?;

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
        percent: percent_f64.map(|v| v as f32),
        webhook_url: row.try_get("webhook_url")?,
        batch_id,
        tier,
        customer_id: row.try_get("customer_id")?,
        created_at,
        error: row.try_get("error")?,
    })
}
