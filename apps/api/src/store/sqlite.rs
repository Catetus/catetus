//! SQLite-backed `JobStoreApi` implementation.
//!
//! Lifted verbatim from the pre-refactor `store.rs` and reshaped to fit
//! the trait. Keeping the SQL byte-for-byte identical is deliberate: the
//! design-partner deploy is on SQLite today, and any behavior drift here
//! would be a silent regression. The trait-level tests in
//! `apps/api/tests/store_trait.rs` exercise this impl alongside the
//! Postgres one to keep the two backends honest.

use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use super::{
    DateTime, Job, JobStatus, JobStoreApi, RatingSummaryRow, StoreError, TeamSignupRow, Tier,
};

/// SQLite-backed job store. Single-file, WAL-journalled, suitable for the
/// single-instance Fly deploy.
#[derive(Clone)]
pub struct SqliteJobStore {
    pool: SqlitePool,
}

impl SqliteJobStore {
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

        sqlx::migrate!("./migrations/sqlite").run(&pool).await?;
        Ok(Self { pool })
    }

    /// In-memory database. Used by unit + integration tests; also handy
    /// for ad-hoc dry-runs that don't want to touch disk. The same
    /// `migrations/sqlite/` directory is replayed so callers get the full
    /// schema (including the billing_events ledger).
    pub async fn in_memory() -> Result<Self, StoreError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::from_str("sqlite::memory:")?.create_if_missing(true),
            )
            .await?;
        sqlx::migrate!("./migrations/sqlite").run(&pool).await?;
        Ok(Self { pool })
    }

    /// Direct pool accessor — kept inherent (not part of the trait) so
    /// SQLite-only tests like `tests/checkout.rs`'s plaintext-leak scan
    /// can run raw SQL. The trait surface stays narrow on purpose; if a
    /// test wants raw SQL it must concretely depend on one backend.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[async_trait]
impl JobStoreApi for SqliteJobStore {
    async fn insert(&self, job: &Job) -> Result<(), StoreError> {
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

    async fn update(&self, job: &Job) -> Result<(), StoreError> {
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

    async fn get(&self, id: &Uuid) -> Result<Option<Job>, StoreError> {
        let row = sqlx::query("SELECT * FROM jobs WHERE id = ?1")
            .bind(id.to_string())
            .fetch_optional(&self.pool)
            .await?;
        row.map(row_to_job).transpose()
    }

    async fn list_by_batch(&self, batch_id: &Uuid) -> Result<Vec<Job>, StoreError> {
        let rows = sqlx::query("SELECT * FROM jobs WHERE batch_id = ?1 ORDER BY created_at ASC")
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

    async fn mark_billing_event_posted(
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

    async fn get_team_signup_by_session(
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

    async fn mark_team_signup_revealed(&self, stripe_session_id: &str) -> Result<bool, StoreError> {
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

    async fn insert_rating(
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

    async fn count_recent_ratings(
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

    async fn summarize_ratings(&self) -> Result<Vec<RatingSummaryRow>, StoreError> {
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

    /// Best-effort audit-event write.
    #[allow(clippy::too_many_arguments)]
    async fn insert_audit_event(
        &self,
        key_prefix: &str,
        route: &str,
        method: &str,
        status: u16,
        body_size: u64,
        duration_ms: u64,
        error: Option<&str>,
    ) -> Result<(), StoreError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT INTO audit_events
                (id, key_prefix, route, method, status, body_size, duration_ms, error, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
        )
        .bind(id)
        .bind(key_prefix)
        .bind(route)
        .bind(method)
        .bind(status as i64)
        .bind(body_size as i64)
        .bind(duration_ms as i64)
        .bind(error)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Last `limit` audit events, newest first.
    async fn list_audit_events(&self, limit: u32) -> Result<Vec<super::AuditEvent>, StoreError> {
        let rows = sqlx::query(
            "SELECT id, key_prefix, route, method, status, body_size, duration_ms, error, created_at \
             FROM audit_events ORDER BY created_at DESC LIMIT ?1",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit).collect()
    }

    /// Last `limit` audit events for a single `key_prefix`, newest
    /// first. Same shape as `list_audit_events` but with a WHERE clause
    /// — used by `GET /v1/me/usage` to scope the customer dashboard
    /// view to the requester's own key. The (key_prefix, created_at)
    /// pair is already indexed by migration 0005 so this is cheap
    /// even at six-figure row counts.
    async fn list_audit_events_by_prefix(
        &self,
        key_prefix: &str,
        limit: u32,
    ) -> Result<Vec<super::AuditEvent>, StoreError> {
        let rows = sqlx::query(
            "SELECT id, key_prefix, route, method, status, body_size, duration_ms, error, created_at \
             FROM audit_events WHERE key_prefix = ?1 ORDER BY created_at DESC LIMIT ?2",
        )
        .bind(key_prefix)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(row_to_audit).collect()
    }
}

fn row_to_audit(row: sqlx::sqlite::SqliteRow) -> Result<super::AuditEvent, StoreError> {
    let status: i64 = row.try_get("status")?;
    let body_size: i64 = row.try_get("body_size")?;
    let duration_ms: i64 = row.try_get("duration_ms")?;
    let created_at_str: String = row.try_get("created_at")?;
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map_err(|e| StoreError::Decode(e.to_string()))?
        .with_timezone(&Utc);
    Ok(super::AuditEvent {
        id: row.try_get("id")?,
        key_prefix: row.try_get("key_prefix")?,
        route: row.try_get("route")?,
        method: row.try_get("method")?,
        status: status as u16,
        body_size: body_size.max(0) as u64,
        duration_ms: duration_ms.max(0) as u64,
        error: row.try_get("error")?,
        created_at,
    })
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
        let store = SqliteJobStore::in_memory().await.expect("store");
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
        assert_eq!(
            got.output_url.as_deref(),
            Some("https://example.com/out.glb")
        );
        assert_eq!(got.tier, Tier::Paid);
    }

    #[tokio::test]
    async fn list_by_batch_returns_grouped_jobs() {
        let store = SqliteJobStore::in_memory().await.expect("store");
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
        let store = SqliteJobStore::in_memory().await.expect("store");
        let got = store.get(&Uuid::new_v4()).await.expect("get");
        assert!(got.is_none());
    }
}
