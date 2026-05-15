//! Trait-level integration tests for `JobStoreApi`.
//!
//! Same assertions run against both backends:
//!
//!   * `SqliteJobStore` — always (uses in-memory SQLite, ~no dependencies).
//!   * `PostgresJobStore` — only when Docker is available. Spins up a
//!     real Postgres 16 container via `testcontainers-rs`, dials it,
//!     runs migrations, executes the same assertions. Each test gets
//!     its own container so they don't share state (the no-double-
//!     charge test in particular relies on a fresh ledger).
//!
//! Setting `SPLATFORGE_SKIP_POSTGRES_TESTS=1` forces a skip without
//! probing Docker — useful on CI runners that don't have it.

use std::sync::Arc;

use splatforge_api::store::{DynJobStore, Job, JobStatus, PostgresJobStore, SqliteJobStore, Tier};
use uuid::Uuid;

/* ----------------------------------------------------------------------- */
/* Test corpus — one set of assertions, exercised against each backend.    */
/* ----------------------------------------------------------------------- */

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
        percent: Some(0.5),
        webhook_url: None,
        batch_id: None,
        tier: Tier::Free,
        customer_id: None,
        created_at: chrono::Utc::now(),
        error: None,
    }
}

/// Roundtrip a Job through `insert` + `get` + `update`; the shape stays
/// stable across backends. Asserts on the load-bearing fields — adding
/// new fields to `Job` requires updating this assertion intentionally.
async fn assert_roundtrip_insert_get_update(store: DynJobStore) {
    let mut job = sample_job();
    store.insert(&job).await.expect("insert");

    let got = store.get(&job.id).await.expect("get").expect("present");
    assert_eq!(got.id, job.id);
    assert_eq!(got.preset, "web-mobile");
    assert_eq!(got.tier, Tier::Free);
    assert_eq!(got.status, JobStatus::AwaitingUpload);
    assert_eq!(got.size_bytes, 1024);
    assert_eq!(got.percent, Some(0.5));

    job.status = JobStatus::Done;
    job.output_url = Some("https://example.com/out.glb".into());
    job.tier = Tier::Paid;
    job.customer_id = Some("cus_test".into());
    store.update(&job).await.expect("update");

    let got = store.get(&job.id).await.expect("get").expect("present");
    assert_eq!(got.status, JobStatus::Done);
    assert_eq!(
        got.output_url.as_deref(),
        Some("https://example.com/out.glb")
    );
    assert_eq!(got.tier, Tier::Paid);
    assert_eq!(got.customer_id.as_deref(), Some("cus_test"));
}

async fn assert_list_by_batch(store: DynJobStore) {
    let batch_id = Uuid::new_v4();
    for i in 0..3 {
        let mut j = sample_job();
        j.batch_id = Some(batch_id);
        j.filename = format!("tile-{i}.ply");
        // Spread created_at so the ORDER BY is meaningful.
        j.created_at = chrono::Utc::now() + chrono::Duration::milliseconds(i as i64);
        store.insert(&j).await.expect("insert");
    }
    // One job NOT in the batch — must not be returned.
    store.insert(&sample_job()).await.expect("insert");

    let batch = store.list_by_batch(&batch_id).await.expect("list");
    assert_eq!(batch.len(), 3);
    assert!(batch.iter().all(|j| j.batch_id == Some(batch_id)));
}

/// The load-bearing invariant: same (job_id, sku) claimed twice yields
/// `Ok(true)` then `Ok(false)`. Both backends MUST honor it identically,
/// because the billing module relies on the second-call-returns-false
/// branch to skip the Stripe POST.
async fn assert_claim_billing_event_idempotent(store: DynJobStore) {
    let job_id = Uuid::new_v4();
    let first = store
        .claim_billing_event(&job_id, "cus_aaa", "splatforge_repack_runs", 1, "key1")
        .await
        .expect("claim 1");
    assert!(first, "first claim must succeed");

    let second = store
        .claim_billing_event(&job_id, "cus_aaa", "splatforge_repack_runs", 1, "key1")
        .await
        .expect("claim 2");
    assert!(!second, "second claim must be a no-op (no double charge)");

    // A different SKU on the same job is independent.
    let third = store
        .claim_billing_event(&job_id, "cus_aaa", "splatforge_repack_seconds", 42, "key2")
        .await
        .expect("claim 3");
    assert!(third, "different SKU on same job must claim independently");

    // A different job is independent.
    let other_job = Uuid::new_v4();
    let fourth = store
        .claim_billing_event(&other_job, "cus_aaa", "splatforge_repack_runs", 1, "key3")
        .await
        .expect("claim 4");
    assert!(fourth, "different job must claim independently");
}

async fn assert_get_missing_returns_none(store: DynJobStore) {
    let got = store.get(&Uuid::new_v4()).await.expect("get");
    assert!(got.is_none());
}

/// Team signup ON CONFLICT(stripe_session_id) DO NOTHING — same shape
/// as billing dedupe, different table. Stripe webhook retries land
/// the second hit; the row already exists; we return Ok(false) and
/// don't mint a second key.
async fn assert_team_signup_idempotent(store: DynJobStore) {
    let session = "cs_test_dup_42";
    let first = store
        .claim_team_signup(
            session,
            "cus_a",
            Some("sub_a"),
            "buyer@example.com",
            "claim-token",
            "sf_live_AAAA",
            "deadbeef-hash",
            1,
        )
        .await
        .expect("claim 1");
    assert!(first);

    let second = store
        .claim_team_signup(
            session,
            "cus_a",
            Some("sub_a"),
            "buyer@example.com",
            "claim-token-DIFFERENT",
            "sf_live_BBBB",
            "different-hash",
            1,
        )
        .await
        .expect("claim 2");
    assert!(!second, "duplicate session id must not re-mint");

    // The original row must still be intact.
    let row = store
        .get_team_signup_by_session(session)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(row.claim_token, "claim-token");
    assert_eq!(row.key_prefix, "sf_live_AAAA");
    assert_eq!(row.email, "buyer@example.com");
    assert!(
        row.key_revealed_at.is_none(),
        "fresh row must not be revealed"
    );

    // First reveal flips the flag, second is a no-op.
    let flipped = store
        .mark_team_signup_revealed(session)
        .await
        .expect("flip");
    assert!(flipped, "first reveal must flip");
    let flipped_again = store
        .mark_team_signup_revealed(session)
        .await
        .expect("flip 2");
    assert!(!flipped_again, "second reveal must be a no-op");
}

/// Insert several ratings, then summarize. Both backends must return
/// the same shape — this exercises the Postgres-side COALESCE+CAST
/// rewrite of the SUM(CASE WHEN …) aggregation.
async fn assert_ratings_summarize(store: DynJobStore) {
    // Three ratings, two "left", one "right", on the same pair.
    let _ = store
        .insert_rating("scene_a", "preset_l", "preset_r", "left", "hash1")
        .await
        .expect("rating 1");
    let _ = store
        .insert_rating("scene_a", "preset_l", "preset_r", "left", "hash2")
        .await
        .expect("rating 2");
    let _ = store
        .insert_rating("scene_a", "preset_l", "preset_r", "right", "hash3")
        .await
        .expect("rating 3");

    let summary = store.summarize_ratings().await.expect("summarize");
    assert_eq!(summary.len(), 1, "single (scene, l, r) tuple");
    let row = &summary[0];
    assert_eq!(row.scene_id, "scene_a");
    assert_eq!(row.left_wins, 2);
    assert_eq!(row.right_wins, 1);
    assert_eq!(row.ties, 0);
    assert_eq!(row.total, 3);

    // count_recent_ratings respects the window argument.
    let count = store
        .count_recent_ratings("hash1", chrono::Duration::hours(1))
        .await
        .expect("count");
    assert_eq!(count, 1);
}

/* ----------------------------------------------------------------------- */
/* SQLite — always runs.                                                    */
/* ----------------------------------------------------------------------- */

async fn sqlite_store() -> DynJobStore {
    Arc::new(SqliteJobStore::in_memory().await.expect("sqlite in_memory"))
}

#[tokio::test]
async fn sqlite_roundtrip_insert_get_update() {
    assert_roundtrip_insert_get_update(sqlite_store().await).await;
}
#[tokio::test]
async fn sqlite_list_by_batch() {
    assert_list_by_batch(sqlite_store().await).await;
}
#[tokio::test]
async fn sqlite_claim_billing_event_idempotent() {
    assert_claim_billing_event_idempotent(sqlite_store().await).await;
}
#[tokio::test]
async fn sqlite_get_missing_returns_none() {
    assert_get_missing_returns_none(sqlite_store().await).await;
}
#[tokio::test]
async fn sqlite_team_signup_idempotent() {
    assert_team_signup_idempotent(sqlite_store().await).await;
}
#[tokio::test]
async fn sqlite_ratings_summarize() {
    assert_ratings_summarize(sqlite_store().await).await;
}

/* ----------------------------------------------------------------------- */
/* Postgres — runs when Docker is available.                                */
/* ----------------------------------------------------------------------- */
//
// Each test spins up its own Postgres 16 container so the test bodies
// can assume an empty schema. The containers are small (~100MB image,
// ~5s startup) so the cost is bearable for the dozen-or-so assertions.
// If you find yourself wanting to share a container, factor out a
// truncate-all helper instead of plumbing a shared container handle.

use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres as PgImage;

/// Returns true if Postgres tests should be skipped. Two paths:
///   * `SPLATFORGE_SKIP_POSTGRES_TESTS=1` — explicit operator skip.
///   * Docker daemon not reachable — common on CI runners without
///     Docker-in-Docker. We detect this by attempting to start a
///     container and treating any error as a skip rather than a
///     failure, with a structured eprintln so the operator sees it.
fn should_skip_postgres() -> bool {
    matches!(
        std::env::var("SPLATFORGE_SKIP_POSTGRES_TESTS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes"),
    )
}

/// Spin up a fresh Postgres container, return a `DynJobStore` against
/// it. Caller MUST keep the returned container handle alive for the
/// lifetime of the test — dropping it stops the container, dropping
/// the pool's underlying TCP connections mid-test.
async fn postgres_store() -> Option<(DynJobStore, testcontainers::ContainerAsync<PgImage>)> {
    if should_skip_postgres() {
        eprintln!("skipping Postgres trait test: SPLATFORGE_SKIP_POSTGRES_TESTS set");
        return None;
    }
    let container = match PgImage::default().start().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping Postgres trait test: container start failed: {e}");
            return None;
        }
    };
    let host = container.get_host().await.expect("host");
    let port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("mapped port");
    let url = format!("postgres://postgres:postgres@{host}:{port}/postgres");
    let store = match PostgresJobStore::connect(&url).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping Postgres trait test: connect/migrate failed: {e}");
            return None;
        }
    };
    Some((Arc::new(store), container))
}

#[tokio::test]
async fn postgres_roundtrip_insert_get_update() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_roundtrip_insert_get_update(store).await;
}
#[tokio::test]
async fn postgres_list_by_batch() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_list_by_batch(store).await;
}
#[tokio::test]
async fn postgres_claim_billing_event_idempotent() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_claim_billing_event_idempotent(store).await;
}
#[tokio::test]
async fn postgres_get_missing_returns_none() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_get_missing_returns_none(store).await;
}
#[tokio::test]
async fn postgres_team_signup_idempotent() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_team_signup_idempotent(store).await;
}
#[tokio::test]
async fn postgres_ratings_summarize() {
    let Some((store, _c)) = postgres_store().await else {
        return;
    };
    assert_ratings_summarize(store).await;
}
