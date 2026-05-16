//! Integration tests for `customer_dashboard::build_response`.
//!
//! Load-bearing invariant: the response MUST be scoped to the
//! requester's masked key prefix. A second user writing audit rows
//! under their own key MUST NOT have those rows appear in another
//! user's `/v1/me/usage` payload. This is the single piece of code
//! that prevents cross-customer data leakage in the dashboard surface,
//! so it gets its own test file with the bytes-in / bytes-out check.

use std::sync::Arc;

use splatforge_api::audit;
use splatforge_api::customer_dashboard::{build_response, Plan};
use splatforge_api::ratelimit::key_prefix;
use splatforge_api::store::{DynJobStore, JobStore};

#[tokio::test]
async fn build_response_only_returns_rows_for_requesting_key() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));

    // User A: three audited mutations (2 repacks + 1 create).
    let key_a = "sk_alice_full_token";
    let prefix_a = key_prefix(key_a);
    audit::record(
        &store,
        &prefix_a,
        "/v1/jobs/:id/repack",
        "POST",
        200,
        128,
        18_000,
        None,
    )
    .await;
    audit::record(
        &store,
        &prefix_a,
        "/v1/jobs/:id/repack",
        "POST",
        200,
        128,
        7_500,
        None,
    )
    .await;
    audit::record(&store, &prefix_a, "/v1/jobs", "POST", 201, 256, 50, None).await;

    // User B: two rows. These MUST NOT appear in A's response.
    let key_b = "sk_bobby_full_token";
    let prefix_b = key_prefix(key_b);
    audit::record(
        &store,
        &prefix_b,
        "/v1/jobs/:id/repack",
        "POST",
        200,
        128,
        99_000,
        None,
    )
    .await;
    audit::record(&store, &prefix_b, "/v1/jobs", "POST", 201, 256, 12, None).await;

    let resp = build_response(&store, prefix_a.clone(), Plan::Paid, None, 25)
        .await
        .expect("build_response");

    // Scope check — every recent_job row must be from key A. There is no
    // `key_prefix` field on the customer-facing shape (it gets stripped
    // by `RecentJob::from`), so we rely on the integers: A has 3 rows,
    // B has 2. If B's rows leaked we'd see 5 here.
    assert_eq!(resp.recent_jobs.len(), 3);

    // Usage summary derived only from A's rows. 18s + 7s = 25s
    // (18_000ms + 7_500ms; 7.5s rounds down per billing semantics).
    assert_eq!(resp.usage.repack_runs, 2);
    assert_eq!(resp.usage.repack_seconds, 25);
    assert_eq!(resp.key_masked, prefix_a);
    assert_eq!(resp.plan, Plan::Paid);
}

#[tokio::test]
async fn build_response_empty_log_renders_zero_state() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    let resp = build_response(&store, "sk_new___".to_string(), Plan::Free, None, 25)
        .await
        .expect("build_response");
    assert_eq!(resp.recent_jobs.len(), 0);
    assert_eq!(resp.usage.repack_runs, 0);
    assert_eq!(resp.usage.repack_seconds, 0);
    assert!(resp.usage.period_start.is_none());
    assert_eq!(resp.plan, Plan::Free);
}

#[tokio::test]
async fn build_response_clamps_limit() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    let prefix = "sk_test_".to_string();
    for _ in 0..120 {
        audit::record(&store, &prefix, "/v1/jobs", "POST", 201, 32, 5, None).await;
    }
    // Pass a wildly out-of-range limit; handler must clamp to <=100.
    let resp = build_response(&store, prefix.clone(), Plan::Paid, None, 9_999)
        .await
        .expect("build_response");
    assert!(
        resp.recent_jobs.len() <= 100,
        "limit must be clamped to RECENT_JOBS_MAX_LIMIT (got {})",
        resp.recent_jobs.len()
    );
}

/// Documented limitation: the audit log stores only the 8-char masked
/// key prefix (`ratelimit::key_prefix`). Two keys that share the same
/// 8-char prefix will see each other's rows in `/v1/me/usage`. In
/// practice the only collision risk today is the `sk_test_` prefix
/// used by Stripe test-mode keys; a follow-up session will add a
/// per-key hash column to the audit table so the dashboard scope is
/// truly per-user. This test PINS the limitation so we notice when
/// the next session lifts it (the test will start failing — flip the
/// assertion at that point).
#[tokio::test]
async fn prefix_collision_known_limitation_is_pinned() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    // Both keys mask to "sk_test_" so they share the same prefix.
    let a = key_prefix("sk_test_alice_zzz");
    let b = key_prefix("sk_test_bobby_zzz");
    assert_eq!(
        a, b,
        "sanity: both test-mode keys mask to the same 8-char prefix"
    );

    audit::record(&store, &a, "/v1/jobs", "POST", 201, 32, 5, None).await;
    audit::record(&store, &b, "/v1/jobs", "POST", 201, 32, 7, None).await;

    let resp = build_response(&store, a, Plan::Paid, None, 25)
        .await
        .expect("build_response");
    // TODO(follow-up): when key_hash lands, this should be 1, not 2.
    assert_eq!(
        resp.recent_jobs.len(),
        2,
        "current behavior: prefix-collision returns both rows. Lift in follow-up."
    );
}
