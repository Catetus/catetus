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
use splatforge_api::ratelimit::{key_fingerprint, key_prefix};
use splatforge_api::store::{DynJobStore, JobStore};

#[tokio::test]
async fn build_response_only_returns_rows_for_requesting_key() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));

    // User A: three audited mutations (2 repacks + 1 create).
    let key_a = "sk_alice_full_token";
    let scope_a = key_fingerprint(key_a);
    audit::record(
        &store,
        &scope_a,
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
        &scope_a,
        "/v1/jobs/:id/repack",
        "POST",
        200,
        128,
        7_500,
        None,
    )
    .await;
    audit::record(&store, &scope_a, "/v1/jobs", "POST", 201, 256, 50, None).await;

    // User B: two rows. These MUST NOT appear in A's response.
    let key_b = "sk_bobby_full_token";
    let scope_b = key_fingerprint(key_b);
    audit::record(
        &store,
        &scope_b,
        "/v1/jobs/:id/repack",
        "POST",
        200,
        128,
        99_000,
        None,
    )
    .await;
    audit::record(&store, &scope_b, "/v1/jobs", "POST", 201, 256, 12, None).await;

    let display_a = key_prefix(key_a);
    let resp = build_response(
        &store,
        scope_a.clone(),
        display_a.clone(),
        Plan::Paid,
        None,
        25,
    )
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
    // Display masked prefix is the 8-char literal, NOT the fingerprint
    // (the fingerprint is the storage scope, not user-visible).
    assert_eq!(resp.key_masked, display_a);
    assert_ne!(resp.key_masked, scope_a, "fingerprint must NOT leak to UI");
    assert_eq!(resp.plan, Plan::Paid);
}

#[tokio::test]
async fn build_response_empty_log_renders_zero_state() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    let resp = build_response(
        &store,
        key_fingerprint("sk_new___nobody_xxxxxxxxxxxxxxxxx"),
        "sk_new___".to_string(),
        Plan::Free,
        None,
        25,
    )
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
    let scope = key_fingerprint("sk_test_clamper_xxxxxxxxxxxxxxx");
    for _ in 0..120 {
        audit::record(&store, &scope, "/v1/jobs", "POST", 201, 32, 5, None).await;
    }
    // Pass a wildly out-of-range limit; handler must clamp to <=100.
    let resp = build_response(
        &store,
        scope.clone(),
        "sk_test_".to_string(),
        Plan::Paid,
        None,
        9_999,
    )
    .await
    .expect("build_response");
    assert!(
        resp.recent_jobs.len() <= 100,
        "limit must be clamped to RECENT_JOBS_MAX_LIMIT (got {})",
        resp.recent_jobs.len()
    );
}

/// P0 cross-customer-leak regression test.
///
/// Every production SplatForge API key is minted with the literal
/// prefix `sf_live_` (see `checkout::KEY_PREFIX_LITERAL`). The legacy
/// `ratelimit::key_prefix` returned exactly 8 chars, so the audit
/// scope key for EVERY paying customer collapsed to the single value
/// `"sf_live_"`. That would have made `/v1/me/usage` return every
/// other customer's audit rows — a catastrophic data leak on the
/// dashboard surface.
///
/// The fix introduces `ratelimit::key_fingerprint` (SHA-256 truncated
/// to 16 hex chars) which is what `audit::record` now writes to the
/// `key_prefix` column. The dashboard query / `build_response` are
/// keyed on the fingerprint, so two distinct `sf_live_…` keys see
/// disjoint audit slices.
#[tokio::test]
async fn two_sf_live_keys_do_not_leak_audit_rows_into_each_others_dashboard() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    // Two real-shape production keys: both start with the canonical
    // `sf_live_` prefix the checkout module mints. The old `key_prefix`
    // helper would return `"sf_live_"` for BOTH — that's the bug.
    let key_a = "sf_live_alice_aaaaaaaaaaaaaaaaa";
    let key_b = "sf_live_bobby_bbbbbbbbbbbbbbbbb";
    assert_eq!(
        key_prefix(key_a),
        key_prefix(key_b),
        "sanity: legacy 8-char prefix collides for any two sf_live_ keys"
    );
    // The fingerprint, by contrast, is per-key.
    let fp_a = key_fingerprint(key_a);
    let fp_b = key_fingerprint(key_b);
    assert_ne!(fp_a, fp_b, "fingerprint must distinguish two sf_live_ keys");

    // Audit middleware writes the fingerprint (not the legacy prefix).
    audit::record(&store, &fp_a, "/v1/jobs", "POST", 201, 32, 5, None).await;
    audit::record(&store, &fp_a, "/v1/jobs/:id/repack", "POST", 200, 0, 9_000, None).await;
    audit::record(&store, &fp_b, "/v1/jobs", "POST", 201, 32, 7, None).await;
    audit::record(&store, &fp_b, "/v1/jobs/:id/repack", "POST", 200, 0, 33_000, None).await;
    audit::record(&store, &fp_b, "/v1/jobs/:id/repack", "POST", 200, 0, 33_000, None).await;

    // Alice's dashboard, scoped on Alice's fingerprint, must return
    // ONLY Alice's rows.
    let resp = build_response(
        &store,
        fp_a.clone(),
        key_prefix(key_a),
        Plan::Paid,
        None,
        25,
    )
    .await
    .expect("build_response");
    assert_eq!(
        resp.recent_jobs.len(),
        2,
        "Alice has exactly 2 audit rows; Bob's MUST NOT leak"
    );
    assert_eq!(
        resp.usage.repack_runs, 1,
        "Alice did 1 successful repack; Bob's 2 must not leak"
    );
    // 9_000 ms -> 9 s. Bob's 66 s of repack must not appear.
    assert_eq!(resp.usage.repack_seconds, 9);
}

/// Pin that the fingerprint is the storage column value, not the
/// masked display prefix. The display prefix (`sf_live_`) is fine to
/// echo back in the UI; the storage column must be opaque + per-key.
#[tokio::test]
async fn key_fingerprint_is_stable_and_collision_resistant() {
    let a1 = key_fingerprint("sf_live_alice_xxxxxxxxxxxxxxxxx");
    let a2 = key_fingerprint("sf_live_alice_xxxxxxxxxxxxxxxxx");
    let b = key_fingerprint("sf_live_alice_xxxxxxxxxxxxxxxxy"); // last char differs
    assert_eq!(a1, a2, "fingerprint must be stable for the same input");
    assert_ne!(a1, b, "single-char difference must change the fingerprint");
    // 16 hex chars = 64 bits of collision resistance. Sized to fit in
    // the existing audit_events.key_prefix column without a migration.
    assert_eq!(a1.len(), 16);
    assert!(a1.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
}
