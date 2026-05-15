//! Integration tests for the audit-event log.
//!
//! Covers:
//!   * route templating (uuid → `:id`),
//!   * mutating-only filter (`GET` is not audited),
//!   * write-then-query roundtrip including order + limit,
//!   * the 1000-row cap from the spec,
//!   * key-prefix masking (full bearer key never persisted).
//!
//! The audit module is best-effort by design (a DB write failure must
//! not surface to the caller), so the "what happens when the DB fails"
//! case is exercised in the unit tests of `audit::record` itself —
//! here we focus on the success path the operator reads back.

use std::sync::Arc;

use splatforge_api::audit::{self, is_audited, route_template, ADMIN_AUDIT_DEFAULT_LIMIT};
use splatforge_api::ratelimit::key_prefix;
use splatforge_api::store::{DynJobStore, JobStore, JobStoreApi};

#[test]
fn route_template_strips_uuid_segment() {
    let uuid = "abcdef01-2345-6789-abcd-ef0123456789";
    assert_eq!(
        route_template("POST", &format!("/v1/jobs/{uuid}/upload")),
        Some("/v1/jobs/:id/upload"),
    );
    assert_eq!(
        route_template("POST", &format!("/v1/jobs/{uuid}/repack")),
        Some("/v1/jobs/:id/repack"),
    );
    assert_eq!(
        route_template("POST", &format!("/v1/jobs/{uuid}/result")),
        Some("/v1/jobs/:id/result"),
    );
}

#[test]
fn read_only_routes_are_not_audited() {
    assert!(!is_audited("GET", "/v1/jobs/abc-123"));
    assert!(!is_audited("GET", "/healthz"));
    assert!(!is_audited("GET", "/openapi.yaml"));
    assert!(!is_audited("GET", "/docs"));
    assert!(!is_audited("GET", "/v1/admin/audit"));
}

#[test]
fn mutating_routes_are_audited() {
    assert!(is_audited("POST", "/v1/jobs"));
    assert!(is_audited("POST", "/v1/jobs/batch"));
    assert!(is_audited("POST", "/v1/jobs/abc-123/upload"));
    assert!(is_audited("POST", "/v1/jobs/abc-123/repack"));
    assert!(is_audited("POST", "/v1/jobs/abc-123/result"));
    assert!(is_audited("POST", "/v1/stripe/webhook"));
}

#[tokio::test]
async fn write_then_query_roundtrip() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    audit::record(
        &store,
        "key_aaa_",
        "/v1/jobs",
        "POST",
        200,
        42,
        12,
        None,
    )
    .await;
    audit::record(
        &store,
        "key_bbb_",
        "/v1/jobs/:id/upload",
        "POST",
        429,
        0,
        3,
        Some("rate-limited"),
    )
    .await;

    let events = store.list_audit_events(10).await.expect("query");
    assert_eq!(events.len(), 2);
    // Newest first.
    assert_eq!(events[0].route, "/v1/jobs/:id/upload");
    assert_eq!(events[0].status, 429);
    assert_eq!(events[0].error.as_deref(), Some("rate-limited"));
    assert_eq!(events[1].route, "/v1/jobs");
    assert_eq!(events[1].status, 200);
}

#[tokio::test]
async fn key_prefix_is_what_lands_in_audit() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    let full_key = "sk_test_super_long_secret_xyz";
    let prefix = key_prefix(full_key);
    audit::record(
        &store,
        &prefix,
        "/v1/jobs",
        "POST",
        200,
        0,
        1,
        None,
    )
    .await;
    let rows = store.list_audit_events(10).await.expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].key_prefix, "sk_test_");
    // Full token MUST NOT appear anywhere in the row.
    assert!(!rows[0].key_prefix.contains("super_long_secret"));
}

#[tokio::test]
async fn list_audit_respects_limit() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    for i in 0..25 {
        audit::record(
            &store,
            "k_",
            "/v1/jobs",
            "POST",
            200,
            0,
            i as u64,
            None,
        )
        .await;
    }
    let small = store.list_audit_events(5).await.expect("query");
    assert_eq!(small.len(), 5);
    let all = store.list_audit_events(100).await.expect("query");
    assert_eq!(all.len(), 25);
}

#[tokio::test]
async fn admin_default_limit_is_1000() {
    // The spec mandates "last 1000 events" — codify the constant so a
    // future operator can't silently drop it.
    assert_eq!(ADMIN_AUDIT_DEFAULT_LIMIT, 1000);
    // And confirm the store can return that many.
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    for _ in 0..1200 {
        audit::record(&store, "k_", "/v1/jobs", "POST", 200, 0, 0, None).await;
    }
    let rows = store
        .list_audit_events(ADMIN_AUDIT_DEFAULT_LIMIT)
        .await
        .expect("query");
    assert_eq!(rows.len(), 1000);
}

#[tokio::test]
async fn error_field_is_optional() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
    audit::record(&store, "k_", "/v1/jobs", "POST", 200, 0, 0, None).await;
    audit::record(
        &store,
        "k_",
        "/v1/jobs",
        "POST",
        400,
        0,
        0,
        Some("bad-request"),
    )
    .await;
    let rows = store.list_audit_events(10).await.expect("query");
    assert_eq!(rows.len(), 2);
    // Newest first: the 400 with the error string.
    assert_eq!(rows[0].error.as_deref(), Some("bad-request"));
    assert_eq!(rows[1].error, None);
}
