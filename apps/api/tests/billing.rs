//! Integration tests for the Stripe-metered billing scaffold.
//!
//! Three properties under test (the same three the deliverable named):
//!
//!   1. Idempotency-key derivation — `idempotency_key_for(job_id, sku)` is
//!      deterministic, namespaced, SKU-scoped, and Stripe-safe.
//!   2. Customer-id mapping — `KeyCustomerMap::parse` handles the
//!      documented env-var format plus the messy real-world variants
//!      (whitespace, blanks, missing colons).
//!   3. No-double-charge invariant — `BillingClient::record_repack_job`
//!      run twice (or N times) for the same job emits exactly one POST
//!      per SKU to Stripe. This is the load-bearing property; the rest
//!      of the module exists to enforce it. Verified end-to-end here
//!      against a hand-rolled Stripe-shaped mock so we exercise both
//!      the ledger gate *and* the Stripe-side `identifier` field.
//!
//! The mock server (in-process tokio listener) is intentionally a
//! handful of lines rather than a `wiremock` dep — we don't need
//! request-matching DSL, just an "increment a counter and reply 200".

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use splatforge_api::billing::{
    idempotency_key_for, BillingClient, KeyCustomerMap, SKU_REPACK_RUNS, SKU_REPACK_SECONDS,
};
use splatforge_api::store::{DynJobStore, JobStore};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use uuid::Uuid;

/* ----------------------------------------------------------------------- */
/* 1. Idempotency-key derivation                                            */
/* ----------------------------------------------------------------------- */

#[test]
fn idempotency_key_is_deterministic() {
    let id = Uuid::parse_str("11111111-2222-3333-4444-555555555555").unwrap();
    let a = idempotency_key_for(&id, SKU_REPACK_RUNS);
    let b = idempotency_key_for(&id, SKU_REPACK_RUNS);
    assert_eq!(a, b);
}

#[test]
fn idempotency_key_is_sku_scoped() {
    let id = Uuid::new_v4();
    assert_ne!(
        idempotency_key_for(&id, SKU_REPACK_RUNS),
        idempotency_key_for(&id, SKU_REPACK_SECONDS),
    );
}

#[test]
fn idempotency_key_fits_stripe_limit() {
    // Stripe's `Idempotency-Key` header / meter event `identifier` is
    // capped at 255 chars. We emit 64-hex + small prefix; 255 is plenty.
    let id = Uuid::new_v4();
    assert!(idempotency_key_for(&id, SKU_REPACK_RUNS).len() <= 255);
    assert!(idempotency_key_for(&id, SKU_REPACK_SECONDS).len() <= 255);
}

#[test]
fn idempotency_key_is_namespaced() {
    // The "sf_" prefix lets humans grep the Stripe dashboard for our
    // events without colliding with anyone else's identifier.
    let id = Uuid::new_v4();
    assert!(idempotency_key_for(&id, SKU_REPACK_RUNS).starts_with("sf_splatforge_repack_runs_"));
    assert!(
        idempotency_key_for(&id, SKU_REPACK_SECONDS).starts_with("sf_splatforge_repack_seconds_")
    );
}

/* ----------------------------------------------------------------------- */
/* 2. Customer-id mapping                                                   */
/* ----------------------------------------------------------------------- */

#[test]
fn customer_map_parses_canonical_form() {
    let m = KeyCustomerMap::parse(Some("key_alpha:cus_aaa,key_beta:cus_bbb".into()));
    assert_eq!(m.lookup("key_alpha"), Some("cus_aaa"));
    assert_eq!(m.lookup("key_beta"), Some("cus_bbb"));
    assert_eq!(m.lookup("missing"), None);
    assert_eq!(m.len(), 2);
}

#[test]
fn customer_map_ignores_malformed_entries() {
    // Garbage entries are dropped, not fatal — keeps the API up even
    // if SPLATFORGE_KEY_CUSTOMERS has a typo. The dropped entries
    // produce a structured warn! at startup, which the operator sees
    // in Fly logs.
    let m = KeyCustomerMap::parse(Some(
        "good:cus_aaa,garbage_no_colon,:cus_missing_key,key:".into(),
    ));
    assert_eq!(m.lookup("good"), Some("cus_aaa"));
    assert_eq!(m.len(), 1);
}

#[test]
fn customer_map_none_yields_empty() {
    assert!(KeyCustomerMap::parse(None).is_empty());
    assert!(KeyCustomerMap::parse(Some("".into())).is_empty());
}

/* ----------------------------------------------------------------------- */
/* 3. No-double-charge invariant                                            */
/* ----------------------------------------------------------------------- */

/// Stripe-shaped mock: counts incoming `POST /v1/billing/meter_events`
/// calls and replies 200 with a JSON body that includes the `identifier`
/// echoed back. Returns `(addr, counter)` so the test can assert on the
/// number of calls received.
async fn spawn_stripe_mock() -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let counter = counter_clone.clone();
            tokio::spawn(async move {
                // Read until end-of-headers; for tests we don't need a full
                // HTTP parser, just enough to tick the counter and emit a
                // valid response. The body is small (form-encoded) so it
                // arrives in the same recv as the headers.
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                counter.fetch_add(1, Ordering::SeqCst);
                // Echo back a Stripe-shaped JSON body. The `identifier`
                // field is what the production code reads to stamp the
                // ledger row.
                let body = r#"{"identifier":"sf_test_event","object":"billing.meter_event"}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (addr, counter)
}

#[tokio::test]
async fn billing_emits_two_events_per_repack_with_seconds() {
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.unwrap());
    let (addr, counter) = spawn_stripe_mock().await;
    let client = BillingClient::with_base_url(
        "sk_test_dummy".into(),
        store.clone(),
        format!("http://{addr}"),
    );
    let job_id = Uuid::new_v4();
    client
        .record_repack_job(job_id, Some("cus_aaa"), 287_000_000, 1000, Some(18))
        .await
        .expect("record");
    // Two events expected: runs (1) + seconds (18). The seconds event
    // only fires because compute_seconds was Some.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "exactly two SKUs per run"
    );
}

#[tokio::test]
async fn billing_is_idempotent_across_retries() {
    // The load-bearing test. Two back-to-back calls with the same
    // job_id must produce exactly two Stripe POSTs (one per SKU), not
    // four. This is the case the Modal callback path can hit when the
    // worker callback fires twice.
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.unwrap());
    let (addr, counter) = spawn_stripe_mock().await;
    let client = BillingClient::with_base_url(
        "sk_test_dummy".into(),
        store.clone(),
        format!("http://{addr}"),
    );
    let job_id = Uuid::new_v4();
    for _ in 0..3 {
        client
            .record_repack_job(job_id, Some("cus_aaa"), 287_000_000, 1000, Some(18))
            .await
            .expect("record");
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "three retries must still produce exactly two events (runs + seconds), not six"
    );
}

#[tokio::test]
async fn billing_runs_only_when_no_seconds() {
    // Synchronous /repack dispatch path: bills the run SKU but doesn't
    // know elapsed time yet. Only the runs event should fire.
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.unwrap());
    let (addr, counter) = spawn_stripe_mock().await;
    let client = BillingClient::with_base_url(
        "sk_test_dummy".into(),
        store.clone(),
        format!("http://{addr}"),
    );
    let job_id = Uuid::new_v4();
    client
        .record_repack_job(job_id, Some("cus_aaa"), 287_000_000, 1000, None)
        .await
        .expect("record");
    assert_eq!(counter.load(Ordering::SeqCst), 1, "only runs SKU emitted");
}

#[tokio::test]
async fn billing_free_tier_emits_no_events() {
    // customer_id = None → free tier → no Stripe calls at all. This is
    // the "free is free" contract from the deliverable.
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.unwrap());
    let (addr, counter) = spawn_stripe_mock().await;
    let client = BillingClient::with_base_url(
        "sk_test_dummy".into(),
        store.clone(),
        format!("http://{addr}"),
    );
    let job_id = Uuid::new_v4();
    client
        .record_repack_job(job_id, None, 287_000_000, 1000, Some(18))
        .await
        .expect("record");
    assert_eq!(counter.load(Ordering::SeqCst), 0, "free tier must not bill");
}

#[tokio::test]
async fn billing_distinct_jobs_emit_distinct_events() {
    // Sanity check: idempotency is per-(job_id, sku), not global. Two
    // different jobs in the same call sequence each emit their own events.
    let store: DynJobStore = Arc::new(JobStore::in_memory().await.unwrap());
    let (addr, counter) = spawn_stripe_mock().await;
    let client = BillingClient::with_base_url(
        "sk_test_dummy".into(),
        store.clone(),
        format!("http://{addr}"),
    );
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    client
        .record_repack_job(a, Some("cus_aaa"), 1_000_000, 1000, Some(5))
        .await
        .unwrap();
    client
        .record_repack_job(b, Some("cus_aaa"), 2_000_000, 1000, Some(7))
        .await
        .unwrap();
    assert_eq!(
        counter.load(Ordering::SeqCst),
        4,
        "2 jobs * 2 SKUs = 4 events"
    );
}
