//! Integration tests for the self-serve Team-tier signup pipeline.
//!
//! Four properties under test:
//!
//!   1. `create_session` issues exactly one Stripe Checkout Session per
//!      legitimate call, mode=subscription, price=$TEAM_PRICE_ID,
//!      success_url carrying `{CHECKOUT_SESSION_ID}` and a fresh
//!      `token=` claim, with Idempotency-Key derived from email+nonce.
//!   2. The webhook is **idempotent** — duplicate
//!      `checkout.session.completed` deliveries mint one key.
//!   3. Plaintext is revealed **exactly once**. The second call to
//!      `/reveal` returns Gone. This is the "key shown twice" anti-test
//!      (`reveal_is_strictly_one_shot`).
//!   4. Plaintext **never** lands on disk — `key_plaintext_never_persisted`
//!      scans every TEXT column of `team_signups` after a happy-path
//!      provision and asserts no value starts with `sf_live_` other
//!      than the safe `key_prefix`.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use splatforge_api::checkout::{
    self, checkout_idempotency_key, hash_key, mint_team_api_key, CheckoutConfig,
    CreateSessionRequest, PendingClaimTokens, PendingKeyCache, RevealRequest, StripeCheckoutClient,
    KEY_DISPLAY_PREFIX_LEN, KEY_PLAINTEXT_LEN, KEY_PREFIX_LITERAL,
};
use splatforge_api::store::{JobStore, JobStoreApi};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Default)]
struct CapturedRequest {
    body: String,
    idempotency_key: Option<String>,
}

async fn spawn_stripe_mock() -> (
    SocketAddr,
    Arc<AtomicUsize>,
    Arc<Mutex<Vec<CapturedRequest>>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let counter = Arc::new(AtomicUsize::new(0));
    let captured: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let counter_clone = counter.clone();
    let captured_clone = captured.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let counter = counter_clone.clone();
            let captured = captured_clone.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let raw = String::from_utf8_lossy(&buf[..n]).to_string();
                let mut idem: Option<String> = None;
                for line in raw.split("\r\n") {
                    if let Some(rest) = line.to_ascii_lowercase().strip_prefix("idempotency-key:") {
                        idem = Some(rest.trim().to_string());
                    }
                }
                let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                captured.lock().await.push(CapturedRequest {
                    body,
                    idempotency_key: idem,
                });
                let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                let body = format!(
                    r#"{{"id":"cs_test_mock_{n:04}","url":"https://checkout.stripe.com/c/pay/cs_test_mock_{n:04}","object":"checkout.session"}}"#
                );
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
    (addr, counter, captured)
}

fn test_config(base_url: String) -> CheckoutConfig {
    CheckoutConfig::from_env("https://splatforge.dev".to_string()).with_overrides(
        Some("sk_test_dummy".into()),
        Some("price_team_test".into()),
        Some(base_url),
    )
}

fn fake_session_event(session_id: &str, customer: &str, email: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("evt_test_{session_id}"),
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": session_id,
                "object": "checkout.session",
                "customer": customer,
                "subscription": format!("sub_{customer}"),
                "customer_email": email,
                "mode": "subscription"
            }
        }
    })
}

/* 1. Key shape + idempotency-key derivation */

#[test]
fn minted_key_matches_workos_branch_contract() {
    // WorkOS branch's auth::mint_key produces:
    //   - 32-char plaintext starting with sf_live_
    //   - 12-char display prefix
    //   - SHA-256 hex hash
    // We MUST match all three or post-merge two key sources diverge.
    let (pt, prefix, hash) = mint_team_api_key();
    assert_eq!(pt.len(), KEY_PLAINTEXT_LEN);
    assert_eq!(prefix.len(), KEY_DISPLAY_PREFIX_LEN);
    assert!(pt.starts_with(KEY_PREFIX_LITERAL));
    assert!(pt.starts_with(&prefix));
    assert_eq!(hash.len(), 64);
    assert_eq!(hash, hash_key(&pt));
    assert!(pt[KEY_PREFIX_LITERAL.len()..]
        .chars()
        .all(|c| c.is_ascii_alphanumeric()));
}

#[test]
fn idempotency_key_includes_email_and_nonce() {
    let same = checkout_idempotency_key("alice@example.com", "n1");
    assert_eq!(same, checkout_idempotency_key("alice@example.com", "n1"));
    assert_ne!(same, checkout_idempotency_key("alice@example.com", "n2"));
    assert_ne!(same, checkout_idempotency_key("bob@example.com", "n1"));
    assert!(same.starts_with("sf_checkout_"));
}

/* 2. create_session shape + Stripe-call accounting */

#[tokio::test]
async fn create_session_posts_team_price_and_subscription_mode() {
    let (addr, counter, captured) = spawn_stripe_mock().await;
    let config = test_config(format!("http://{addr}"));
    let client = StripeCheckoutClient::new("sk_test_dummy".into(), format!("http://{addr}"));
    let pending_tokens = PendingClaimTokens::new();
    let resp = checkout::create_session_and_register(
        &config,
        &client,
        &pending_tokens,
        CreateSessionRequest {
            email: "alice@example.com".into(),
            nonce: Some("nonce-1".into()),
        },
    )
    .await
    .expect("session");
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    assert!(resp.url.starts_with("https://checkout.stripe.com/"));
    assert!(resp.session_id.starts_with("cs_test_mock_"));

    let req = captured.lock().await.pop().expect("captured");
    assert!(
        req.body.contains("mode=subscription"),
        "expected mode=subscription; got {:?}",
        req.body
    );
    assert!(
        req.body
            .contains("line_items%5B0%5D%5Bprice%5D=price_team_test"),
        "expected team price in form body; got {:?}",
        req.body
    );
    assert!(
        req.body.contains("CHECKOUT_SESSION_ID") && req.body.contains("token%3D"),
        "expected success_url to contain Stripe placeholder + claim token; got {:?}",
        req.body
    );
    let expected_idem = checkout_idempotency_key("alice@example.com", "nonce-1");
    assert_eq!(req.idempotency_key.as_deref(), Some(expected_idem.as_str()));
}

#[tokio::test]
async fn create_session_refuses_empty_or_malformed_email() {
    let (addr, _, _) = spawn_stripe_mock().await;
    let config = test_config(format!("http://{addr}"));
    let client = StripeCheckoutClient::new("sk_test_dummy".into(), format!("http://{addr}"));
    let pending_tokens = PendingClaimTokens::new();
    let err = checkout::create_session_and_register(
        &config,
        &client,
        &pending_tokens,
        CreateSessionRequest {
            email: "".into(),
            nonce: None,
        },
    )
    .await
    .expect_err("empty");
    assert!(matches!(err, checkout::CheckoutError::BadRequest(_)));

    let err = checkout::create_session_and_register(
        &config,
        &client,
        &pending_tokens,
        CreateSessionRequest {
            email: "no-at-sign".into(),
            nonce: None,
        },
    )
    .await
    .expect_err("missing @");
    assert!(matches!(err, checkout::CheckoutError::BadRequest(_)));
}

/* 3. Webhook idempotency + provisioning */

async fn happy_path() -> (JobStore, PendingKeyCache, PendingClaimTokens, String) {
    let store = JobStore::in_memory().await.expect("store");
    let pending_keys = PendingKeyCache::new();
    let pending_tokens = PendingClaimTokens::new();
    let (addr, _, _) = spawn_stripe_mock().await;
    let config = test_config(format!("http://{addr}"));
    let client = StripeCheckoutClient::new("sk_test_dummy".into(), format!("http://{addr}"));
    let resp = checkout::create_session_and_register(
        &config,
        &client,
        &pending_tokens,
        CreateSessionRequest {
            email: "alice@example.com".into(),
            nonce: Some("n1".into()),
        },
    )
    .await
    .expect("session");
    let event = fake_session_event(&resp.session_id, "cus_alpha", "alice@example.com");
    checkout::provision_from_session(&store, &pending_keys, &pending_tokens, &event)
        .await
        .expect("provision");
    (store, pending_keys, pending_tokens, resp.session_id)
}

#[tokio::test]
async fn webhook_idempotent_under_double_delivery() {
    let store = JobStore::in_memory().await.expect("store");
    let pending_keys = PendingKeyCache::new();
    let pending_tokens = PendingClaimTokens::new();
    let event = fake_session_event("cs_test_dup", "cus_dup", "dup@example.com");
    pending_tokens
        .insert("cs_test_dup".into(), "tok123".into())
        .await;
    for _ in 0..3 {
        checkout::provision_from_session(&store, &pending_keys, &pending_tokens, &event)
            .await
            .expect("provision");
    }
    let row = store
        .get_team_signup_by_session("cs_test_dup")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(row.email, "dup@example.com");
    assert_eq!(row.stripe_customer_id, "cus_dup");
    assert!(row.key_revealed_at.is_none());
}

#[tokio::test]
async fn webhook_rejects_event_missing_required_fields() {
    let store = JobStore::in_memory().await.expect("store");
    let pending_keys = PendingKeyCache::new();
    let pending_tokens = PendingClaimTokens::new();
    let bad = serde_json::json!({
        "id": "evt_bad",
        "type": "checkout.session.completed",
        "data": { "object": { "id": "cs_bad", "customer_email": "x@y.com" }}
    });
    let err = checkout::provision_from_session(&store, &pending_keys, &pending_tokens, &bad)
        .await
        .expect_err("missing customer");
    assert!(matches!(err, checkout::CheckoutError::BadRequest(_)));
}

/* 4. Reveal: exactly-once + plaintext-never-persisted */

#[tokio::test]
async fn reveal_returns_plaintext_and_authorization_header() {
    let (store, pending_keys, _pending_tokens, session_id) = happy_path().await;
    let row = store
        .get_team_signup_by_session(&session_id)
        .await
        .unwrap()
        .unwrap();
    let resp = checkout::reveal_key(
        &store,
        &pending_keys,
        RevealRequest {
            session_id: session_id.clone(),
            token: row.claim_token.clone(),
        },
    )
    .await
    .expect("reveal");
    assert!(resp.api_key.starts_with("sf_live_"));
    assert_eq!(resp.api_key.len(), 32);
    // Authorization header is exactly what the WorkOS-branch auth
    // middleware expects: `Bearer sf_live_<24>` byte-for-byte.
    assert_eq!(
        resp.authorization_header,
        format!("Bearer {}", resp.api_key)
    );
    assert_eq!(resp.email, "alice@example.com");
    assert_eq!(resp.key_prefix, row.key_prefix);
    assert_eq!(hash_key(&resp.api_key), row.key_hash);
}

#[tokio::test]
async fn reveal_is_strictly_one_shot() {
    // The "key shown twice" anti-test. If this passes, the
    // plaintext-once invariant is broken.
    let (store, pending_keys, _pending_tokens, session_id) = happy_path().await;
    let row = store
        .get_team_signup_by_session(&session_id)
        .await
        .unwrap()
        .unwrap();
    let token = row.claim_token.clone();
    checkout::reveal_key(
        &store,
        &pending_keys,
        RevealRequest {
            session_id: session_id.clone(),
            token: token.clone(),
        },
    )
    .await
    .expect("first reveal");
    let err = checkout::reveal_key(
        &store,
        &pending_keys,
        RevealRequest {
            session_id: session_id.clone(),
            token: token.clone(),
        },
    )
    .await
    .expect_err("second reveal");
    assert!(
        matches!(err, checkout::CheckoutError::Gone),
        "second reveal must be Gone; got {:?}",
        err
    );
    // Even post simulated process restart (fresh cache) the DB still
    // says revealed -> Gone.
    let fresh_cache = PendingKeyCache::new();
    let err = checkout::reveal_key(&store, &fresh_cache, RevealRequest { session_id, token })
        .await
        .expect_err("post-restart reveal");
    assert!(matches!(err, checkout::CheckoutError::Gone));
}

#[tokio::test]
async fn reveal_rejects_bad_token() {
    let (store, pending_keys, _pending_tokens, session_id) = happy_path().await;
    let err = checkout::reveal_key(
        &store,
        &pending_keys,
        RevealRequest {
            session_id: session_id.clone(),
            token: "not-the-real-token".into(),
        },
    )
    .await
    .expect_err("bad token");
    assert!(matches!(err, checkout::CheckoutError::Forbidden));
    // Critically: the DB column is NOT flipped. The legitimate
    // buyer can still reveal with the correct token.
    let row = store
        .get_team_signup_by_session(&session_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        row.key_revealed_at.is_none(),
        "bad-token attempt must not consume the one-shot reveal"
    );
}

#[tokio::test]
async fn reveal_for_unknown_session_returns_not_found() {
    let store = JobStore::in_memory().await.expect("store");
    let cache = PendingKeyCache::new();
    let err = checkout::reveal_key(
        &store,
        &cache,
        RevealRequest {
            session_id: "cs_test_does_not_exist".into(),
            token: "anything".into(),
        },
    )
    .await
    .expect_err("unknown session");
    assert!(matches!(err, checkout::CheckoutError::NotFound));
}

#[tokio::test]
async fn key_plaintext_never_persisted() {
    // Defence-in-depth: scan every column of every row and confirm the
    // plaintext is nowhere to be found. The server can't show the key
    // again because the server doesn't have it.
    let (store, _pending_keys, _pending_tokens, _session_id) = happy_path().await;

    use sqlx::{Column, Row};
    let pool = store.pool();
    let rows = sqlx::query("SELECT * FROM team_signups")
        .fetch_all(pool)
        .await
        .expect("query");
    for r in rows {
        for col in r.columns() {
            let name = col.name();
            let val: Option<String> = r.try_get(name).ok();
            let Some(s) = val else { continue };
            if name == "key_prefix" {
                assert!(
                    s.len() <= KEY_DISPLAY_PREFIX_LEN,
                    "key_prefix longer than {}; leaked secret material: {s}",
                    KEY_DISPLAY_PREFIX_LEN
                );
                continue;
            }
            assert!(
                !s.starts_with(KEY_PREFIX_LITERAL),
                "column {name} contains a plaintext-looking value: {s}"
            );
        }
    }
}

#[tokio::test]
async fn webhook_recovers_when_pending_token_cache_is_empty() {
    // Simulate: the API process restarted between create-session and
    // the webhook. The webhook must still provision a row — it
    // generates a fresh claim_token, and /reveal correctly refuses
    // the buyer's stale URL (fail closed → support escalates).
    let store = JobStore::in_memory().await.expect("store");
    let pending_keys = PendingKeyCache::new();
    let pending_tokens = PendingClaimTokens::new();
    let event = fake_session_event("cs_test_orphan", "cus_orphan", "orphan@example.com");
    checkout::provision_from_session(&store, &pending_keys, &pending_tokens, &event)
        .await
        .expect("provision");
    let row = store
        .get_team_signup_by_session("cs_test_orphan")
        .await
        .unwrap()
        .unwrap();
    assert!(!row.claim_token.is_empty());
    let err = checkout::reveal_key(
        &store,
        &pending_keys,
        RevealRequest {
            session_id: "cs_test_orphan".into(),
            token: "the-buyer's-token-from-success_url".into(),
        },
    )
    .await
    .expect_err("orphaned token must not reveal");
    assert!(matches!(err, checkout::CheckoutError::Forbidden));
}
