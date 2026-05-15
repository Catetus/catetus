//! Stripe-metered billing for the paid tier.
//!
//! ## Why this exists
//!
//! `POST /v1/jobs/:id/repack` dispatches to a Modal A100 (~$0.05-$0.12 per
//! scene). We need usage-based billing so the paid tier is monetizable.
//! Two SKUs are emitted per repack run:
//!
//!   * `splatforge_repack_runs`    — 1 unit per repack call (flat per-job fee)
//!   * `splatforge_repack_seconds` — 1 unit per second of compute (per-second
//!                                   cost). Reported on the worker callback
//!                                   when we have the elapsed time.
//!
//! Stripe's modern (2024+) usage-based pattern is the Billing Meter Events
//! API: `POST /v1/billing/meter_events` with form-encoded
//! `event_name`, `payload[stripe_customer_id]`, `payload[value]`, plus an
//! `identifier` for idempotency. We derive that identifier deterministically
//! from `(job_id, sku)` so a retried call cannot double-bill — Stripe
//! dedupes on identifier *and* we dedupe in our own ledger before we even
//! make the network call.
//!
//! ## Modes
//!
//! `BillingClient::live(secret)` makes real test-mode (or live-mode, gated
//! on `STRIPE_LIVE_MODE=true`) calls to Stripe. `BillingClient::dry_run()`
//! logs the would-be charge and short-circuits the network call — used
//! when `STRIPE_SECRET_KEY` is unset (local dev, CI) so the rest of the
//! API works without Stripe credentials.
//!
//! ## Free tier
//!
//! Free-tier jobs MUST NOT emit billing events. `record_repack_job` is
//! only called from the paid `/repack` handler, and `Job.customer_id ==
//! None` short-circuits before any network call. The free path is free.

use std::time::Duration;

use sha2::{Digest, Sha256};
use tracing::{info, instrument, warn};
use uuid::Uuid;

use crate::store::DynJobStore;

/// SKU emitted per repack run (flat fee — 1 unit per call).
pub const SKU_REPACK_RUNS: &str = "splatforge_repack_runs";

/// SKU emitted per second of compute. Reported when the worker callback
/// includes elapsed wall-clock time for the run.
pub const SKU_REPACK_SECONDS: &str = "splatforge_repack_seconds";

/// Stripe API base. Same endpoint serves test-mode and live-mode keys —
/// the key prefix (`sk_test_` vs `sk_live_`) decides which environment
/// Stripe routes the call to. We refuse to use a `sk_live_` key unless
/// `STRIPE_LIVE_MODE=true` is set explicitly, so a misconfigured env
/// var can't ship real charges.
const STRIPE_API_BASE: &str = "https://api.stripe.com";

/// HTTP timeout for Stripe meter-event posts. The meter API is fast
/// (<200ms p99 in practice); a 10s ceiling caps the worst-case stall on
/// the callback path so a Stripe outage can't block the job-result
/// pipeline.
const STRIPE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum BillingError {
    #[error("stripe request transport: {0}")]
    Transport(String),
    #[error("stripe rejected meter event: {status}: {body}")]
    Stripe { status: u16, body: String },
    #[error("billing ledger: {0}")]
    Store(#[from] crate::store::StoreError),
}

/// Where to send Stripe traffic. `Live` mode hits the real API; `DryRun`
/// logs the would-be payload and returns success. Both share the same
/// ledger semantics (claim → idempotent), so swapping modes never
/// changes the no-double-charge invariant.
#[derive(Clone)]
enum Backend {
    Live {
        http: reqwest::Client,
        secret: String,
        base_url: String,
    },
    DryRun,
}

#[derive(Clone)]
pub struct BillingClient {
    backend: Backend,
    /// Ledger used to dedupe (job_id, sku) before we ever talk to Stripe.
    /// Sharing the `DynJobStore` (Arc<dyn JobStoreApi + Send + Sync>) from
    /// `AppState` keeps the billing path transactional with the rest of
    /// the job state, and lets the same client run against SQLite or
    /// Postgres without any code change in this file.
    store: DynJobStore,
}

impl BillingClient {
    /// Live mode — hits the real Stripe Billing API. Pass the test-mode
    /// secret (`sk_test_...`) unless `STRIPE_LIVE_MODE=true` is set.
    pub fn live(secret: String, store: DynJobStore) -> Self {
        Self::with_base_url(secret, store, STRIPE_API_BASE.to_string())
    }

    /// Live-mode with a caller-supplied base URL. Used by integration
    /// tests to point at a local Stripe-shaped mock server. The base URL
    /// must NOT have a trailing slash (we append `/v1/billing/...`).
    pub fn with_base_url(secret: String, store: DynJobStore, base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(STRIPE_TIMEOUT)
            .build()
            .expect("reqwest client");
        Self {
            backend: Backend::Live {
                http,
                secret,
                base_url: base_url.trim_end_matches('/').to_string(),
            },
            store,
        }
    }

    /// Construct from environment. Returns `(client, mode)` where mode is
    /// `"live"` (real Stripe), `"test"` (real Stripe, test-mode key), or
    /// `"dry-run"` (no Stripe credentials — log only).
    pub fn from_env(store: DynJobStore) -> (Self, &'static str) {
        match std::env::var("STRIPE_SECRET_KEY").ok().filter(|s| !s.is_empty()) {
            Some(secret) => {
                let live_mode = std::env::var("STRIPE_LIVE_MODE")
                    .map(|v| matches!(v.as_str(), "true" | "1" | "yes"))
                    .unwrap_or(false);
                let is_live_key = secret.starts_with("sk_live_");
                if is_live_key && !live_mode {
                    warn!(
                        "STRIPE_SECRET_KEY is a sk_live_ key but STRIPE_LIVE_MODE != true; \
                         refusing to use the live key, falling back to dry-run mode"
                    );
                    return (Self::dry_run(store), "dry-run");
                }
                let mode = if is_live_key { "live" } else { "test" };
                (Self::live(secret, store), mode)
            }
            None => (Self::dry_run(store), "dry-run"),
        }
    }

    /// Dry-run mode. Used when Stripe isn't configured (local dev, CI,
    /// or while the operator is still in the manual-charging stage).
    /// Logs the would-be charge at INFO and returns success.
    pub fn dry_run(store: DynJobStore) -> Self {
        Self {
            backend: Backend::DryRun,
            store,
        }
    }

    /// Record one repack run for billing. Emits *both* SKUs:
    ///   * `splatforge_repack_runs` — 1 unit (per-job flat fee)
    ///   * `splatforge_repack_seconds` — `compute_seconds` units, only if
    ///     `compute_seconds.is_some()` (callback path knows the elapsed
    ///     time; the synchronous dispatch path generally doesn't)
    ///
    /// `customer_id` is the Stripe customer (`cus_xxx`). Free-tier jobs
    /// pass `None` and short-circuit before any network call.
    ///
    /// Idempotent: same `job_id` → same Stripe identifier. The ledger
    /// rejects duplicate (job_id, sku) tuples *before* we hit Stripe,
    /// and Stripe also dedupes on the `identifier` field, so we're safe
    /// against both local retries and webhook double-fires.
    #[instrument(skip(self), fields(job_id = %job_id, sku))]
    pub async fn record_repack_job(
        &self,
        job_id: Uuid,
        customer_id: Option<&str>,
        scene_size_bytes: u64,
        iterations: u32,
        compute_seconds: Option<u64>,
    ) -> Result<(), BillingError> {
        let Some(customer_id) = customer_id else {
            // Free tier (or paid-key-without-customer-mapping). The
            // free pipeline is free; we do not emit a meter event.
            info!(
                %job_id,
                scene_size_bytes,
                iterations,
                "billing skip: no customer_id"
            );
            return Ok(());
        };

        // 1 run, always.
        self.post_meter_event(job_id, customer_id, SKU_REPACK_RUNS, 1).await?;

        // Compute-seconds when available. The /repack synchronous handler
        // doesn't know elapsed time; the Modal callback does. Both call
        // paths funnel through here.
        if let Some(secs) = compute_seconds {
            if secs > 0 {
                self.post_meter_event(job_id, customer_id, SKU_REPACK_SECONDS, secs)
                    .await?;
            }
        }
        Ok(())
    }

    /// Internal: claim → POST. The claim is the no-double-charge gate;
    /// Stripe's own idempotency is belt-and-braces.
    async fn post_meter_event(
        &self,
        job_id: Uuid,
        customer_id: &str,
        sku: &str,
        units: u64,
    ) -> Result<(), BillingError> {
        let idempotency_key = idempotency_key_for(&job_id, sku);

        // 1. Claim the ledger slot. UNIQUE(job_id, sku) means concurrent
        //    callers serialize and only one wins; the loser gets `false`
        //    and short-circuits without contacting Stripe.
        let fresh = self
            .store
            .claim_billing_event(&job_id, customer_id, sku, units, &idempotency_key)
            .await?;
        if !fresh {
            info!(
                %job_id, sku, units,
                "billing dedupe: ledger already has this (job_id, sku); skipping Stripe call"
            );
            return Ok(());
        }

        // 2. Post to Stripe (or log in dry-run).
        match &self.backend {
            Backend::DryRun => {
                info!(
                    %job_id,
                    sku,
                    units,
                    customer_id,
                    idempotency_key,
                    "billing dry-run: would post meter event"
                );
                Ok(())
            }
            Backend::Live { http, secret, base_url } => {
                let url = format!("{base_url}/v1/billing/meter_events");
                let form = [
                    ("event_name", sku.to_string()),
                    ("payload[stripe_customer_id]", customer_id.to_string()),
                    ("payload[value]", units.to_string()),
                    ("identifier", idempotency_key.clone()),
                ];
                let resp = http
                    .post(&url)
                    .basic_auth(secret, Some(""))
                    // Stripe's own idempotency-key header. Belt-and-braces:
                    // even if Stripe ever changes how `identifier` dedupes,
                    // this header is the canonical "do not double-process".
                    .header("Idempotency-Key", &idempotency_key)
                    .form(&form)
                    .send()
                    .await
                    .map_err(|e| BillingError::Transport(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    warn!(%job_id, sku, %status, %body, "stripe meter event failed");
                    return Err(BillingError::Stripe {
                        status: status.as_u16(),
                        body,
                    });
                }
                let body: serde_json::Value =
                    resp.json().await.map_err(|e| BillingError::Transport(e.to_string()))?;
                let stripe_event_id = body
                    .get("identifier")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&idempotency_key)
                    .to_string();
                let _ = self
                    .store
                    .mark_billing_event_posted(&job_id, sku, &stripe_event_id)
                    .await;
                info!(%job_id, sku, units, %stripe_event_id, "stripe meter event posted");
                Ok(())
            }
        }
    }
}

/// Derive a stable, Stripe-safe idempotency key for a (job_id, sku) pair.
///
/// Stripe allows up to 255 characters for the meter event `identifier`
/// (and the `Idempotency-Key` header). We use `sha256(job_id || ":" ||
/// sku || ":billing")` and hex-encode — short, collision-resistant, and
/// deterministic so a retry computes the same key. The literal suffix
/// `:billing` namespaces the hash so we never collide with any other
/// hash-of-job-id our infrastructure produces.
pub fn idempotency_key_for(job_id: &Uuid, sku: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(job_id.as_bytes());
    hasher.update(b":");
    hasher.update(sku.as_bytes());
    hasher.update(b":billing");
    let digest = hasher.finalize();
    // 64 hex chars — well under Stripe's 255-char cap. Prefix with the
    // SKU so a human inspecting the ledger can scan SKUs without
    // running sha256sum.
    format!("sf_{sku}_{}", hex::encode(digest))
}

/* ---------- key → customer mapping ---------- */

/// Resolved at startup from `SPLATFORGE_KEY_CUSTOMERS`. Format:
///
///   key1:cus_xxx,key2:cus_yyy
///
/// Unknown keys map to `None` (free tier — no billing).
#[derive(Clone, Default)]
pub struct KeyCustomerMap {
    inner: std::collections::HashMap<String, String>,
}

impl KeyCustomerMap {
    pub fn parse(raw: Option<String>) -> Self {
        let Some(raw) = raw else {
            return Self::default();
        };
        let mut inner = std::collections::HashMap::new();
        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            // We split on the *last* ':' so a key that happens to contain a
            // colon (legal in our bearer alphabet) still parses; the
            // customer id `cus_xxx` never contains one.
            let (key, customer) = match entry.rsplit_once(':') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => {
                    warn!(entry, "SPLATFORGE_KEY_CUSTOMERS: entry missing ':' separator, skipping");
                    continue;
                }
            };
            if key.is_empty() || customer.is_empty() {
                warn!(entry, "SPLATFORGE_KEY_CUSTOMERS: empty key or customer id, skipping");
                continue;
            }
            if !customer.starts_with("cus_") {
                warn!(
                    customer,
                    "SPLATFORGE_KEY_CUSTOMERS: customer id does not start with 'cus_'; \
                     accepting but this is probably a typo"
                );
            }
            inner.insert(key.to_string(), customer.to_string());
        }
        Self { inner }
    }

    pub fn lookup(&self, key: &str) -> Option<&str> {
        self.inner.get(key).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/* ---------- webhook signature verification ---------- */

/// Result of parsing/verifying a Stripe-Signature header.
#[derive(Debug, thiserror::Error)]
pub enum WebhookError {
    #[error("missing Stripe-Signature header")]
    MissingSignature,
    #[error("malformed Stripe-Signature header: {0}")]
    Malformed(String),
    #[error("webhook timestamp outside tolerance ({delta_s}s > {tolerance_s}s)")]
    StaleTimestamp { delta_s: i64, tolerance_s: i64 },
    #[error("no v1 signature matched the payload")]
    BadSignature,
    #[error("invalid JSON body: {0}")]
    BadJson(String),
}

/// Default tolerance for the Stripe-Signature timestamp. Matches the
/// Stripe-recommended 5-minute window from
/// https://docs.stripe.com/webhooks/signatures.
pub const WEBHOOK_DEFAULT_TOLERANCE_SECS: i64 = 300;

/// Verify a Stripe webhook signature against the raw request body and
/// return the parsed JSON event. Caller is responsible for handing us the
/// raw bytes — any framework that re-serializes the body invalidates the
/// signature.
///
/// Constant-time comparison via `subtle::ConstantTimeEq` so a timing
/// attack can't probe the signing secret.
pub fn verify_webhook(
    raw_body: &[u8],
    sig_header: Option<&str>,
    secret: &str,
    now_unix: i64,
    tolerance_s: i64,
) -> Result<serde_json::Value, WebhookError> {
    use hmac::{Hmac, Mac};
    use subtle::ConstantTimeEq;

    let header = sig_header.ok_or(WebhookError::MissingSignature)?;
    let mut timestamp: Option<i64> = None;
    let mut v1_sigs: Vec<&str> = Vec::new();
    for part in header.split(',') {
        let (k, v) = part
            .split_once('=')
            .ok_or_else(|| WebhookError::Malformed(format!("missing '=' in part: {part}")))?;
        match k.trim() {
            "t" => {
                timestamp = Some(
                    v.trim()
                        .parse::<i64>()
                        .map_err(|e| WebhookError::Malformed(format!("bad t=: {e}")))?,
                );
            }
            "v1" => v1_sigs.push(v.trim()),
            _ => { /* ignore v0 / unknown schemes */ }
        }
    }
    let ts = timestamp.ok_or_else(|| WebhookError::Malformed("missing t=".into()))?;
    if v1_sigs.is_empty() {
        return Err(WebhookError::Malformed("no v1= signature present".into()));
    }
    let delta = (now_unix - ts).abs();
    if delta > tolerance_s {
        return Err(WebhookError::StaleTimestamp {
            delta_s: delta,
            tolerance_s,
        });
    }

    // signed_payload = "{t}.{raw_body}"
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(secret.as_bytes())
        .expect("HMAC-SHA256 accepts any key length");
    mac.update(ts.to_string().as_bytes());
    mac.update(b".");
    mac.update(raw_body);
    let expected = mac.finalize().into_bytes();
    let expected_hex = hex::encode(expected);

    let mut ok = false;
    for sig in v1_sigs {
        // Same-length constant-time compare; differing lengths short-circuit
        // to false but in constant time relative to `sig`.
        if sig.len() == expected_hex.len()
            && sig.as_bytes().ct_eq(expected_hex.as_bytes()).into()
        {
            ok = true;
            break;
        }
    }
    if !ok {
        return Err(WebhookError::BadSignature);
    }

    serde_json::from_slice(raw_body).map_err(|e| WebhookError::BadJson(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::JobStore;
    use hmac::{Hmac, Mac};
    use std::sync::Arc;

    #[test]
    fn idempotency_key_is_deterministic_and_namespaced() {
        let id = Uuid::parse_str("11111111-2222-3333-4444-555555555555").unwrap();
        let a = idempotency_key_for(&id, SKU_REPACK_RUNS);
        let b = idempotency_key_for(&id, SKU_REPACK_RUNS);
        assert_eq!(a, b, "same input must produce same key");
        assert!(a.starts_with("sf_splatforge_repack_runs_"));
        let c = idempotency_key_for(&id, SKU_REPACK_SECONDS);
        assert_ne!(a, c, "different SKU must produce different key");
        assert!(a.len() <= 255, "stripe Idempotency-Key cap is 255 chars");
    }

    #[test]
    fn idempotency_key_differs_per_job() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        assert_ne!(
            idempotency_key_for(&a, SKU_REPACK_RUNS),
            idempotency_key_for(&b, SKU_REPACK_RUNS),
        );
    }

    #[test]
    fn key_customer_map_parses_basic_form() {
        let m = KeyCustomerMap::parse(Some("key1:cus_aaa,key2:cus_bbb".into()));
        assert_eq!(m.lookup("key1"), Some("cus_aaa"));
        assert_eq!(m.lookup("key2"), Some("cus_bbb"));
        assert_eq!(m.lookup("key3"), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn key_customer_map_tolerates_whitespace_and_blanks() {
        let m = KeyCustomerMap::parse(Some(" key1 : cus_aaa , , key2:cus_bbb ".into()));
        assert_eq!(m.lookup("key1"), Some("cus_aaa"));
        assert_eq!(m.lookup("key2"), Some("cus_bbb"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn key_customer_map_handles_none_and_empty() {
        assert!(KeyCustomerMap::parse(None).is_empty());
        assert!(KeyCustomerMap::parse(Some("".into())).is_empty());
        assert!(KeyCustomerMap::parse(Some("garbage_no_colon".into())).is_empty());
    }

    #[tokio::test]
    async fn no_double_charge_invariant() {
        // Same (job_id, sku) claimed twice -> second call returns false
        // and (in production) skips the Stripe POST. This is the
        // load-bearing invariant the whole module exists to enforce.
        let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
        let job_id = Uuid::new_v4();
        let first = store
            .claim_billing_event(&job_id, "cus_aaa", SKU_REPACK_RUNS, 1, "key1")
            .await
            .expect("claim");
        let second = store
            .claim_billing_event(&job_id, "cus_aaa", SKU_REPACK_RUNS, 1, "key1")
            .await
            .expect("claim");
        assert!(first, "first claim must succeed");
        assert!(!second, "second claim must be a no-op (no double charge)");

        // A *different* SKU on the same job is a separate claim — both
        // SKUs are emitted per repack run.
        let third = store
            .claim_billing_event(&job_id, "cus_aaa", SKU_REPACK_SECONDS, 42, "key2")
            .await
            .expect("claim");
        assert!(third, "different SKU on same job must claim independently");
    }

    #[tokio::test]
    async fn dry_run_does_not_charge_but_records_ledger() {
        let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
        let client = BillingClient::dry_run(store.clone());
        let job_id = Uuid::new_v4();
        client
            .record_repack_job(job_id, Some("cus_aaa"), 287_000_000, 1000, Some(18))
            .await
            .expect("dry-run record");
        // Two SKUs emitted -> two ledger rows -> retry must be a no-op.
        client
            .record_repack_job(job_id, Some("cus_aaa"), 287_000_000, 1000, Some(18))
            .await
            .expect("dry-run retry");
    }

    #[tokio::test]
    async fn free_tier_emits_no_events() {
        let store: DynJobStore = Arc::new(JobStore::in_memory().await.expect("store"));
        let client = BillingClient::dry_run(store.clone());
        let job_id = Uuid::new_v4();
        // customer_id = None -> free tier -> short-circuit, no ledger row.
        client
            .record_repack_job(job_id, None, 287_000_000, 1000, Some(18))
            .await
            .expect("free record");
        // After a "free" call, claiming for real must still succeed
        // because the previous call wrote nothing to the ledger.
        let fresh = store
            .claim_billing_event(&job_id, "cus_aaa", SKU_REPACK_RUNS, 1, "k")
            .await
            .expect("claim");
        assert!(fresh, "free-tier call must not poison the ledger");
    }

    fn sign(secret: &str, ts: i64, body: &[u8]) -> String {
        let mut mac =
            <Hmac<Sha256> as Mac>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(ts.to_string().as_bytes());
        mac.update(b".");
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    #[test]
    fn webhook_verify_accepts_valid_signature() {
        let secret = "whsec_test";
        let body = br#"{"id":"evt_1","type":"customer.subscription.updated"}"#;
        let ts = 1_700_000_000_i64;
        let sig = sign(secret, ts, body);
        let header = format!("t={ts},v1={sig}");
        let v = verify_webhook(body, Some(&header), secret, ts + 1, 300).expect("verify");
        assert_eq!(v["id"], "evt_1");
    }

    #[test]
    fn webhook_verify_rejects_tampered_body() {
        let secret = "whsec_test";
        let body = br#"{"id":"evt_1"}"#;
        let ts = 1_700_000_000_i64;
        let sig = sign(secret, ts, body);
        let header = format!("t={ts},v1={sig}");
        let tampered = br#"{"id":"evt_2"}"#;
        let err = verify_webhook(tampered, Some(&header), secret, ts, 300)
            .expect_err("tampered body must fail");
        assert!(matches!(err, WebhookError::BadSignature));
    }

    #[test]
    fn webhook_verify_rejects_stale_timestamp() {
        let secret = "whsec_test";
        let body = br#"{}"#;
        let ts = 1_700_000_000_i64;
        let sig = sign(secret, ts, body);
        let header = format!("t={ts},v1={sig}");
        // 10 minutes later, default tolerance is 5 minutes.
        let err = verify_webhook(body, Some(&header), secret, ts + 600, 300)
            .expect_err("stale must fail");
        assert!(matches!(err, WebhookError::StaleTimestamp { .. }));
    }

    #[test]
    fn webhook_verify_rejects_missing_header() {
        let err = verify_webhook(b"{}", None, "whsec_test", 0, 300)
            .expect_err("missing header must fail");
        assert!(matches!(err, WebhookError::MissingSignature));
    }
}
