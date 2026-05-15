//! Self-serve Team-tier signup via Stripe Checkout.
//!
//! ## Why this module exists
//!
//! Before today, the only way to become a paying SplatForge customer
//! was the operator manually mapping a bearer key to a Stripe customer
//! id in `SPLATFORGE_KEY_CUSTOMERS`. That gate doesn't scale past the
//! design-partner cohort. This module wires the full self-serve loop:
//!
//!   1. Buyer hits `/pricing` → POSTs `/v1/checkout/create-session`.
//!   2. We create a Stripe Checkout Session for the Team-tier price
//!      ($99/mo per seat) and return its URL.
//!   3. Buyer pays. Stripe POSTs `checkout.session.completed` to
//!      `/v1/checkout/webhook`.
//!   4. The webhook mints a fresh `sf_live_<24chars>` API key,
//!      records it in `team_signups` (SHA-256 hash + display prefix
//!      only — plaintext NEVER hits disk), and caches the plaintext
//!      in memory keyed by the session id for at most 10 minutes.
//!   5. Stripe redirects the buyer to `/welcome?session_id=…&token=…`.
//!   6. `/welcome` calls `/v1/checkout/reveal`, gets the plaintext
//!      ONCE, displays it with a "copy now" warning, and the row is
//!      flipped to `key_revealed_at NOT NULL` so a second call 410s.
//!
//! ## The "exactly once" invariant
//!
//! The plaintext key crosses the wire exactly once, on the
//! `/v1/checkout/reveal` response. Three layers enforce this:
//!
//!   * **DB:** `mark_team_signup_revealed` does an atomic
//!     `UPDATE … WHERE key_revealed_at IS NULL` and returns false on
//!     the second hit. The endpoint refuses the second hit with 410.
//!   * **Memory:** The plaintext lives in `PendingKeyCache` (a
//!     `Mutex<HashMap>`). On a successful reveal the entry is
//!     `remove`'d, so even if the DB flag fails open the cache is
//!     empty.
//!   * **TTL:** `PendingKeyCache::sweep_expired` drops entries older
//!     than 10 minutes. A buyer who closes their browser before
//!     hitting `/welcome` cannot recover the key — they email
//!     support, which rotates them a fresh one. This is the single
//!     biggest customer-loss risk in the funnel; see CHECKOUT.md.
//!
//! ## Webhook idempotency
//!
//! Stripe retries `checkout.session.completed` on any non-2xx and on
//! network timeouts. We rely on two layers:
//!
//!   * `verify_webhook` (in `billing.rs`) rejects events with a stale
//!     `t=` timestamp — replay-after-5-minutes is rejected by the
//!     signature gate before we touch the DB.
//!   * `claim_team_signup` uses
//!     `INSERT … ON CONFLICT(stripe_session_id) DO NOTHING`. A retry
//!     that lands a second time finds the row present, returns
//!     `Ok(false)`, and we 200 back to Stripe without re-minting a key.
//!
//! The combination means: two simultaneous webhook deliveries for the
//! same session id can't both mint keys, and the SECOND mint can't
//! overwrite the plaintext that the first one cached. Tested in
//! `tests/checkout.rs::webhook_idempotent_under_double_delivery`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::distributions::Alphanumeric;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;
use tracing::{info, instrument, warn};

use crate::store::{JobStoreApi, StoreError, TeamSignupRow};

/// Prefix every freshly-minted SplatForge API key starts with. Matches
/// the WorkOS branch's `auth::KEY_PREFIX_LITERAL` so once that branch
/// merges, keys minted by this module are byte-for-byte indistinguishable
/// from keys minted by the dashboard / SSO admin path.
pub const KEY_PREFIX_LITERAL: &str = "sf_live_";
/// Total plaintext length: `sf_live_` (8) + 24 random alphanumeric.
/// Lines up with `auth::KEY_PLAINTEXT_LEN = 32` on the WorkOS branch.
pub const KEY_PLAINTEXT_LEN: usize = 32;
/// How many chars of the plaintext are safe to display. `sf_live_XXXX` —
/// distinguishable in the dashboard while revealing only 4 bits of
/// post-prefix randomness.
pub const KEY_DISPLAY_PREFIX_LEN: usize = 12;

/// Length of the per-session `claim_token` that gates `/reveal`. 32
/// alphanumeric chars = ~190 bits of entropy. The buyer never sees
/// or types this — it round-trips through `success_url`.
pub const CLAIM_TOKEN_LEN: usize = 32;

/// How long a webhook may sit before the buyer hits `/welcome`.
/// 10 minutes covers the realistic worst case (buyer closes the
/// Stripe tab, opens it on their phone, walks to a desk, pastes the
/// success URL). After this, the in-memory plaintext is GC'd and the
/// reveal endpoint returns 410. The customer emails support and the
/// operator rotates them a fresh key via the WorkOS-branch admin API.
pub const PENDING_KEY_TTL: Duration = Duration::from_secs(10 * 60);

/// Stripe API base. Mirrors `billing.rs::STRIPE_API_BASE` — the same
/// constant on both sides keeps swapping to a mock server (in tests) a
/// one-arg change.
pub const STRIPE_API_BASE: &str = "https://api.stripe.com";

/// HTTP timeout for the create-session call. Checkout session creation
/// is normally sub-second; 10s caps the worst-case stall at the same
/// budget the meter-event poster uses.
pub const STRIPE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum CheckoutError {
    #[error("stripe not configured (set STRIPE_SECRET_KEY)")]
    NotConfigured,
    #[error("stripe team price not configured (set STRIPE_TEAM_PRICE_ID)")]
    PriceNotConfigured,
    #[error("stripe request transport: {0}")]
    Transport(String),
    #[error("stripe rejected request: {status}: {body}")]
    Stripe { status: u16, body: String },
    #[error("stripe response missing field: {0}")]
    BadResponse(String),
    #[error("store: {0}")]
    Store(#[from] StoreError),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("not found")]
    NotFound,
    #[error("gone (already revealed once)")]
    Gone,
    #[error("forbidden (bad claim token)")]
    Forbidden,
}

/// Top-level configuration for the checkout module. Built from env
/// once at startup; carried in `AppState` so handlers don't re-parse.
#[derive(Clone)]
pub struct CheckoutConfig {
    /// `sk_test_…` or `sk_live_…`. `None` means dry-run mode (the
    /// `/create-session` endpoint returns 503 with a clear error, the
    /// welcome page renders a stub).
    pub stripe_secret: Option<String>,
    /// Stripe price id for the Team tier. Provisioned in the Stripe
    /// dashboard at $99/mo per seat. We deliberately do NOT mint this
    /// from code (see `tasks/scripts/stripe-bootstrap.sh` rationale —
    /// pricing is a commercial decision that lives in the dashboard).
    pub team_price_id: Option<String>,
    /// Base URL of the public web app. `success_url` is built as
    /// `{public_site_url}/welcome?session_id=…&token=…`.
    pub public_site_url: String,
    /// Override for the Stripe API base. Production = `https://api.stripe.com`;
    /// tests inject `http://127.0.0.1:PORT` for the in-process mock.
    pub stripe_base_url: String,
    /// If false (the default), refuse to create live-mode sessions
    /// even if a `sk_live_` key is configured. Same belt-and-braces
    /// pattern as `BillingClient::from_env`.
    pub live_mode: bool,
}

impl CheckoutConfig {
    pub fn from_env(public_site_url: String) -> Self {
        let secret = std::env::var("STRIPE_SECRET_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let live_mode = std::env::var("STRIPE_LIVE_MODE")
            .map(|v| matches!(v.as_str(), "true" | "1" | "yes"))
            .unwrap_or(false);
        if let Some(s) = &secret {
            if s.starts_with("sk_live_") && !live_mode {
                warn!(
                    "STRIPE_SECRET_KEY is a sk_live_ key but STRIPE_LIVE_MODE != true — \
                     /v1/checkout/create-session will refuse to create real-money sessions"
                );
            }
        }
        Self {
            stripe_secret: secret,
            team_price_id: std::env::var("STRIPE_TEAM_PRICE_ID")
                .ok()
                .filter(|s| !s.is_empty()),
            public_site_url: public_site_url.trim_end_matches('/').to_string(),
            stripe_base_url: STRIPE_API_BASE.to_string(),
            live_mode,
        }
    }

    /// Test-only override. Lets `tests/checkout.rs` point at the
    /// in-process Stripe-mock without poking env vars.
    pub fn with_overrides(
        mut self,
        stripe_secret: Option<String>,
        team_price_id: Option<String>,
        stripe_base_url: Option<String>,
    ) -> Self {
        if let Some(s) = stripe_secret {
            self.stripe_secret = Some(s);
        }
        if let Some(p) = team_price_id {
            self.team_price_id = Some(p);
        }
        if let Some(b) = stripe_base_url {
            self.stripe_base_url = b;
        }
        self
    }

    /// Returns true if `create_session` will produce a real Stripe URL
    /// instead of a 503. The welcome-page stub uses this same check.
    pub fn is_live(&self) -> bool {
        let Some(secret) = self.stripe_secret.as_deref() else {
            return false;
        };
        if secret.starts_with("sk_live_") && !self.live_mode {
            return false;
        }
        self.team_price_id.is_some()
    }
}

/// One pending (paid-but-not-yet-revealed) key. Lives in memory only;
/// crash-safety for this window is acceptable because the buyer can
/// re-trigger via support — losing the plaintext is preferable to
/// persisting it.
#[derive(Clone)]
struct PendingKey {
    plaintext: String,
    minted_at: Instant,
}

/// Process-local cache of revealed-once plaintext keys, keyed by Stripe
/// session id. Bounded by `PENDING_KEY_TTL` (10 min) — anything older
/// is dropped on the next sweep.
#[derive(Default)]
pub struct PendingKeyCache {
    inner: Mutex<HashMap<String, PendingKey>>,
}

impl PendingKeyCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, session_id: String, plaintext: String) {
        let mut g = self.inner.lock().await;
        g.insert(
            session_id,
            PendingKey {
                plaintext,
                minted_at: Instant::now(),
            },
        );
    }

    /// Returns the plaintext if present AND fresh (< TTL). Removes the
    /// entry in either case — this is a one-shot read. Stale entries
    /// are returned as `None` so the caller errors out with `Gone`.
    pub async fn take_if_fresh(&self, session_id: &str) -> Option<String> {
        let mut g = self.inner.lock().await;
        let Some(entry) = g.remove(session_id) else {
            return None;
        };
        if entry.minted_at.elapsed() > PENDING_KEY_TTL {
            return None;
        }
        Some(entry.plaintext)
    }

    pub async fn sweep_expired(&self) {
        let mut g = self.inner.lock().await;
        g.retain(|_, v| v.minted_at.elapsed() <= PENDING_KEY_TTL);
    }
}

/* ---------- minting + idempotency ---------- */

/// Generate a fresh plaintext key (`sf_live_<24 alphanumeric>`) and
/// return `(plaintext, display_prefix, sha256_hex)`. Plaintext is
/// returned to the caller exactly once; only the prefix + hash are
/// persisted in `team_signups`. Matches the `auth::mint_key` contract
/// on the WorkOS SSO branch byte-for-byte (same prefix, same length,
/// same hash function).
pub fn mint_team_api_key() -> (String, String, String) {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(KEY_PLAINTEXT_LEN - KEY_PREFIX_LITERAL.len())
        .map(char::from)
        .collect();
    let plaintext = format!("{KEY_PREFIX_LITERAL}{suffix}");
    debug_assert_eq!(plaintext.len(), KEY_PLAINTEXT_LEN);
    let prefix = plaintext[..KEY_DISPLAY_PREFIX_LEN].to_string();
    let hash = hash_key(&plaintext);
    (plaintext, prefix, hash)
}

/// Hex-encoded SHA-256 of a plaintext key. Same contract as
/// `auth::hash_key` on the WorkOS branch.
pub fn hash_key(plaintext: &str) -> String {
    hex::encode(Sha256::digest(plaintext.as_bytes()))
}

/// Random claim token returned via `success_url`. Without this, anyone
/// who could enumerate session ids (e.g. by snooping the merchant
/// dashboard) could steal a fresh customer's key. Pure entropy — never
/// derived from the session id.
pub fn mint_claim_token() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(CLAIM_TOKEN_LEN)
        .map(char::from)
        .collect()
}

/// Stripe Idempotency-Key for the create-session POST. Stripe caches
/// the response under this key for 24h, so a buyer who double-clicks
/// the Team CTA gets the same checkout URL back instead of a second
/// session id. Format: `sf_checkout_<sha256(email || ":" || nonce)>`.
///
/// The nonce is a per-request UUID, not deterministic — we *want* a
/// fresh nonce on each genuine click (so the buyer can retry after a
/// validation error). But within a single request, the idempotency
/// must survive a transport retry of *that exact request*, hence
/// hashing instead of re-randomising on every Stripe call.
pub fn checkout_idempotency_key(email: &str, nonce: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(email.as_bytes());
    hasher.update(b":");
    hasher.update(nonce.as_bytes());
    hasher.update(b":checkout");
    let digest = hasher.finalize();
    format!("sf_checkout_{}", hex::encode(digest))
}

/* ---------- stripe client ---------- */

/// Response from `POST /v1/checkout/create-session`. The client
/// redirects the browser to `url`.
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub url: String,
    pub session_id: String,
}

/// What the buyer-facing `/welcome` page receives from
/// `/v1/checkout/reveal`. `key` is the plaintext, returned EXACTLY
/// ONCE — the row is flipped to `key_revealed_at NOT NULL` before
/// this struct is built, and the in-memory cache entry is removed.
#[derive(Debug, Serialize)]
pub struct RevealResponse {
    pub api_key: String,
    pub key_prefix: String,
    pub email: String,
    /// "Bearer sf_live_xxx" — the exact `Authorization` header value
    /// the buyer should send. Computed here so the frontend doesn't
    /// have to re-invent the prefix.
    pub authorization_header: String,
}

/// Lightweight client around the Stripe API endpoints we touch
/// (`POST /v1/checkout/sessions`). Kept separate from `billing.rs`'s
/// `BillingClient` because the surfaces don't overlap and conflating
/// them would force every checkout test to spin up a metered-billing
/// ledger.
#[derive(Clone)]
pub struct StripeCheckoutClient {
    http: reqwest::Client,
    secret: String,
    base_url: String,
}

impl StripeCheckoutClient {
    pub fn new(secret: String, base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(STRIPE_TIMEOUT)
            .build()
            .expect("reqwest client");
        Self {
            http,
            secret,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// `POST /v1/checkout/sessions` with form-encoded params. Returns
    /// `(session_id, url)`.
    #[instrument(skip(self, success_url, cancel_url), fields(email = %email))]
    pub async fn create_session(
        &self,
        price_id: &str,
        email: &str,
        success_url: &str,
        cancel_url: &str,
        idempotency_key: &str,
    ) -> Result<(String, String), CheckoutError> {
        let url = format!("{}/v1/checkout/sessions", self.base_url);
        // Stripe's Checkout Session API. `mode=subscription` because
        // the Team tier is recurring ($99/mo per seat). `line_items[0]`
        // is the Team price; quantity is left at 1 here — the seats
        // model is "buy one base seat, add more via the customer
        // portal". `client_reference_id` is the per-request nonce so
        // we can correlate the webhook back to the local intent log
        // even before the session id is known (e.g. abandoned carts).
        let form = [
            ("mode", "subscription"),
            ("customer_email", email),
            ("line_items[0][price]", price_id),
            ("line_items[0][quantity]", "1"),
            ("success_url", success_url),
            ("cancel_url", cancel_url),
            // Capture the email + a billing-portal-friendly customer
            // record. Without `allow_promotion_codes`, the dashboard
            // CLI can't apply launch discounts later.
            ("allow_promotion_codes", "true"),
            // Bill on Stripe's side; we don't need to collect a tax
            // jurisdiction here. Automatic tax becomes an operator
            // toggle once we cross the SaaS-tax thresholds.
            ("automatic_tax[enabled]", "false"),
        ];
        let resp = self
            .http
            .post(&url)
            .basic_auth(&self.secret, Some(""))
            .header("Idempotency-Key", idempotency_key)
            .form(&form)
            .send()
            .await
            .map_err(|e| CheckoutError::Transport(e.to_string()))?;
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            warn!(%status, body = %body_text, "stripe create_session failed");
            return Err(CheckoutError::Stripe {
                status: status.as_u16(),
                body: body_text,
            });
        }
        let body: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| CheckoutError::Transport(format!("decoding stripe response: {e}")))?;
        let session_id = body
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckoutError::BadResponse("missing `id`".to_string()))?
            .to_string();
        let url = body
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckoutError::BadResponse("missing `url`".to_string()))?
            .to_string();
        info!(session_id, "stripe checkout session created");
        Ok((session_id, url))
    }
}

/* ---------- request shapes ---------- */

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub email: String,
    /// Optional caller-supplied nonce for the idempotency key. The
    /// frontend mints this so a fast double-click on the Team CTA gets
    /// the same Stripe session back instead of two parallel sessions.
    /// Omitted? We mint one ourselves.
    #[serde(default)]
    pub nonce: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RevealRequest {
    pub session_id: String,
    pub token: String,
}

/* ---------- end-to-end provisioning ---------- */

/// Validate + create a Stripe Checkout Session for the Team tier.
///
/// Pure function over the (config, client, store) tuple so the route
/// handler in `main.rs` is a 5-line shim and the test in
/// `tests/checkout.rs` doesn't need an axum runtime.
#[instrument(skip(config, client))]
pub async fn create_session(
    config: &CheckoutConfig,
    client: &StripeCheckoutClient,
    req: CreateSessionRequest,
) -> Result<CreateSessionResponse, CheckoutError> {
    let email = req.email.trim();
    if email.is_empty() || !email.contains('@') {
        return Err(CheckoutError::BadRequest(
            "email is required and must look like an email".to_string(),
        ));
    }
    let Some(price_id) = config.team_price_id.as_deref() else {
        return Err(CheckoutError::PriceNotConfigured);
    };
    let nonce = req.nonce.unwrap_or_else(mint_claim_token);
    let idem = checkout_idempotency_key(email, &nonce);
    // success_url carries `{CHECKOUT_SESSION_ID}` literal which Stripe
    // substitutes server-side before redirecting. We also append the
    // claim_token via a placeholder we own — but Stripe doesn't
    // substitute caller-defined placeholders. Workaround: pre-mint the
    // token here and embed it directly. The webhook will see this
    // same token in the session metadata? No — metadata isn't on the
    // success_url path. Simpler: bind token to session_id in the DB
    // on webhook receipt. The success URL only needs session_id; the
    // welcome page sends BOTH session_id AND a fresh `token` query
    // param. But the buyer doesn't have the token from Stripe's
    // redirect — only session_id.
    //
    // Resolution: the token IS the session_id-prefix of the row's
    // `claim_token`. We pass session_id only through Stripe's
    // {CHECKOUT_SESSION_ID} substitution. The buyer's /welcome reads
    // session_id from the query, posts that to /reveal, and the
    // server compares server-stored claim_token to nothing — the
    // session_id IS the secret. To raise the bar above "guessable
    // session id", we additionally require the reveal call to arrive
    // within PENDING_KEY_TTL (10 min) of the webhook. A session_id
    // alone is 32 chars of cs_test_… entropy — not predictable, but
    // not high-stakes-secret-grade either. Belt: the claim_token IS
    // exposed via the welcome URL by issuing it through the success
    // URL's path. Stripe lets us put `?session_id={CHECKOUT_SESSION_ID}`
    // — any other static path component is preserved. We embed the
    // claim_token in success_url before sending it to Stripe, so the
    // buyer's URL looks like:
    //   /welcome?session_id={CHECKOUT_SESSION_ID}&token=<random32>
    // Stripe interpolates session_id; token is static-from-its-pov.
    // The webhook handler then stores the same token in the DB row;
    // /reveal compares query-token to DB-token.
    let claim_token = mint_claim_token();
    let success_url = format!(
        "{}/welcome?session_id={{CHECKOUT_SESSION_ID}}&token={claim_token}",
        config.public_site_url
    );
    let cancel_url = format!("{}/pricing?canceled=1", config.public_site_url);

    let (session_id, url) = client
        .create_session(price_id, email, &success_url, &cancel_url, &idem)
        .await?;

    // Stash the claim_token under the idempotency key so the webhook
    // (which sees the session id, not the idem key) can recover it.
    // We sidestep the side channel entirely by relying on Stripe's
    // own metadata field instead — see below note. For now, the
    // claim_token round-trips through the success_url and is what
    // /reveal checks against; the webhook generates a parallel token
    // and ignores the one in the URL? No — that would break the
    // "buyer presents the URL we gave them" property.
    //
    // Final design (see CHECKOUT.md): the webhook does NOT see the
    // claim_token directly. The success_url contains the buyer's copy;
    // the buyer POSTs it to /reveal; the server cross-references it
    // against an in-memory map keyed by session_id that's populated
    // by the create-session call. The DB row's `claim_token` column
    // is filled in by the webhook from THE SAME in-memory map. If
    // the API restarts between create-session and the webhook, the
    // map is lost — the customer's URL still works because the
    // webhook regenerates a NEW token, but then /reveal sees a
    // mismatch and refuses. This is fail-closed (customer emails
    // support) which is the right side of the safety boundary.
    let _ = claim_token; // placeholder for the in-memory map plumb below
    Ok(CreateSessionResponse { url, session_id })
}

/// In-memory map from Stripe session id -> the claim_token we baked
/// into that session's success_url. Lives alongside `PendingKeyCache`
/// in `AppState`. Cleared on the same 10-minute sweep.
#[derive(Default)]
pub struct PendingClaimTokens {
    inner: Mutex<HashMap<String, (String, Instant)>>,
}

impl PendingClaimTokens {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, session_id: String, claim_token: String) {
        let mut g = self.inner.lock().await;
        g.insert(session_id, (claim_token, Instant::now()));
    }

    pub async fn take(&self, session_id: &str) -> Option<String> {
        let mut g = self.inner.lock().await;
        let (token, when) = g.remove(session_id)?;
        if when.elapsed() > PENDING_KEY_TTL {
            return None;
        }
        Some(token)
    }

    pub async fn sweep_expired(&self) {
        let mut g = self.inner.lock().await;
        g.retain(|_, (_, t)| t.elapsed() <= PENDING_KEY_TTL);
    }
}

/// Full-fat version of `create_session` that also registers the
/// claim_token in `pending_tokens`. The naked `create_session` exists
/// only because tests want to exercise the Stripe call shape without
/// reaching into the AppState plumbing.
pub async fn create_session_and_register(
    config: &CheckoutConfig,
    client: &StripeCheckoutClient,
    pending_tokens: &PendingClaimTokens,
    req: CreateSessionRequest,
) -> Result<CreateSessionResponse, CheckoutError> {
    let email = req.email.trim().to_string();
    if email.is_empty() || !email.contains('@') {
        return Err(CheckoutError::BadRequest(
            "email is required and must look like an email".to_string(),
        ));
    }
    let Some(price_id) = config.team_price_id.as_deref() else {
        return Err(CheckoutError::PriceNotConfigured);
    };
    let nonce = req.nonce.unwrap_or_else(mint_claim_token);
    let idem = checkout_idempotency_key(&email, &nonce);
    let claim_token = mint_claim_token();
    let success_url = format!(
        "{}/welcome?session_id={{CHECKOUT_SESSION_ID}}&token={claim_token}",
        config.public_site_url
    );
    let cancel_url = format!("{}/pricing?canceled=1", config.public_site_url);
    let (session_id, url) = client
        .create_session(price_id, &email, &success_url, &cancel_url, &idem)
        .await?;
    pending_tokens.insert(session_id.clone(), claim_token).await;
    Ok(CreateSessionResponse { url, session_id })
}

/// Provision a brand-new Team-tier customer from a verified
/// `checkout.session.completed` event.
///
/// The flow:
///   1. Pull session id + customer id + email from the event object.
///   2. Mint a fresh `sf_live_<24>` plaintext key.
///   3. INSERT into `team_signups` with ON CONFLICT DO NOTHING — if a
///      row already exists for this session id, we early-return
///      `Ok(())` (Stripe webhook retry; nothing to do).
///   4. Stash the plaintext in `pending_keys` keyed by session id, for
///      `/v1/checkout/reveal` to pick up.
///
/// Returns `Ok(())` for both fresh provisions AND idempotent retries.
/// Errors only on malformed events or DB transport failure — the
/// caller (the webhook route) MUST return 2xx on `Ok(())` so Stripe
/// stops retrying.
#[instrument(skip(store, pending_keys, pending_tokens, event))]
pub async fn provision_from_session(
    store: &(dyn JobStoreApi + Send + Sync),
    pending_keys: &PendingKeyCache,
    pending_tokens: &PendingClaimTokens,
    event: &serde_json::Value,
) -> Result<(), CheckoutError> {
    // Stripe wraps the session under data.object. We accept both the
    // wrapped form (real webhook) and the bare form (operator manually
    // POSTing a session for replay) by checking object.id existence.
    let session = event
        .pointer("/data/object")
        .or_else(|| event.get("object").filter(|v| v.is_object()))
        .unwrap_or(event);

    let session_id = session
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CheckoutError::BadRequest("event missing data.object.id".to_string()))?;
    let customer_id = session
        .get("customer")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CheckoutError::BadRequest("event missing customer".to_string()))?;
    let subscription_id = session.get("subscription").and_then(|v| v.as_str());
    let email = session
        .get("customer_email")
        .and_then(|v| v.as_str())
        .or_else(|| {
            session
                .pointer("/customer_details/email")
                .and_then(|v| v.as_str())
        })
        .unwrap_or("");
    if email.is_empty() {
        return Err(CheckoutError::BadRequest(
            "event missing customer_email".to_string(),
        ));
    }

    // Recover the claim_token we minted in create-session. If the
    // process restarted between then and now, we mint a fresh one and
    // accept the trade-off: /reveal will reject the buyer's URL and
    // they'll email support. Better than fail-open.
    let claim_token = pending_tokens
        .take(session_id)
        .await
        .unwrap_or_else(mint_claim_token);

    let (plaintext, prefix, hash) = mint_team_api_key();
    let fresh = store
        .claim_team_signup(
            session_id,
            customer_id,
            subscription_id,
            email,
            &claim_token,
            &prefix,
            &hash,
            1, /* seats — buyer can add more in the customer portal */
        )
        .await?;
    if !fresh {
        info!(
            session_id,
            customer_id, "team signup already provisioned; webhook retry — no key minted"
        );
        return Ok(());
    }
    pending_keys.insert(session_id.to_string(), plaintext).await;
    // Belt-and-braces: also drop the claim token back so /reveal can
    // accept it. The DB now has the same value, so this is purely a
    // cache warm-up for the immediate post-checkout reveal.
    pending_tokens
        .insert(session_id.to_string(), claim_token)
        .await;
    info!(
        session_id,
        customer_id,
        email,
        key_prefix = prefix,
        "team-tier API key provisioned; awaiting reveal at /welcome"
    );
    // IMPORTANT: never log the plaintext. The prefix is the only
    // human-readable handle that ever hits the log pipeline.
    Ok(())
}

/// Serve the one-time plaintext key to the legitimate buyer.
///
/// This is the load-bearing "exactly once" point. Three gates, in
/// order:
///
///   1. `mark_team_signup_revealed` — atomic UPDATE that returns false
///      if `key_revealed_at` is already non-null. This is the
///      transactional invariant; even if the cache lies, the DB tells
///      the truth.
///   2. Constant-time claim_token compare — refuses a query with a
///      bad token even if the session id was guessed.
///   3. `pending_keys.take_if_fresh` — if the cache has expired (10
///      min TTL) or was wiped by a restart, we return `Gone`. The
///      hash is on disk so we *could* re-mint, but that would change
///      the contract from "key shown once" to "key reshown from disk
///      at the operator's whim" — refusing is the safe default.
#[instrument(skip(store, pending_keys))]
pub async fn reveal_key(
    store: &(dyn JobStoreApi + Send + Sync),
    pending_keys: &PendingKeyCache,
    req: RevealRequest,
) -> Result<RevealResponse, CheckoutError> {
    let signup: TeamSignupRow = store
        .get_team_signup_by_session(&req.session_id)
        .await?
        .ok_or(CheckoutError::NotFound)?;

    // Compare the URL-supplied token to the stored one in constant
    // time. The stored token is just a random string; the constant-time
    // compare is defence-in-depth — if a future change made the
    // comparison shortcircuit on first byte mismatch a remote
    // timing-oracle attack could enumerate the token.
    if !ct_eq_str(&signup.claim_token, &req.token) {
        return Err(CheckoutError::Forbidden);
    }
    if signup.key_revealed_at.is_some() {
        // The DB has already been flipped. Refuse, even if a stale
        // entry happens to still live in the cache.
        return Err(CheckoutError::Gone);
    }

    // Pull the plaintext from the cache BEFORE flipping the DB column.
    // If the cache is empty/expired we return Gone without poisoning
    // the row — the buyer can email support, the operator can rotate
    // a fresh key, and the (still-clean) row gets a fresh plaintext
    // with `mark_team_signup_revealed` correctly remaining false.
    let Some(plaintext) = pending_keys.take_if_fresh(&req.session_id).await else {
        return Err(CheckoutError::Gone);
    };

    // The flip. If this fails (concurrent reveal raced us between the
    // get + the take), we don't have a way to put the plaintext back
    // — and we don't try. The single-row UPDATE either succeeds, in
    // which case we hand the plaintext to the legitimate buyer, or
    // it fails and we return Gone. The cache eviction above already
    // removed the entry, so a second concurrent caller sees None and
    // also returns Gone. Both attackers and the legitimate buyer get
    // a deterministic single-shot semantic.
    let flipped = store.mark_team_signup_revealed(&req.session_id).await?;
    if !flipped {
        return Err(CheckoutError::Gone);
    }

    let authorization_header = format!("Bearer {plaintext}");
    Ok(RevealResponse {
        api_key: plaintext,
        key_prefix: signup.key_prefix,
        email: signup.email,
        authorization_header,
    })
}

/// Constant-time string compare, length-independent (constant in the
/// length of the *shorter* side; differing lengths short-circuit). The
/// `subtle` crate exposes this on `&[u8]` only.
fn ct_eq_str(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/* ---------- tests ---------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_has_documented_shape() {
        let (plaintext, prefix, hash) = mint_team_api_key();
        assert_eq!(plaintext.len(), KEY_PLAINTEXT_LEN, "32 chars total");
        assert!(plaintext.starts_with(KEY_PREFIX_LITERAL));
        // Suffix is alphanumeric only.
        assert!(plaintext[KEY_PREFIX_LITERAL.len()..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric()));
        assert_eq!(prefix.len(), KEY_DISPLAY_PREFIX_LEN);
        assert!(plaintext.starts_with(&prefix));
        // SHA-256 hex = 64 chars.
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, hash_key(&plaintext));
    }

    #[test]
    fn distinct_mints_produce_distinct_keys() {
        // 24 alphanumeric chars = ~143 bits of entropy. A collision in
        // a 1000-iter loop would be a bug, not bad luck.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let (pt, _, _) = mint_team_api_key();
            assert!(seen.insert(pt), "duplicate key — RNG misconfigured?");
        }
    }

    #[test]
    fn idempotency_key_deterministic_per_email_and_nonce() {
        let a = checkout_idempotency_key("alice@example.com", "n1");
        let b = checkout_idempotency_key("alice@example.com", "n1");
        assert_eq!(a, b);
        // Different nonce -> different key (so retries after a failed
        // submit don't dedupe against the original attempt).
        assert_ne!(a, checkout_idempotency_key("alice@example.com", "n2"));
        // Different email -> different key.
        assert_ne!(a, checkout_idempotency_key("bob@example.com", "n1"));
        assert!(a.starts_with("sf_checkout_"));
    }

    #[tokio::test]
    async fn pending_cache_returns_plaintext_once_then_none() {
        let cache = PendingKeyCache::new();
        cache
            .insert("cs_test_aaa".into(), "sf_live_AAA".into())
            .await;
        let first = cache.take_if_fresh("cs_test_aaa").await;
        assert_eq!(first.as_deref(), Some("sf_live_AAA"));
        let second = cache.take_if_fresh("cs_test_aaa").await;
        assert!(second.is_none(), "second take must be None — one-shot");
    }
}
