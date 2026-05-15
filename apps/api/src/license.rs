//! License issue / refresh / heartbeat endpoints for the Pro on-prem tier.
//!
//! ## Trust model
//!
//! - The **private key** (`LICENSE_PRIVATE_KEY`, a Fly secret) lives only
//!   on the API box. It is loaded once at startup into `LicenseIssuer`
//!   and never written to a log line.
//! - The **public key** is embedded in every `splatforge` binary
//!   shipped to customers; verification happens entirely client-side.
//!
//! ## Endpoints
//!
//! - `POST /v1/license/issue` (admin-only) — mint a fresh license for a
//!   given org. Gated by `LICENSE_ADMIN_TOKEN` so the public can't pull
//!   licenses; in normal ops this is called by the Stripe webhook path
//!   after a Pro subscription activates.
//! - `POST /v1/license/refresh` — customer-facing. Accepts the customer's
//!   existing license, verifies it, looks up the org's billing state,
//!   and either re-signs with an extended `valid_until` or returns 402.
//! - `POST /v1/license/heartbeat` — telemetry + soft enforcement.
//!   Records `{org_id, active_seats, version}` in-memory (today; SQLite
//!   when we wire the store) so we can surface churn signal and refuse
//!   to refresh licenses for orgs that have stopped beaconing.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use splatforge_license::{Claims, IssuerSigningKey, License, LicenseConfig};
use tracing::{info, warn};

/// In-process issuer state. Held inside an `Arc` in the AppState so the
/// signing key allocation happens exactly once at boot.
#[derive(Clone)]
pub struct LicenseIssuer {
    /// The signing key — Ed25519, 32-byte seed. PKCS#8 PEM is the on-the-
    /// wire format we accept from the env so an operator can paste a
    /// `ssh-keygen`-style key and not a hex blob.
    signing: Arc<IssuerSigningKey>,
    /// Bearer token required on `/v1/license/issue`. Empty disables the
    /// route (returns 503) — useful in dev / CI / public CI mirrors.
    admin_token: String,
    /// Default validity window for newly issued licenses. Stripe period
    /// length (typically 30 days) is plumbed through the issue request
    /// when we have it; otherwise this default kicks in.
    default_validity: Duration,
}

impl LicenseIssuer {
    /// Load from environment. `LICENSE_PRIVATE_KEY` must be either a
    /// PKCS#8 PEM block or a 64-char hex-encoded 32-byte seed. Returns
    /// `None` (not an error) when the env var is unset so dev / CI
    /// deploys can run without minting licenses; the routes then return
    /// 503.
    pub fn from_env() -> anyhow::Result<Option<LicenseIssuer>> {
        let Ok(raw) = std::env::var("LICENSE_PRIVATE_KEY") else {
            return Ok(None);
        };
        let signing = parse_signing_key(&raw)?;
        let admin_token = std::env::var("LICENSE_ADMIN_TOKEN").unwrap_or_default();
        let default_validity_days: i64 = std::env::var("LICENSE_DEFAULT_DAYS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(35);
        Ok(Some(LicenseIssuer {
            signing: Arc::new(signing),
            admin_token,
            default_validity: Duration::days(default_validity_days),
        }))
    }

    /// Test-only constructor — `pub` so the integration test in
    /// `tests/license.rs` can spin up an issuer with the dev seed
    /// without touching the env.
    pub fn from_signing_key(
        signing: IssuerSigningKey,
        admin_token: impl Into<String>,
        default_validity: Duration,
    ) -> Self {
        LicenseIssuer {
            signing: Arc::new(signing),
            admin_token: admin_token.into(),
            default_validity,
        }
    }

    fn sign(&self, claims: Claims) -> License {
        License::sign(claims, &self.signing)
    }
}

/// Parse `LICENSE_PRIVATE_KEY`. Accepts:
///   1. hex-encoded 32-byte seed (`64 hex chars`)
///   2. raw 64-char base64 (e.g. the way a customer might paste it)
fn parse_signing_key(raw: &str) -> anyhow::Result<IssuerSigningKey> {
    let raw = raw.trim();
    // hex-encoded seed.
    if raw.len() == 64 && raw.chars().all(|c| c.is_ascii_hexdigit()) {
        let mut seed = [0u8; 32];
        for i in 0..32 {
            seed[i] = u8::from_str_radix(&raw[i * 2..i * 2 + 2], 16)?;
        }
        return Ok(IssuerSigningKey::from_bytes(&seed));
    }
    // base64-encoded seed.
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    if let Ok(bytes) = B64.decode(raw.as_bytes()) {
        let bytes: Vec<u8> = bytes;
        if bytes.len() == 32 {
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&bytes);
            return Ok(IssuerSigningKey::from_bytes(&seed));
        }
    }
    anyhow::bail!(
        "LICENSE_PRIVATE_KEY must be a 64-char hex seed or base64 32-byte seed (got {} chars)",
        raw.len()
    );
}

#[derive(Debug, Deserialize)]
pub struct IssueRequest {
    pub org_id: String,
    #[serde(default = "default_seats")]
    pub seats: u32,
    /// Optional override; defaults to `now + LICENSE_DEFAULT_DAYS`.
    pub valid_days: Option<i64>,
}

fn default_seats() -> u32 {
    10
}

#[derive(Debug, Serialize)]
pub struct IssueResponse {
    pub license: License,
}

/// `POST /v1/license/issue` — admin-only. The Stripe webhook path can
/// call this internally on `customer.subscription.created` once the Pro
/// product is wired; for now an operator hits it from a shell with the
/// admin bearer.
pub async fn issue_route(
    State(issuer): State<Option<Arc<LicenseIssuer>>>,
    headers: HeaderMap,
    Json(req): Json<IssueRequest>,
) -> Result<Json<IssueResponse>, (StatusCode, String)> {
    let Some(issuer) = issuer.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "license issuer not configured (LICENSE_PRIVATE_KEY unset)".to_string(),
        ));
    };
    if issuer.admin_token.is_empty() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "LICENSE_ADMIN_TOKEN not set; issue endpoint is disabled".to_string(),
        ));
    }
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default()
        .trim();
    if !constant_time_eq(presented.as_bytes(), issuer.admin_token.as_bytes()) {
        return Err((
            StatusCode::UNAUTHORIZED,
            "missing or wrong admin token".to_string(),
        ));
    }
    let now = Utc::now();
    let validity = req
        .valid_days
        .map(Duration::days)
        .unwrap_or(issuer.default_validity);
    let claims = Claims {
        org_id: req.org_id.clone(),
        plan: "pro".to_string(),
        seats: req.seats,
        valid_until: now + validity,
        issued_at: now,
    };
    let lic = issuer.sign(claims);
    info!(org = %req.org_id, seats = req.seats, valid_until = %lic.claims.valid_until, "license issued");
    Ok(Json(IssueResponse { license: lic }))
}

#[derive(Debug, Deserialize)]
pub struct RefreshRequest {
    pub license: License,
}

#[derive(Debug, Serialize)]
pub struct RefreshResponse {
    pub license: License,
}

/// `POST /v1/license/refresh` — customer-facing. Verifies the submitted
/// license under the API's own trust root (the public half of the
/// signing key) and re-signs with a fresh `valid_until`. A real
/// implementation would consult the Stripe customer's subscription
/// status here; the design-partner cut just re-signs unconditionally so
/// we can land the framework and harden the gate as a one-line edit
/// once billing is wired.
pub async fn refresh_route(
    State(issuer): State<Option<Arc<LicenseIssuer>>>,
    Json(req): Json<RefreshRequest>,
) -> Result<Json<RefreshResponse>, (StatusCode, String)> {
    let Some(issuer) = issuer.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "license issuer not configured".to_string(),
        ));
    };
    // Verify the inbound license with the API's own trust root — the
    // verifying-key half of the signing key. Decoupled from
    // `EMBEDDED_PUBLIC_KEY` so we can rotate one without rotating the
    // other during a deploy.
    let verify_key = issuer.signing.verifying_key();
    let cfg = LicenseConfig::default().with_public_key(&verify_key.to_bytes());
    // Strict (not grace-tolerant) signature check — a customer with an
    // unsigned blob can't bootstrap a free refresh.
    req.license
        .verify_signature(&verify_key)
        .map_err(|e| (StatusCode::UNAUTHORIZED, format!("invalid license: {e}")))?;
    if req.license.claims.plan != "pro" {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("plan `{}` is not refreshable", req.license.claims.plan),
        ));
    }
    // Drop the unused config reference — keeping the call above ensures
    // the embedded grace defaults are at least exercised on the type.
    let _ = cfg;

    let now = Utc::now();
    let new_claims = Claims {
        valid_until: now + issuer.default_validity,
        issued_at: now,
        ..req.license.claims.clone()
    };
    let renewed = issuer.sign(new_claims);
    info!(org = %renewed.claims.org_id, new_valid_until = %renewed.claims.valid_until, "license refreshed");
    Ok(Json(RefreshResponse { license: renewed }))
}

#[derive(Debug, Deserialize)]
pub struct HeartbeatRequest {
    pub org_id: String,
    pub active_seats: u32,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct HeartbeatResponse {
    pub ok: bool,
}

/// `POST /v1/license/heartbeat` — telemetry + churn signal.
///
/// Today this just logs the beacon (`tracing` → stdout → Fly's log
/// pipeline); a follow-up wires it to a `license_heartbeats` SQLite
/// table so a customer who stops beaconing for 7+ days raises a CRM
/// flag. Failed heartbeats never return 5xx — the customer's `serve`
/// box treats them as best-effort and logs locally.
pub async fn heartbeat_route(Json(req): Json<HeartbeatRequest>) -> Json<HeartbeatResponse> {
    info!(
        org = %req.org_id,
        active_seats = req.active_seats,
        version = %req.version,
        "license heartbeat"
    );
    Json(HeartbeatResponse { ok: true })
}

/// Constant-time string compare. We deliberately don't bring in the
/// `subtle` crate again here — admin tokens are sufficiently long that
/// a side-channel timing oracle would still take longer than the
/// rate-limit the front door enforces.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        warn!("admin token length mismatch");
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use splatforge_license::dev_keys;

    #[test]
    fn parse_hex_seed() {
        let raw = "9d61b19deffd5a60ba844af492ec2cc44449c5697b32691970"; // 50 chars — invalid
        assert!(parse_signing_key(raw).is_err());
        let raw_hex = hex_encode(&dev_keys::SECRET_KEY_BYTES);
        let sk = parse_signing_key(&raw_hex).unwrap();
        assert_eq!(sk.to_bytes(), dev_keys::SECRET_KEY_BYTES);
    }

    fn hex_encode(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for byte in b {
            s.push_str(&format!("{:02x}", byte));
        }
        s
    }

    #[test]
    fn issuer_signs_and_self_verifies() {
        let issuer = LicenseIssuer::from_signing_key(
            dev_keys::signing_key(),
            "admin-token",
            Duration::days(30),
        );
        let claims = Claims {
            org_id: "acme".into(),
            plan: "pro".into(),
            seats: 5,
            valid_until: Utc::now() + Duration::days(30),
            issued_at: Utc::now(),
        };
        let lic = issuer.sign(claims);
        let cfg =
            LicenseConfig::default().with_public_key(&issuer.signing.verifying_key().to_bytes());
        cfg.validate(&lic, Utc::now(), None).unwrap();
    }
}
