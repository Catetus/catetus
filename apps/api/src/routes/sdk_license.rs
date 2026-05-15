//! SDK licensing HTTP surface.
//!
//! Three endpoints:
//!
//!   * `POST /v1/sdk/license`         — issue a new (plugin, domain) license
//!   * `POST /v1/sdk/license/verify`  — verify a presented token + Origin
//!   * `POST /v1/sdk/beacon`          — per-MAU telemetry beacon (verifies
//!                                      token, records MAU sample, 204s)
//!   * `POST /v1/sdk/pricing/preview` — pure-compute MAU quote
//!
//! License issuance is gated by an operator-only bearer token (separate
//! from the `SPLATFORGE_API_KEYS` set — see `SDK_OPERATOR_KEY`) so the
//! self-serve `/sdk` page doesn't accidentally mint licenses for free.
//! Verify + beacon are open: the JWT signature is the auth, and the
//! `Origin` header is the binding.

use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::pricing::{
    mint_sdk_license, normalize_domain, preview_sdk_cost, verify_sdk_license, LicenseError,
    SDK_LICENSE_TTL_SECS, SDK_PLUGIN_TYPES,
};

/// Operator-only env var. Without this set, license issuance is
/// disabled (the route returns 503). A separate secret means the
/// public `SPLATFORGE_API_KEYS` set never accidentally grants license
/// issuance.
pub const SDK_OPERATOR_KEY_ENV: &str = "SPLATFORGE_SDK_OPERATOR_KEY";

/// Env var holding the HMAC secret used to sign SDK license JWTs. We
/// deliberately don't fall back to a hard-coded default — a missing
/// secret should disable the endpoint, not silently sign with a
/// well-known value.
pub const SDK_LICENSE_SECRET_ENV: &str = "SPLATFORGE_SDK_LICENSE_SECRET";

/// Subset of `AppState` the SDK routes care about. Built from the
/// `main.rs` `AppState` at registration time; lets the handlers stay
/// independent of the larger AppState struct so a future refactor
/// doesn't ripple through every test.
#[derive(Clone)]
pub struct SdkLicenseState {
    pub license_secret: Option<Vec<u8>>,
    pub operator_key: Option<String>,
}

impl SdkLicenseState {
    pub fn from_env() -> Self {
        let license_secret = std::env::var(SDK_LICENSE_SECRET_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.into_bytes());
        let operator_key = std::env::var(SDK_OPERATOR_KEY_ENV)
            .ok()
            .filter(|s| !s.is_empty());
        if license_secret.is_none() {
            warn!(
                "SPLATFORGE_SDK_LICENSE_SECRET is unset — /v1/sdk/license{{,/verify}} return 503. \
                 Set this to a high-entropy random string (>=32 bytes) to enable SDK licensing."
            );
        }
        if operator_key.is_none() {
            warn!(
                "SPLATFORGE_SDK_OPERATOR_KEY is unset — /v1/sdk/license issuance is disabled. \
                 Set this to gate license minting to the operator."
            );
        }
        Self {
            license_secret,
            operator_key,
        }
    }

    /// Test-only override.
    pub fn for_test(license_secret: Option<Vec<u8>>, operator_key: Option<String>) -> Self {
        Self {
            license_secret,
            operator_key,
        }
    }
}

/* ---------- issue ---------- */

#[derive(Debug, Deserialize)]
pub struct IssueRequest {
    pub plugin: String,
    pub domain: String,
    /// Optional override; defaults to `SDK_LICENSE_TTL_SECS` (1 year).
    #[serde(default)]
    pub ttl_secs: Option<u64>,
    /// Optional caller-supplied license id. Mostly for tests; in
    /// production the server mints one.
    #[serde(default)]
    pub kid: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IssueResponse {
    pub token: String,
    pub plugin: String,
    pub domain: String,
    pub iat: u64,
    pub exp: u64,
    pub kid: String,
}

/// `POST /v1/sdk/license` — operator-gated.
///
/// Requires `Authorization: Bearer <SDK_OPERATOR_KEY>`. Refuses if
/// either the operator key or the signing secret is unset.
pub async fn issue_license(
    State(state): State<SdkLicenseState>,
    headers: HeaderMap,
    Json(req): Json<IssueRequest>,
) -> Result<Json<IssueResponse>, SdkRouteError> {
    let Some(operator) = state.operator_key.as_deref() else {
        return Err(SdkRouteError::Unavailable(
            "SDK license issuance not enabled (set SPLATFORGE_SDK_OPERATOR_KEY)".to_string(),
        ));
    };
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or_default();
    if presented != operator {
        return Err(SdkRouteError::Unauthorized);
    }
    let Some(secret) = state.license_secret.as_deref() else {
        return Err(SdkRouteError::Unavailable(
            "SDK license signing secret not configured".to_string(),
        ));
    };

    let now = crate::pricing::now_unix();
    let ttl = req.ttl_secs.unwrap_or(SDK_LICENSE_TTL_SECS);
    let kid = req.kid.unwrap_or_else(|| {
        // Stable-ish opaque id: hex of (now || plugin || domain). Not
        // a secret; just lets the operator revoke a specific issue.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(now.to_le_bytes());
        h.update(req.plugin.as_bytes());
        h.update(b":");
        h.update(req.domain.as_bytes());
        let digest = h.finalize();
        format!("lic_{}", &hex::encode(digest)[..16])
    });
    let (token, claims) = mint_sdk_license(&req.plugin, &req.domain, secret, now, ttl, &kid)?;
    info!(plugin = %req.plugin, domain = %req.domain, %kid, "sdk license minted");
    Ok(Json(IssueResponse {
        token,
        plugin: claims.sub,
        domain: claims.aud,
        iat: claims.iat,
        exp: claims.exp,
        kid: claims.kid,
    }))
}

/* ---------- verify ---------- */

#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub token: String,
    /// Caller-supplied domain (e.g. from `window.location.host`). The
    /// route ALSO reads the `Origin` request header and refuses if it
    /// disagrees — domain in the body is convenience for tools that
    /// can't set headers; `Origin` is the load-bearing check for
    /// browser-originated calls.
    pub domain: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub plugin: String,
    pub domain: String,
    pub expires_at: u64,
    pub kid: String,
}

/// `POST /v1/sdk/license/verify` — verify a presented license against
/// `(token, domain, Origin header)`. Returns 200 + claims on success;
/// non-2xx with a structured error otherwise. No auth required: the
/// token signature IS the auth.
pub async fn verify_license_route(
    State(state): State<SdkLicenseState>,
    headers: HeaderMap,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, SdkRouteError> {
    let Some(secret) = state.license_secret.as_deref() else {
        return Err(SdkRouteError::Unavailable(
            "SDK license signing secret not configured".to_string(),
        ));
    };
    // Cross-check Origin header if present. Browsers always send it;
    // server-to-server callers may not — those just rely on the body
    // `domain` field. When both are present they must agree.
    if let Some(origin) = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        let origin_norm = normalize_domain(origin)?;
        let body_norm = normalize_domain(&req.domain)?;
        if origin_norm != body_norm {
            return Err(SdkRouteError::License(LicenseError::DomainMismatch {
                bound: origin_norm,
                presented: body_norm,
            }));
        }
    }

    let now = crate::pricing::now_unix();
    let claims = verify_sdk_license(&req.token, secret, now, &req.domain)?;
    Ok(Json(VerifyResponse {
        valid: true,
        plugin: claims.sub,
        domain: claims.aud,
        expires_at: claims.exp,
        kid: claims.kid,
    }))
}

/* ---------- beacon ---------- */

#[derive(Debug, Deserialize)]
pub struct BeaconRequest {
    pub token: String,
    pub domain: String,
    /// MAU snapshot the SDK is reporting. The server doesn't enforce
    /// the count today; it logs it for the operator to roll up into
    /// the monthly invoice. (When we wire a real time-series store
    /// for MAU rollups, this becomes a write.)
    pub mau: u32,
}

/// `POST /v1/sdk/beacon` — telemetry endpoint the SDK plugins call
/// once per session. Verifies the license, records the MAU value
/// (just a log line today), responds 204.
pub async fn beacon(
    State(state): State<SdkLicenseState>,
    headers: HeaderMap,
    Json(req): Json<BeaconRequest>,
) -> Result<StatusCode, SdkRouteError> {
    let Some(secret) = state.license_secret.as_deref() else {
        return Err(SdkRouteError::Unavailable(
            "SDK license signing secret not configured".to_string(),
        ));
    };
    if let Some(origin) = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        let origin_norm = normalize_domain(origin)?;
        let body_norm = normalize_domain(&req.domain)?;
        if origin_norm != body_norm {
            return Err(SdkRouteError::License(LicenseError::DomainMismatch {
                bound: origin_norm,
                presented: body_norm,
            }));
        }
    }
    let now = crate::pricing::now_unix();
    let claims = verify_sdk_license(&req.token, secret, now, &req.domain)?;
    info!(
        plugin = %claims.sub,
        domain = %claims.aud,
        kid = %claims.kid,
        mau = req.mau,
        "sdk beacon received"
    );
    Ok(StatusCode::NO_CONTENT)
}

/* ---------- pricing preview ---------- */

#[derive(Debug, Deserialize)]
pub struct SdkPreviewRequest {
    pub plugin: String,
    pub mau: u32,
}

/// `POST /v1/sdk/pricing/preview` — pure-compute SDK MAU quote.
pub async fn sdk_preview(Json(req): Json<SdkPreviewRequest>) -> impl IntoResponse {
    if !SDK_PLUGIN_TYPES.contains(&req.plugin.as_str()) {
        // Return a 200 with the quote anyway — better DX for a calculator
        // that's exploring possible plugin names. The form will reject
        // server-side at issuance time.
    }
    let preview = preview_sdk_cost(&req.plugin, req.mau);
    Json(preview)
}

/* ---------- errors ---------- */

#[derive(Debug, thiserror::Error)]
pub enum SdkRouteError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("service unavailable: {0}")]
    Unavailable(String),
    #[error("license: {0}")]
    License(#[from] LicenseError),
}

impl IntoResponse for SdkRouteError {
    fn into_response(self) -> axum::response::Response {
        let (status, body) = match &self {
            SdkRouteError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                serde_json::json!({ "error": "unauthorized" }),
            ),
            SdkRouteError::Unavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({ "error": msg }),
            ),
            SdkRouteError::License(e) => {
                let status = match e {
                    LicenseError::Expired
                    | LicenseError::NotYetValid
                    | LicenseError::BadSignature
                    | LicenseError::DomainMismatch { .. } => StatusCode::FORBIDDEN,
                    LicenseError::UnknownPlugin(_)
                    | LicenseError::InvalidDomain(_)
                    | LicenseError::Malformed(_) => StatusCode::BAD_REQUEST,
                };
                (status, serde_json::json!({ "error": e.to_string() }))
            }
        };
        (status, Json(body)).into_response()
    }
}
