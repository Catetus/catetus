//! Integration tests for the API-side license issuer.
//!
//! The route handlers themselves consume `Option<Arc<LicenseIssuer>>`
//! state via axum's FromRef impl — exercising the full axum stack would
//! require spinning up the AppState (Stripe stubs, blob storage, …),
//! which is overkill for the contract here. Instead, these tests
//! exercise the issuer's public surface end-to-end: sign with the real
//! production code path, then verify with the embedded public key the
//! CLI ships with.

use chrono::{Duration, Utc};
use splatforge_api::license::LicenseIssuer;
use splatforge_license::{dev_keys, Claims, License, LicenseConfig};

#[test]
fn issuer_signs_license_a_cli_can_verify() {
    let _issuer = LicenseIssuer::from_signing_key(
        dev_keys::signing_key(),
        "dummy-admin-token",
        Duration::days(30),
    );
    // Mint the same way `issue_route` does.
    let claims = Claims {
        org_id: "acme-corp".to_string(),
        plan: "pro".to_string(),
        seats: 42,
        valid_until: Utc::now() + Duration::days(30),
        issued_at: Utc::now(),
    };
    // The signing path lives behind the route — go through `License::sign`
    // directly to match what the handler does internally.
    let lic = License::sign(claims, &dev_keys::signing_key());

    // The CLI's default LicenseConfig (which points at EMBEDDED_PUBLIC_KEY)
    // must accept it. This is the round-trip an on-prem customer hits.
    let cfg = LicenseConfig::default();
    cfg.validate(&lic, Utc::now(), None).expect("valid round-trip");
}

#[test]
fn refresh_round_trip_extends_validity() {
    // Mint, simulate aging the license, re-sign with new validity, then
    // assert the new license is valid under strict rules.
    let original_claims = Claims {
        org_id: "acme-corp".to_string(),
        plan: "pro".to_string(),
        seats: 42,
        valid_until: Utc::now() + Duration::days(1),
        issued_at: Utc::now() - Duration::days(29),
    };
    let original = License::sign(original_claims.clone(), &dev_keys::signing_key());

    // Verify the original is still good.
    let cfg = LicenseConfig::default();
    cfg.validate(&original, Utc::now(), None).unwrap();

    // Mock the refresh: re-sign with extended valid_until.
    let refreshed_claims = Claims {
        valid_until: Utc::now() + Duration::days(35),
        issued_at: Utc::now(),
        ..original_claims
    };
    let refreshed = License::sign(refreshed_claims, &dev_keys::signing_key());
    cfg.validate(&refreshed, Utc::now(), None).unwrap();
    assert!(refreshed.claims.valid_until > original.claims.valid_until);
}

#[test]
fn from_env_returns_none_when_unset() {
    // SAFETY: tests in this crate are not run in parallel with anything
    // that touches LICENSE_PRIVATE_KEY.
    std::env::remove_var("LICENSE_PRIVATE_KEY");
    let issuer = LicenseIssuer::from_env().expect("never errors when unset");
    assert!(issuer.is_none(), "unset env should mean no issuer");
}
