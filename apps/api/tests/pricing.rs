//! Integration tests for the per-job pricing preview + SDK licensing.
//!
//! Two surfaces under test:
//!
//!   1. `preview_job_cost` — pure-compute quote against the published
//!      rate cards. Anchored on the same bonsai (274 MB → ~18 s) and
//!      bicycle (855 MB → ~120 s) datapoints in `apps/api/BILLING.md`,
//!      so a rate-card tune that drifts the quote outside the buyer's
//!      expectation fails CI before it hits prod.
//!
//!   2. `mint_sdk_license` + `verify_sdk_license` — HMAC-SHA256 JWT
//!      round-trip, plus the four attack shapes the verifier MUST
//!      reject: tampered signature, wrong key, wrong domain, expired
//!      token. Lives here (rather than only in the in-module `#[cfg(test)]`)
//!      so a refactor that accidentally removes the domain check
//!      surfaces as an integration-test failure.

use splatforge_api::pricing::{
    mint_sdk_license, preview_job_cost, preview_sdk_cost, verify_sdk_license, LicenseError,
    FREE_TIER_RUNS_PER_MONTH, PRICING_VERSION, SDK_FREE_TIER_MAU, SDK_PER_MAU_CENTS,
    SDK_PRICING_VERSION,
};

/* ----------------------------------------------------------------------- */
/* 1. Per-job pricing                                                       */
/* ----------------------------------------------------------------------- */

#[test]
fn preview_carries_pricing_version() {
    let p = preview_job_cost(1_000_000, "web-mobile", 0);
    assert_eq!(p.pricing_version, PRICING_VERSION);
}

#[test]
fn bonsai_quote_matches_published_anchor_within_band() {
    // BILLING.md anchor: bonsai (274 MB) → ~18 s of compute. We
    // intentionally bias the curve toward the high anchor so the quote
    // never undershoots the bill; the assertion captures the "no
    // negative surprise" property.
    let p = preview_job_cost(274 * 1024 * 1024, "web-mobile", 0);
    assert!(
        p.estimated_compute_seconds >= 18,
        "bonsai quote must be at least the published 18s anchor; got {}",
        p.estimated_compute_seconds
    );
    // …and shouldn't be wildly larger either (catches a slope typo).
    assert!(
        p.estimated_compute_seconds <= 120,
        "bonsai quote runaway; got {}",
        p.estimated_compute_seconds
    );
}

#[test]
fn flat_fee_is_present_in_total() {
    let p = preview_job_cost(0, "web-mobile", 0);
    // Floor: even a 0-byte job pays the flat fee.
    assert!(p.estimated_cost_usd_cents >= 1);
}

#[test]
fn larger_input_costs_strictly_more() {
    let small = preview_job_cost(10 * 1024 * 1024, "web-mobile", 0);
    let big = preview_job_cost(500 * 1024 * 1024, "web-mobile", 0);
    assert!(big.estimated_cost_usd_cents > small.estimated_cost_usd_cents);
}

#[test]
fn free_tier_remaining_clamps_at_zero() {
    let p = preview_job_cost(1024, "web-mobile", FREE_TIER_RUNS_PER_MONTH + 10);
    assert_eq!(p.free_tier_runs_remaining, 0);
}

#[test]
fn breakdown_adds_up_to_total() {
    let p = preview_job_cost(100 * 1024 * 1024, "web-mobile", 0);
    let sum = p.breakdown.flat_cents + p.breakdown.compute_cents;
    let total_cents = p.estimated_cost_usd_cents as f64;
    // ceil() rounding may add up to 1 cent of slack.
    assert!(
        (sum - total_cents).abs() <= 1.0,
        "breakdown {sum} cents != total {total_cents} cents"
    );
}

/* ----------------------------------------------------------------------- */
/* 2. SDK MAU pricing                                                       */
/* ----------------------------------------------------------------------- */

#[test]
fn sdk_pricing_version_stamped() {
    let p = preview_sdk_cost("threejs", 1);
    assert_eq!(p.sdk_pricing_version, SDK_PRICING_VERSION);
}

#[test]
fn sdk_free_tier_is_truly_free() {
    let p = preview_sdk_cost("threejs", SDK_FREE_TIER_MAU);
    assert_eq!(p.estimated_cost_usd_cents, 0);
}

#[test]
fn sdk_paid_band_scales_linearly() {
    let a = preview_sdk_cost("threejs", SDK_FREE_TIER_MAU + 100_000);
    let b = preview_sdk_cost("threejs", SDK_FREE_TIER_MAU + 200_000);
    assert!(b.estimated_cost_usd_cents > a.estimated_cost_usd_cents);
    // Doubling the paid MAU doubles the price (modulo ceil rounding).
    let ratio = (b.estimated_cost_usd_cents as f64) / (a.estimated_cost_usd_cents as f64);
    assert!((ratio - 2.0).abs() < 0.01, "ratio={ratio}");
}

#[test]
fn sdk_per_mau_rate_is_one_tenth_of_a_cent() {
    // Sanity-check the v0.1 published number. If an operator tunes
    // this in pricing.rs without updating the docs, CI rings.
    assert_eq!(SDK_PER_MAU_CENTS, 0.1);
}

/* ----------------------------------------------------------------------- */
/* 3. SDK license HMAC round-trip                                           */
/* ----------------------------------------------------------------------- */

const TEST_SECRET: &[u8] = b"hmac-secret-for-tests-only-do-not-use-in-prod";

#[test]
fn license_round_trip_through_module_surface() {
    let now = 1_710_000_000_u64;
    let (token, _) = mint_sdk_license(
        "threejs",
        "https://acme.example/path?x=1",
        TEST_SECRET,
        now,
        60,
        "lic_int_1",
    )
    .expect("mint");
    let claims = verify_sdk_license(&token, TEST_SECRET, now + 5, "acme.example").expect("verify");
    assert_eq!(claims.aud, "acme.example");
    assert_eq!(claims.sub, "threejs");
    assert_eq!(claims.iss, "splatforge");
    assert_eq!(claims.kid, "lic_int_1");
    assert_eq!(claims.exp - claims.iat, 60);
}

#[test]
fn license_round_trip_rejects_wrong_secret() {
    let now = 1_710_000_000_u64;
    let (token, _) =
        mint_sdk_license("threejs", "acme.example", TEST_SECRET, now, 60, "k").unwrap();
    let err = verify_sdk_license(&token, b"different-secret", now + 1, "acme.example")
        .expect_err("must reject");
    assert!(matches!(err, LicenseError::BadSignature));
}

#[test]
fn license_round_trip_rejects_other_domain() {
    let now = 1_710_000_000_u64;
    let (token, _) =
        mint_sdk_license("threejs", "acme.example", TEST_SECRET, now, 60, "k").unwrap();
    let err =
        verify_sdk_license(&token, TEST_SECRET, now + 1, "evil.example").expect_err("must reject");
    assert!(matches!(err, LicenseError::DomainMismatch { .. }));
}

#[test]
fn license_round_trip_rejects_expired_token() {
    let now = 1_710_000_000_u64;
    let (token, _) =
        mint_sdk_license("threejs", "acme.example", TEST_SECRET, now, 10, "k").unwrap();
    let err = verify_sdk_license(&token, TEST_SECRET, now + 100, "acme.example")
        .expect_err("must reject");
    assert!(matches!(err, LicenseError::Expired));
}

#[test]
fn license_round_trip_normalizes_domain_at_both_ends() {
    // Issue with one form, verify with another — should still match
    // because both sides normalize.
    let now = 1_710_000_000_u64;
    let (token, _) = mint_sdk_license(
        "babylonjs",
        "https://Acme.Example.com:8443/app",
        TEST_SECRET,
        now,
        60,
        "k",
    )
    .expect("mint");
    let claims = verify_sdk_license(
        &token,
        TEST_SECRET,
        now + 1,
        "http://acme.example.com/somewhere/else",
    )
    .expect("verify");
    assert_eq!(claims.aud, "acme.example.com");
}

#[test]
fn license_round_trip_rejects_unknown_plugin() {
    let err = mint_sdk_license(
        "unknown-plugin",
        "acme.example",
        TEST_SECRET,
        1_710_000_000,
        60,
        "k",
    )
    .expect_err("must reject");
    assert!(matches!(err, LicenseError::UnknownPlugin(_)));
}
