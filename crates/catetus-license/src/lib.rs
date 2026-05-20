#![deny(clippy::all)]
//! `catetus-license` — Ed25519-signed license file format + verifier for
//! the Catetus Pro on-prem tier.
//!
//! ## File format
//!
//! `catetus.lic` is plain UTF-8 JSON with a stable shape:
//!
//! ```json
//! {
//!   "org_id": "acme-corp",
//!   "plan": "pro",
//!   "seats": 25,
//!   "valid_until": "2027-01-01T00:00:00Z",
//!   "issued_at": "2026-05-15T12:00:00Z",
//!   "signature": "<base64 Ed25519 signature over the canonical claims>"
//! }
//! ```
//!
//! `signature` is computed over the **canonical claims bytes** — a
//! deterministic JSON serialization of every field *except* `signature`.
//! That gives the customer a file they can `cat` and `jq` while preserving
//! a tamper-evident envelope. Re-serializing through the same encoder is
//! what makes the verification stable across whitespace / key-order
//! differences in the on-disk JSON.
//!
//! ## Trust root
//!
//! The verifier embeds a single Ed25519 public key (32 bytes). The matching
//! private key lives in the API box as a Fly secret (`LICENSE_PRIVATE_KEY`).
//! A `catetus` binary built from this repo will only honor licenses
//! signed by that key — there is deliberately no key rotation surface yet;
//! rotating requires a binary rebuild + customer redeploy, which is the
//! right amount of friction for v1.
//!
//! Until we cut the first production release `EMBEDDED_PUBLIC_KEY` is the
//! dev keypair in `dev_keys` (RFC 8032 §7.1 test vector 1) so a clean
//! checkout can sign + verify end-to-end with zero setup.
//!
//! ## Offline grace
//!
//! - `is_valid_strict(now)` — refuses if `now >= valid_until`.
//! - `is_valid_with_grace(now, last_refresh, grace)` — accepts an expired
//!   license *as long as* `now - last_refresh < grace`. This is what
//!   `catetus serve` calls so a customer can survive a 7-day API
//!   outage / firewall block without their box flatlining.

use std::path::Path;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default offline-grace window. Customers running `catetus serve`
/// behind an air-gapped firewall keep working for up to 7 days after a
/// license's `valid_until` passes, provided they could reach the refresh
/// endpoint at least once before the network broke.
pub const DEFAULT_OFFLINE_GRACE_DAYS: i64 = 7;

/// Public key bytes the verifier trusts. Until first prod release this
/// is `dev_keys::PUBLIC_KEY_BYTES`; `embedded_public_key_matches_dev_seed`
/// pins them together so they can't drift.
pub const EMBEDDED_PUBLIC_KEY: [u8; PUBLIC_KEY_LENGTH] = dev_keys::PUBLIC_KEY_BYTES;

/// Errors surfaced by the license verifier. Each variant maps 1:1 to a
/// CLI exit code so `catetus license status` can report a useful
/// reason.
#[derive(Debug, Error)]
pub enum LicenseError {
    #[error("license file not found at {0}")]
    NotFound(String),
    #[error("license file is not valid JSON: {0}")]
    Malformed(String),
    #[error("license signature is invalid")]
    BadSignature,
    #[error("license expired at {0} (no offline grace remaining)")]
    Expired(DateTime<Utc>),
    #[error("license plan is `{0}`, expected `pro`")]
    WrongPlan(String),
    #[error("license seats={got} below minimum required ({need})")]
    InsufficientSeats { got: u32, need: u32 },
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("cryptography error: {0}")]
    Crypto(String),
}

impl From<serde_json::Error> for LicenseError {
    fn from(e: serde_json::Error) -> Self {
        LicenseError::Malformed(e.to_string())
    }
}

/// The canonical claims serialized for signing. Same shape as `License`
/// minus the `signature` field. The signer and the verifier both
/// re-serialize through this struct so whitespace / field-order
/// differences in the on-disk JSON never affect the signature check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Claims {
    pub org_id: String,
    pub plan: String,
    pub seats: u32,
    pub valid_until: DateTime<Utc>,
    /// When the license was minted. Cosmetic; not consulted at verify
    /// time. Persisted so support can correlate a customer's license
    /// against the API audit log.
    pub issued_at: DateTime<Utc>,
}

impl Claims {
    /// Canonical bytes signed by the issuer. Keys emit in struct order
    /// via serde, which is stable as long as we don't reorder this
    /// struct — which would invalidate every existing license, which is
    /// exactly why the field set is small and pinned.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("Claims serialization is infallible")
    }
}

/// Full on-disk license payload. `signature` is base64-encoded Ed25519
/// over `Claims::canonical_bytes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    #[serde(flatten)]
    pub claims: Claims,
    /// base64 Ed25519 signature, 64 raw bytes / 88 base64 chars.
    pub signature: String,
}

impl License {
    /// Sign a fresh license with the given private key. Used by the API's
    /// `/v1/license/issue` and `/v1/license/refresh` handlers; never by
    /// the CLI.
    pub fn sign(claims: Claims, signing_key: &SigningKey) -> License {
        let sig = signing_key.sign(&claims.canonical_bytes());
        License {
            claims,
            signature: B64.encode(sig.to_bytes()),
        }
    }

    /// Verify the signature against the supplied verifying key. Strict
    /// signature check only — does **not** check expiry.
    pub fn verify_signature(&self, key: &VerifyingKey) -> Result<(), LicenseError> {
        let raw = B64
            .decode(self.signature.as_bytes())
            .map_err(|e| LicenseError::Crypto(format!("signature is not valid base64: {e}")))?;
        if raw.len() != SIGNATURE_LENGTH {
            return Err(LicenseError::Crypto(format!(
                "signature length {} (expected {SIGNATURE_LENGTH})",
                raw.len()
            )));
        }
        let mut buf = [0u8; SIGNATURE_LENGTH];
        buf.copy_from_slice(&raw);
        let sig = Signature::from_bytes(&buf);
        key.verify(&self.claims.canonical_bytes(), &sig)
            .map_err(|_| LicenseError::BadSignature)
    }

    /// Strict expiry check.
    pub fn is_valid_strict(&self, now: DateTime<Utc>) -> bool {
        now < self.claims.valid_until
    }

    /// Accept an expired license as long as the last successful refresh
    /// was within `grace`. The predicate `catetus serve` calls on
    /// startup.
    pub fn is_valid_with_grace(
        &self,
        now: DateTime<Utc>,
        last_refresh: Option<DateTime<Utc>>,
        grace: Duration,
    ) -> bool {
        if self.is_valid_strict(now) {
            return true;
        }
        match last_refresh {
            Some(t) => now - t < grace,
            None => false,
        }
    }

    /// Read + JSON-parse a license file. Does **not** verify signature.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<License, LicenseError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => LicenseError::NotFound(path.display().to_string()),
            _ => LicenseError::Io(e),
        })?;
        let lic: License = serde_json::from_slice(&bytes)?;
        Ok(lic)
    }

    /// Atomic-ish write: serialize pretty (so customer ops can read it)
    /// then rename into place so a crash mid-write can't leave a
    /// half-license.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<(), LicenseError> {
        let path = path.as_ref();
        let tmp = path.with_extension("lic.tmp");
        let body = serde_json::to_vec_pretty(self)?;
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Verifier configuration. Defaults to the embedded production public
/// key and a 7-day offline grace.
#[derive(Debug, Clone)]
pub struct LicenseConfig {
    pub verifying_key: VerifyingKey,
    pub grace: Duration,
}

impl Default for LicenseConfig {
    fn default() -> Self {
        let key = VerifyingKey::from_bytes(&EMBEDDED_PUBLIC_KEY)
            .expect("embedded public key is well-formed");
        LicenseConfig {
            verifying_key: key,
            grace: Duration::days(DEFAULT_OFFLINE_GRACE_DAYS),
        }
    }
}

impl LicenseConfig {
    pub fn with_public_key(mut self, bytes: &[u8; PUBLIC_KEY_LENGTH]) -> Self {
        self.verifying_key = VerifyingKey::from_bytes(bytes).expect("public key is well-formed");
        self
    }

    pub fn with_grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    /// Full validation pass: signature, plan, expiry-with-grace.
    pub fn validate(
        &self,
        license: &License,
        now: DateTime<Utc>,
        last_refresh: Option<DateTime<Utc>>,
    ) -> Result<(), LicenseError> {
        license.verify_signature(&self.verifying_key)?;
        if license.claims.plan != "pro" {
            return Err(LicenseError::WrongPlan(license.claims.plan.clone()));
        }
        if !license.is_valid_with_grace(now, last_refresh, self.grace) {
            return Err(LicenseError::Expired(license.claims.valid_until));
        }
        Ok(())
    }
}

/// Compile-time dev keypair. The seed is RFC 8032 §7.1 test vector 1 —
/// DO NOT issue real customer licenses from this key; production uses
/// `LICENSE_PRIVATE_KEY` in Fly secrets, and rotating the trust root
/// means bumping `EMBEDDED_PUBLIC_KEY` and rebuilding every shipped
/// binary.
pub mod dev_keys {
    use ed25519_dalek::{SigningKey, VerifyingKey, SECRET_KEY_LENGTH};

    /// RFC 8032 §7.1 test vector 1 — deterministic 32-byte seed.
    pub const SECRET_KEY_BYTES: [u8; SECRET_KEY_LENGTH] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];

    /// RFC 8032 §7.1 test vector 1 public key. Pinned via a #[test] so
    /// it can never drift from the seed above.
    pub const PUBLIC_KEY_BYTES: [u8; 32] = [
        0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07,
        0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07,
        0x51, 0x1a,
    ];

    pub fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&SECRET_KEY_BYTES)
    }

    pub fn verifying_key() -> VerifyingKey {
        VerifyingKey::from_bytes(&PUBLIC_KEY_BYTES).expect("dev public key is well-formed")
    }
}

/// Re-exports so downstream crates don't have to take a direct dep on
/// `ed25519-dalek` for the common types.
pub use ed25519_dalek::{SigningKey as IssuerSigningKey, VerifyingKey as IssuerVerifyingKey};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_claims(valid_until: DateTime<Utc>) -> Claims {
        Claims {
            org_id: "acme-corp".to_string(),
            plan: "pro".to_string(),
            seats: 25,
            valid_until,
            issued_at: Utc::now(),
        }
    }

    #[test]
    fn round_trip_sign_verify() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() + Duration::days(30));
        let lic = License::sign(claims, &sk);
        let cfg = LicenseConfig::default();
        cfg.validate(&lic, Utc::now(), None).expect("valid");
    }

    #[test]
    fn expired_license_is_rejected() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() - Duration::days(1));
        let lic = License::sign(claims, &sk);
        let cfg = LicenseConfig::default();
        let err = cfg.validate(&lic, Utc::now(), None).unwrap_err();
        assert!(matches!(err, LicenseError::Expired(_)), "got {err:?}");
    }

    #[test]
    fn offline_grace_extends_expired_license() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() - Duration::days(2));
        let lic = License::sign(claims, &sk);
        let cfg = LicenseConfig::default().with_grace(Duration::days(7));
        let last = Some(Utc::now() - Duration::days(3));
        cfg.validate(&lic, Utc::now(), last)
            .expect("grace covers it");
    }

    #[test]
    fn offline_grace_does_not_extend_beyond_window() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() - Duration::days(2));
        let lic = License::sign(claims, &sk);
        let cfg = LicenseConfig::default().with_grace(Duration::days(7));
        let last = Some(Utc::now() - Duration::days(8));
        let err = cfg.validate(&lic, Utc::now(), last).unwrap_err();
        assert!(matches!(err, LicenseError::Expired(_)), "got {err:?}");
    }

    #[test]
    fn invalid_signature_is_rejected() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() + Duration::days(30));
        let mut lic = License::sign(claims, &sk);
        let mut raw = B64.decode(lic.signature.as_bytes()).unwrap();
        raw[0] ^= 0xff;
        lic.signature = B64.encode(&raw);
        let cfg = LicenseConfig::default();
        let err = cfg.validate(&lic, Utc::now(), None).unwrap_err();
        assert!(matches!(err, LicenseError::BadSignature), "got {err:?}");
    }

    #[test]
    fn tampered_claims_invalidate_signature() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() + Duration::days(30));
        let mut lic = License::sign(claims, &sk);
        lic.claims.seats = 9999;
        let cfg = LicenseConfig::default();
        let err = cfg.validate(&lic, Utc::now(), None).unwrap_err();
        assert!(matches!(err, LicenseError::BadSignature), "got {err:?}");
    }

    #[test]
    fn wrong_plan_is_rejected() {
        let sk = dev_keys::signing_key();
        let mut claims = sample_claims(Utc::now() + Duration::days(30));
        claims.plan = "team".to_string();
        let lic = License::sign(claims, &sk);
        let cfg = LicenseConfig::default();
        let err = cfg.validate(&lic, Utc::now(), None).unwrap_err();
        assert!(matches!(err, LicenseError::WrongPlan(_)), "got {err:?}");
    }

    #[test]
    fn write_and_read_round_trip() {
        let sk = dev_keys::signing_key();
        let claims = sample_claims(Utc::now() + Duration::days(30));
        let lic = License::sign(claims, &sk);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("catetus.lic");
        lic.write_to_path(&path).unwrap();
        let loaded = License::read_from_path(&path).unwrap();
        let cfg = LicenseConfig::default();
        cfg.validate(&loaded, Utc::now(), None).unwrap();
    }

    #[test]
    fn embedded_public_key_matches_dev_seed() {
        let sk = SigningKey::from_bytes(&dev_keys::SECRET_KEY_BYTES);
        let derived = sk.verifying_key();
        assert_eq!(derived.to_bytes(), dev_keys::PUBLIC_KEY_BYTES);
    }
}
