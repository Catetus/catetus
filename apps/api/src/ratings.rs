//! Helpers for the `fidelity-ml v0.4` human pairwise-rating collector.
//!
//! Lives in the library surface (not `main.rs`) so the integration
//! tests under `tests/ratings.rs` can exercise the hash + validation
//! invariants without spinning up the full Axum app. The HTTP handlers
//! in `main.rs` still own the routing, body parsing, and rate-limit
//! gate — this module is purely the pure-function core they call into.

use axum::http::HeaderMap;
use sha2::{Digest, Sha256};

/// Rolling-hour cap on ratings from a single respondent. Documented
/// here (not in `main.rs`) so test code can reference the same
/// constant the production gate uses.
pub const RATING_RATE_LIMIT_PER_HOUR: i64 = 100;

/// Accepted `winner` values. "skip" is deliberately excluded — the
/// page handles skips client-side by just fetching the next pair.
pub const VALID_WINNERS: &[&str] = &["left", "right", "tie"];

/// Bounded allowlist of preset names so the page can't fill the
/// table with garbage. New presets get added here when they ship in
/// `splatbench-v0.json`.
pub const KNOWN_PRESETS: &[&str] = &[
    "lossless-repack",
    "web-mobile",
    "size-min",
    "differentiable-repack",
];

/// Compute the respondent hash from request headers. The input
/// plaintext is **never** persisted — the only thing that hits the
/// database is the hex digest. Order, separator, and fallback values
/// are stable so the same browser hitting the page twice produces
/// the same hash, which is what makes the rate-limit query work.
///
/// We hash the *first* `X-Forwarded-For` token because Fly / Vercel
/// append their own proxy hop, and we want the visitor's IP, not the
/// edge POP. Falling back to a fixed `"unknown"` keeps the hash space
/// from collapsing to zero entries (which would defeat the rate limit
/// when running behind a misconfigured proxy).
///
/// The hash MUST be deterministic for identical inputs and MUST
/// differ when either IP or UA changes — those properties are
/// asserted in `tests/ratings.rs`.
pub fn respondent_hash(headers: &HeaderMap) -> String {
    let ip = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| headers.get("x-real-ip").and_then(|v| v.to_str().ok()))
        .unwrap_or("unknown");
    let ua = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");
    let mut hasher = Sha256::new();
    hasher.update(ip.as_bytes());
    hasher.update(b"|");
    hasher.update(ua.as_bytes());
    hex::encode(hasher.finalize())
}

/// Validate a rating payload before it touches the database.
/// Returns `Err(reason)` with a human-readable message that the HTTP
/// handler can surface verbatim as the 400 body.
pub fn validate_rating(
    scene_id: &str,
    left_preset: &str,
    right_preset: &str,
    winner: &str,
) -> Result<(), String> {
    if !VALID_WINNERS.contains(&winner) {
        return Err(format!(
            "winner must be one of {VALID_WINNERS:?}; got {winner:?}"
        ));
    }
    if left_preset == right_preset {
        return Err("left_preset and right_preset must differ".to_string());
    }
    if !KNOWN_PRESETS.contains(&left_preset) || !KNOWN_PRESETS.contains(&right_preset) {
        return Err(format!(
            "unknown preset (expected one of {KNOWN_PRESETS:?})"
        ));
    }
    if scene_id.is_empty() || scene_id.len() > 128 {
        return Err("scene_id must be 1..128 chars".to_string());
    }
    Ok(())
}
