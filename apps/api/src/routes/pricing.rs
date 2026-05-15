//! `POST /v1/pricing/preview` — quote-before-you-pay.
//!
//! Pure HTTP shim over `pricing::preview_job_cost`. No DB access, no
//! Stripe call, no auth gate — anyone can ask for a quote. The math
//! is the same math the meter-emit path will charge against, so a
//! customer-facing calculator and the eventual invoice agree to the
//! cent.

use axum::{extract::Json, response::IntoResponse};
use serde::Deserialize;

use crate::pricing;

#[derive(Debug, Deserialize)]
pub struct PreviewRequest {
    /// Input size in bytes. We allow `u64` so a 855 MB bicycle scene
    /// fits without overflow even on the JSON-number-to-u64 path
    /// (serde_json clamps at 2^53 — 9 PB, plenty).
    pub size_bytes: u64,
    /// Preset id — matches the `/v1/jobs` payload. Unknown presets
    /// fall back to a `web-mobile`-shaped curve rather than 400ing,
    /// so a brand-new preset can be quoted before the rate card is
    /// updated.
    pub preset: String,
    /// How many free-tier runs the caller has already used this
    /// billing period. Optional; defaults to 0 (full allotment
    /// remaining). Today the API doesn't track per-customer usage so
    /// the route always passes whatever the caller sends — the
    /// frontend stores a localStorage counter, the eventual auth
    /// branch will replace this with a server-side lookup.
    #[serde(default)]
    pub free_tier_runs_used_this_month: u32,
}

/// `POST /v1/pricing/preview` handler. Returns the full
/// `PricePreview` struct as JSON. No errors today — every payload is
/// quotable (we don't gate on `size_bytes` upper bound here because
/// the calculator should still tell a curious user "your 10 GB scene
/// would cost $X" even if the optimize endpoint would reject it).
pub async fn preview(Json(req): Json<PreviewRequest>) -> impl IntoResponse {
    let preview = pricing::preview_job_cost(
        req.size_bytes,
        &req.preset,
        req.free_tier_runs_used_this_month,
    );
    Json(preview)
}
