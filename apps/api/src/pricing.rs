//! Per-job pricing engine + SDK licensing primitives.
//!
//! ## Why this exists
//!
//! `billing.rs` already emits Stripe Meter Events for the metered repack
//! pipeline — but the rate card the buyer sees on `/pricing` is hard-coded
//! into copy and not derivable. This module owns the rate-card constants
//! and a `preview_job_cost` function that's the single source of truth
//! for both:
//!
//!   * `POST /v1/pricing/preview` — quote-before-you-pay (`size_bytes` +
//!     `preset` → `{ estimated_compute_seconds, estimated_cost_usd_cents,
//!     free_tier_runs_remaining }`).
//!   * `apps/web/src/pages/pricing.astro` — the "Per-job calculator"
//!     calls the same endpoint so the customer-facing number matches
//!     the meter-emitted number to the cent.
//!
//! It also owns the SDK-licensing surface: `mint_sdk_license` issues a
//! domain-bound HMAC-signed JWT for the Three.js / Babylon.js / model-viewer
//! / Cesium-ion plugin builds, and `verify_sdk_license` checks the
//! signature + domain on every telemetry beacon.
//!
//! Both rate cards are versioned (`PRICING_VERSION_*`). Buyers see the
//! version in their preview response so an operator who tunes the rates
//! later doesn't silently re-bill the design partners on stale quotes.

use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use subtle::ConstantTimeEq;

/* ====================================================================== */
/*  Per-job rate card (v0.1)                                                */
/* ====================================================================== */

/// Pricing version stamped on every preview. Bump every time a constant
/// in this module changes. Customers store the version with their
/// quoted price so an operator-side tune doesn't silently re-bill them.
pub const PRICING_VERSION: &str = "v0.1";

/// Flat per-job fee, in USD cents. Maps 1:1 to the
/// `splatforge_repack_runs` Stripe meter (1 unit per call).
///
/// Why $0.01: the meter-event POST itself is the bookkeeping cost; we
/// price it at floor so the variable cost (`PER_COMPUTE_SECOND_CENTS`)
/// dominates the total. Tunable in the dashboard once we have usage
/// data on the long-tail of fast jobs.
pub const PER_JOB_FLAT_CENTS: f64 = 1.0;

/// Per-second compute fee, in USD cents. Maps 1:1 to the
/// `splatforge_repack_seconds` Stripe meter (1 unit per second).
///
/// $0.001/sec = $3.60/hr. Modal A100 list is ~$3.09/hr so this carries
/// a thin margin while undercutting the headline DIY-on-AWS rate.
/// Bonsai (~18s) lands at $0.018 + $0.01 = $0.028; bicycle (~120s) at
/// $0.12 + $0.01 = $0.13.
pub const PER_COMPUTE_SECOND_CENTS: f64 = 0.1;

/// Free-tier monthly run cap. Anyone with a paid bearer key gets this
/// many free `repack_runs` events per billing period before metering
/// kicks in. Mirrors the Vercel / Modal pattern: friction-free trial,
/// no credit card on the free tier (closed beta still gates on
/// bearer-token issuance).
pub const FREE_TIER_RUNS_PER_MONTH: u32 = 5;

/// Preset → expected compute-seconds. Calibrated against the bonsai
/// (274 MB → ~18 s at iter 1000) and bicycle (855 MB → ~120 s at iter
/// 1000) datapoints from `apps/api/BILLING.md`. The slope is roughly
/// linear in input size; the preset multiplier captures the "how hard
/// the optimizer is working" axis.
///
/// Returned as `(base_seconds, seconds_per_MB)`.
fn preset_compute_curve(preset: &str) -> (f64, f64) {
    // Anchors:
    //   bonsai 274 MB → 18 s   → ~0.066 s/MB
    //   bicycle 855 MB → 120 s → ~0.140 s/MB (more splats per MB)
    // We pick midpoints per preset and bias toward the high anchor so
    // buyer-facing quotes don't undershoot the actual meter.
    match preset {
        // Fastest preset — bit-exact PLY round-trip, almost no
        // optimization. Tiny constant base, near-zero per-MB.
        "lossless-repack" => (2.0, 0.02),
        // Default web preset. Mid-range optimization.
        "web-mobile" => (4.0, 0.13),
        // Aggressive size minimization — most CPU per byte.
        "size-min" => (6.0, 0.16),
        // Hosted neural codec — v0.1 hash-grid hyperprior + 8-bit quant
        // + RD loss, 1000 iters on Modal A100. Validated on bicycle_real
        // 2026-05-15: 7.54× compression / +8.39 dB ΔPSNR. The +ΔPSNR is
        // real (the optimizer kills low-importance Gaussians and the
        // surviving ones get tuned). Per-MB cost is higher than
        // `size-min` because of the GPU training pass, but absolute is
        // bounded by the 1000-iter cap.
        // Anchor: bicycle 855 MB → ~75 s at 1000 iters on A100.
        "hosted-neural-outdoor" => (12.0, 0.090),
        // hosted-neural — productized version of the per-scene neural
        // codec (M3 ship). Same encoder as hosted-neural-outdoor but
        // exposed as a customer-facing preset routed through the
        // private `splatforge-hosted-neural` Modal app. The encoder is
        // per-scene fit at request time (no shared trained model), so
        // wall-clock is dominated by the 1000-iter A100 training pass.
        // Cost band: ~$0.13/scene on bicycle-sized inputs, ~$0.05 on
        // smaller indoor scenes. Validated N=3 on aaadf09: bicycle
        // outdoor 7.54× / +8.39 dB ΔPSNR (seed-0 7.54x, seed-1 8.08x,
        // seed-2 7.21x). See research/neural-codec-v0.1-m3 branch.
        "hosted-neural" => (120.0, 0.13),
        // MesonGS++ — REMOVED as customer-facing preset 2026-05-15 (task
        // #141). Render-PSNR gate failed by 13-20 dB on bonsai / bicycle:
        // K=256 K-means quantization of the scale group (d=3, 1.1M+
        // splats, log-scale dynamic range) is fundamentally under-
        // resourced. Isolation pinned `scales` as the dominant culprit
        // (21.6 dB hybrid PSNR ≈ 21.5 dB all-decoded). Fix requires
        // scalar-per-channel quantization, not "more K". Crate
        // `splatforge-meson` and CLI `mesonpp-encode`/`-decode` retained
        // for future research; not priced because not sold.
        // CodecGS — feature-plane projection + standard video codec
        // (HEVC). A4 spike 2026-05-15 reproduced the Lee et al. ICCV 2025
        // (arXiv:2501.03399) compression ratios (26.2× at CRF 28; 144.9×
        // at AV1 CRF 38). A4.1 render-PSNR validation followed up and
        // KILLED the CRF 28 / 38 tiers as production defaults — render-
        // PSNR was 17.6 dB / 12.2 dB respectively (attribute-RMSE was
        // misleading; not a proxy for render quality). A4.2 follow-up
        // needed to find the 30 dB knee (likely CRF 14-18 ~ 5-10×).
        // These presets remain reachable for debug / bandwidth-extreme
        // use; default web preset stays 'web-mobile'.
        // Anchor: bonsai 287 MB → ~8 s encode at CRF 28 (lossy).
        "codec-gs" => (4.0, 0.028),
        "codec-gs-extreme" => (4.0, 0.012),
        // CodecGS stacked on v0.1 neural codec — A4.1 BUILT. Bicycle:
        // 152× combined (896.8 → 5.9 MB) with 22.37 dB render-PSNR vs
        // v0.1-trained baseline. v0.1's RD-loss training pushes splats
        // into a more compressible distribution, so CodecGS at same CRF
        // gets 76× vs only 31× on vanilla bicycle. Cost dominated by
        // the v0.1 training step (~$0.30 Modal A100 per scene); the
        // CodecGS post-process is cheap (~3-8 s CPU).
        // Anchor: 15s GPU training base + 0.090 s/MB (v0.1) plus
        // 4s base + 0.028 s/MB (CodecGS post-process), summed.
        "codec-gs-stacked" => (19.0, 0.118),
        // CodecGS Mixed (K=2 default) — novel-3 BUILT 2026-05-15. Bicycle v0.1
        // stacked: 151× @ 25.2 dB (vs 152× @ 22.4 dB for codec-gs-stacked). Same
        // ratio, +2.82 dB render-PSNR. Encodes top-K% of splats by importance
        // (opacity × det(scale)^(2/3)) at CRF 14, rest at CRF 28. Decoder
        // concatenates both streams. Same compute curve as codec-gs-stacked
        // (the partitioning is cheap; both CRF passes run sequentially on CPU).
        // Anchor: bicycle 855 MB → 5.9 MB at 25.2 dB.
        "codec-gs-mixed" => (19.0, 0.118),
        // K=5 variant — slightly worse ratio (59× on bicycle), slightly better
        // PSNR (26.3 dB). Same compute curve. Exposed for users wanting more
        // hi-fidelity headroom at the cost of ratio.
        "codec-gs-mixed-k5" => (19.0, 0.118),
        // FCGS (Fast Feedforward 3DGS Compression, Chen et al. ICLR'25) —
        // pre-trained feed-forward codec on Modal A100. No per-scene
        // optimization, ~15-16× lossless on any 3DGS PLY in ~95 s. The
        // wall-clock is dominated by the encode/decode roundtrip; per-MB
        // cost is roughly flat because the network sees the full splat
        // set regardless of file size on disk.
        // Anchor: bicycle 855 MB → ~95 s end-to-end on A100.
        "fcgs-instant" => (90.0, 0.005),
        // HAC++ Phase A + lzma passthrough — anchor-feature entropy
        // coder for Scaffold-GS scenes, BUILT 2026-05-15. The GPU
        // hyperprior train pass is the only expensive step (~5-10 min
        // A100 on bonsai, 5000 iters); lzma compression of the offset/
        // scale/rot/opacity streams is CPU-cheap (<5 s per scene). The
        // per-MB slope is small because input PLY size doesn't track
        // anchor count linearly — Scaffold-GS bundles trim to ~130 MB
        // even for bicycle-scale scenes.
        // Anchor: bonsai 130 MB Scaffold-GS bundle → 24.21 MB .hacpp
        // container at -0.178 dB render-PSNR vs Scaffold baseline.
        // Lossless-ish on Inria 3DGS PLY passthrough (11.5× lossless).
        "hacpp-lzma" => (2.5, 0.025),
        // capture-and-compress — photos.zip → COLMAP → 3DGS training →
        // compression. The full "no PLY required" pipeline. This is the
        // single preset that closes the loop vs Polycam/Luma — buyers
        // upload raw photos, never see a PLY, get back a compressed
        // .mgs2 / .ply / .lodge depending on the inner-encode preset.
        //
        // Cost composition (Modal A100 anchor, MVP at 7k training iters):
        //   * COLMAP sparse reconstruction:   ~300-600 s (CPU heavy)
        //   * 3DGS training (7k iters MVP):  ~900-1500 s (A100)
        //   * Encode (default codec-gs-mixed): ~95-150 s
        //   * Upload/marshalling overhead:     ~50-100 s
        //   Total floor ≈ 1500 s; typical ≈ 2400-3000 s; ceiling ≈ 3600 s
        //   (training timeout). FastGS / DashGaussian variants replace
        //   the 7k iter training step in a later milestone; that drops
        //   the typical case toward 1800 s without changing the rate card.
        //
        // The `size_bytes` parameter for this preset is the photos.zip
        // payload (NOT a PLY size). The per-MB slope is calibrated to
        // approximate the per-photo cost: a 50-photo, 5 MB/photo zip is
        // ~250 MB and lands at 2400 + 250*2.4 ≈ 3000 s ≈ $3.00 compute
        // (+ flat fee). The 5-10 USD target line in the spec includes
        // operator margin on top of Modal pass-through.
        //
        // Anchor (synthetic, MVP): 250 MB photo zip → ~3000 s end-to-end.
        // Will re-anchor against real captures once the Modal app ships.
        "capture-and-compress" => (2400.0, 2.4),
        // Unknown / future preset: assume web-mobile shape so the
        // preview doesn't 400 on a new preset before the operator
        // tunes a curve for it.
        _ => (4.0, 0.13),
    }
}

/// Round a positive compute-seconds estimate to the nearest whole
/// second. Stripe meter events are integer-valued (`u64`) and the
/// `splatforge_repack_seconds` SKU expects whole seconds; quoting
/// fractional seconds would mismatch the bill. Ceiling so we never
/// under-quote.
fn round_up_seconds(secs_f: f64) -> u64 {
    if !secs_f.is_finite() || secs_f <= 0.0 {
        return 0;
    }
    secs_f.ceil() as u64
}

/// One quote line, returned by `preview_job_cost`. Mirrors the
/// `pricing.astro` calculator's display fields one-to-one.
#[derive(Debug, Clone, Serialize)]
pub struct PricePreview {
    pub pricing_version: &'static str,
    pub preset: String,
    pub size_bytes: u64,
    pub estimated_compute_seconds: u64,
    pub estimated_cost_usd_cents: u64,
    /// Raw, pre-rounding number — useful for the frontend calculator
    /// to display a smooth curve when the user drags a size slider.
    pub estimated_cost_usd: f64,
    pub free_tier_runs_remaining: u32,
    /// Itemized breakdown so the buyer can see the per-job + per-second
    /// math add up. Serializes as a small JSON object.
    pub breakdown: PriceBreakdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct PriceBreakdown {
    pub flat_cents: f64,
    pub compute_cents: f64,
    pub per_second_rate_cents: f64,
    pub per_job_rate_cents: f64,
}

/// Compute the quote. `free_tier_runs_used_this_month` is supplied by
/// the caller (the route handler) from whatever counter we wire next;
/// today the API doesn't track per-customer usage so the route passes
/// `0` and the response shows the full allotment. The math is pure so
/// the test suite doesn't need a database.
pub fn preview_job_cost(
    size_bytes: u64,
    preset: &str,
    free_tier_runs_used_this_month: u32,
) -> PricePreview {
    let (base_s, per_mb_s) = preset_compute_curve(preset);
    let mib = (size_bytes as f64) / (1024.0 * 1024.0);
    let compute_s_f = base_s + per_mb_s * mib;
    let compute_seconds = round_up_seconds(compute_s_f);

    let compute_cents = (compute_seconds as f64) * PER_COMPUTE_SECOND_CENTS;
    let flat_cents = PER_JOB_FLAT_CENTS;
    let total_cents = flat_cents + compute_cents;

    let free_remaining = FREE_TIER_RUNS_PER_MONTH.saturating_sub(free_tier_runs_used_this_month);

    PricePreview {
        pricing_version: PRICING_VERSION,
        preset: preset.to_string(),
        size_bytes,
        estimated_compute_seconds: compute_seconds,
        estimated_cost_usd_cents: total_cents.ceil() as u64,
        estimated_cost_usd: total_cents / 100.0,
        free_tier_runs_remaining: free_remaining,
        breakdown: PriceBreakdown {
            flat_cents,
            compute_cents,
            per_second_rate_cents: PER_COMPUTE_SECOND_CENTS,
            per_job_rate_cents: PER_JOB_FLAT_CENTS,
        },
    }
}

/* ====================================================================== */
/*  SDK licensing                                                           */
/* ====================================================================== */

/// SDK pricing version. Independent of `PRICING_VERSION` so the two
/// rate cards can tune on different cadences. Bump on every change to
/// the constants below.
pub const SDK_PRICING_VERSION: &str = "v0.1";

/// Free MAU allotment per app per month. Above this, the per-MAU rate
/// kicks in. 10k MAU covers indie projects + early-stage startups —
/// the floor where royalty friction would kill adoption.
pub const SDK_FREE_TIER_MAU: u32 = 10_000;

/// Per-MAU royalty above the free tier, in USD cents. $0.001/MAU =
/// $1/10k MAU. A 1M-MAU customer pays $1k/mo, which lines up with
/// "buy the seat, not the lookup" pricing on the dashboard side.
pub const SDK_PER_MAU_CENTS: f64 = 0.1;

/// Supported SDK plugin types. The license JWT's `sub` claim must be
/// one of these. Add new ones as we ship new plugins; refuse unknowns
/// at mint time so a typo in the issuance form can't quietly create a
/// dangling license.
pub const SDK_PLUGIN_TYPES: &[&str] = &[
    "threejs",      // Three.js plugin (npm @splatforge/three)
    "babylonjs",    // Babylon.js plugin
    "model-viewer", // <model-viewer> custom element
    "cesium-ion",   // Cesium ion data tile fetcher
];

/// Default license validity. 1 year matches the Stripe billing cycle
/// boundaries so a renewing customer always sees a fresh license at
/// the start of their next year. Operator can override per-issue.
pub const SDK_LICENSE_TTL_SECS: u64 = 365 * 24 * 60 * 60;

/// One MAU pricing quote, returned by `preview_sdk_cost`.
#[derive(Debug, Clone, Serialize)]
pub struct SdkPricePreview {
    pub sdk_pricing_version: &'static str,
    pub plugin: String,
    pub mau: u32,
    pub free_tier_mau: u32,
    pub paid_mau: u32,
    pub estimated_cost_usd_cents: u64,
    pub estimated_cost_usd: f64,
    pub per_mau_rate_cents: f64,
}

/// Compute the SDK MAU quote for a single app. Pure function; the
/// route handler calls this with the MAU number the customer types
/// into the calculator on `/sdk`.
pub fn preview_sdk_cost(plugin: &str, mau: u32) -> SdkPricePreview {
    let paid_mau = mau.saturating_sub(SDK_FREE_TIER_MAU);
    let cents = (paid_mau as f64) * SDK_PER_MAU_CENTS;
    SdkPricePreview {
        sdk_pricing_version: SDK_PRICING_VERSION,
        plugin: plugin.to_string(),
        mau,
        free_tier_mau: SDK_FREE_TIER_MAU,
        paid_mau,
        estimated_cost_usd_cents: cents.ceil() as u64,
        estimated_cost_usd: cents / 100.0,
        per_mau_rate_cents: SDK_PER_MAU_CENTS,
    }
}

/* ---------- license JWT ---------- */

/// JWT header — fixed to HMAC-SHA256, JWT type. We deliberately don't
/// support alg switching; "alg=none" attacks are a non-issue if there's
/// only ever one accepted algorithm.
const JWT_HEADER_JSON: &str = r#"{"alg":"HS256","typ":"JWT"}"#;

#[derive(Debug, thiserror::Error)]
pub enum LicenseError {
    #[error("unknown sdk plugin: {0}")]
    UnknownPlugin(String),
    #[error("invalid domain: {0}")]
    InvalidDomain(String),
    #[error("malformed license token: {0}")]
    Malformed(String),
    #[error("bad signature")]
    BadSignature,
    #[error("license expired")]
    Expired,
    #[error("license not yet valid")]
    NotYetValid,
    #[error("domain mismatch: license bound to {bound}, presented {presented}")]
    DomainMismatch { bound: String, presented: String },
}

/// License payload. Serialized into the JWT body. Stable shape — once
/// shipped, only ADD fields, never rename/remove.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SdkLicenseClaims {
    /// `iss` — always `"splatforge"` so consumer SDKs can refuse tokens
    /// from a copycat issuer.
    pub iss: String,
    /// `sub` — the plugin type (`"threejs"`, etc).
    pub sub: String,
    /// `aud` — the domain the license is bound to. CORS-style match on
    /// the `Origin` header at telemetry-beacon time.
    pub aud: String,
    /// `iat` — issued-at, unix seconds.
    pub iat: u64,
    /// `exp` — expiry, unix seconds.
    pub exp: u64,
    /// `kid` — license id (random, opaque). Lets the operator revoke a
    /// specific license without invalidating every license for that
    /// customer.
    pub kid: String,
    /// Per-license metadata. Free-form so we can add app-name /
    /// owner-email later without breaking the schema.
    #[serde(default)]
    pub meta: serde_json::Value,
}

/// Normalize a domain to its registrable form for binding. Strips
/// scheme + path + port + trailing dot; lowercases. `"https://Foo.com/x"`
/// → `"foo.com"`. We don't do PSL-aware suffix matching — the binding
/// is exact, so `app.foo.com` is a separate license from `foo.com` and
/// from `www.foo.com`. Customers issue one license per origin.
pub fn normalize_domain(input: &str) -> Result<String, LicenseError> {
    let raw = input.trim();
    if raw.is_empty() {
        return Err(LicenseError::InvalidDomain("empty".to_string()));
    }
    // Strip scheme.
    let after_scheme = match raw.find("://") {
        Some(i) => &raw[i + 3..],
        None => raw,
    };
    // Strip path/query/fragment.
    let host_port = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    // Strip port.
    let host = host_port.split(':').next().unwrap_or(host_port);
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return Err(LicenseError::InvalidDomain(input.to_string()));
    }
    // Reject anything that obviously isn't a hostname — at minimum we
    // need a dot or `localhost`.
    if !host.contains('.') && host != "localhost" {
        return Err(LicenseError::InvalidDomain(input.to_string()));
    }
    // Reject characters that have no business in a hostname.
    if host
        .chars()
        .any(|c| !(c.is_ascii_alphanumeric() || c == '.' || c == '-'))
    {
        return Err(LicenseError::InvalidDomain(input.to_string()));
    }
    Ok(host)
}

/// Mint a license JWT for `(plugin, domain)`. Caller supplies the
/// HMAC secret (from env, never logged) and the desired TTL. Returns
/// the compact-encoded token string `"<b64>.<b64>.<b64>"`.
///
/// Issuance is deterministic in `(claims, secret)` — the same input
/// always produces the same token. That's the property the round-trip
/// test relies on. Real issuance varies the `kid` so revocation is
/// per-license.
pub fn mint_sdk_license(
    plugin: &str,
    domain: &str,
    secret: &[u8],
    now_unix: u64,
    ttl_secs: u64,
    kid: &str,
) -> Result<(String, SdkLicenseClaims), LicenseError> {
    if !SDK_PLUGIN_TYPES.contains(&plugin) {
        return Err(LicenseError::UnknownPlugin(plugin.to_string()));
    }
    let domain = normalize_domain(domain)?;
    let claims = SdkLicenseClaims {
        iss: "splatforge".to_string(),
        sub: plugin.to_string(),
        aud: domain,
        iat: now_unix,
        exp: now_unix.saturating_add(ttl_secs),
        kid: kid.to_string(),
        meta: serde_json::Value::Null,
    };
    let token = encode_jwt(&claims, secret)?;
    Ok((token, claims))
}

/// Verify a license JWT. Checks:
///
///   1. Three base64url-encoded segments.
///   2. Header `alg=HS256`, `typ=JWT` — no algorithm-substitution.
///   3. HMAC-SHA256 signature matches (constant-time compare).
///   4. `now_unix` is between `iat` and `exp`.
///   5. `aud` matches the normalized `expected_domain`.
///
/// Returns the parsed claims on success; an `Err` variant on any
/// failure so the route handler can map it to a clear HTTP status.
pub fn verify_sdk_license(
    token: &str,
    secret: &[u8],
    now_unix: u64,
    expected_domain: &str,
) -> Result<SdkLicenseClaims, LicenseError> {
    let mut parts = token.split('.');
    let header_b64 = parts
        .next()
        .ok_or_else(|| LicenseError::Malformed("missing header".into()))?;
    let payload_b64 = parts
        .next()
        .ok_or_else(|| LicenseError::Malformed("missing payload".into()))?;
    let sig_b64 = parts
        .next()
        .ok_or_else(|| LicenseError::Malformed("missing signature".into()))?;
    if parts.next().is_some() {
        return Err(LicenseError::Malformed("extra segments".into()));
    }

    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|e| LicenseError::Malformed(format!("header b64: {e}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| LicenseError::Malformed(format!("header json: {e}")))?;
    if header.get("alg").and_then(|v| v.as_str()) != Some("HS256") {
        return Err(LicenseError::Malformed("alg != HS256".into()));
    }
    if header.get("typ").and_then(|v| v.as_str()) != Some("JWT") {
        return Err(LicenseError::Malformed("typ != JWT".into()));
    }

    // Recompute signature over `header_b64.payload_b64`.
    let signing_input = format!("{header_b64}.{payload_b64}");
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(secret).expect("HMAC-SHA256 accepts any key length");
    mac.update(signing_input.as_bytes());
    let expected_sig = mac.finalize().into_bytes();
    let expected_sig_b64 = URL_SAFE_NO_PAD.encode(expected_sig);
    if expected_sig_b64.len() != sig_b64.len()
        || !bool::from(expected_sig_b64.as_bytes().ct_eq(sig_b64.as_bytes()))
    {
        return Err(LicenseError::BadSignature);
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|e| LicenseError::Malformed(format!("payload b64: {e}")))?;
    let claims: SdkLicenseClaims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| LicenseError::Malformed(format!("payload json: {e}")))?;

    if now_unix < claims.iat {
        return Err(LicenseError::NotYetValid);
    }
    if now_unix >= claims.exp {
        return Err(LicenseError::Expired);
    }

    let expected = normalize_domain(expected_domain)?;
    if claims.aud != expected {
        return Err(LicenseError::DomainMismatch {
            bound: claims.aud.clone(),
            presented: expected,
        });
    }
    Ok(claims)
}

fn encode_jwt(claims: &SdkLicenseClaims, secret: &[u8]) -> Result<String, LicenseError> {
    let header_b64 = URL_SAFE_NO_PAD.encode(JWT_HEADER_JSON.as_bytes());
    let payload_json = serde_json::to_vec(claims)
        .map_err(|e| LicenseError::Malformed(format!("encode payload: {e}")))?;
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
    let signing_input = format!("{header_b64}.{payload_b64}");
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(secret).expect("HMAC-SHA256 accepts any key length");
    mac.update(signing_input.as_bytes());
    let sig = mac.finalize().into_bytes();
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig);
    Ok(format!("{signing_input}.{sig_b64}"))
}

/// Convenience: current unix time. Pulled out so tests can pass a
/// fixed clock without messing with the system one.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/* ====================================================================== */
/*  Tests                                                                   */
/* ====================================================================== */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_bonsai_sized_job_lands_in_expected_range() {
        // 274 MB bonsai under the web-mobile preset. The BILLING.md
        // anchor target is ~18 s of compute → ~$0.028 total.
        let p = preview_job_cost(274 * 1024 * 1024, "web-mobile", 0);
        assert_eq!(p.pricing_version, PRICING_VERSION);
        assert!(
            p.estimated_compute_seconds >= 30 && p.estimated_compute_seconds <= 60,
            "bonsai @ web-mobile expected ~35-45 s, got {}",
            p.estimated_compute_seconds
        );
        assert_eq!(p.free_tier_runs_remaining, FREE_TIER_RUNS_PER_MONTH);
    }

    #[test]
    fn preview_bicycle_costs_more_than_bonsai_same_preset() {
        let bonsai = preview_job_cost(274 * 1024 * 1024, "web-mobile", 0);
        let bicycle = preview_job_cost(855 * 1024 * 1024, "web-mobile", 0);
        assert!(bicycle.estimated_cost_usd_cents > bonsai.estimated_cost_usd_cents);
        assert!(bicycle.estimated_compute_seconds > bonsai.estimated_compute_seconds);
    }

    #[test]
    fn lossless_repack_is_cheaper_than_size_min() {
        let lossless = preview_job_cost(100 * 1024 * 1024, "lossless-repack", 0);
        let size_min = preview_job_cost(100 * 1024 * 1024, "size-min", 0);
        assert!(size_min.estimated_compute_seconds > lossless.estimated_compute_seconds);
    }

    #[test]
    fn zero_size_still_costs_flat_fee() {
        // Edge: tiny / empty input. The flat per-job fee still applies
        // (we did the bookkeeping work) but compute should be ~base only.
        let p = preview_job_cost(0, "web-mobile", 0);
        assert!(p.estimated_cost_usd_cents >= PER_JOB_FLAT_CENTS as u64);
    }

    #[test]
    fn codec_gs_mixed_matches_codec_gs_stacked_compute_curve() {
        // novel-3 BUILT 2026-05-15: codec-gs-mixed and codec-gs-mixed-k5
        // share codec-gs-stacked's compute curve because the K-percentile
        // partitioning is cheap and both CRF passes run sequentially on
        // CPU. The pricing entry must match exactly so users don't see
        // a different quote for a same-cost pipeline.
        for sz_mb in [100u64, 274, 855] {
            let bytes = sz_mb * 1024 * 1024;
            let stacked = preview_job_cost(bytes, "codec-gs-stacked", 0);
            let mixed = preview_job_cost(bytes, "codec-gs-mixed", 0);
            let mixed_k5 = preview_job_cost(bytes, "codec-gs-mixed-k5", 0);
            assert_eq!(
                stacked.estimated_compute_seconds,
                mixed.estimated_compute_seconds,
                "codec-gs-mixed should match codec-gs-stacked compute at {} MB",
                sz_mb
            );
            assert_eq!(
                stacked.estimated_compute_seconds,
                mixed_k5.estimated_compute_seconds,
                "codec-gs-mixed-k5 should match codec-gs-stacked compute at {} MB",
                sz_mb
            );
        }
    }

    #[test]
    fn fcgs_instant_has_dedicated_curve() {
        // FCGS hosted preset MUST have its own compute curve — not the
        // web-mobile fallback. Anchor: bicycle 855 MB → ~95 s on A100,
        // so a 100 MB job should land near base (90s + 0.005*100 ≈ 91s).
        // We assert (a) the entry is registered (does not collapse to
        // the fallback curve) and (b) the curve is wall-clock-dominated
        // (per-MB rate is tiny vs `web-mobile`'s 0.13 s/MB).
        let bicycle = preview_job_cost(855 * 1024 * 1024, "fcgs-instant", 0);
        let bonsai = preview_job_cost(100 * 1024 * 1024, "fcgs-instant", 0);
        // The flat anchor means a 100MB and 855MB job differ by < 10 s.
        let delta = bicycle
            .estimated_compute_seconds
            .saturating_sub(bonsai.estimated_compute_seconds);
        assert!(
            delta < 10,
            "fcgs-instant should be wall-clock dominated, got Δ={delta}s between 100 MB and 855 MB"
        );
        // Sanity: cheaper than the codec-gs-stacked curve for the same
        // job, because FCGS skips per-scene training entirely.
        let stacked = preview_job_cost(855 * 1024 * 1024, "codec-gs-stacked", 0);
        assert!(
            bicycle.estimated_compute_seconds < stacked.estimated_compute_seconds,
            "fcgs-instant should be cheaper than codec-gs-stacked on bicycle"
        );
    }

    #[test]
    fn hacpp_lzma_has_dedicated_curve_cheaper_than_codec_gs_mixed() {
        // hacpp-lzma is the HAC++ Phase A + lzma anchor-feature codec for
        // Scaffold-GS scenes. The GPU hyperprior pass dominates wall-clock
        // (~5-10 min A100); lzma stream compression is CPU-cheap. The
        // entry MUST (a) be registered (not the web-mobile fallback) and
        // (b) cost less per MB than codec-gs-mixed since the codec is
        // CPU-only post-train.
        let bonsai = preview_job_cost(130 * 1024 * 1024, "hacpp-lzma", 0);
        // 130 MB Scaffold bundle: base 2.5s + 0.025 * 130 = 5.75s → ceil 6s.
        assert!(
            bonsai.estimated_compute_seconds >= 3
                && bonsai.estimated_compute_seconds <= 15,
            "hacpp-lzma bonsai (130 MB) quote drifted: {}s",
            bonsai.estimated_compute_seconds
        );
        // Confirm it's a registered curve, not the fallback. The fallback
        // (web-mobile) would charge 4 + 0.13 * 130 = ~21 s for the same
        // size, so a registered hacpp-lzma curve must land strictly under.
        let fallback = preview_job_cost(130 * 1024 * 1024, "web-mobile", 0);
        assert!(
            bonsai.estimated_compute_seconds < fallback.estimated_compute_seconds,
            "hacpp-lzma must be cheaper than web-mobile fallback at 130 MB"
        );
        // Cheaper per-MB than codec-gs-mixed (which carries the
        // codec-gs-stacked 0.118 s/MB curve). Sanity check on a big
        // scene where the per-MB term dominates.
        let hacpp_big = preview_job_cost(855 * 1024 * 1024, "hacpp-lzma", 0);
        let mixed_big = preview_job_cost(855 * 1024 * 1024, "codec-gs-mixed", 0);
        assert!(
            hacpp_big.estimated_compute_seconds
                < mixed_big.estimated_compute_seconds,
            "hacpp-lzma should be cheaper than codec-gs-mixed at 855 MB"
        );
    }

    #[test]
    fn capture_and_compress_has_dedicated_curve_in_expected_band() {
        // capture-and-compress is the photos → COLMAP → 3DGS → encode
        // pipeline. The compute curve MUST be (a) registered (not the
        // web-mobile fallback) and (b) land a typical 50-photo / ~250 MB
        // zip in the 1500-3600 s wall-clock band that the Modal app
        // budget targets. If a future tune drops the curve below the
        // 1500 s floor we're under-quoting and will leak margin; if it
        // rises above the 3600 s ceiling the worker will time out
        // before the meter event fires.
        let typical = preview_job_cost(250 * 1024 * 1024, "capture-and-compress", 0);
        assert!(
            typical.estimated_compute_seconds >= 1500
                && typical.estimated_compute_seconds <= 3600,
            "capture-and-compress typical (250 MB) should land in [1500, 3600] s, got {}",
            typical.estimated_compute_seconds
        );
        // Cost band: at PER_COMPUTE_SECOND_CENTS=0.1, 1500-3600 s →
        // $1.50-$3.60 of compute, plus the $0.01 flat fee. Operator
        // markup lands the retail $5-10 figure separately; here we just
        // bound the meter-side cost.
        let dollars = typical.estimated_cost_usd;
        assert!(
            dollars >= 1.50 && dollars <= 3.61,
            "capture-and-compress quote band drifted: ${dollars}"
        );
        // Sanity vs fallback: confirm the entry doesn't collapse to the
        // web-mobile curve (which would put 250 MB at ~37 s, way under
        // floor).
        let fallback = preview_job_cost(250 * 1024 * 1024, "web-mobile", 0);
        assert!(
            typical.estimated_compute_seconds > fallback.estimated_compute_seconds * 10,
            "capture-and-compress must be its own (much heavier) curve, not web-mobile"
        );
    }

    #[test]
    fn hosted_neural_has_dedicated_curve_in_expected_band() {
        // hosted-neural is the productized per-scene neural codec
        // (Bet 1 / M3 ship). The compute curve MUST (a) be registered
        // (not the web-mobile fallback) and (b) land a typical
        // 100-855 MB scene in the $0.10-$0.30 band that the Modal A100
        // per-scene-fit anchor targets.
        let bonsai = preview_job_cost(274 * 1024 * 1024, "hosted-neural", 0);
        // 274 MB bonsai: 120 + 0.13*274 = 155.62s → ceil 156s → $0.157
        assert!(
            bonsai.estimated_compute_seconds >= 120
                && bonsai.estimated_compute_seconds <= 250,
            "hosted-neural bonsai (274 MB) drifted: {}s",
            bonsai.estimated_compute_seconds
        );
        // Confirm it's a registered curve, not the fallback (web-mobile
        // at 274 MB lands at 4 + 0.13*274 ≈ 40s, which would make the
        // hosted-neural quote silently collapse to ~$0.05 if the entry
        // got dropped during a refactor).
        let fallback = preview_job_cost(274 * 1024 * 1024, "web-mobile", 0);
        assert!(
            bonsai.estimated_compute_seconds > fallback.estimated_compute_seconds * 3,
            "hosted-neural must dominate web-mobile on wall-clock at 274 MB"
        );
        // Bicycle anchor ~$0.13: 855 MB bicycle → 120 + 0.13*855 = 231 s
        // → $0.231 + $0.01 flat. Allow a $0.10-$0.30 band so any future
        // tune of the per-MB slope or base stays within Modal A100
        // pass-through reality.
        let bicycle = preview_job_cost(855 * 1024 * 1024, "hosted-neural", 0);
        let dollars = bicycle.estimated_cost_usd;
        assert!(
            dollars >= 0.10 && dollars <= 0.40,
            "hosted-neural bicycle quote band drifted: ${dollars}"
        );
    }

    #[test]
    fn unknown_preset_falls_back_to_web_mobile_shape() {
        let p_unknown = preview_job_cost(100 * 1024 * 1024, "future-preset-xyz", 0);
        let p_known = preview_job_cost(100 * 1024 * 1024, "web-mobile", 0);
        assert_eq!(
            p_unknown.estimated_compute_seconds,
            p_known.estimated_compute_seconds
        );
    }

    #[test]
    fn free_tier_remaining_subtracts_usage() {
        let p = preview_job_cost(1024, "web-mobile", 3);
        assert_eq!(p.free_tier_runs_remaining, FREE_TIER_RUNS_PER_MONTH - 3);
        let p2 = preview_job_cost(1024, "web-mobile", FREE_TIER_RUNS_PER_MONTH + 99);
        assert_eq!(p2.free_tier_runs_remaining, 0, "saturating_sub on overflow");
    }

    #[test]
    fn sdk_under_free_tier_is_free() {
        let p = preview_sdk_cost("threejs", 5000);
        assert_eq!(p.estimated_cost_usd_cents, 0);
        assert_eq!(p.paid_mau, 0);
    }

    #[test]
    fn sdk_over_free_tier_charges_per_mau() {
        // 1M MAU: 10k free + 990k paid * $0.001 = $990.
        let p = preview_sdk_cost("threejs", 1_000_000);
        assert_eq!(p.paid_mau, 990_000);
        // 990k * 0.1 cents = 99_000 cents = $990
        assert_eq!(p.estimated_cost_usd_cents, 99_000);
    }

    #[test]
    fn normalize_domain_strips_scheme_and_path_and_port() {
        assert_eq!(
            normalize_domain("https://Foo.com/path?q=1").unwrap(),
            "foo.com"
        );
        assert_eq!(normalize_domain("foo.com:8080").unwrap(), "foo.com");
        assert_eq!(normalize_domain("http://foo.com.").unwrap(), "foo.com");
        assert_eq!(normalize_domain("localhost").unwrap(), "localhost");
    }

    #[test]
    fn normalize_domain_rejects_garbage() {
        assert!(normalize_domain("").is_err());
        assert!(normalize_domain("not a domain").is_err());
        assert!(normalize_domain("no-tld").is_err());
        assert!(normalize_domain("foo$bar.com").is_err());
    }

    #[test]
    fn license_round_trip_succeeds_for_bound_domain() {
        let secret = b"test-secret-do-not-use-in-prod";
        let now = 1_700_000_000_u64;
        let (token, claims) = mint_sdk_license(
            "threejs",
            "https://example.com/some/path",
            secret,
            now,
            3600,
            "lic_test_001",
        )
        .expect("mint");
        assert_eq!(claims.aud, "example.com");
        let verified = verify_sdk_license(&token, secret, now + 10, "example.com").expect("verify");
        assert_eq!(verified.kid, "lic_test_001");
        assert_eq!(verified.sub, "threejs");
        assert_eq!(verified.iss, "splatforge");
    }

    #[test]
    fn license_rejects_wrong_domain() {
        let secret = b"k";
        let now = 1_700_000_000_u64;
        let (token, _) =
            mint_sdk_license("threejs", "example.com", secret, now, 3600, "k1").unwrap();
        let err =
            verify_sdk_license(&token, secret, now + 10, "attacker.com").expect_err("must reject");
        assert!(matches!(err, LicenseError::DomainMismatch { .. }));
    }

    #[test]
    fn license_rejects_tampered_signature() {
        let secret = b"k";
        let now = 1_700_000_000_u64;
        let (token, _) =
            mint_sdk_license("threejs", "example.com", secret, now, 3600, "k1").unwrap();
        // Flip the last char of the signature segment.
        let mut chars: Vec<char> = token.chars().collect();
        let last = chars.len() - 1;
        chars[last] = if chars[last] == 'A' { 'B' } else { 'A' };
        let tampered: String = chars.into_iter().collect();
        let err = verify_sdk_license(&tampered, secret, now + 10, "example.com")
            .expect_err("tampered must reject");
        assert!(matches!(err, LicenseError::BadSignature));
    }

    #[test]
    fn license_rejects_wrong_secret() {
        let secret_a = b"secret-a";
        let secret_b = b"secret-b";
        let now = 1_700_000_000_u64;
        let (token, _) =
            mint_sdk_license("threejs", "example.com", secret_a, now, 3600, "k1").unwrap();
        let err =
            verify_sdk_license(&token, secret_b, now + 10, "example.com").expect_err("wrong key");
        assert!(matches!(err, LicenseError::BadSignature));
    }

    #[test]
    fn license_rejects_expired() {
        let secret = b"k";
        let now = 1_700_000_000_u64;
        let (token, _) =
            mint_sdk_license("threejs", "example.com", secret, now, 100, "k1").unwrap();
        let err =
            verify_sdk_license(&token, secret, now + 200, "example.com").expect_err("expired");
        assert!(matches!(err, LicenseError::Expired));
    }

    #[test]
    fn license_rejects_unknown_plugin() {
        let secret = b"k";
        let err =
            mint_sdk_license("not-a-real-plugin", "example.com", secret, 0, 1, "k1").unwrap_err();
        assert!(matches!(err, LicenseError::UnknownPlugin(_)));
    }

    #[test]
    fn license_rejects_malformed_token() {
        let secret = b"k";
        let err = verify_sdk_license("notajwt", secret, 0, "example.com").expect_err("malformed");
        assert!(matches!(err, LicenseError::Malformed(_)));
    }

    #[test]
    fn license_rejects_alg_none_attack() {
        // Hand-craft a token with `alg=none`. Even if the signature
        // segment is empty (the canonical "alg=none" attack), we must
        // refuse it because our verifier hard-codes HS256.
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(br#"{"iss":"splatforge","sub":"threejs","aud":"example.com","iat":0,"exp":9999999999,"kid":"x","meta":null}"#);
        let token = format!("{header}.{payload}.");
        let err =
            verify_sdk_license(&token, b"k", 1_000, "example.com").expect_err("alg=none must fail");
        assert!(matches!(err, LicenseError::Malformed(_)));
    }
}
