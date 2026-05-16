//! Per-API-key token-bucket rate limiter.
//!
//! ## Design
//!
//! One token-bucket per `(api_key, route_class)` pair. Each bucket holds at
//! most `capacity` tokens and refills at `capacity / window` tokens per
//! second (so the documented "60 per hour" caps refill smoothly over the
//! window rather than resetting at the top of the hour). A request consumes
//! one token; an empty bucket returns 429.
//!
//! Token buckets — not fixed windows — because:
//!   * Bursts are forgiving: a quiet user can spend their whole hour at once.
//!   * Sustained users see a *steady* rate, not the saw-tooth of a fixed
//!     window where everyone slams the gate at HH:00:01.
//!   * Math is trivial and stays correct across clock skew (uses monotonic
//!     `Instant`, not wall-clock).
//!
//! ## Storage
//!
//! `Mutex<HashMap<(key, route_class), BucketState>>`. Single-process is
//! fine on Fly today — we run one machine. If we scale horizontally each
//! machine will enforce its own per-key cap, so the effective per-user
//! cap doubles per added machine. The migration path is to swap this
//! type for a Redis-backed implementation behind the same `Limiter`
//! interface — see `PRODUCTION-HARDENING.md`.
//!
//! ## Identity
//!
//! Per-key, NOT per-IP. Fly puts the API behind their proxy; behind-proxy
//! IPs are either the proxy or `X-Forwarded-For` which is trivially
//! spoofable. The bearer token is the only stable identity we have for
//! programmatic clients. Unauthenticated requests (only hit `/healthz`,
//! the worker callback, and the Stripe webhook) bypass rate-limiting
//! entirely — they're either trusted (worker / Stripe both sign their
//! requests) or harmless (`/healthz`).
//!
//! ## Tunability
//!
//! Limits read from env at startup; an operator can edit `fly secrets set`
//! to retune without a code change. Defaults match the spec:
//!
//!   * `/v1/jobs` POST          — 60/h free, 600/h paid
//!   * `/v1/jobs/:id/upload`    — 10/h free, 100/h paid
//!   * `/v1/jobs/:id/repack`    — 5/h (any tier)
//!   * `/v1/jobs/:id` GET       — 600/h (any tier)
//!   * `/v1/jobs/batch`         — 6/h paid (free not allowed to batch)
//!
//! Env var format: `SPLATFORGE_RATE_LIMITS=create_free=60,create_paid=600,...`
//! Unknown keys are ignored; missing keys fall back to the defaults above.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Logical route classes that have their own rate-limit bucket. Mapping
/// from concrete HTTP route to class lives in `classify_route` so the
/// middleware doesn't have to know the bucket key shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteClass {
    /// `POST /v1/jobs`
    CreateJob,
    /// `POST /v1/jobs/batch` — paid-only (free tier gets 429 with a
    /// different message so the operator can tell a budget-exhausted
    /// paid user apart from a free user trying to batch).
    CreateBatch,
    /// `POST /v1/jobs/:id/upload`
    Upload,
    /// `POST /v1/jobs/:id/repack` — paid-tier-only, very expensive.
    Repack,
    /// `GET /v1/jobs/:id` — high cap because clients poll.
    GetJob,
}

impl RouteClass {
    /// Diagnostic string — surfaces in operator logs when an operator
    /// runs `RUST_LOG=splatforge_api::ratelimit=debug`. Public so future
    /// instrumentation code can pull it without re-deriving from the
    /// enum.
    pub fn name(self) -> &'static str {
        match self {
            RouteClass::CreateJob => "create_job",
            RouteClass::CreateBatch => "create_batch",
            RouteClass::Upload => "upload",
            RouteClass::Repack => "repack",
            RouteClass::GetJob => "get_job",
        }
    }
}

/// Tier the *caller* presents — looked up against the paid-key set in
/// the surrounding middleware. The Limiter doesn't know about Stripe;
/// it just picks the free or paid capacity for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Free,
    Paid,
}

/// Static caps + window per (class, tier). All windows are 1 hour today;
/// the per-class window is a field so we can shorten Repack later if it
/// turns out 5/h is too coarse, without touching the bucket math.
#[derive(Debug, Clone, Copy)]
pub struct Cap {
    pub capacity: u32,
    pub window: Duration,
}

impl Cap {
    pub fn per_hour(n: u32) -> Self {
        Self {
            capacity: n,
            window: Duration::from_secs(3600),
        }
    }

    /// Refill rate in tokens per second. Stored as a float because the
    /// only place it's used is multiplied by elapsed seconds — keeping
    /// it floating-point avoids a "divide every refill" branch.
    fn refill_per_sec(self) -> f64 {
        self.capacity as f64 / self.window.as_secs_f64()
    }
}

/// Limits table. Wide on purpose: every (class, tier) is a separate
/// configurable cap so the operator can move them independently.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub create_free: Cap,
    pub create_paid: Cap,
    pub batch_paid: Cap,
    pub upload_free: Cap,
    pub upload_paid: Cap,
    pub repack: Cap,
    pub get_job: Cap,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            create_free: Cap::per_hour(60),
            create_paid: Cap::per_hour(600),
            batch_paid: Cap::per_hour(6),
            upload_free: Cap::per_hour(10),
            upload_paid: Cap::per_hour(100),
            repack: Cap::per_hour(5),
            get_job: Cap::per_hour(600),
        }
    }
}

impl Limits {
    /// Parse the `SPLATFORGE_RATE_LIMITS` env var.
    ///
    /// Format: comma-separated `name=N` pairs where `N` is the per-hour
    /// cap. Unknown names are silently ignored — the operator can leave
    /// commented-out tunings in place during incident response without
    /// the binary refusing to boot.
    ///
    /// Names: `create_free`, `create_paid`, `batch_paid`, `upload_free`,
    /// `upload_paid`, `repack`, `get_job`.
    pub fn from_env(raw: Option<&str>) -> Self {
        let mut out = Self::default();
        let Some(raw) = raw else { return out };
        for entry in raw.split(',') {
            let Some((name, value)) = entry.split_once('=') else {
                continue;
            };
            let name = name.trim();
            let Ok(n) = value.trim().parse::<u32>() else {
                continue;
            };
            if n == 0 {
                continue;
            }
            let cap = Cap::per_hour(n);
            match name {
                "create_free" => out.create_free = cap,
                "create_paid" => out.create_paid = cap,
                "batch_paid" => out.batch_paid = cap,
                "upload_free" => out.upload_free = cap,
                "upload_paid" => out.upload_paid = cap,
                "repack" => out.repack = cap,
                "get_job" => out.get_job = cap,
                _ => {} // ignore unknown — see doc comment.
            }
        }
        out
    }

    /// Resolve the bucket capacity for a (class, tier) pair.
    ///
    /// `CreateBatch` is paid-only: free callers don't get a bucket
    /// allocated, they hit the explicit `free can't batch` 403 path
    /// in the surrounding middleware. Free-tier callers hitting the
    /// repack class shouldn't reach here (paid-key gate runs first),
    /// but if they do they get the same 5/h cap as paid — better to
    /// rate-limit than to panic.
    pub fn cap_for(&self, class: RouteClass, tier: Tier) -> Cap {
        match (class, tier) {
            (RouteClass::CreateJob, Tier::Free) => self.create_free,
            (RouteClass::CreateJob, Tier::Paid) => self.create_paid,
            (RouteClass::CreateBatch, _) => self.batch_paid,
            (RouteClass::Upload, Tier::Free) => self.upload_free,
            (RouteClass::Upload, Tier::Paid) => self.upload_paid,
            (RouteClass::Repack, _) => self.repack,
            (RouteClass::GetJob, _) => self.get_job,
        }
    }
}

/// In-bucket state. We track `tokens` as a float so partial refill (e.g.
/// 0.7 of a token accumulated since last check) doesn't get rounded
/// down to zero and stall a slow-but-steady user. `last_refill` is a
/// monotonic Instant so wall-clock skew can't run the bucket backwards.
#[derive(Debug, Clone, Copy)]
struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

/// Outcome of a single take attempt. `Retry-After` is in seconds (rounded
/// up); `remaining` is `floor(tokens)` *after* the attempt — for the
/// allow case that's tokens-left, for the deny case it's 0.
#[derive(Debug, Clone, Copy)]
pub enum Decision {
    Allow { remaining: u32 },
    Deny { retry_after_s: u64, remaining: u32 },
}

/// The limiter itself. Cheap to clone (just an `Arc<Mutex>` internally
/// when wrapped) — wire one instance into `AppState`.
pub struct Limiter {
    /// Lock around the map. A `DashMap` would shave a microsecond per
    /// request, but at our throughput this is invisible against I/O
    /// latency and gives us a simpler crash story.
    buckets: Mutex<HashMap<(String, RouteClass), BucketState>>,
    limits: Limits,
    /// Clock injection point — tests pass a `MockClock` so they can
    /// fast-forward without sleeping. Production uses `Instant::now`
    /// via the `RealClock` zero-sized type.
    clock: Box<dyn Clock + Send + Sync>,
}

/// Trait so tests can fast-forward without sleeping.
pub trait Clock {
    fn now(&self) -> Instant;
}

pub struct RealClock;
impl Clock for RealClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

impl Limiter {
    pub fn new(limits: Limits) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            limits,
            clock: Box::new(RealClock),
        }
    }

    /// Test-only constructor. Public because integration tests under
    /// `tests/` live in a separate compilation unit and can't reach
    /// `#[cfg(test)]` items.
    pub fn with_clock(limits: Limits, clock: Box<dyn Clock + Send + Sync>) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            limits,
            clock,
        }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Attempt to take one token for `(key, class, tier)`. Returns
    /// `Decision::Allow` if a token was available (and consumes it);
    /// `Decision::Deny` if the bucket is empty (and does NOT consume).
    ///
    /// Allocates a fresh full bucket on first sight of a (key, class)
    /// pair so a brand-new key gets its full burst on its first call.
    pub fn take(&self, key: &str, class: RouteClass, tier: Tier) -> Decision {
        let cap = self.limits.cap_for(class, tier);
        let refill_rate = cap.refill_per_sec();
        let now = self.clock.now();
        let mut map = self.buckets.lock().expect("ratelimit mutex poisoned");
        let state = map.entry((key.to_string(), class)).or_insert(BucketState {
            tokens: cap.capacity as f64,
            last_refill: now,
        });
        // Refill since last touch. Saturate at `cap.capacity` — a long
        // idle period mustn't overflow into negative deficit.
        let elapsed = now.saturating_duration_since(state.last_refill);
        let refilled = elapsed.as_secs_f64() * refill_rate;
        state.tokens = (state.tokens + refilled).min(cap.capacity as f64);
        state.last_refill = now;

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            return Decision::Allow {
                remaining: state.tokens.floor() as u32,
            };
        }
        // Time until next whole token: (1 - tokens) / refill_rate seconds.
        // Round up so we never tell the client to retry too early.
        let deficit = 1.0 - state.tokens;
        let wait_s = (deficit / refill_rate).ceil() as u64;
        Decision::Deny {
            retry_after_s: wait_s.max(1),
            remaining: 0,
        }
    }
}

/// Mask a key for *display*. We only ever surface the first 8
/// characters — enough to disambiguate keys in a small operator pool
/// without giving an attacker a foothold if the audit log is ever
/// leaked. Anything shorter than 8 chars is returned as-is (test keys).
///
/// **This function is NOT suitable for per-user scoping** of the audit
/// table: every production key is minted with the literal `sf_live_`
/// prefix (`checkout::KEY_PREFIX_LITERAL`), so two distinct customers
/// collide on the same 8-char prefix. Use `key_fingerprint` for any
/// storage / query that must isolate one user from another.
pub fn key_prefix(key: &str) -> String {
    // Operate on chars, not bytes, so a multibyte rune doesn't get sliced.
    let mut out = String::with_capacity(8);
    for (i, c) in key.chars().enumerate() {
        if i >= 8 {
            break;
        }
        out.push(c);
    }
    out
}

/// Per-key opaque fingerprint for audit + per-user storage scoping.
///
/// Returns a 16-char lowercase hex string = SHA-256(key) truncated to
/// the first 64 bits. Properties:
///
///   * **Stable** - same input -> same output across processes / restarts
///     (no random salt; we explicitly want the same key to map to the
///     same row on a redeploy).
///   * **Per-user** - two distinct keys, even ones that share the same
///     literal display prefix (`sf_live_...`), produce distinct
///     fingerprints with overwhelming probability. We expect <1k keys for
///     the lifetime of the beta; the birthday bound on 64 bits is ~4B,
///     so collisions are not a practical concern at this scale.
///   * **Non-reversible** - a leaked audit log doesn't expose the raw
///     bearer token; an attacker would have to bruteforce SHA-256 with
///     no other constraint than length.
///   * **Schema-compatible** - 16 chars fits the existing
///     `audit_events.key_prefix TEXT` column without a migration.
///
/// This replaces `key_prefix` as the audit-table identity column. The
/// 8-char `key_prefix` is still used for human-readable masking in the
/// `/v1/me/usage` response (`key_masked` field).
pub fn key_fingerprint(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(key.as_bytes());
    // First 8 bytes -> 16 hex chars. 64 bits of collision resistance is
    // ample at <1k expected keys; sized to fit existing schema.
    hex::encode(&digest[..8])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// Fast-forwardable clock for the tests in this module.
    struct TestClock {
        inner: Arc<Mutex<Instant>>,
    }
    impl Clock for TestClock {
        fn now(&self) -> Instant {
            *self.inner.lock().unwrap()
        }
    }

    #[test]
    fn defaults_match_spec() {
        let l = Limits::default();
        assert_eq!(l.create_free.capacity, 60);
        assert_eq!(l.create_paid.capacity, 600);
        assert_eq!(l.upload_free.capacity, 10);
        assert_eq!(l.upload_paid.capacity, 100);
        assert_eq!(l.repack.capacity, 5);
        assert_eq!(l.get_job.capacity, 600);
        assert_eq!(l.batch_paid.capacity, 6);
        assert_eq!(l.create_free.window, Duration::from_secs(3600));
    }

    #[test]
    fn env_override_parses_known_keys_and_ignores_unknown() {
        let l = Limits::from_env(Some("create_free=10,unknown=999,repack=2"));
        assert_eq!(l.create_free.capacity, 10);
        assert_eq!(l.repack.capacity, 2);
        // Defaults for everything else.
        assert_eq!(l.upload_paid.capacity, 100);
    }

    #[test]
    fn zero_caps_in_env_are_rejected() {
        // A "0" cap would deny every request forever. Treat as a typo
        // and fall back to the default rather than blackholing traffic.
        let l = Limits::from_env(Some("create_free=0"));
        assert_eq!(l.create_free.capacity, 60);
    }

    #[test]
    fn key_prefix_caps_at_eight() {
        assert_eq!(key_prefix("sk_test_abcdef_long"), "sk_test_");
        assert_eq!(key_prefix("short"), "short");
    }

    #[test]
    fn key_fingerprint_distinguishes_sf_live_keys() {
        // Real bug: every production key starts with `sf_live_`, so the
        // legacy `key_prefix` collapsed every customer to the same scope.
        // The fingerprint must be per-key.
        let a = key_fingerprint("sf_live_alice_aaaaaaaaaaaaaaaaa");
        let b = key_fingerprint("sf_live_bobby_bbbbbbbbbbbbbbbbb");
        assert_ne!(a, b);
        assert_eq!(a.len(), 16);
        assert_eq!(b.len(), 16);
        // Lowercase hex only.
        for c in a.chars() {
            assert!(c.is_ascii_hexdigit() && !c.is_uppercase(), "non-hex char {c}");
        }
    }

    #[test]
    fn key_fingerprint_is_stable() {
        // Same input on two independent calls -> same output. This is
        // what lets a re-issued process find the same audit rows.
        let k = "sf_live_stable_xxxxxxxxxxxxxxxxx";
        assert_eq!(key_fingerprint(k), key_fingerprint(k));
    }

    #[test]
    fn key_fingerprint_handles_empty_and_unicode() {
        // Bad-faith inputs: empty string and multibyte runes must not
        // panic. SHA-256 accepts arbitrary bytes; we hash the UTF-8.
        let e = key_fingerprint("");
        assert_eq!(e.len(), 16);
        // SHA256("") is well-known: e3b0c44298fc1c14...; truncated to 16.
        assert_eq!(e, "e3b0c44298fc1c14");
        // Unicode in a (synthetic) key must not panic and must differ
        // from a plain-ASCII variant.
        let u = key_fingerprint("sf_live_\u{1F4A9}xxxxxxxxxxxxxxxxx");
        let a = key_fingerprint("sf_live_!xxxxxxxxxxxxxxxxxxxxxxx");
        assert_ne!(u, a);
        assert_eq!(u.len(), 16);
    }

    fn make_limiter(limits: Limits, start: Instant) -> (Limiter, Arc<Mutex<Instant>>) {
        let clock = Arc::new(Mutex::new(start));
        let l = Limiter::with_clock(
            limits,
            Box::new(TestClock {
                inner: clock.clone(),
            }),
        );
        (l, clock)
    }

    #[test]
    fn first_burst_consumes_exactly_capacity_then_429s() {
        let now = Instant::now();
        let limits = Limits {
            create_free: Cap {
                capacity: 3,
                window: Duration::from_secs(3600),
            },
            ..Limits::default()
        };
        let (l, _clk) = make_limiter(limits, now);
        for i in 0..3 {
            match l.take("k", RouteClass::CreateJob, Tier::Free) {
                Decision::Allow { remaining } => assert_eq!(remaining, 2 - i),
                Decision::Deny { .. } => panic!("got 429 within burst at i={i}"),
            }
        }
        // 4th call must 429.
        match l.take("k", RouteClass::CreateJob, Tier::Free) {
            Decision::Deny { retry_after_s, .. } => {
                assert!(retry_after_s >= 1);
            }
            Decision::Allow { .. } => panic!("4th call inside burst must 429"),
        }
    }

    #[test]
    fn bucket_refills_at_documented_rate() {
        // 3 tokens / 30s → 0.1 tokens/sec. After 10s we have +1 token.
        let now = Instant::now();
        let limits = Limits {
            create_free: Cap {
                capacity: 3,
                window: Duration::from_secs(30),
            },
            ..Limits::default()
        };
        let (l, clk) = make_limiter(limits, now);
        // Drain.
        for _ in 0..3 {
            assert!(matches!(
                l.take("k", RouteClass::CreateJob, Tier::Free),
                Decision::Allow { .. }
            ));
        }
        assert!(matches!(
            l.take("k", RouteClass::CreateJob, Tier::Free),
            Decision::Deny { .. }
        ));
        // Fast-forward 10s → +1 token.
        *clk.lock().unwrap() = now + Duration::from_secs(10);
        assert!(matches!(
            l.take("k", RouteClass::CreateJob, Tier::Free),
            Decision::Allow { .. }
        ));
        // Immediately afterwards we should be empty again.
        assert!(matches!(
            l.take("k", RouteClass::CreateJob, Tier::Free),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn free_and_paid_keys_have_independent_buckets() {
        let now = Instant::now();
        let (l, _) = make_limiter(Limits::default(), now);
        // Free-tier cap is 60; drain it.
        for _ in 0..60 {
            assert!(matches!(
                l.take("free", RouteClass::CreateJob, Tier::Free),
                Decision::Allow { .. }
            ));
        }
        assert!(matches!(
            l.take("free", RouteClass::CreateJob, Tier::Free),
            Decision::Deny { .. }
        ));
        // Paid bucket is wholly untouched.
        assert!(matches!(
            l.take("paid", RouteClass::CreateJob, Tier::Paid),
            Decision::Allow { remaining: 599 }
        ));
    }

    #[test]
    fn different_route_classes_are_isolated() {
        let now = Instant::now();
        let (l, _) = make_limiter(Limits::default(), now);
        // Drain repack to 0 (5/h).
        for _ in 0..5 {
            assert!(matches!(
                l.take("k", RouteClass::Repack, Tier::Paid),
                Decision::Allow { .. }
            ));
        }
        assert!(matches!(
            l.take("k", RouteClass::Repack, Tier::Paid),
            Decision::Deny { .. }
        ));
        // Same key, different class — fresh bucket.
        assert!(matches!(
            l.take("k", RouteClass::GetJob, Tier::Paid),
            Decision::Allow { .. }
        ));
    }
}
