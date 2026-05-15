//! Integration tests for the per-API-key token-bucket rate limiter.
//!
//! Covers the four properties named in the production-hardening spec:
//!   (a) burst N+1 returns 429
//!   (b) bucket refills at expected rate
//!   (c) free vs paid keys see different caps
//!   (d) per-class isolation (separate buckets per route class)
//!
//! Plus a small set of guard tests around env parsing + key masking
//! because those are the operator-facing tunability knobs and a typo
//! in `SPLATFORGE_RATE_LIMITS` shouldn't silently disable a class.
//!
//! Clock is injected (no `std::thread::sleep`) so the suite runs in
//! milliseconds and is deterministic on a busy CI runner.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use splatforge_api::ratelimit::{
    key_prefix, Cap, Clock, Decision, Limiter, Limits, RouteClass, Tier,
};

/* ---------- test clock ---------- */

struct TestClock {
    inner: Arc<Mutex<Instant>>,
}
impl Clock for TestClock {
    fn now(&self) -> Instant {
        *self.inner.lock().unwrap()
    }
}

fn make_clock(start: Instant) -> (Box<dyn Clock + Send + Sync>, Arc<Mutex<Instant>>) {
    let inner = Arc::new(Mutex::new(start));
    let clk: Box<dyn Clock + Send + Sync> = Box::new(TestClock { inner: inner.clone() });
    (clk, inner)
}

fn advance(clk: &Arc<Mutex<Instant>>, dt: Duration) {
    let mut g = clk.lock().unwrap();
    *g += dt;
}

/* ---------- (a) burst N+1 returns 429 ---------- */

#[test]
fn burst_consumes_capacity_then_429s() {
    let (clk_box, _clk) = make_clock(Instant::now());
    let limits = Limits {
        create_free: Cap {
            capacity: 5,
            window: Duration::from_secs(3600),
        },
        ..Limits::default()
    };
    let l = Limiter::with_clock(limits, clk_box);
    for _ in 0..5 {
        match l.take("k_free", RouteClass::CreateJob, Tier::Free) {
            Decision::Allow { .. } => {}
            Decision::Deny { .. } => panic!("denied inside burst capacity"),
        }
    }
    match l.take("k_free", RouteClass::CreateJob, Tier::Free) {
        Decision::Deny { retry_after_s, remaining } => {
            assert!(retry_after_s >= 1, "retry_after_s must be at least 1");
            assert_eq!(remaining, 0);
        }
        Decision::Allow { .. } => panic!("N+1th call must 429"),
    }
}

#[test]
fn default_create_free_caps_at_60() {
    // Defaults from the spec: 60/h free, 600/h paid for /v1/jobs.
    let (clk_box, _clk) = make_clock(Instant::now());
    let l = Limiter::with_clock(Limits::default(), clk_box);
    for _ in 0..60 {
        assert!(matches!(
            l.take("kf", RouteClass::CreateJob, Tier::Free),
            Decision::Allow { .. }
        ));
    }
    assert!(matches!(
        l.take("kf", RouteClass::CreateJob, Tier::Free),
        Decision::Deny { .. }
    ));
}

/* ---------- (b) bucket refills at expected rate ---------- */

#[test]
fn refill_returns_tokens_after_window_elapses() {
    // 6 tokens / 60s → 0.1 tok/s.
    let start = Instant::now();
    let (clk_box, clk) = make_clock(start);
    let limits = Limits {
        repack: Cap {
            capacity: 6,
            window: Duration::from_secs(60),
        },
        ..Limits::default()
    };
    let l = Limiter::with_clock(limits, clk_box);
    // Drain.
    for _ in 0..6 {
        assert!(matches!(
            l.take("k", RouteClass::Repack, Tier::Paid),
            Decision::Allow { .. }
        ));
    }
    assert!(matches!(
        l.take("k", RouteClass::Repack, Tier::Paid),
        Decision::Deny { .. }
    ));
    // Advance 10s -> +1 token.
    advance(&clk, Duration::from_secs(10));
    assert!(matches!(
        l.take("k", RouteClass::Repack, Tier::Paid),
        Decision::Allow { .. }
    ));
    // Drained again.
    assert!(matches!(
        l.take("k", RouteClass::Repack, Tier::Paid),
        Decision::Deny { .. }
    ));
    // Advance the full window — bucket back at capacity.
    advance(&clk, Duration::from_secs(60));
    for _ in 0..6 {
        assert!(matches!(
            l.take("k", RouteClass::Repack, Tier::Paid),
            Decision::Allow { .. }
        ));
    }
}

#[test]
fn refill_does_not_exceed_capacity() {
    // Bucket capacity is the *ceiling* — leaving a key idle for a year
    // doesn't grant a year's worth of tokens on the next request.
    let start = Instant::now();
    let (clk_box, clk) = make_clock(start);
    let limits = Limits {
        create_free: Cap {
            capacity: 3,
            window: Duration::from_secs(10),
        },
        ..Limits::default()
    };
    let l = Limiter::with_clock(limits, clk_box);
    // Spend one token, then idle for an hour.
    assert!(matches!(
        l.take("k", RouteClass::CreateJob, Tier::Free),
        Decision::Allow { remaining: 2 }
    ));
    advance(&clk, Duration::from_secs(3600));
    // Capacity is 3, NOT 3 + 3600*0.3.
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
}

/* ---------- (c) free vs paid see different caps ---------- */

#[test]
fn free_and_paid_caps_diverge_for_create_job() {
    // The spec calls for 60/h free, 600/h paid. Same route class, same
    // limiter — the tier controls the cap. We don't need to enumerate
    // all 660 tokens; we exercise the boundary (free at 60 → 429,
    // paid still allowed).
    let (clk_box, _) = make_clock(Instant::now());
    let l = Limiter::with_clock(Limits::default(), clk_box);
    for _ in 0..60 {
        assert!(matches!(
            l.take("free_key", RouteClass::CreateJob, Tier::Free),
            Decision::Allow { .. }
        ));
    }
    assert!(matches!(
        l.take("free_key", RouteClass::CreateJob, Tier::Free),
        Decision::Deny { .. }
    ));
    // Paid key has a totally separate bucket and a 10x cap.
    for _ in 0..61 {
        assert!(matches!(
            l.take("paid_key", RouteClass::CreateJob, Tier::Paid),
            Decision::Allow { .. }
        ));
    }
}

#[test]
fn free_and_paid_caps_diverge_for_upload() {
    // 10/h free, 100/h paid per the spec.
    let (clk_box, _) = make_clock(Instant::now());
    let l = Limiter::with_clock(Limits::default(), clk_box);
    for _ in 0..10 {
        assert!(matches!(
            l.take("kf", RouteClass::Upload, Tier::Free),
            Decision::Allow { .. }
        ));
    }
    assert!(matches!(
        l.take("kf", RouteClass::Upload, Tier::Free),
        Decision::Deny { .. }
    ));
    for _ in 0..11 {
        assert!(matches!(
            l.take("kp", RouteClass::Upload, Tier::Paid),
            Decision::Allow { .. }
        ));
    }
}

#[test]
fn repack_caps_at_5_for_any_tier() {
    // Repack is expensive (A100). Per spec, 5/h regardless of tier.
    let (clk_box, _) = make_clock(Instant::now());
    let l = Limiter::with_clock(Limits::default(), clk_box);
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
}

#[test]
fn keys_are_isolated_from_each_other() {
    // A noisy neighbour on the same plan must not affect a separate key.
    let (clk_box, _) = make_clock(Instant::now());
    let limits = Limits {
        create_free: Cap {
            capacity: 2,
            window: Duration::from_secs(3600),
        },
        ..Limits::default()
    };
    let l = Limiter::with_clock(limits, clk_box);
    // Drain key A.
    assert!(matches!(l.take("a", RouteClass::CreateJob, Tier::Free), Decision::Allow { .. }));
    assert!(matches!(l.take("a", RouteClass::CreateJob, Tier::Free), Decision::Allow { .. }));
    assert!(matches!(l.take("a", RouteClass::CreateJob, Tier::Free), Decision::Deny { .. }));
    // Key B has its own bucket.
    assert!(matches!(l.take("b", RouteClass::CreateJob, Tier::Free), Decision::Allow { remaining: 1 }));
}

/* ---------- (d) per-class isolation ---------- */

#[test]
fn route_classes_use_independent_buckets() {
    // The CreateJob bucket emptying must not affect Upload / GetJob /
    // Repack / Batch for the same key.
    let (clk_box, _) = make_clock(Instant::now());
    let l = Limiter::with_clock(Limits::default(), clk_box);
    for _ in 0..60 {
        l.take("k", RouteClass::CreateJob, Tier::Free);
    }
    assert!(matches!(
        l.take("k", RouteClass::CreateJob, Tier::Free),
        Decision::Deny { .. }
    ));
    // Other classes are still fresh.
    assert!(matches!(
        l.take("k", RouteClass::GetJob, Tier::Free),
        Decision::Allow { .. }
    ));
    assert!(matches!(
        l.take("k", RouteClass::Upload, Tier::Free),
        Decision::Allow { .. }
    ));
    assert!(matches!(
        l.take("k", RouteClass::Repack, Tier::Paid),
        Decision::Allow { .. }
    ));
    assert!(matches!(
        l.take("k", RouteClass::CreateBatch, Tier::Paid),
        Decision::Allow { .. }
    ));
}

/* ---------- env tunability ---------- */

#[test]
fn env_override_applies_per_class() {
    let l = Limits::from_env(Some(
        "create_free=2,create_paid=5,repack=1,upload_free=3,batch_paid=2",
    ));
    assert_eq!(l.create_free.capacity, 2);
    assert_eq!(l.create_paid.capacity, 5);
    assert_eq!(l.repack.capacity, 1);
    assert_eq!(l.upload_free.capacity, 3);
    assert_eq!(l.batch_paid.capacity, 2);
    // Untouched classes keep their defaults.
    assert_eq!(l.get_job.capacity, 600);
    assert_eq!(l.upload_paid.capacity, 100);
}

#[test]
fn env_override_ignores_typos() {
    // A typo in SPLATFORGE_RATE_LIMITS shouldn't blackhole a class.
    let l = Limits::from_env(Some("create_freee=2,not_a_class=5"));
    assert_eq!(l.create_free.capacity, 60, "typo'd key must fall back to default");
}

/* ---------- key masking ---------- */

#[test]
fn key_prefix_never_leaks_full_token() {
    let full = "sk_test_super_secret_token_abcdef";
    let prefix = key_prefix(full);
    assert_eq!(prefix.len(), 8);
    assert!(full.starts_with(&prefix));
    // Reverse-lookup must be impossible from the prefix alone.
    assert_ne!(prefix, full);
}

#[test]
fn key_prefix_handles_short_keys_safely() {
    // A dev-mode short key (e.g. test fixture) returns as-is, not panic.
    let p = key_prefix("a1b2");
    assert_eq!(p, "a1b2");
}
