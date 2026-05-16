//! Customer-facing usage dashboard.
//!
//! Backs `GET /v1/me/usage` — the JSON the public `/dashboard` page on
//! splatforge.dev renders. Every field is derived from data the bearer
//! token already has access to; this module never reaches across keys.
//!
//! ## Why this is separate from `admin_audit`
//!
//! `GET /v1/admin/audit` is operator-only (gated on
//! `SPLATFORGE_ADMIN_API_KEYS`) and returns the audit trail for the
//! entire deployment. The customer dashboard surfaces a strict *subset*
//! of that data scoped to one masked key prefix. Conflating the two
//! handlers would mean a customer key with the audit scope could read
//! the whole table; keeping them split makes the per-key filter the
//! single load-bearing piece of code that prevents cross-user leakage.
//!
//! ## Plan tier
//!
//! Resolved from the same `AppState` knobs the auth middleware already
//! uses:
//!
//!   * key in `paid_api_keys`  → `"paid"`
//!   * otherwise               → `"free"`
//!
//! There's no per-customer email column in our store yet (Stripe
//! Checkout writes it on the team_signups path, but free keys are
//! provisioned by hand). For first-session scaffolding we surface
//! `email = None` when the lookup fails and let the page render a
//! "Sign in to see email" stub.
//!
//! ## Usage counters
//!
//! Today's billing-meter totals (`splatforge_repack_runs`,
//! `splatforge_repack_seconds`) live in Stripe, not our DB — we POST
//! them and forget. For the first cut of the dashboard we derive a
//! local approximation from the audit log: count `POST
//! /v1/jobs/:id/repack` rows with status 2xx and read their
//! `duration_ms`. This stays bit-correct against the local ledger
//! even when Stripe is in dry-run mode (CI / staging), and lets the
//! page render a usage line on day one. A follow-up session will
//! switch to the authoritative Stripe meter readout via the
//! `/v1/billing/meter_event_summaries` API.

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::store::{AuditEvent, DynJobStore, StoreError};

/// Hard cap on how many recent-jobs rows the dashboard returns per
/// request. 25 is the spec deliverable; we hard-cap server-side so a
/// curious caller can't pull the whole audit table by inflating the
/// query string.
pub const RECENT_JOBS_DEFAULT_LIMIT: u32 = 25;
pub const RECENT_JOBS_MAX_LIMIT: u32 = 100;

/// Plan tier exposed to the customer. Stable string, NOT the same enum
/// as `ratelimit::Tier` (which carries an internal-only `Paid` variant
/// without product-marketing names). Keeping these decoupled lets us
/// rename the marketing tier without churning the limiter code.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Plan {
    Free,
    Paid,
}

impl Plan {
    pub fn label(self) -> &'static str {
        match self {
            Plan::Free => "Free",
            Plan::Paid => "Paid",
        }
    }
}

/// Aggregate usage counters for the period covered by the audit log.
/// `period_start` is the timestamp of the oldest audited row we found
/// for this key — a first-cut approximation of "start of the current
/// billing period". A follow-up session will resolve the real billing
/// anchor from the Stripe subscription.
#[derive(Debug, Clone, Serialize)]
pub struct UsageSummary {
    pub repack_runs: u64,
    pub repack_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period_start: Option<DateTime<Utc>>,
}

/// One row in the dashboard's "Recent jobs" table. Derived from
/// `AuditEvent` but with a customer-facing shape — we strip the raw
/// `body_size` (operator-only signal) and rename `key_prefix` away
/// (the customer already knows their own key).
#[derive(Debug, Clone, Serialize)]
pub struct RecentJob {
    pub timestamp: DateTime<Utc>,
    pub route: String,
    pub method: String,
    pub status: u16,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<AuditEvent> for RecentJob {
    fn from(ev: AuditEvent) -> Self {
        Self {
            timestamp: ev.created_at,
            route: ev.route,
            method: ev.method,
            status: ev.status,
            duration_ms: ev.duration_ms,
            error: ev.error,
        }
    }
}

/// The full JSON returned by `GET /v1/me/usage`.
#[derive(Debug, Clone, Serialize)]
pub struct DashboardResponse {
    pub plan: Plan,
    /// 8-char masked key prefix (e.g. `sk_test_`). NEVER the full token
    /// — even though the requester already presented it, we still mask
    /// in case this response is screenshotted into a support ticket.
    pub key_masked: String,
    /// Email on file, if known. `None` for free keys provisioned by hand
    /// (no Checkout flow). The page falls back to "(unknown — contact
    /// support to attach)" when this is null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub usage: UsageSummary,
    pub recent_jobs: Vec<RecentJob>,
}

/// Build the dashboard response for a single bearer key.
///
/// `key_prefix` MUST come from `ratelimit::key_prefix` — the audit
/// table is queried by the masked form. `plan` and `email` are
/// resolved by the caller (they need access to `AppState`); this
/// function's job is just to assemble the response from the store.
pub async fn build_response(
    store: &DynJobStore,
    key_prefix: String,
    plan: Plan,
    email: Option<String>,
    limit: u32,
) -> Result<DashboardResponse, StoreError> {
    let limit = limit.clamp(1, RECENT_JOBS_MAX_LIMIT);
    let events = store
        .list_audit_events_by_prefix(&key_prefix, limit)
        .await?;
    let usage = summarize(&events);
    let recent_jobs = events.into_iter().map(RecentJob::from).collect();
    Ok(DashboardResponse {
        plan,
        key_masked: key_prefix,
        email,
        usage,
        recent_jobs,
    })
}

/// Local approximation of the Stripe meter totals from the audit rows
/// we already have. See module docs for the "why local first" rationale.
fn summarize(events: &[AuditEvent]) -> UsageSummary {
    let mut repack_runs = 0_u64;
    let mut repack_seconds = 0_u64;
    let mut oldest: Option<DateTime<Utc>> = None;
    for ev in events {
        if ev.route == "/v1/jobs/:id/repack" && (200..300).contains(&ev.status) {
            repack_runs += 1;
            // duration_ms → seconds, rounded down. Matches what the
            // billing module reports to Stripe (worker callback rounds
            // the same way).
            repack_seconds += ev.duration_ms / 1000;
        }
        oldest = match oldest {
            Some(prev) if prev <= ev.created_at => Some(prev),
            _ => Some(ev.created_at),
        };
    }
    UsageSummary {
        repack_runs,
        repack_seconds,
        period_start: oldest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn mk_event(route: &str, status: u16, duration_ms: u64, ts: i64) -> AuditEvent {
        AuditEvent {
            id: format!("evt-{ts}"),
            key_prefix: "sk_test_".to_string(),
            route: route.to_string(),
            method: "POST".to_string(),
            status,
            body_size: 0,
            duration_ms,
            error: None,
            created_at: Utc.timestamp_opt(ts, 0).single().unwrap(),
        }
    }

    #[test]
    fn summarize_counts_only_successful_repacks() {
        let events = vec![
            // Successful repack — counts.
            mk_event("/v1/jobs/:id/repack", 200, 18_000, 1_700_000_000),
            // Failed repack — does NOT count.
            mk_event("/v1/jobs/:id/repack", 500, 2_000, 1_700_000_500),
            // Different route — never counts.
            mk_event("/v1/jobs", 201, 100, 1_700_001_000),
            // Another success — counts.
            mk_event("/v1/jobs/:id/repack", 200, 7_500, 1_700_002_000),
        ];
        let s = summarize(&events);
        assert_eq!(s.repack_runs, 2);
        // 18 + 7 = 25 (7.5 rounds down to 7, matching billing semantics).
        assert_eq!(s.repack_seconds, 25);
        assert_eq!(
            s.period_start,
            Some(Utc.timestamp_opt(1_700_000_000, 0).single().unwrap())
        );
    }

    #[test]
    fn summarize_handles_empty_log() {
        let s = summarize(&[]);
        assert_eq!(s.repack_runs, 0);
        assert_eq!(s.repack_seconds, 0);
        assert!(s.period_start.is_none());
    }

    #[test]
    fn plan_serializes_as_lowercase() {
        let json = serde_json::to_string(&Plan::Paid).unwrap();
        assert_eq!(json, "\"paid\"");
    }

    #[test]
    fn recent_job_strips_body_size_and_key() {
        let ev = mk_event("/v1/jobs", 201, 42, 1_700_000_000);
        let row: RecentJob = ev.into();
        let json = serde_json::to_value(&row).unwrap();
        // No `key_prefix` leakage, no `body_size` leakage.
        assert!(json.get("key_prefix").is_none());
        assert!(json.get("body_size").is_none());
        assert_eq!(json["route"], "/v1/jobs");
    }
}
