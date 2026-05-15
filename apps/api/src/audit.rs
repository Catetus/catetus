//! Audit log for mutating /v1 routes.
//!
//! Persistence lives in `store::JobStore::insert_audit_event`; this
//! module is the thin coordination layer that:
//!
//!   * Knows which routes to log (mutating /v1 only — see `is_mutating`).
//!   * Masks the bearer key to its 8-char prefix before writing.
//!   * Treats the write as best-effort: a DB failure here MUST NOT
//!     surface to the user.
//!
//! ## What's logged
//!
//! `(timestamp, key_prefix, route_template, method, status, body_size,
//! duration_ms, optional short error)`. We do not log:
//!
//!   * The raw bearer token (only first 8 chars).
//!   * Request/response bodies (could contain user-supplied URLs with
//!     embedded credentials).
//!   * IP addresses (Fly puts us behind a proxy; not useful + GDPR).
//!
//! ## Read path
//!
//! `GET /v1/admin/audit` returns the last 1000 events (most recent
//! first) as JSON. Gated on the `SPLATFORGE_ADMIN_API_KEYS` env var —
//! a separate set from the regular API keys so a leaked customer key
//! can't read the audit trail.

use tracing::warn;

use crate::store::DynJobStore;

/// Hard cap on the number of audit rows returned by the admin endpoint.
/// Matches the spec deliverable. Larger queries would have to page,
/// which we'll add when the operator actually has > 1000 events to
/// scroll through.
pub const ADMIN_AUDIT_DEFAULT_LIMIT: u32 = 1000;

/// Templated routes the audit log writes for. Concrete request paths
/// like `/v1/jobs/abc123/upload` are mapped to their *template* form
/// (`/v1/jobs/:id/upload`) before insert, so the route column is
/// queryable by class rather than by specific job id.
pub fn route_template(method: &str, path: &str) -> Option<&'static str> {
    // Read-only routes deliberately not in the list — we only audit
    // mutating routes per the spec. `GET /v1/jobs/:id` is high-volume
    // and read-only; auditing it would dwarf the useful rows.
    match (method, classify_path(path)?) {
        ("POST", "/v1/jobs") => Some("/v1/jobs"),
        ("POST", "/v1/jobs/batch") => Some("/v1/jobs/batch"),
        ("POST", "/v1/jobs/:id/upload") => Some("/v1/jobs/:id/upload"),
        ("POST", "/v1/jobs/:id/repack") => Some("/v1/jobs/:id/repack"),
        ("POST", "/v1/jobs/:id/result") => Some("/v1/jobs/:id/result"),
        ("POST", "/v1/stripe/webhook") => Some("/v1/stripe/webhook"),
        _ => None,
    }
}

/// `true` if the request should generate an audit row. Convenience
/// wrapper around `route_template` for the middleware.
pub fn is_audited(method: &str, path: &str) -> bool {
    route_template(method, path).is_some()
}

/// Coarse path classifier. Maps `/v1/jobs/<uuid>/upload` →
/// `/v1/jobs/:id/upload`. Intentionally string-based — `axum::Path`
/// has already validated the uuid by the time the route handler
/// fires, but the audit middleware runs around the whole request so
/// it sees the raw path. Anything we don't recognize is ignored
/// (returned as `None`) and the audit middleware short-circuits.
fn classify_path(path: &str) -> Option<&'static str> {
    if path == "/v1/jobs" {
        return Some("/v1/jobs");
    }
    if path == "/v1/jobs/batch" {
        return Some("/v1/jobs/batch");
    }
    if path == "/v1/stripe/webhook" {
        return Some("/v1/stripe/webhook");
    }
    // /v1/jobs/<id>/<tail>
    let rest = path.strip_prefix("/v1/jobs/")?;
    let (head, tail) = rest.split_once('/').unwrap_or((rest, ""));
    if head.is_empty() {
        return None;
    }
    // We don't validate the UUID shape here — middleware logs every
    // mutating request, even if the uuid is malformed (the handler
    // will 400 it; we still want the audit row for the attempt).
    match tail {
        "upload" => Some("/v1/jobs/:id/upload"),
        "repack" => Some("/v1/jobs/:id/repack"),
        "result" => Some("/v1/jobs/:id/result"),
        _ => None,
    }
}

/// Write one audit event. Best-effort: a DB error is logged at WARN
/// and otherwise swallowed. The caller has already responded to the
/// client by this point — propagating an error here would do nothing
/// but spam logs with confusing transactional rollbacks.
///
/// `key_prefix` should already be masked by `ratelimit::key_prefix`
/// — we don't re-mask here so test code can write a known prefix.
#[allow(clippy::too_many_arguments)]
pub async fn record(
    store: &DynJobStore,
    key_prefix: &str,
    route: &str,
    method: &str,
    status: u16,
    body_size: u64,
    duration_ms: u64,
    error: Option<&str>,
) {
    if let Err(e) = store
        .insert_audit_event(
            key_prefix,
            route,
            method,
            status,
            body_size,
            duration_ms,
            error,
        )
        .await
    {
        warn!(
            error = %e,
            route,
            method,
            status,
            "audit log write failed; request response was unaffected"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_strip_uuid() {
        assert_eq!(
            route_template("POST", "/v1/jobs/abc-123/upload"),
            Some("/v1/jobs/:id/upload")
        );
        assert_eq!(
            route_template("POST", "/v1/jobs/abc-123/repack"),
            Some("/v1/jobs/:id/repack")
        );
        assert_eq!(
            route_template("POST", "/v1/jobs/abc-123/result"),
            Some("/v1/jobs/:id/result")
        );
        assert_eq!(route_template("POST", "/v1/jobs"), Some("/v1/jobs"));
        assert_eq!(route_template("POST", "/v1/jobs/batch"), Some("/v1/jobs/batch"));
    }

    #[test]
    fn read_only_routes_not_audited() {
        // GETs are read-only — the spec deliberately limits the audit
        // log to mutating routes to keep the volume sane.
        assert!(!is_audited("GET", "/v1/jobs/abc-123"));
        assert!(!is_audited("GET", "/healthz"));
        assert!(!is_audited("GET", "/openapi.yaml"));
    }

    #[test]
    fn unknown_paths_not_audited() {
        assert!(!is_audited("POST", "/v1/jobs/abc-123/teleport"));
        assert!(!is_audited("DELETE", "/v1/jobs"));
    }
}
