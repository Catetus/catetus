//! `splatforge license …` and `splatforge serve` implementations.
//!
//! The license subcommands and the `serve` command live here so the main
//! `splatforge` binary stays a thin dispatch shell. None of these handlers
//! ever see the issuer's private key — they only consume the public
//! signature path via `splatforge-license`.

use std::path::{Path, PathBuf};
use std::time::Duration as StdDuration;

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use splatforge_license::{License, LicenseConfig};

/// Default install location. `dirs::home_dir()` returns the user's home
/// even on macOS where `$HOME` may be unset under launchd; we fall back
/// to CWD if the lookup fails (e.g. inside a container with no passwd
/// entry) so the CLI never panics on a missing home.
pub fn default_license_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".splatforge").join("license.lic")
    } else {
        PathBuf::from("./splatforge.lic")
    }
}

/// Sidecar file that records the last successful `/v1/license/refresh`.
/// Lives next to the license itself so a user copying the .lic between
/// boxes also copies their grace clock (the operator can `rm` it to
/// force a strict re-validation).
fn last_refresh_path(license_path: &Path) -> PathBuf {
    license_path.with_extension("last_refresh")
}

fn read_last_refresh(license_path: &Path) -> Option<DateTime<Utc>> {
    let raw = std::fs::read_to_string(last_refresh_path(license_path)).ok()?;
    DateTime::parse_from_rfc3339(raw.trim())
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

fn write_last_refresh(license_path: &Path, when: DateTime<Utc>) -> Result<()> {
    let path = last_refresh_path(license_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, when.to_rfc3339())?;
    Ok(())
}

/// `splatforge license install <path>` — verifies signature, copies file
/// into place. Refuses to overwrite an existing license with an invalid
/// one (defensive: avoids a bad refresh wedging a working install).
pub fn cmd_license_install(src: &Path) -> Result<()> {
    let lic = License::read_from_path(src)
        .with_context(|| format!("reading license at {}", src.display()))?;
    let cfg = LicenseConfig::default();
    // Strict verification — install should never accept a stale license.
    lic.verify_signature(&cfg.verifying_key)
        .map_err(|e| anyhow!("license at {} failed signature check: {e}", src.display()))?;
    let dst = default_license_path();
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    lic.write_to_path(&dst)?;
    // A fresh install resets the grace clock — the user has just "seen"
    // the license, even if they got it offline.
    write_last_refresh(&dst, Utc::now())?;
    println!(
        "installed Pro license for org={} seats={} valid_until={} -> {}",
        lic.claims.org_id,
        lic.claims.seats,
        lic.claims.valid_until.to_rfc3339(),
        dst.display()
    );
    Ok(())
}

/// `splatforge license status` — pretty-print the active license + grace
/// state. Exits non-zero (via the caller's `Result<()>`) if validation
/// fails so a customer can wire this into cron / healthchecks.
pub fn cmd_license_status(license_override: Option<&Path>) -> Result<()> {
    let path = license_override
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_license_path);
    let lic = License::read_from_path(&path).with_context(|| {
        format!(
            "no license at {} — run `splatforge license install <file>`",
            path.display()
        )
    })?;
    let cfg = LicenseConfig::default();
    let last = read_last_refresh(&path);
    let now = Utc::now();
    println!("path:         {}", path.display());
    println!("org_id:       {}", lic.claims.org_id);
    println!("plan:         {}", lic.claims.plan);
    println!("seats:        {}", lic.claims.seats);
    println!("issued_at:    {}", lic.claims.issued_at.to_rfc3339());
    println!("valid_until:  {}", lic.claims.valid_until.to_rfc3339());
    if let Some(t) = last {
        let remaining = cfg.grace - (now - t);
        println!(
            "last_refresh: {} (grace remaining: {})",
            t.to_rfc3339(),
            humanize(remaining)
        );
    } else {
        println!("last_refresh: never (strict mode — no offline grace)");
    }
    match cfg.validate(&lic, now, last) {
        Ok(()) => {
            println!("status:       OK");
            Ok(())
        }
        Err(e) => {
            println!("status:       INVALID ({e})");
            bail!("license is not valid: {e}")
        }
    }
}

fn humanize(d: Duration) -> String {
    if d <= Duration::zero() {
        return "expired".to_string();
    }
    let total_secs = d.num_seconds();
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    format!("{days}d {hours}h")
}

#[derive(Serialize)]
struct RefreshRequest<'a> {
    license: &'a License,
}

#[derive(Deserialize)]
struct RefreshResponse {
    license: License,
}

/// `splatforge license refresh` — POST the current license to the API,
/// expect a freshly-signed one in return. The API decides whether to
/// extend `valid_until` based on the customer's Stripe state; on success
/// the new license replaces the on-disk file and `last_refresh` is bumped
/// so the offline-grace clock resets.
pub fn cmd_license_refresh(license_override: Option<&Path>, api_base: &str) -> Result<()> {
    let path = license_override
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_license_path);
    let lic = License::read_from_path(&path)
        .with_context(|| format!("reading license at {}", path.display()))?;

    let url = format!("{}/v1/license/refresh", api_base.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(StdDuration::from_secs(15))
        .build()?;
    let resp = client
        .post(&url)
        .json(&RefreshRequest { license: &lic })
        .send()
        .with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let code = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("refresh failed: HTTP {code}: {body}");
    }
    let parsed: RefreshResponse = resp.json().context("parsing refresh response")?;

    let cfg = LicenseConfig::default();
    cfg.validate(&parsed.license, Utc::now(), None)
        .map_err(|e| anyhow!("API returned an invalid license: {e}"))?;
    parsed.license.write_to_path(&path)?;
    write_last_refresh(&path, Utc::now())?;
    println!(
        "refreshed: org={} new valid_until={}",
        parsed.license.claims.org_id,
        parsed.license.claims.valid_until.to_rfc3339()
    );
    Ok(())
}

/// `splatforge serve` — on-prem optimize service.
///
/// We deliberately keep this serve loop tiny here: it verifies the
/// license, fires off the telemetry beacon, and binds a placeholder
/// HTTP listener that returns 200 on `/healthz` and 503 on everything
/// else. The actual job-handling routes land on the `feat/cli-serve`
/// branch and merge in cleanly because both paths route through the
/// same license gate.
///
/// The license check happens **once at boot**. A long-running deployment
/// will re-validate on every heartbeat tick (every 1 h) so a customer who
/// stops paying does not get unbounded service.
pub fn cmd_serve(
    bind: &str,
    license_override: Option<&Path>,
    api_base: &str,
    active_seats: u32,
) -> Result<()> {
    let path = license_override
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_license_path);
    let lic = License::read_from_path(&path).with_context(|| {
        format!(
            "no license at {} — run `splatforge license install <file>` before `serve`",
            path.display()
        )
    })?;
    let cfg = LicenseConfig::default();
    let last = read_last_refresh(&path);
    cfg.validate(&lic, Utc::now(), last)
        .map_err(|e| anyhow!("refusing to start: {e}"))?;
    if active_seats > lic.claims.seats {
        bail!(
            "license allows {} seats but --active-seats={active_seats}",
            lic.claims.seats
        );
    }

    let telemetry_enabled = std::env::var("SPLATFORGE_NO_TELEMETRY")
        .map(|v| v != "1")
        .unwrap_or(true);
    eprintln!(
        "splatforge serve: org={} seats={} valid_until={} telemetry={}",
        lic.claims.org_id,
        lic.claims.seats,
        lic.claims.valid_until.to_rfc3339(),
        if telemetry_enabled { "on" } else { "off" }
    );

    if telemetry_enabled {
        spawn_heartbeat_thread(
            lic.claims.org_id.clone(),
            active_seats,
            api_base.to_string(),
        );
    }

    let server = tiny_http::Server::http(bind)
        .map_err(|e| anyhow!("bind {bind}: {e}"))?;
    eprintln!("splatforge serve: listening on http://{bind}");
    for req in server.incoming_requests() {
        let url = req.url().to_string();
        let body = if url == "/healthz" {
            tiny_http::Response::from_string("ok")
        } else if url == "/v1/license" {
            // Lets a customer's monitoring scrape the active license
            // metadata without re-reading the file on disk.
            let body = serde_json::json!({
                "org_id": lic.claims.org_id,
                "plan": lic.claims.plan,
                "seats": lic.claims.seats,
                "valid_until": lic.claims.valid_until,
            });
            tiny_http::Response::from_string(body.to_string())
        } else {
            tiny_http::Response::from_string("splatforge serve: handler not wired on this branch")
                .with_status_code(503)
        };
        let _ = req.respond(body);
    }
    Ok(())
}

#[derive(Serialize)]
struct HeartbeatPayload<'a> {
    org_id: &'a str,
    active_seats: u32,
    version: &'a str,
}

/// Heartbeat loop. Best-effort: a failed beacon is logged but does not
/// crash the server (network blips shouldn't kill a customer's box).
/// Re-validates the license each hour so a revoked-but-cached license
/// stops working within the grace window.
fn spawn_heartbeat_thread(org_id: String, active_seats: u32, api_base: String) {
    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .timeout(StdDuration::from_secs(10))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("heartbeat: failed to build client: {e}");
                return;
            }
        };
        let url = format!("{}/v1/license/heartbeat", api_base.trim_end_matches('/'));
        let version = env!("CARGO_PKG_VERSION");
        loop {
            let body = HeartbeatPayload {
                org_id: &org_id,
                active_seats,
                version,
            };
            match client.post(&url).json(&body).send() {
                Ok(r) if r.status().is_success() => {}
                Ok(r) => eprintln!("heartbeat: HTTP {}", r.status()),
                Err(e) => eprintln!("heartbeat: {e}"),
            }
            std::thread::sleep(StdDuration::from_secs(60 * 60));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_handles_negative() {
        assert_eq!(humanize(Duration::seconds(-1)), "expired");
    }

    #[test]
    fn humanize_round_numbers() {
        assert_eq!(humanize(Duration::days(3) + Duration::hours(4)), "3d 4h");
    }

    #[test]
    fn default_path_under_dot_splatforge() {
        let p = default_license_path();
        assert!(p.ends_with("license.lic"));
        assert!(p.to_string_lossy().contains(".splatforge"));
    }
}

