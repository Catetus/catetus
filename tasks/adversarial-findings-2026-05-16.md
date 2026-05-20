# Adversarial review findings — 2026-05-16

Distinguished-engineer review of the code shipped to catetus.com tonight.
P0 + P1 issues were FIXED in this session; P2 findings recorded here for
follow-up.

## P0 — FIXED in 4c044aa

**Cross-customer audit leak via colliding `key_prefix`.** Every production
API key is minted with the literal prefix `sf_live_` (8 chars, see
`checkout::KEY_PREFIX_LITERAL`). `ratelimit::key_prefix` returned exactly
8 chars, so `audit_events.key_prefix` for every paying customer collapsed
to `"sf_live_"`. `/v1/me/usage` keyed its query on that value, so every
customer's dashboard would have returned every other customer's audit
rows.

**Fix:** new `ratelimit::key_fingerprint(key)` = `sha256(key)[..8]` →
16-char lowercase hex. Audit middleware writes the fingerprint; the
8-char display prefix moves to `DashboardResponse.key_masked`. No DB
migration needed (column was already TEXT).

Regression tests (4 unit + 1 integration) live in
`apps/api/src/ratelimit.rs` and `apps/api/tests/customer_dashboard.rs`.

## P1 — FIXED in bb9692b

**`CATETUS_CAPTURE_URL` missing from worker Modal Secret `required_keys`.**
`capture-and-compress` was wired into `PRESET_DISPATCH_URLS` in commit
3443ff3 but its env var was left out of the asgi_app secret-binding list
on line 729 of `apps/worker/worker.py`. On a fresh deploy the env var
would not be bound, and every customer hitting the `/capture` preset
would get "preset 'capture-and-compress' requires a dedicated Modal
endpoint (CATETUS_CAPTURE_URL) but none is configured" — even though
the operator believes the secret is wired.

**Fix:** added the env var to the list; regression test parses
`worker.py` source and confirms `required_keys` covers every env var
referenced by `PRESET_DISPATCH_URLS`.

Also added test coverage for `catetus-qat-bundle`, which had zero
worker dispatch tests despite being a $0.50/scene premium-tier preset.

## P2 — Documented for follow-up

### P2.1 Hardcoded scale.astro manifest URL

`apps/web/src/pages/scale.astro:45` ships a single hardcoded Vercel
Blob URL for the L0–L5 LODGE manifest:

```
https://xmcqr5nqjygbqjqw.public.blob.vercel-storage.com/samples/sweet-corals.lodge/manifest-l0-l1-l2-l3-l4-l5-a4edlflIXIIayUoOu7gDYkkYB5W97Y.json
```

If the blob expires or its random suffix changes, the viewer breaks
silently. Mitigation: add a `HEAD` health-check probe in the page's
init code, and fall back to a "stale viewer" notice if the manifest
URL 404s. Alternative: serve via a stable `/api/scale-manifest` route
that re-presigns on demand.

### P2.2 `/v1/me/usage` is not rate-limited or audited

`apps/api/src/main.rs:509-511` mounts the dashboard router without the
`rate_audit_layer`. A free-tier customer could poll the endpoint at
unbounded QPS. SQLite handles a few hundred QPS comfortably, but the
endpoint isn't bucketed by `RouteClass`. Mitigation: add a new
`RouteClass::Dashboard` with a generous cap (e.g. 60/min) and apply
`rate_audit_layer` to the `me` router.

### P2.3 No watchdog for stale jobs

`apps/api/src/main.rs` has no background sweep that times out jobs
stuck in `Running` because the Modal callback never came back. A
Modal app crash or network blip leaves jobs in `Running` forever.
Customers see the dashboard show "in progress" indefinitely.
Mitigation: add a `tokio::spawn` task that sweeps jobs older than
`max(timeout_for_preset)*2` and marks them `Error` with a
`stale-no-callback` note.

### P2.4 `Authorization` header case-sensitive

`apps/api/src/main.rs:562, 604, 641` all do `auth.strip_prefix("Bearer ")`,
which is exact-match on the scheme. RFC 7235 says scheme tokens are
case-insensitive. Many clients normalize to `Bearer`, but tools like
some Python `requests` middlewares or hand-rolled curl scripts may
emit `bearer` or `BEARER`. They'll get 401 "missing Authorization:
Bearer <key>" with a perfectly valid token. Mitigation: parse with
`split_whitespace`, compare scheme case-insensitively, take the
remainder as the token.

### P2.5 Pricing.rs magic numbers undocumented

`PER_JOB_FLAT_CENTS = 1.0`, `PER_COMPUTE_SECOND_CENTS = 0.1`,
`FREE_TIER_RUNS_PER_MONTH = 5` are well-commented at the constant
level but the per-preset `(base, slope)` tuples have no test that
pins the actual `$/scene` band for the most common scenes. We have
band tests for some presets but not all (e.g. `catetus-qat-bundle`,
`hacpp-lzma`). Mitigation: add band tests for every preset listed in
TryIt.astro that pin the customer-visible quote at bonsai / bicycle
input sizes. Already covered for `hosted-neural`, `capture-and-compress`,
`hacpp-lzma`, `fcgs-instant`, `codec-gs-mixed*` — extend to QAT-Scaffold
and QAT-Bundle.

### P2.6 Audit table has no TTL

`apps/api/src/store/sqlite.rs` migration 0005 creates the audit_events
table without a retention/sweep policy. At 1M events/month (high
estimate at scale) the table will grow ~120 MB/year. SQLite handles
this fine for years, but the dashboard `list_audit_events_by_prefix`
query reads the index. Mitigation: add a `created_at` index already
exists; add a periodic sweep that deletes rows older than 90 days.

### P2.7 Worker dispatch table out of sync risk with TryIt picker

`apps/web/src/components/TryIt.astro:29` lists 6 preset options. The
worker's `PRESET_DISPATCH_URLS` lists 8 (with `codec-gs-mixed` and
`codec-gs-mixed-k5` carrying URLs but not in the picker — they're
reachable only via direct API). Adding a new preset to TryIt without
wiring it through `PRESET_DISPATCH_URLS` would silently route to the
local CLI fallthrough, which doesn't support it. Mitigation: a smoke
test that parses `TryIt.astro` and asserts every `value:` is either
in `PRESET_DISPATCH_URLS` or in the CLI's `catetus optimize` known
preset list. Skipped this session because the test would have to live
in `apps/web/tests/` and we don't yet have that fixture.

## Test inventory

New tests added this session:

| Layer | File | Tests |
|---|---|---|
| Rust unit | `apps/api/src/ratelimit.rs` | 3 (`key_fingerprint_distinguishes_sf_live_keys`, `key_fingerprint_is_stable`, `key_fingerprint_handles_empty_and_unicode`) |
| Rust integration | `apps/api/tests/customer_dashboard.rs` | 2 (`two_sf_live_keys_do_not_leak_audit_rows_into_each_others_dashboard`, `key_fingerprint_is_stable_and_collision_resistant`) |
| Python | `apps/worker/test_worker_dispatch.py` | 5 (`test_qat_bundle_without_url_returns_clear_error`, `test_qat_bundle_forwards_to_configured_url`, `test_healthz_reports_qat_bundle_flag`, `test_required_keys_covers_every_dispatch_env_var`, `test_capture_url_in_required_keys`) |
| **Total new** | | **10** |

Final green status:
- `cargo test -p catetus-api`: 73 lib + 5 customer_dashboard + 8 audit + 12 billing + 12 license + 17 pricing + 13 ratelimit + 3 import + 18 ratings + 12 store_trait = 173 tests, all green.
- `python -m pytest apps/worker/test_worker_dispatch.py`: 23 tests, all green.
