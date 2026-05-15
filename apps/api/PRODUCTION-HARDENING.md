# Production hardening — `splatforge-api`

What's in place, what's not, and the env vars to tune at runtime.
This is the operator's reference, not a roadmap doc. Pair with
`BILLING.md` (Stripe metering) and `README.md` (general operations).

## In place

### 1. OpenAPI 3.1 spec
- File: [`openapi.yaml`](./openapi.yaml). Hand-written.
- Served by the binary itself at `GET /openapi.yaml`. Baked into
  the binary at build time via `include_str!` — there is no sidecar
  file to mount or CDN dependency.
- Swagger UI rendered at `GET /docs` (single static HTML, pulls
  `swagger-ui-dist@5.17.14` from unpkg). Pinned version to avoid
  silent UI drift.

### 2. Token-bucket rate limiter (per API key, per route class)
- Module: [`src/ratelimit.rs`](./src/ratelimit.rs).
- Storage: in-memory `Mutex<HashMap<(key, route_class), BucketState>>`.
  Single-process; survives until the Fly machine restarts.
- Identity: per-bearer-token, NOT per-IP. Fly puts us behind their
  proxy; `X-Forwarded-For` is trivially spoofable.
- Default caps (per hour):

  | Route class                 | Free   | Paid   |
  | --------------------------- | ------ | ------ |
  | `POST /v1/jobs`             | 60     | 600    |
  | `POST /v1/jobs/batch`       | (403)  | 6      |
  | `POST /v1/jobs/:id/upload`  | 10     | 100    |
  | `POST /v1/jobs/:id/repack`  | 5      | 5      |
  | `GET  /v1/jobs/:id`         | 600    | 600    |

  Free callers hitting `/v1/jobs/batch` get a structured `403` with
  `error: "batch endpoint requires a paid-tier API key"`, audited
  with `error: free-tier-batch-forbidden`.

- On 429 the response carries `Retry-After: <seconds>` and
  `X-RateLimit-Remaining: 0`. Allowed responses also surface
  `X-RateLimit-Remaining` so clients can adapt before they hit the gate.
- Clock-injectable for tests (`Limiter::with_clock`).

### 3. Audit log
- Migration: [`migrations/0003_audit_events.sql`](./migrations/0003_audit_events.sql).
- Module: [`src/audit.rs`](./src/audit.rs).
- Mutating /v1 routes only — read-only `GET` is not audited (volume
  would dwarf the useful rows).
- Per-row data:
  - `key_prefix` (first 8 chars of the bearer token, NEVER the full token)
  - templated `route` (`/v1/jobs/:id/upload`, not the concrete uuid)
  - `method`, HTTP `status`, `body_size`, `duration_ms`
  - optional short `error` string for the operator
  - `created_at`
- Best-effort: a DB-insert failure is logged at `WARN` and otherwise
  swallowed. It MUST NOT propagate to the user response — that's a
  hard constraint from the spec.
- Read endpoint: `GET /v1/admin/audit?limit=N`. Requires a bearer key
  from `SPLATFORGE_ADMIN_API_KEYS` (separate set from `SPLATFORGE_API_KEYS`).
  Default and max `N` is 1000.

### 4. Tests
- `apps/api/tests/ratelimit.rs` — burst, refill, free vs paid caps,
  per-class isolation, env tunability, key masking.
- `apps/api/tests/audit.rs` — write/query roundtrip, key-prefix masking,
  templated routes, mutating-only filter, 1000-row admin cap.
- `cargo test -p splatforge-api` — 59 tests across lib unit tests +
  3 integration suites. All passing.

## What's NOT in place

- **SOC2 Type-I**: 12-month target per the v2 plan Q2/Q3. The audit
  log is the start of the "tamper-evident operations log" control,
  but we don't have continuous control monitoring, no vendor risk
  reviews, no formal incident-response runbook, no change-management
  process review. The audit log here is operator-readable, not
  cryptographically signed.
- **PII handling**: jobs carry user-supplied `label` strings and
  `webhook_url` values that may contain personally identifying or
  credential-bearing data. We do not redact these in the DB. The
  audit log explicitly avoids logging request bodies for this reason.
- **Multi-instance rate limiting**: see "Operational risks" below.
- **Geo / IP-level rate limiting**: out of scope. The bearer-key gate
  is the identity boundary. An attacker without a key can only hit
  `/healthz` (rate-limited only by Fly's edge), the Stripe webhook
  (HMAC-gated), and the worker callback (per-job uuid).
- **Audit log tamper protection**: rows can be deleted by anyone with
  write access to the SQLite file (i.e. anyone with shell access to
  the Fly machine). For SOC2 we'd ship rows to an append-only
  external store (CloudWatch Logs, S3 with object lock, etc.).
- **PCI scope**: no card data ever touches this API. Stripe Checkout
  + the Billing Customer Portal handle the payment surface end-to-end.
  See `BILLING.md`.

## Env vars to tune at runtime

| Variable                       | Default                  | Purpose                                                                  |
| ------------------------------ | ------------------------ | ------------------------------------------------------------------------ |
| `SPLATFORGE_RATE_LIMITS`       | (defaults above)         | Comma-separated `name=N` per-hour caps; unknown names ignored.           |
| `SPLATFORGE_ADMIN_API_KEYS`    | (empty → /audit is 401)  | Comma-separated bearer tokens for `/v1/admin/audit`.                     |
| `SPLATFORGE_API_KEYS`          | (empty → no auth, dev)   | Comma-separated tokens for `/v1/jobs*`.                                  |
| `SPLATFORGE_PAID_API_KEYS`     | (empty → free can repack)| Subset of API_KEYS additionally allowed on `/repack` and `/batch`.       |
| `DATABASE_URL`                 | `sqlite://data/jobs.db`  | SQLite path. Audit table is in the same DB.                              |
| `STRIPE_WEBHOOK_SECRET`        | (empty → webhook = 401)  | HMAC key for `/v1/stripe/webhook`. See `BILLING.md`.                     |

### Tuning examples

Lower the `/repack` cap during an A100 capacity crunch:

```bash
fly secrets set SPLATFORGE_RATE_LIMITS=repack=2
fly deploy   # required for new env to take effect — see below
```

Bump paid uploads from 100/h to 500/h for a specific customer rollout:

```bash
fly secrets set SPLATFORGE_RATE_LIMITS=upload_paid=500
fly deploy
```

Disable the admin endpoint (e.g. during a key-rotation window):

```bash
fly secrets unset SPLATFORGE_ADMIN_API_KEYS
fly deploy
```

> Note: env-var changes on Fly require a deploy to take effect because
> we read them once at boot. The audit + limiter state is recreated on
> every deploy.

## Operational risks (in priority order)

### 1. Single-process rate-limit state, machine restart resets buckets
Fly's `auto_stop_machines=stop` config (see `fly.toml`) will stop
the machine after idle. On the next request, the machine restarts
fresh and every bucket is re-allocated at full capacity. A
determined caller can intentionally drive an idle stop (one bot
hitting `/healthz` every five minutes won't keep us up) and then
burn through 60 free `/v1/jobs` calls per machine-cycle.

**Mitigation today**: keep `auto_stop_machines = off` while this
limiter is in place, OR accept that the per-hour cap is best-case
"60 per hour per machine instance". Document on the dashboard.

**Migration path**: swap the `Mutex<HashMap>` for a Redis-backed
implementation behind the same `Limiter` interface. Redis's
`INCR` + `EXPIRE` is a 2-line token bucket; Upstash Redis on
Fly is `fly redis create` and a `REDIS_URL` env var. Changes
are confined to `src/ratelimit.rs` — none of the middleware
needs to know.

### 2. Audit log on the same SQLite as jobs
A misbehaving migration could lock the jobs table and incidentally
freeze audit writes. Today both tables share a single connection
pool. Best-effort writes mean a frozen audit table won't fail user
requests, but we lose forensic data during the incident.

**Mitigation**: monitor `audit_events` row growth. If we see a
gap, cross-check against Fly's request logs.

### 3. Audit log unbounded growth
1000 free jobs/day × 7 mutating routes × audit row ≈ 7k rows/day.
At ~200 bytes/row that's 1.4 MB/day, 500 MB/year. Fine until we
6× growth. No automated rotation today.

**Mitigation**: cron a quarterly `DELETE FROM audit_events WHERE
created_at < date('now', '-90 days')` and re-VACUUM. Add a
proper rotation job once we onboard a second design partner.

### 4. No per-tenant isolation for the SQLite file
A SQL injection in any handler exposes every tenant's job rows.
We don't use string concatenation in queries (sqlx prepared
statements only), but the surface should be re-audited every
quarter.

**Mitigation**: `sqlx::query!` macro everywhere. The store
module already uses parameterised queries; future handlers must
keep that pattern.

## What changed in this release

- Migration `0003_audit_events.sql` (new table; no schema change to
  existing tables).
- New modules: `audit.rs`, `ratelimit.rs`.
- New routes: `GET /openapi.yaml`, `GET /openapi.json` (406 by
  design — points caller at YAML), `GET /docs`, `GET /v1/admin/audit`.
- New middleware: rate-limit + audit wrapper on every mutating /v1
  route. Order: auth → rate-limit/audit → handler.
- New env vars: `SPLATFORGE_RATE_LIMITS`, `SPLATFORGE_ADMIN_API_KEYS`.

No breaking changes to existing /v1 payloads. Existing clients keep
working; they only see 429s when they exceed the documented caps.
