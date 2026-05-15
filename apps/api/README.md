# `splatforge-api`

Hosted optimize endpoint. Axum HTTP service that issues presigned upload URLs,
enqueues optimize jobs against the Modal worker (`apps/worker`), serves job
status + result downloads, and routes paid-tier requests to the A100
differentiable-repack worker.

The actual splat work happens in the Python workers — this crate stays
HTTP-light so it can run on any standard PaaS (Fly, Render, Railway,
DigitalOcean) without rewriting handlers.

## Quick start (no external services)

`splatforge-api` runs against a local SQLite database and stub backends when
the storage and worker env vars are absent, so you can hit the routes from a
fresh checkout without provisioning anything:

```bash
cargo run -p splatforge-api --release
# In another terminal:
curl -sS http://localhost:8080/healthz
# → {"ok":true,"service":"splatforge-api","version":"0.1.0"}

curl -sS -X POST http://localhost:8080/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"preset":"web-mobile","filename":"scene.ply","size_bytes":1234}'
# → {"id":"…","status":"awaiting-upload","upload_url":"blob://stub/jobs/…","upload_method":"PUT",…}
```

Stub mode returns a `blob://stub/...` sentinel URL instead of a real presigned
HTTPS URL, and the modal-client surface short-circuits to a "no worker
configured" error so the API contract still exercises end-to-end.

## Routes

| Method | Path                       | Purpose                                                          |
| ------ | -------------------------- | ---------------------------------------------------------------- |
| GET    | `/healthz`                 | liveness probe + version                                         |
| POST   | `/v1/jobs`                 | create free-tier job (upload or URL-mode)                        |
| POST   | `/v1/jobs/batch`           | create up to 100 jobs atomically with a shared `batch_id`        |
| GET    | `/v1/jobs/:id`             | poll job status; result/error are populated on finish            |
| POST   | `/v1/jobs/:id/upload`      | stream splat bytes through to Blob (proxy-upload mode)           |
| POST   | `/v1/jobs/:id/repack`      | paid-tier: dispatch to A100 differentiable repack                |
| POST   | `/v1/jobs/:id/result`      | worker → API callback; updates terminal status & fires webhooks  |

Free-tier jobs run the deterministic CPU optimize pipeline; paid-tier
`/repack` calls dispatch to a separate Modal app that runs gsplat on an
A100 to compress against a fixed byte budget. Repack requires the job to
already be `Done` so we have a baseline render to validate against.

## Configuration

All env vars are optional in dev (the service degrades to stub backends),
but `SPLATFORGE_API_KEYS` and the Modal URLs are required in production:

| Variable                       | Purpose                                                                  |
| ------------------------------ | ------------------------------------------------------------------------ |
| `SPLATFORGE_API_BIND`          | bind address; default `0.0.0.0:8080`                                     |
| `SPLATFORGE_PUBLIC_BASE_URL`   | publicly reachable base URL; baked into worker callback URLs             |
| `DATABASE_URL`                 | SQLite path; default `sqlite://data/jobs.db`                             |
| `SPLATFORGE_API_KEYS`          | comma-separated bearer tokens for free-tier routes                        |
| `SPLATFORGE_PAID_API_KEYS`     | bearer tokens additionally accepted on `/repack` (must subset API_KEYS)  |
| `SPLATFORGE_MODAL_URL`         | Modal free-tier `/enqueue` endpoint                                      |
| `SPLATFORGE_MODAL_REPACK_URL`  | Modal A100 `/enqueue` endpoint for `/repack`                             |
| `BLOB_READ_WRITE_TOKEN`        | Vercel Blob write token; presigning calls require it                     |
| `RUST_LOG`                     | tracing filter; default `splatforge_api=info,tower_http=info`            |

## Architecture

```
            ┌───────────────┐   /enqueue    ┌─────────────────┐
client ────►│ splatforge-api│ ─────────────►│ Modal free-tier │
            │  (Axum, this) │               │ (apps/worker)   │
            └──────┬────────┘               └────────┬────────┘
                   │ /enqueue                         │ callback
                   ▼                                  ▼
            ┌─────────────────┐    /repack    ┌──────────────────┐
            │ Modal A100      │ ◄─────────────┤ /v1/jobs/:id/result
            │ (diff-repack)   │               └──────────────────┘
            └─────────────────┘
                                              ┌─────────────────┐
                                              │ SQLite jobs.db  │
                                              │ (persistent)    │
                                              └─────────────────┘
```

Jobs persist to SQLite via sqlx; the surface in `store.rs` is the only
place that depends on the concrete backend so swapping to Postgres later
is a one-file change. The blob backend (`store::BlobBackend`) and modal
client (`modal_client::ModalClient`) are trait-shaped so R2/S3 +
self-hosted worker swaps don't touch the handler code.

## Tests

```bash
cargo test -p splatforge-api
```

In-memory SQLite is used for unit tests; the same `migrations/` directory
is replayed so the schema under test matches production.

## Deployment (Fly.io)

```bash
fly launch --no-deploy --config apps/api/fly.toml
fly volumes create splatforge_api_data --size 10 --region iad
fly secrets set \
  SPLATFORGE_API_KEYS=key1,key2 \
  SPLATFORGE_PAID_API_KEYS=paidkey1 \
  SPLATFORGE_MODAL_URL=https://...modal.run/enqueue \
  SPLATFORGE_MODAL_REPACK_URL=https://...modal.run/enqueue \
  BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...
fly deploy
```

The Dockerfile expects the workspace context (path deps), so build from
the repo root or use `--config apps/api/fly.toml --dockerfile apps/api/Dockerfile`.

## Smoke test

```bash
API_URL=https://splatforge-api.fly.dev API_KEY=... ./scripts/smoke.sh
```

Creates a URL-mode job against a known-public splat, polls until terminal,
fetches the resulting `.glb` and validates the GLB magic. Pass `PAID=1
PAID_API_KEY=...` to also exercise `/repack`. Exits non-zero on any failure.

See [`docs/architecture.md`](../../docs/architecture.md) for how this fits
into the broader hosted pipeline.
