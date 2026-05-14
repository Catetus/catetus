# `splatforge-api`

Hosted optimize endpoint. Axum HTTP service that issues presigned upload URLs,
enqueues optimize jobs against the Modal worker (`apps/worker`), and serves job
status + result downloads.

The actual splat work happens in the Python worker — this crate stays
HTTP-light so it can run on any standard PaaS (or Modal's `web_endpoint`)
without rewriting handlers.

## Quick start (no external services)

`splatforge-api` runs against in-memory state and stub backends when the
storage and worker env vars are absent, so you can hit the routes from a fresh
checkout without provisioning anything:

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

| Method | Path                     | Purpose                                                |
| ------ | ------------------------ | ------------------------------------------------------ |
| GET    | `/healthz`               | liveness probe + version                               |
| POST   | `/v1/jobs`               | create job + receive presigned upload URL              |
| GET    | `/v1/jobs/:id`           | poll job status; result/error are populated on finish  |
| POST   | `/v1/jobs/:id/upload`    | confirm the upload completed; enqueues the optimize    |

The presign URL is returned from `POST /v1/jobs`; the client uploads the splat
to it directly (`PUT`), then calls `POST /v1/jobs/:id/upload` so the API
enqueues the job against the Modal worker. The worker's status webhook (TBD,
SPEC-0014) updates the job to `succeeded` / `failed`.

## Configuration

All env vars are optional in dev (the service degrades to stub backends), but
required in production:

| Variable                  | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `SPLATFORGE_API_BIND`     | bind address; default `0.0.0.0:8080`                       |
| `SPLATFORGE_MODAL_URL`    | Modal worker `/enqueue` endpoint                           |
| `BLOB_READ_WRITE_TOKEN`   | Vercel Blob write token; presigning calls require it       |
| `RUST_LOG`                | tracing filter; default `splatforge_api=info,tower_http=info` |

## Architecture

```
            ┌───────────────┐  presign       ┌─────────────────┐
client ────►│ splatforge-api│ ──────────────►│ Vercel Blob     │
            │  (Axum, this) │                │  (uploads)      │
            └──────┬────────┘                └─────────────────┘
                   │ /enqueue                       ▲
                   ▼                                │ pull
            ┌─────────────────┐                     │
            │ Modal worker    │ ────────────────────┘
            │ (apps/worker)   │  pulls splat → runs splatforge CLI
            └─────────────────┘
```

The job store is in-memory today (`store::JobStore`) — swapped for Postgres
when we outgrow the single-instance deploy. The blob backend (`store::BlobBackend`)
and modal client (`modal_client::ModalClient`) are trait-shaped so R2/S3 +
self-hosted worker swaps don't touch the handler code.

## Tests

```bash
cargo test -p splatforge-api --release
```

No integration tests yet — the surface is small enough that the smoke-curl
sequence above is the canonical correctness check. The trait surface in
`store.rs` / `modal_client.rs` is unit-testable; tests land alongside the
first non-stub backend implementation.

## Deployment

Two target environments today:

1. **DigitalOcean droplet** at `splatforge-api.fly.dev` — preferred. Cross-compile for
   `x86_64-unknown-linux-gnu`, install as a systemd unit, front with Caddy for
   auto-TLS on `api.splatforge.dev`. Not yet wired up; SSH access pending.

2. **Modal `web_endpoint`** — requires a thin Python wrapper around the Rust
   binary. Less clean but lives next to the worker.

Either way, set the env vars above and ensure the Vercel Blob token has
`READ_WRITE` scope on the `splatforge-blobs` store.

See [`docs/architecture.md`](../../docs/architecture.md) for how this fits
into the broader hosted pipeline.
