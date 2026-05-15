# Capture-tool import endpoints

Three POST routes that resolve a share URL from a third-party capture app
(Luma, Polycam, Scaniverse) to the underlying splat asset and feed it into
the standard SplatForge optimize pipeline. The motivation: skip the
"download from app → re-upload to us" round trip that designers and field
operators were doing manually.

All three endpoints live under the same bearer-token auth as `/v1/jobs`
(set `Authorization: Bearer <key>`) and share a 10 imports/min/key
sliding-window rate limit. Hitting the limit returns `429`.

On success, the response carries the standard `job_id` you already poll
via `GET /v1/jobs/<id>` — no SDK changes required.

## `POST /v1/import/luma`

Resolves a `https://lumalabs.ai/capture/<id>` (or `/embed/<id>`) share URL
to the capture's `gaussian_splat_url`. We require `status == "complete"`
on the Luma side; still-processing captures return `415` so you can prompt
the user to retry later.

### Request

```json
{
  "share_url": "https://lumalabs.ai/capture/abc-123",
  "label": "Q3 walkthrough"   // optional, flows through to the Job
}
```

### Response (200)

```json
{
  "job_id": "0e2a3a4f-...",
  "source_url": "https://cdn-luma.com/scenes/abc-123/scene.ply",
  "provider": "luma"
}
```

### Errors

| Status | Cause                                                                  |
| ------ | ---------------------------------------------------------------------- |
| `400`  | Share URL didn't match `https://lumalabs.ai/(capture\|embed)/<id>`     |
| `400`  | Resolved asset URL is off-CDN (not under `lumalabs.ai` / `cdn-luma.com`) |
| `415`  | Capture is still processing, or isn't a Gaussian splat                 |
| `429`  | Burned through the 10/min budget                                       |
| `502`  | Luma REST returned 5xx / timed out                                     |

## `POST /v1/import/polycam`

Resolves a `https://poly.cam/capture/<id>` (or `polycam.com/capture/<id>`)
share URL to `https://glcdn.poly.cam/<id>.ply`. We issue a `HEAD` probe
first so a non-exported capture returns `415` instead of getting queued
and failing inside the worker.

### Request

```json
{
  "share_url": "https://poly.cam/capture/xyz",
  "label": null
}
```

### Response (200)

```json
{
  "job_id": "0e2a3a4f-...",
  "source_url": "https://glcdn.poly.cam/xyz.ply",
  "provider": "polycam"
}
```

### Errors

| Status | Cause                                                                  |
| ------ | ---------------------------------------------------------------------- |
| `400`  | Share URL didn't match the documented pattern                          |
| `415`  | No PLY export — user needs to hit the Polycam app's `Export → PLY`     |
| `429`  | Rate limit                                                             |
| `502`  | CDN returned 5xx                                                       |

## `POST /v1/import/scaniverse`

Scaniverse exposes USDZ by default; we don't yet ship a USDZ-without-PLY
converter in the worker. The handler probes `https://scans.scaniverse.com/<id>.ply`
— some flagship users export a sibling PLY manually — and **returns `415`**
with a clear message when no PLY exists:

> Scaniverse USDZ-without-PLY not yet supported — convert to PLY via the
> desktop app and re-share, or upload the `.ply` directly to `/v1/jobs`.

### Request

```json
{
  "share_url": "https://scaniverse.com/scan/scan-77"
}
```

### Errors

| Status | Cause                                                                  |
| ------ | ---------------------------------------------------------------------- |
| `400`  | Share URL didn't match `https://scaniverse.com/scan/<id>`              |
| `415`  | USDZ-without-PLY not supported (the common case until we ship a converter) |
| `429`  | Rate limit                                                             |

## Implementation notes

- **Security.** Every resolved asset URL is rechecked against a per-provider
  host allowlist (`*.lumalabs.ai`, `*.cdn-luma.com`, `*.amazonaws.com` for
  Luma; `*.poly.cam` / `*.polycam.com` for Polycam; `*.scaniverse.com` for
  Scaniverse) **and** against the same private-IP-literal block list that
  `/v1/jobs` URL-mode uses. A compromised provider response can't trick the
  worker into hitting `169.254.169.254` or an attacker-controlled host.
- **Rate limit.** In-memory sliding-window, keyed on the bearer token.
  Bucket resets after 60 s without traffic. The implementation lives in
  `apps/api/src/routes/import.rs::ImportRateLimiter`.
- **Tests.** `apps/api/tests/import.rs` exercises every provider arm and
  the rate limiter against `wiremock`-backed stubs — no live HTTP in CI.
- **Env overrides.** Operators can repoint any of the three providers via
  `SPLATFORGE_LUMA_API_BASE`, `SPLATFORGE_POLYCAM_CDN_BASE`,
  `SPLATFORGE_SCANIVERSE_CDN_BASE` (useful for staging or smoke tests
  against a wiremock recorder).

## Provider URL-shape reference

| Provider   | Share URL shape                                          | Resolved asset URL                                |
| ---------- | -------------------------------------------------------- | ------------------------------------------------- |
| Luma       | `https://lumalabs.ai/capture/<id>` or `/embed/<id>`      | `gaussian_splat_url` from `/api/v2/captures/<id>` |
| Polycam    | `https://poly.cam/capture/<id>` or `polycam.com/...`     | `https://glcdn.poly.cam/<id>.ply`                 |
| Scaniverse | `https://scaniverse.com/scan/<id>`                       | `https://scans.scaniverse.com/<id>.ply` (if it exists) |
