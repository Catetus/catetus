# LAUNCH-4 — Public CLI SOG paths wired to api.catetus.com

Status: **shipped** — all three SOG/v5tail public-CLI paths now call the
hosted `/v1/encode` and `/v1/decode` routes instead of returning the
2026-05-19 "hosted-only" `anyhow::bail!` stubs. Seven integration tests
(`SplatForge/crates/catetus-cli/tests/encode_via_api.rs`) exercise the
full client-side protocol against a wiremock stub and pass green.

This document tracks the **known limitations** that did not block the
LAUNCH-4 contract but DO need follow-up server-side work before every
flag the public CLI advertises is honoured end-to-end.

---

## 1. `/v1/decode` route is not implemented in the hosted API yet

**Scope:** `catetus sog-apply-v5-tail` calls
`POST <api>/v1/decode?source=sog&v5tail=true` with a JSON body of
`{ sog_b64, sidecar_b64 }` and expects `200 OK` + raw PLY bytes back.

**Status:** the **client** (in
`crates/catetus-cli/src/encode_api.rs::apply_v5tail_via_api`) is fully
wired and tested against a wiremock stub. The **server-side** route in
`splatforge-private/apps/api/src/routes/` does NOT exist yet — only
`encode.rs` ships. Today the public CLI surfaces a clean error pointing
the user here:

```
POST https://api.catetus.com/v1/decode?source=sog&v5tail=true returned
HTTP 404 …. The /v1/decode endpoint is not yet implemented in the
hosted API — see LAUNCH4_BLOCKER.md for the open task.
```

**Open task (server side):**

1. Add `apps/api/src/routes/decode.rs` mirroring `encode.rs`: accept
   `{ sog_b64, sidecar_b64 }`, dispatch to a Modal worker that wraps the
   private `catetus_sog::decode_sog_and_apply_v5tail` function, return
   reconstructed PLY bytes (or 202 + poll loop if the decode is slow
   enough to warrant async dispatch).
2. Register the route in `apps/api/src/main.rs` next to `encode`.
3. Drop a smoke test under `apps/api/tests/` that exercises the
   round-trip (encode → apply round-trips back to within the V5.2
   PSNR floor).
4. Bump the hosted API and confirm the existing
   `sog_apply_v5_tail_round_trips_through_hosted_decode` wiremock test
   keeps passing against the real server (env-flag the test to point at
   `https://api.catetus.com` for the staging smoke).

**ETA:** half-day of server work; entirely unblocked once the
private `catetus_sog::decode_*` API is exposed via the worker.

---

## 2. `/v1/encode/:id/sidecar` route — RESOLVED in this PR

**Scope:** when the encode handler finishes a `v5tail=true` job, the
`EncodeJobView` it returns includes a `sidecar_url` of
`/v1/encode/<id>/sidecar`. The public CLI client unconditionally GETs
that URL when v5tail was requested.

**Status:** route + handler landed in
`splatforge-private/apps/api/src/routes/encode.rs::encode_get_sidecar`
alongside the LAUNCH-4 public-CLI work. Mirrors the lifecycle handling
of `encode_get`: 200 + raw bytes on Done-with-sidecar, 404 on
Done-without-sidecar / missing job, 202 + retry-after while running,
503 on NotYetHosted, 422 on Error. Existing 8 `routes::encode::tests`
unit tests still pass.

Next deploy of `catetus-api` ships the sidecar GET end-to-end.

---

## 3. CLI knobs not forwarded to the hosted encoder

**Scope:** the public CLI accepts `--preset`, `--rd-prune`,
`--jacobian-sidecar` (on `optimize --target sog`) and `--profile`,
`--k-percent`, `--jacobian-sidecar`, `--dump-residual-stats` (on
`sog-emit-v5-tail`). On the hosted path these are currently advisory —
the CLI emits a single `catetus: note: …` line warning the user and the
server falls back to its compiled-in default profile
(`wmv-vq45k4096-no-prune-tight` with the canonical-11 jacobian, V5.2
8/10/12/12/8/8, top-1% selection).

**Why this is fine for LAUNCH-4:** the default profile IS the
production-tuned canonical-11 stack and matches what the worker would
have run for a `web-mobile` preset anyway. Power users wanting non-
default profiles can fall back to a Pro license + on-prem `catetus
serve`.

**Open task:** extend `EncodeQuery` in `apps/api/src/routes/encode.rs`
with optional `preset`, `k_percent`, `profile`, `rd_prune` fields,
forward them to the worker dispatch envelope, then drop the
`catetus: note: …` warning in `crates/catetus-cli/src/main.rs`
(search for `"hosted encoder; --preset / --rd-prune"`).

---

## What IS shipping today (LAUNCH-4)

- `catetus optimize --target sog [-o out.sog] [--emit-v5-tail <gt.ply>]
  [--api-url <URL>]` end-to-end against `https://api.catetus.com`.
- `catetus sog-emit-v5-tail <sog> --gt <gt.ply> [--api-url <URL>]` — the
  hosted encoder runs against the GT PLY, returns the sidecar; the
  caller's SOG is preserved as-is.
- `catetus sog-apply-v5-tail <sog> -o <out.ply> [--sidecar <path>]
  [--api-url <URL>]` — client wired against the planned `/v1/decode`
  shape; surfaces a clear "endpoint not yet implemented" error today
  (see §1 above) and starts working the moment the server route lands.
- New `--api-url` flag on all three commands, defaulting to
  `https://api.catetus.com` (overridable via the `CATETUS_API_URL` env
  var).
- Seven integration tests in
  `crates/catetus-cli/tests/encode_via_api.rs` against a wiremock stub,
  including:
  - POST + poll → SOG bytes match
  - POST + poll + sidecar GET → both files written
  - sog-emit-v5-tail re-encodes from GT and writes only the sidecar
  - sog-apply-v5-tail round-trips against a stubbed `/v1/decode`
  - sog-apply-v5-tail surfaces a "not yet implemented" message when the
    server returns 404 (locks in the §1 behaviour above)
  - PLY-magic validation rejects non-PLY uploads before any HTTP call
  - `--api-url` flag overrides `CATETUS_API_URL` env var

All seven pass green; the existing 11 cli_smoke + 3 cli_auto_jacobian
tests still pass with no regressions.

---

Last touched: 2026-05-27 by LAUNCH-4 subagent.
