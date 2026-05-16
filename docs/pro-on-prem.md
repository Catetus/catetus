# SplatForge Pro — On-Prem License & `splatforge serve`

Pro is the enterprise tier: customers run the optimize pipeline inside
their own VPC, pay per seat, and never send splats to splatforge.dev.
This doc covers the operator contract — how to install a license, run
`splatforge serve`, refresh, and opt out of telemetry.

## License file (`splatforge.lic`)

Plain UTF-8 JSON, Ed25519-signed. The signature is over the canonical
serialization of every field except `signature` — re-encode through
the same struct (`splatforge_license::Claims`) and the signature
remains stable across whitespace / key-order differences.

```json
{
  "org_id": "acme-corp",
  "plan": "pro",
  "seats": 25,
  "valid_until": "2027-01-01T00:00:00Z",
  "issued_at": "2026-05-15T12:00:00Z",
  "signature": "<base64 Ed25519 signature>"
}
```

The public key lives inside every `splatforge` binary; the private
key only exists on the SplatForge API box (`LICENSE_PRIVATE_KEY` Fly
secret). Rotating the trust root means bumping
`EMBEDDED_PUBLIC_KEY` in `crates/splatforge-license/src/lib.rs` and
re-cutting every shipped binary — deliberately high-friction for v1.

## CLI

```sh
# Sales hands you splatforge.lic over a secure channel.
splatforge license install ./splatforge.lic
splatforge license status        # exits 0 if valid (with or without grace)
splatforge license refresh       # pulls a fresh signature from the API
splatforge serve --bind 0.0.0.0:8080 --active-seats 25
```

`splatforge license install` writes the file to
`~/.splatforge/license.lic` (override with `--license <path>` on any
license subcommand). It refuses to overwrite an existing license with
one that fails signature verification — a corrupt refresh can't wedge
a working install.

## Offline grace

`splatforge serve` accepts an expired license for up to **7 days**
after `valid_until` *as long as* the last successful
`splatforge license refresh` happened within that window. Persist
ence lives in `~/.splatforge/license.last_refresh` (an RFC 3339
timestamp). Delete that file to force strict re-validation.

## Telemetry beacon

`splatforge serve` heartbeats `POST /v1/license/heartbeat` every hour
with `{org_id, active_seats, version}`. The beacon is used for
license enforcement (an org that stops beaconing won't get a refresh)
and as a churn signal.

To opt out, set:

```sh
export SPLATFORGE_NO_TELEMETRY=1
```

When opted out, the beacon loop is never started. Note that opting
out **does not** extend the offline-grace window — a customer who
both opts out and lets their license expire will still see `serve`
refuse to start after 7 days.

## API endpoints

| Endpoint                       | Auth                                           | Purpose                                       |
| ------------------------------ | ---------------------------------------------- | --------------------------------------------- |
| `POST /v1/license/issue`       | `Authorization: Bearer $LICENSE_ADMIN_TOKEN`   | Admin-only — mint a fresh license             |
| `POST /v1/license/refresh`     | Inbound license's Ed25519 signature            | Customer-facing — extend `valid_until`        |
| `POST /v1/license/heartbeat`   | None (best-effort telemetry)                   | Record seat usage / version                   |

Set `LICENSE_PRIVATE_KEY` (PKCS#8 PEM or 64-char hex seed) and
`LICENSE_ADMIN_TOKEN` on the API box to enable the issuer. Without
`LICENSE_PRIVATE_KEY` the endpoints return 503; without
`LICENSE_ADMIN_TOKEN` the `issue` route is disabled (still 503).
This means a CI mirror with no secrets can keep building the API
binary unchanged.

## Tests

```sh
cargo test -p splatforge-license -p splatforge-api -p splatforge-cli
```

Round-trip sign/verify, expired-license rejection, offline-grace
bypass within 7 days, invalid-signature rejection, and tampered-claims
rejection all live in `crates/splatforge-license/src/lib.rs`.
End-to-end API issuer → CLI verifier round-trip lives in
`apps/api/tests/license.rs`.
