# SPEC-0002 — PLY Ingest

**Status:** Implemented (Phase 1)
**Crate:** `catetus-ply`

## Goal

Read common Gaussian-Splat PLY files (Inria 3DGS / INRIA-style) and convert them into `SplatIR`.

## Supported fields (v1)

| Field            | Required | Notes |
| ---------------- | -------- | ----- |
| `x, y, z`        | yes      | float32 position |
| `scale_0..2`     | yes      | log-scale per axis |
| `rot_0..3`       | yes      | quaternion (w, x, y, z) — normalized on import |
| `opacity`        | yes      | logit-space; sigmoid applied on import |
| `f_dc_0..2`      | yes      | SH DC term (RGB) |
| `f_rest_*`       | optional | higher-order SH coefficients |
| `nx, ny, nz`     | ignored  | discarded |

## Formats

* `format binary_little_endian 1.0` — primary
* `format ascii 1.0` — supported, slower path
* `format binary_big_endian 1.0` — error with clear message

## Acceptance tests

```gherkin
Feature: PLY ingest

Scenario: Parse valid binary PLY
  Given fixture "tiny/basic_binary.ply"
  When I run "catetus analyze tiny/basic_binary.ply"
  Then the command exits 0
  And the report says format is "ply"
  And splatCount is 3

Scenario: Reject PLY missing rotation
  Given fixture "invalid/missing_rotation.ply"
  When I run "catetus analyze invalid/missing_rotation.ply"
  Then the command exits non-zero
  And stderr includes "missing required rotation fields"

Scenario: Reject truncated binary
  Given fixture "invalid/truncated_binary.ply"
  When I parse it
  Then a parse error with code "truncated_payload" is returned
```
