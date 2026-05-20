# ParticleField3DGaussianSplat Conformance Report

Catetus ships a self-contained conformance suite for the OpenUSD 26.03
`ParticleField3DGaussianSplat` schema. This document is the canonical
report Pixar / Apple / the AOUSD core spec WG opens when reviewing the
submission.

## Suite at a glance

- **23 normative clauses** evaluated per asset (≥ the 15-clause floor the
  reviewing engineer asked for).
- **8 fixture files** under `fixtures/`, generated deterministically from
  Rust — re-running the generator produces byte-identical output.
- **Pure Rust**, no Pixar/OpenUSD library dependency.
- The validator is also wired as a Cargo integration test, so
  `cargo test -p catetus-usd-conformance` is the canonical pass/fail gate.

## How to run

```bash
# Validate a single asset:
cargo run -p catetus-usd-conformance --bin catetus-usd-validate -- \
    crates/catetus-usd-conformance/fixtures/01_valid_minimal.usda

# Machine-readable JSON form:
cargo run -p catetus-usd-conformance --bin catetus-usd-validate -- \
    crates/catetus-usd-conformance/fixtures/01_valid_minimal.usda --json

# Regenerate fixtures:
crates/catetus-usd-conformance/scripts/generate-fixtures.sh

# Run the full conformance test matrix:
cargo test -p catetus-usd-conformance
```

Exit codes from the CLI:

| Exit code | Meaning                                              |
|-----------|------------------------------------------------------|
| 0         | All clauses passed (skips allowed).                  |
| 1         | At least one clause failed.                          |
| 2         | Validator-level error (file unreadable, bad USDC, …).|

## Clauses

| ID                          | Required | Description                                                                                                                                       |
|-----------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `USDA_MAGIC`                | MUST     | File MUST begin with `#usda 1.0` magic line.                                                                                                      |
| `PRIM_PARTICLE_FIELD`       | MUST     | At least one `def ParticleField3DGaussianSplat` prim MUST be present.                                                                             |
| `ROOT_XFORM`                | SHOULD   | ParticleField3DGaussianSplat prim SHOULD be a descendant of a root `def Xform`.                                                                   |
| `UP_AXIS_VALID`             | MUST     | Layer metadata `upAxis`, when authored, MUST be `"Y"` or `"Z"` per UsdGeomTokens.                                                                 |
| `METERS_PER_UNIT_POSITIVE`  | MUST     | Layer metadata `metersPerUnit`, when authored, MUST be a positive number.                                                                         |
| `ATTR_POINTS`               | MUST     | ParticleField3DGaussianSplat MUST author `points` (inherited from `GeomPoints`).                                                                  |
| `ATTR_POINTS_TYPE`          | MUST     | `points` attribute MUST be typed `point3f[]`.                                                                                                     |
| `ATTR_ORIENTATIONS`         | MUST     | ParticleField3DGaussianSplat MUST author `orientations`.                                                                                          |
| `ATTR_ORIENTATIONS_TYPE`    | MUST     | `orientations` attribute MUST be typed `quatf[]`.                                                                                                 |
| `ATTR_SCALES`               | MUST     | ParticleField3DGaussianSplat MUST author `scales`.                                                                                                |
| `ATTR_SCALES_TYPE`          | MUST     | `scales` attribute MUST be typed `float3[]`.                                                                                                      |
| `ATTR_OPACITIES`            | MUST     | ParticleField3DGaussianSplat MUST author `opacities`.                                                                                             |
| `ATTR_OPACITIES_RANGE`      | MUST     | `opacities` values MUST lie in `[0, 1]` (post-sigmoid convention — see SPEC-GAPS #4).                                                             |
| `ATTR_COLORS_DC`            | MUST     | ParticleField3DGaussianSplat MUST author `colorsDC`.                                                                                              |
| `ATTR_COLORS_DC_RANGE`      | MUST     | `colorsDC` values MUST lie in `[0, 1]` per `color3f` convention.                                                                                  |
| `ATTR_WIDTHS_OPTIONAL`      | MUST     | When authored, `widths` (inherited from `GeomPoints`) MUST be typed `float[]`.                                                                    |
| `ATTR_VELOCITIES_OPTIONAL`  | MUST     | When authored, `velocities` (inherited from `GeomPoints`) MUST be typed `vector3f[]`.                                                             |
| `COUNTS_AGREE`              | MUST     | Lengths of `points`, `orientations`, `scales`, `opacities`, `colorsDC` MUST agree (one element per splat).                                        |
| `EXTENT_CONSISTENT`         | MUST     | When authored, `extent` (`2 × float3`) MUST enclose every authored point.                                                                         |
| `QUATS_NORMALIZED`          | MUST     | `orientations` quaternions MUST be unit-length within `1e-3` tolerance.                                                                           |
| `SH_COEFFS_COUNT`           | MUST     | When authored, `custom float[] catetus:shCoefficients` count MUST equal `splat_count * (degree+1)^2 * 3` for `degree ∈ {0,1,2,3}`.             |
| `DISPLAYCOLOR_INTERP`       | MUST     | When authored, `primvars:displayColor:interpolation` MUST be `"vertex"` or `"varying"` (see SPEC-GAPS #10).                                       |
| `SCHEMA_REQUIRED_ATTRS`     | MUST     | All five mandatory schema attributes (`points`, `orientations`, `scales`, `opacities`, `colorsDC`) MUST be authored on every prim.                |

## Fixture corpus

| File                                       | Container | Intent                                                                              |
|--------------------------------------------|-----------|-------------------------------------------------------------------------------------|
| `01_valid_minimal.usda`                    | USDA      | Minimal valid 1-splat prim, identity quaternion, all five required attrs.           |
| `02_valid_particle_field.usda`             | USDA      | 3-splat scene with non-identity quaternions and varied opacity/color.               |
| `03_valid_dense.usda`                      | USDA      | 64-splat 4×4×4 grid; exercises the array path at non-trivial size.                  |
| `04_valid_with_sh.usda`                    | USDA      | Adds `custom float[] catetus:shCoefficients` (degree 3); `SH_COEFFS_COUNT` PASS. |
| `05_valid_minimal.usdc`                    | USDC      | Binary form of fixture 01; validates the USDC reader path end-to-end.               |
| `06_invalid_no_orientations.usda`          | USDA      | Negative: removes the `quatf[] orientations` line. `ATTR_ORIENTATIONS` → FAIL.      |
| `07_invalid_opacity_out_of_range.usda`     | USDA      | Negative: one opacity bumped to `1.5`. `ATTR_OPACITIES_RANGE` → FAIL.               |
| `08_invalid_count_mismatch.usda`           | USDA      | Negative: `opacities` length truncated by one. `COUNTS_AGREE` → FAIL.               |

## Sample output

Running the validator against `01_valid_minimal.usda`:

```
ParticleField3DGaussianSplat conformance report for crates/catetus-usd-conformance/fixtures/01_valid_minimal.usda (usda)
clause                       status detail
----------------------------------------------------------------------------
USDA_MAGIC                   PASS
PRIM_PARTICLE_FIELD          PASS
ROOT_XFORM                   PASS
UP_AXIS_VALID                PASS
METERS_PER_UNIT_POSITIVE     PASS
ATTR_POINTS                  PASS
ATTR_POINTS_TYPE             PASS
ATTR_ORIENTATIONS            PASS
ATTR_ORIENTATIONS_TYPE       PASS
ATTR_SCALES                  PASS
ATTR_SCALES_TYPE             PASS
ATTR_OPACITIES               PASS
ATTR_OPACITIES_RANGE         PASS
ATTR_COLORS_DC               PASS
ATTR_COLORS_DC_RANGE         PASS
ATTR_WIDTHS_OPTIONAL         SKIP   `widths` not authored
ATTR_VELOCITIES_OPTIONAL     SKIP   `velocities` not authored
COUNTS_AGREE                 PASS
EXTENT_CONSISTENT            SKIP   `extent` not authored
QUATS_NORMALIZED             PASS
SH_COEFFS_COUNT              SKIP   shCoefficients not authored
DISPLAYCOLOR_INTERP          SKIP   displayColor interpolation not authored
SCHEMA_REQUIRED_ATTRS        PASS

summary: 18 pass, 0 fail, 5 skip (of 23 clauses)
```

Sample *failing* output from `07_invalid_opacity_out_of_range.usda`:

```
ATTR_OPACITIES_RANGE         FAIL   opacities[0]=1.5 outside [0, 1]
…
```

## Open question for the AOUSD core spec WG

The largest spec ambiguity uncovered while authoring the suite is the
**`displayColor` interpolation convention** for per-splat color
(`SH_COEFFS_COUNT` and `DISPLAYCOLOR_INTERP` clauses). The schema is silent
on:

- whether `colorsDC` is intended to *replace* `primvars:displayColor` or
  to live alongside it,
- which `interpolation` value (`vertex` vs `varying`) viewers should
  assume when the primvar is authored without an explicit token, and
- the layout convention for higher-order spherical harmonics (see SPEC-GAPS
  #1; the validator accepts any of the four canonical band counts via
  `SH_COEFFS_COUNT`).

The validator picks the strictest defensible reading on each — see
`crates/catetus-usd/SPEC-GAPS.md` — and the choices are surfaced as
clause IDs so a future schema clarification can flip the bit without
breaking the public report contract.

## CI

A GitHub Action (`.github/workflows/usd-conformance.yml`) runs
`cargo test -p catetus-usd-conformance` on every PR. The validator
binary is also built and dry-run against the committed fixture corpus.
