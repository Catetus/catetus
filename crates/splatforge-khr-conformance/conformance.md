# KHR_gaussian_splatting Conformance Report

SplatForge ships a self-contained conformance suite for the Khronos
`KHR_gaussian_splatting` glTF extension, currently in **Release
Candidate** status. This document is the canonical written report; the
live machine-generated matrix is at
[splatforge.dev/khr-conformance](https://splatforge.dev/khr-conformance).

The suite tracks the RC text at KhronosGroup/glTF commit
[`63770cc7`](https://github.com/KhronosGroup/glTF/commit/63770cc70a3709cf101a42cece0bdf602b37e2e7)
("Editorial review (#2567)", 2026-04-15). Each time Khronos lands an
editorial pass on the extension we re-audit the clause list, bump the
crate minor version, and update `CHANGELOG.md`.

## Suite at a glance

- **23 normative clauses** evaluated per asset (mapped 1:1 to RC text paragraphs).
- **10 fixture files** under `fixtures/`, generated deterministically from
  Rust — re-running the generator produces byte-identical output.
- **Pure Rust**, no Python. Single binary entry point.
- The validator is wired as a Cargo integration test, so `cargo test -p
  splatforge-khr-conformance` is the canonical pass/fail gate.

## How to run

```bash
# Validate a single asset:
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb

# Machine-readable JSON form:
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb --json

# Regenerate fixtures (byte-deterministic):
crates/splatforge-khr-conformance/scripts/generate-fixtures.sh

# Run the full conformance test matrix:
cargo test -p splatforge-khr-conformance
```

Exit codes from the CLI:

| Exit code | Meaning                                              |
|-----------|------------------------------------------------------|
| 0         | All clauses passed (skips allowed).                  |
| 1         | At least one clause failed.                          |
| 2         | Validator-level error (file unreadable, bad GLB, …). |

## Clauses

| ID                    | Description                                                                                                                              |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `EXT_USED`            | Root `extensionsUsed` array MUST list `"KHR_gaussian_splatting"`.                                                                        |
| `ASSET_VERSION`       | `asset.version` MUST be `"2.0"` per glTF 2.0.                                                                                            |
| `PRIM_EXT`            | At least one `mesh.primitives[i].extensions["KHR_gaussian_splatting"]` block MUST be present.                                            |
| `PRIM_MODE_POINTS`    | The primitive carrying `KHR_gaussian_splatting` MUST set `mode` to `POINTS` (0).                                                          |
| `EXT_KERNEL`          | The extension object MUST declare a string `kernel`.                                                                                     |
| `EXT_COLOR_SPACE`     | The extension object MUST declare a string `colorSpace`.                                                                                 |
| `EXT_PROJECTION`      | If `projection` is present it MUST be a string (defaults to `"perspective"`).                                                            |
| `EXT_SORTING`         | If `sortingMethod` is present it MUST be a string (defaults to `"cameraDistance"`).                                                       |
| `ATTR_POSITION`       | The primitive's attributes object MUST declare a `POSITION` accessor.                                                                    |
| `ATTR_ROTATION`       | Attributes MUST declare `KHR_gaussian_splatting:ROTATION`.                                                                               |
| `ATTR_SCALE`          | Attributes MUST declare `KHR_gaussian_splatting:SCALE`.                                                                                  |
| `ATTR_OPACITY`        | Attributes MUST declare `KHR_gaussian_splatting:OPACITY`.                                                                                |
| `ATTR_SH_DC`          | Attributes MUST declare `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0`.                                                                     |
| `ACC_POSITION`        | `POSITION` accessor MUST be `VEC3` (FLOAT or normalized integer under `KHR_mesh_quantization`).                                          |
| `ACC_ROTATION`        | `KHR_gaussian_splatting:ROTATION` MUST be `VEC4` (FLOAT, normalized BYTE, or normalized SHORT — unit quaternion `xyzw`).                  |
| `ACC_SCALE`           | `KHR_gaussian_splatting:SCALE` MUST be `VEC3` (FLOAT or unsigned-integer normalized variants).                                           |
| `ACC_OPACITY`         | `KHR_gaussian_splatting:OPACITY` MUST be `SCALAR` (FLOAT or normalized UByte/UShort).                                                    |
| `ACC_SH_COEF`         | Every `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` accessor MUST be `VEC3 FLOAT`.                                                         |
| `ACC_POSITION_MINMAX` | `POSITION` accessor MUST provide both `min` and `max` arrays (glTF 2.0 §3.6.2.4).                                                        |
| `SH_DEGREES_FULL`     | SH degrees MUST be fully defined — each declared degree provides all `(2l+1)` coefficients, and using degree `l` requires degrees `0..l-1`. |
| `ACC_COUNTS_AGREE`    | All per-splat accessors MUST share the same `count`.                                                                                     |
| `BUFFERVIEW_BOUNDS`   | Every referenced accessor's bufferView MUST be in range and its byte footprint MUST fit inside the parent buffer.                        |
| `ATTRS_KNOWN_ONLY`    | Any `KHR_gaussian_splatting:*` attribute key MUST match one of the spec-defined names.                                                   |

## Fixture corpus

| File                                  | Container | Intent                                                                                |
|---------------------------------------|-----------|---------------------------------------------------------------------------------------|
| `01_valid_baseline.glb`               | GLB       | Minimal valid 4-splat scene, FLOAT accessors only.                                    |
| `02_valid_baseline.gltf`              | glTF + bin| Same scene as 01 but as external `.gltf` + `buffers/chunk_0000.bin` sidecar.          |
| `03_valid_quantized.glb`              | GLB       | Integer accessors via `KHR_mesh_quantization` (normalized signed-short ROTATION, etc).|
| `04_valid_with_sh.glb`                | GLB       | Adds SH degree-1 coefficient accessors (`SH_DEGREE_1_COEF_0..2`).                     |
| `05_valid_default_methods.glb`        | GLB       | Omits optional `projection` + `sortingMethod` to exercise the RC defaults branch.     |
| `06_invalid_missing_ext_used.glb`     | GLB       | Negative: drops `KHR_gaussian_splatting` from `extensionsUsed`. `EXT_USED` → FAIL.    |
| `07_invalid_no_rotation.gltf`         | glTF      | Negative: removes `KHR_gaussian_splatting:ROTATION`. `ATTR_ROTATION` → FAIL.          |
| `08_invalid_rotation_vec3.gltf`       | glTF      | Negative: ROTATION accessor type is `VEC3`. `ACC_ROTATION` → FAIL.                    |
| `09_invalid_position_no_minmax.gltf`  | glTF      | Negative: POSITION accessor missing `min`/`max`. `ACC_POSITION_MINMAX` → FAIL.        |
| `10_invalid_count_mismatch.gltf`      | glTF      | Negative: `OPACITY.count` set to 7 instead of 4. `ACC_COUNTS_AGREE` → FAIL.           |

## Sample output

Running the validator against `01_valid_baseline.glb`:

```
KHR_gaussian_splatting conformance report for crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb (glb)
clause                   status detail
----------------------------------------------------------------------
EXT_USED                 PASS
ASSET_VERSION            PASS
PRIM_EXT                 PASS
PRIM_MODE_POINTS         PASS
EXT_KERNEL               PASS
EXT_COLOR_SPACE          PASS
EXT_PROJECTION           PASS
EXT_SORTING              PASS
ATTR_POSITION            PASS
ATTR_ROTATION            PASS
ATTR_SCALE               PASS
ATTR_OPACITY             PASS
ATTR_SH_DC               PASS
ACC_POSITION             PASS
ACC_ROTATION             PASS
ACC_SCALE                PASS
ACC_OPACITY              PASS
ACC_SH_COEF              SKIP   no SH coefficient accessors declared
ACC_POSITION_MINMAX      PASS
SH_DEGREES_FULL          PASS
ACC_COUNTS_AGREE         PASS
BUFFERVIEW_BOUNDS        PASS
ATTRS_KNOWN_ONLY         PASS

summary: 22 pass, 0 fail, 1 skip (of 23 clauses)
```

`ACC_SH_COEF` skips when no SH coefficient attributes are declared —
fixture 04 exercises it.

## Open questions for the working group

See [`(private outreach draft)`](../../(private outreach draft))
for the three normative-ambiguity questions we'd like the working group
to resolve: `COLOR_0` fallback enforcement, `colorSpace` enumeration
strictness, and the `KHR_mesh_quantization` requirement when quantized
OPACITY accessors are used.

## CI

A GitHub Action (`.github/workflows/khr-conformance.yml`) runs
`cargo test -p splatforge-khr-conformance` on every PR. The validator
binary is also built and dry-run against the committed fixture corpus.
