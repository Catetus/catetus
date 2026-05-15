# KHR_gaussian_splatting Conformance Report

SplatForge ships a self-contained conformance suite for the Khronos
`KHR_gaussian_splatting` glTF extension (RC, Feb 2026). This document is
the canonical report Khronos opens when reviewing the submission.

## Suite at a glance

- **28 normative clauses** evaluated per asset — the original 23 for the
  base `KHR_gaussian_splatting` extension, plus 5 added for the
  `KHR_gaussian_splatting_compression_spz` sub-extension that SplatForge
  ships as the second Khronos extension in this repository.
- **13 fixture files** under `fixtures/`, generated deterministically from
  Rust — re-running the generator produces byte-identical output.
- **Pure Rust**, no Python. Single binary entry point.
- The validator is also wired as a Cargo integration test, so `cargo test
  -p splatforge-khr-conformance` is the canonical pass/fail gate.

## How to run

```bash
# Validate a single asset:
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb

# Machine-readable JSON form:
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb --json

# Regenerate fixtures:
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

| ID                    | Required | Description                                                                                                                                          |
|-----------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EXT_USED`            | MUST     | Root `extensionsUsed` array MUST list `"KHR_gaussian_splatting"`.                                                                                    |
| `EXT_REQUIRED`        | SHOULD   | `extensionsRequired` SHOULD list `"KHR_gaussian_splatting"` when the asset cannot render without it.                                                 |
| `ASSET_VERSION`       | MUST     | `asset.version` MUST be `"2.0"` per glTF 2.0.                                                                                                        |
| `PRIM_EXT`            | MUST     | At least one `mesh.primitives[i].extensions["KHR_gaussian_splatting"]` block MUST be present.                                                        |
| `ATTRS_OBJECT`        | MUST     | The `KHR_gaussian_splatting` block on a primitive MUST contain an `attributes` object.                                                               |
| `ATTR_POSITION`       | MUST     | The attributes object MUST declare a `POSITION` accessor.                                                                                            |
| `ATTR_ROTATION`       | MUST     | The attributes object MUST declare a `_ROTATION` accessor.                                                                                           |
| `ATTR_SCALE`          | MUST     | The attributes object MUST declare a `_SCALE` accessor.                                                                                              |
| `ATTR_OPACITY`        | MUST     | The attributes object MUST declare a `_OPACITY` accessor.                                                                                            |
| `ATTR_COLOR_DC`       | MUST     | The attributes object MUST declare a `_COLOR_DC` accessor.                                                                                           |
| `ACC_POSITION`        | MUST     | `POSITION` accessor MUST be `VEC3` (componentType `FLOAT`, or normalized `UNSIGNED_SHORT` / `UNSIGNED_BYTE` under `KHR_mesh_quantization`).          |
| `ACC_ROTATION`        | MUST     | `_ROTATION` accessor MUST be `VEC4 FLOAT` (unit quaternion `xyzw`).                                                                                  |
| `ACC_SCALE`           | MUST     | `_SCALE` accessor MUST be `VEC3` (FLOAT or normalized integer).                                                                                      |
| `ACC_OPACITY`         | MUST     | `_OPACITY` accessor MUST be `SCALAR` (FLOAT or normalized integer in `[0, 1]`).                                                                      |
| `ACC_COLOR_DC`        | MUST     | `_COLOR_DC` accessor MUST be `VEC3` (FLOAT or normalized integer in `[0, 1]`).                                                                       |
| `ACC_COLOR_SH`        | MUST     | When present, `_COLOR_SH` accessor MUST be `SCALAR FLOAT` with `count = splat_count * 45`. *(See "Open question" below.)*                            |
| `ACC_POSITION_MINMAX` | MUST     | `POSITION` accessor MUST provide both `min` and `max` arrays (glTF 2.0 §3.6.2.4).                                                                    |
| `SH_DEGREE_RANGE`     | MUST     | `shDegree` MUST be an integer in `[0, 3]`; when `_COLOR_SH` is absent it MUST be `0`.                                                                |
| `ACC_COUNTS_AGREE`    | MUST     | All per-splat accessors (`POSITION`, `_ROTATION`, `_SCALE`, `_OPACITY`, `_COLOR_DC`) MUST share the same `count`.                                    |
| `BUFFERVIEW_BOUNDS`   | MUST     | Every referenced accessor's bufferView MUST be in range and its byte footprint MUST fit inside the parent buffer.                                    |
| `SPZ_DECLARED`        | MUST     | If `KHR_gaussian_splatting_compression_spz` appears anywhere in the asset it MUST be listed in `extensionsUsed`.                                     |
| `SPZ_CONSISTENT`      | MUST     | If a primitive declares `KHR_gaussian_splatting_compression_spz` it MUST also declare `KHR_gaussian_splatting` on the same primitive.                |
| `ATTRS_KNOWN_ONLY`    | MUST     | The `KHR_gaussian_splatting` attributes object MUST NOT contain unknown attribute keys (only the six reserved names are permitted).                  |
| `SPZ_EXT_PRESENT`     | MUST     | If `KHR_gaussian_splatting_compression_spz` is in `extensionsUsed`, at least one primitive MUST declare it under its own `extensions` object.        |
| `SPZ_VERSION`         | MUST     | `KHR_gaussian_splatting_compression_spz.version` MUST be `2` (current SPZ wire format).                                                              |
| `SPZ_BUFFERVIEW`      | MUST     | `KHR_gaussian_splatting_compression_spz.bufferView` MUST be an in-range index and the view MUST fit inside its buffer.                               |
| `SPZ_BLOB_MAGIC`      | MUST     | The bytes referenced by the SPZ bufferView MUST start with the SPZ magic `0x5053_4e47` (`SNPS` LE).                                                  |
| `SPZ_DECODED_COUNT`   | MUST     | The `splat_count` decoded from the SPZ header MUST match the primitive's declared `splatCount` (extension field or `_OPACITY` accessor count).       |

## Fixture corpus

| File                                  | Container | Intent                                                                                |
|---------------------------------------|-----------|---------------------------------------------------------------------------------------|
| `01_valid_baseline.glb`               | GLB       | Minimal valid 4-splat scene, FLOAT accessors only.                                    |
| `02_valid_baseline.gltf`              | glTF + bin| Same scene as 01 but as external `.gltf` + `buffers/chunk_0000.bin` sidecar.          |
| `03_valid_quantized.glb`              | GLB       | Same scene, integer accessors via `KHR_mesh_quantization`.                            |
| `04_valid_with_sh.glb`                | GLB       | Same scene plus `_COLOR_SH` (degree 3).                                               |
| `05_valid_spz_stub.glb`               | GLB       | Declares the optional `KHR_gaussian_splatting_compression_spz` sub-extension.         |
| `06_invalid_missing_ext_used.glb`     | GLB       | Negative: drops `KHR_gaussian_splatting` from `extensionsUsed`. `EXT_USED` → FAIL.    |
| `07_invalid_no_rotation.gltf`         | glTF      | Negative: removes `_ROTATION` from the attributes object. `ATTR_ROTATION` → FAIL.     |
| `08_invalid_rotation_vec3.gltf`       | glTF      | Negative: `_ROTATION` accessor type is `VEC3`. `ACC_ROTATION` → FAIL.                 |
| `09_invalid_position_no_minmax.gltf`  | glTF      | Negative: `POSITION` accessor missing `min`/`max`. `ACC_POSITION_MINMAX` → FAIL.      |
| `10_invalid_count_mismatch.gltf`      | glTF      | Negative: `_OPACITY.count` set to 7 instead of 4. `ACC_COUNTS_AGREE` → FAIL.          |
| `11_valid_spz_compressed.glb`         | GLB       | End-to-end `KHR_gaussian_splatting_compression_spz`: SPZ blob embedded in the BIN chunk. |
| `12_invalid_spz_missing_ext_used.glb` | GLB       | Negative: primitive declares SPZ but root `extensionsUsed` omits it. `SPZ_DECLARED` → FAIL. |
| `13_invalid_spz_wrong_magic.glb`      | GLB       | Negative: SPZ blob's first four bytes zeroed. `SPZ_BLOB_MAGIC` → FAIL.                  |

## Sample output

Running the validator against `01_valid_baseline.glb`:

```
KHR_gaussian_splatting conformance report for crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb (glb)
clause                   status detail
----------------------------------------------------------------------
EXT_USED                 PASS
EXT_REQUIRED             PASS
ASSET_VERSION            PASS
PRIM_EXT                 PASS
ATTRS_OBJECT             PASS
ATTR_POSITION            PASS
ATTR_ROTATION            PASS
ATTR_SCALE               PASS
ATTR_OPACITY             PASS
ATTR_COLOR_DC            PASS
ACC_POSITION             PASS
ACC_ROTATION             PASS
ACC_SCALE                PASS
ACC_OPACITY              PASS
ACC_COLOR_DC             PASS
ACC_COLOR_SH             SKIP   _COLOR_SH not declared
ACC_POSITION_MINMAX      PASS
SH_DEGREE_RANGE          PASS
ACC_COUNTS_AGREE         PASS
BUFFERVIEW_BOUNDS        PASS
SPZ_DECLARED             SKIP   SPZ not present in asset
SPZ_CONSISTENT           SKIP   SPZ not present in asset
ATTRS_KNOWN_ONLY         PASS

summary: 20 pass, 0 fail, 3 skip (of 23 clauses)
```

Sample *failing* output from `08_invalid_rotation_vec3.gltf`:

```
ACC_ROTATION             FAIL   _ROTATION.type="VEC3", want one of ["VEC4"]
…
summary: 19 pass, 1 fail, 3 skip (of 23 clauses)
```

## Open question for the KHR working group

The single largest spec ambiguity uncovered while authoring the suite is the
**wire layout of `_COLOR_SH`**:

- **Reading A** (strict glTF accessor semantics):
  `_COLOR_SH` is `SCALAR FLOAT` with `count = splat_count * 45`, one
  accessor element per SH coefficient.
- **Reading B** (per-splat record):
  `_COLOR_SH` is `SCALAR FLOAT` with `count = splat_count` and a 180-byte
  stride, treating the 45 floats per splat as a packed record.

Existing implementations (SplatForge's writer, several Three.js prototypes
seen in PRs against the extension repo) silently disagree about which
reading is canonical. `ACC_COLOR_SH` in this suite enforces Reading A —
this is the choice we recommend Khronos formalise, because it is the only
reading that satisfies glTF 2.0 §3.6.1 ("count is the number of elements
of `type`").

Fixture `04_valid_with_sh.glb` deliberately writes Reading B so a manual
review of the validator output makes the disagreement visible. The
companion test `valid_with_sh_exercises_sh_clause` documents the choice
in code.

## CI

A GitHub Action (`.github/workflows/khr-conformance.yml`) runs
`cargo test -p splatforge-khr-conformance` on every PR. The validator
binary is also built and dry-run against the committed fixture corpus.
