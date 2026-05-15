# KHR_gaussian_splatting Conformance — Submission Notes

This document is the cover-letter for the SplatForge submission to the
Khronos `KHR_gaussian_splatting` working group. The detailed clause table
and sample output live in
[`crates/splatforge-khr-conformance/conformance.md`](../crates/splatforge-khr-conformance/conformance.md).

## What we are submitting

Three artifacts, all in this repository:

1. **`splatforge-khr-validate`** — a pure-Rust CLI that loads a `.gltf` or
   `.glb` asset and reports per-clause pass/fail for every normative
   requirement in the RC text. Single static binary, no Python, no native
   dependencies beyond the Rust standard library and `serde_json`.
2. **A 10-file fixture corpus** under
   `crates/splatforge-khr-conformance/fixtures/`. Five valid fixtures
   covering FLOAT accessors, `KHR_mesh_quantization` integer accessors,
   spherical-harmonic coefficients, and the `KHR_gaussian_splatting_compression_spz`
   sub-extension declaration; five negative fixtures that exercise each
   class of invalid asset the validator catches.
3. **A conformance test matrix** (`cargo test -p splatforge-khr-conformance`)
   that regenerates the corpus from scratch, validates every fixture, and
   asserts both byte-determinism of the corpus and the validator's verdict.

## How the fixtures were generated

Every fixture is produced by the `splatforge-khr-fixtures` binary, which
calls `splatforge-gltf::{write_gltf, write_glb}` with seeded inputs (no
clocks, no entropy). To rebuild the corpus from a clean tree:

```bash
crates/splatforge-khr-conformance/scripts/generate-fixtures.sh
```

The Rust integration test `fixtures_are_byte_deterministic` runs the
generator twice and asserts every byte matches, so a one-line change to
the writer surfaces immediately.

The negative fixtures are produced by writing a valid baseline into an
isolated staging directory, parsing its JSON, mutating exactly one field
(remove an attribute, change a `type`, etc.), and re-serialising. This
keeps the negative fixtures close to real-world output instead of
hand-crafted stubs.

## How to invoke the validator

```bash
# Human-readable report:
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb

# Machine-readable JSON (suitable for CI ingestion):
cargo run -p splatforge-khr-conformance --bin splatforge-khr-validate -- \
    crates/splatforge-khr-conformance/fixtures/01_valid_baseline.glb --json
```

Exit codes: `0` if every clause passed (skips allowed), `1` if any clause
failed, `2` for a validator-level error (file unreadable, malformed GLB).

## Pass / fail criteria

An asset is "conformant" iff `splatforge-khr-validate` exits 0. Skipped
clauses are permitted — for example, the `SPZ_*` clauses skip on any asset
that does not declare the sub-extension. The full list of clauses and
their MUST/SHOULD classification lives in
[`crates/splatforge-khr-conformance/conformance.md`](../crates/splatforge-khr-conformance/conformance.md).

## Open question we want the WG to settle

The validator's `ACC_COLOR_SH` clause enforces the strict glTF accessor
reading: `_COLOR_SH` is `SCALAR FLOAT` with `count = splat_count * 45`.
We have observed at least one in-the-wild writer (SplatForge's own,
historically) emit `count = splat_count` with an implicit 45-float stride.
The two readings produce semantically identical byte buffers but make the
validator output diverge sharply. We are submitting the fixture
`04_valid_with_sh.glb` deliberately under the alternate reading so the
working group can see the disagreement and resolve it in the RC text. The
recommended resolution is "Reading A" — it is the only reading consistent
with glTF 2.0 §3.6.1.

## Where this is going

We will open a Khronos issue at
`https://github.com/KhronosGroup/glTF/issues` linking to this commit and
the conformance directory once the submission is filed. Until that issue
exists, the PR-side gate (`.github/workflows/khr-conformance.yml`)
guarantees the suite stays green on every change to either the validator
or the underlying `splatforge-gltf` writer.

The v2 engineering plan
(`splatforge-private/docs/engplan-prd-v2.md`) makes the Khronos submission
a non-negotiable Q1 deliverable: KHR_gaussian_splatting RC landed Feb 2026
and ratifies Q2 2026, and at the time of writing this document **no
reference implementation exists**. The conformance suite plus the
underlying `splatforge-gltf` crate land SplatForge as the first.
