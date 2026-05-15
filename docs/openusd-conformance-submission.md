# OpenUSD `ParticleField3DGaussianSplat` Conformance — Submission Notes

This document is the cover-letter for the SplatForge submission to the
OpenUSD / AOUSD core spec working group, companion to the public forum
post in [`docs/standards-outreach/openusd-forum-post.md`](standards-outreach/openusd-forum-post.md).
The detailed clause table and sample output live in
[`crates/splatforge-usd-conformance/conformance.md`](../crates/splatforge-usd-conformance/conformance.md).

## What we are submitting

Three artifacts, all in this repository:

1. **`splatforge-usd-validate`** — a pure-Rust CLI that loads a `.usda`
   or `.usdc` asset and reports per-clause pass/fail for every normative
   requirement in the OpenUSD 26.03 `ParticleField3DGaussianSplat`
   schema. Single static binary, no Python, no `libusd`, no native
   dependencies beyond the Rust standard library and `serde_json`.
2. **An 8-file fixture corpus** under
   `crates/splatforge-usd-conformance/fixtures/`. Five valid fixtures
   covering minimal / typical / dense scenes, spherical-harmonic
   coefficients, and the USDC binary form; three negative fixtures that
   each exercise a different class of invalid asset the validator catches.
3. **A conformance test matrix** (`cargo test -p splatforge-usd-conformance`)
   that regenerates the corpus from scratch, validates every fixture, and
   asserts both byte-determinism of the corpus and the validator's
   verdict.

## How the fixtures were generated

Every fixture is produced by the `splatforge-usd-fixtures` binary, which
calls `splatforge_usd::{write_usda, write_usdc, render_usda}` with seeded
inputs (no clocks, no entropy). To rebuild the corpus from a clean tree:

```bash
crates/splatforge-usd-conformance/scripts/generate-fixtures.sh
```

The Rust integration test `fixtures_are_byte_deterministic` runs the
generator twice and asserts every byte matches, so a one-line change to
the writer surfaces immediately.

The negative fixtures are produced by writing a valid baseline,
re-rendering it as canonical USDA, mutating exactly one line (drop an
attribute, change one value, truncate one array), and re-serialising.
This keeps the negative fixtures close to real-world output instead of
hand-crafted stubs.

The USDC fixture (`05_valid_minimal.usdc`) round-trips through
`splatforge_usd::write_usdc`'s version-0.0.1 binary encoder; the validator
decodes it via `splatforge_usd::read_usdc`, re-emits canonical USDA via
`render_usda`, and runs the same clause matrix as for the textual
fixtures. This proves the binary and textual encodings are
schema-equivalent.

## How to invoke the validator

```bash
# Human-readable report:
cargo run -p splatforge-usd-conformance --bin splatforge-usd-validate -- \
    crates/splatforge-usd-conformance/fixtures/01_valid_minimal.usda

# Machine-readable JSON (suitable for CI ingestion):
cargo run -p splatforge-usd-conformance --bin splatforge-usd-validate -- \
    crates/splatforge-usd-conformance/fixtures/01_valid_minimal.usda --json

# As a sibling-binary to the splatforge CLI:
splatforge spec-check crates/splatforge-usd-conformance/fixtures/01_valid_minimal.usda
# (auto-detects the OpenUSD validator from the .usda/.usdc extension and shells
#  out to `splatforge-usd-validate`; override the resolved binary via the
#  `SPLATFORGE_USD_VALIDATE` env var for local dev.)
```

Exit codes: `0` if every clause passed (skips allowed), `1` if any
clause failed, `2` for a validator-level error (file unreadable,
malformed USDC).

## Pass / fail criteria

An asset is "conformant" iff `splatforge-usd-validate` exits 0. Skipped
clauses are permitted — for example, `ATTR_WIDTHS_OPTIONAL` skips on any
prim that does not author the inherited `GeomPoints.widths`. The full
list of clauses and their MUST/SHOULD classification lives in
[`crates/splatforge-usd-conformance/conformance.md`](../crates/splatforge-usd-conformance/conformance.md).

## Open questions we want the WG to settle

The validator enforces the strictest defensible reading of every
ambiguous clause; the choices are surfaced as clause IDs so a future
schema clarification can flip the bit without breaking the public report
contract. The full taxonomy lives in
[`crates/splatforge-usd/SPEC-GAPS.md`](../crates/splatforge-usd/SPEC-GAPS.md);
the items that bit us during validator construction are:

- **`displayColor` interpolation convention** (clause
  `DISPLAYCOLOR_INTERP`). The schema is silent on whether `colorsDC` is
  intended to *replace* `primvars:displayColor` or to live alongside it,
  and which `interpolation` value (`vertex` vs `varying`) viewers should
  assume when the primvar is authored without an explicit token. The
  validator accepts both `vertex` and `varying`. **Ask:** specify one.
- **SH coefficient layout** (clause `SH_COEFFS_COUNT`). 26.03 has no
  schema slot for higher-order spherical harmonics; we author them into
  `custom float[] splatforge:shCoefficients` and accept any of the four
  canonical band counts. **Ask:** define `shCoefficients` + `shDegree`
  slots in the next schema rev, or formally document the `custom`
  convention.
- **Quaternion convention** (clause `ATTR_ORIENTATIONS_TYPE` and
  `QUATS_NORMALIZED`). USDA prints quaternions as `(w, x, y, z)` while
  the on-disk `GfQuatf` stores them as `(x, y, z, w)`. **Ask:** add a
  one-line note to the schema doc.

## Where this is going

We will open an [OpenUSD Forum](https://forum.aousd.org/) thread under
"Schemas & Specifications" linking to this commit and the conformance
directory once the submission is filed. The forum post is drafted at
[`docs/standards-outreach/openusd-forum-post.md`](standards-outreach/openusd-forum-post.md).

Until that thread exists, the PR-side gate
(`.github/workflows/usd-conformance.yml`) guarantees the suite stays
green on every change to either the validator or the underlying
`splatforge-usd` writer.

The v2 engineering plan
(`splatforge-private/docs/engplan-prd-v2.md`) makes the OpenUSD
submission a non-negotiable Q1 deliverable, paired with the Khronos
`KHR_gaussian_splatting` submission tracked in
[`docs/khr-conformance-submission.md`](khr-conformance-submission.md):
OpenUSD 26.03 landed the `ParticleField3DGaussianSplat` schema, and at
the time of writing **no reference implementation pair (writer +
validator) exists outside Pixar's own `usdcat`**. The conformance suite
plus the underlying `splatforge-usd` crate land SplatForge as the first.
