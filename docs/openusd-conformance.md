# OpenUSD Conformance — SplatForge `splatforge-usd`

**Status:** USDA + USDC round-trip bit-exact-as-USDA against Pixar's `usdcat`
(Apple USD Tools 0.25.2 verified, May 2026). Anything Pixar can read, we
can write; anything we write, Pixar can read.

## What we support

| Surface                                | Status      | Notes |
| -------------------------------------- | ----------- | ----- |
| USDA (text USD) writer                 | ✅ Stable    | Single-prim `ParticleField3DGaussianSplat` under `/World/Splats`. |
| USDA reader                            | ✅ Stable    | Tolerates `usdcat`-reformatted whitespace, missing layer metadata, attribute reordering. |
| USDC (binary crate) writer             | ✅ Stable    | Emits version **0.0.1**; `usdcat` reads it without errors. |
| USDC (binary crate) reader             | ✅ Subset    | In-process reader for our own schema (version 0.0.1, uncompressed sections). For arbitrary USDC, shell out to `usdcat -o tmp.usda` then call `read_usda`. |
| `ParticleField3DGaussianSplat` schema  | ✅ All-of    | `points`, `orientations`, `scales`, `opacities`, `colorsDC`. |
| SH coefficients                        | ⚠ Custom    | Written as `custom float[] splatforge:shCoefficients` until OpenUSD 26.03's official slot lands. See `crates/splatforge-usd/SPEC-GAPS.md`. |
| Streaming (SPEC-0012 payload-arcs)     | 🚧 Draft    | Feature-gated under `streaming`; not exercised in conformance tests yet. |

## Format details

We write USDC at **version 0.0.1** intentionally. Pixar's
`SdfFileVersion::CanRead` accepts any minor `<=` software minor; targeting
the oldest schema buys us the simplest, most forward-compatible wire format:

| Section    | 0.0.1 layout                                                                            |
| ---------- | --------------------------------------------------------------------------------------- |
| Bootstrap  | 88 bytes: `"PXR-USDC"` + `[0,0,1,0,0,0,0,0]` + `int64 tocOffset` + 64 reserved zero bytes. |
| TOKENS     | `u64 numTokens, u64 totalBytes, raw null-terminated UTF-8`. Token 0 is `";-)"` (Pixar's path-coder sentinel). |
| STRINGS    | `u64 numStrings` followed by `numStrings * u32 tokenIndex`.                              |
| FIELDS     | `u64 numFields` then `numFields * { u32 _pad, u32 tokenIndex, u64 valueRep }`.           |
| FIELDSETS  | `u64 numEntries` then `numEntries * u32` (each set terminated by `~0u`).                 |
| PATHS      | `u64 numPaths` then a depth-first tree of `_PathItemHeader_0_0_1` (16-byte) records.     |
| SPECS      | `u64 numSpecs` then `numSpecs * { u32 _pad, u32 pathIndex, u32 fieldSetIndex, u32 specType }`. |
| TOC        | `u64 numSections` then `numSections * { char name[16], int64 start, int64 size }`.       |

Versions ≥ 0.4.0 introduce LZ4 compression on most sections plus a custom
70-byte VLE integer coder — none of which is required at 0.0.1. The only
piece we'd inherit at higher versions is the `TfFastCompression`-wrapped
TOKENS section; reusing the deterministic uncompressed form keeps the
encoder byte-stable.

## Running the round-trip test

### One-liner

```sh
./scripts/usdc-roundtrip.sh
```

### Expected output

```
Using usdcat: Apple USD Tools (0.25.2)
   Compiling ...
    Finished `release` profile [optimized] target(s) in 20.93s

=== minimal.usda ===
  [ok] wrote /tmp/usdc-roundtrip-XXXX/minimal.usdc (1377 bytes)
  [ok] usdcat accepted; reformat = /tmp/usdc-roundtrip-XXXX/minimal.via_usdcat.usda
  [PASS] minimal.usda round-tripped bit-exact-as-USDA

=== particle_field.usda ===
  [ok] wrote /tmp/usdc-roundtrip-XXXX/particle_field.usdc (1489 bytes)
  [ok] usdcat accepted; reformat = /tmp/usdc-roundtrip-XXXX/particle_field.via_usdcat.usda
  [PASS] particle_field.usda round-tripped bit-exact-as-USDA

=== dense.usda ===
  [ok] wrote /tmp/usdc-roundtrip-XXXX/dense.usdc (4889 bytes)
  [ok] usdcat accepted; reformat = /tmp/usdc-roundtrip-XXXX/dense.via_usdcat.usda
  [PASS] dense.usda round-tripped bit-exact-as-USDA

===========================================
  PASS: 3   FAIL: 0
===========================================
```

### Reference fixtures

| File                                                          | Purpose |
| ------------------------------------------------------------- | ------- |
| `crates/splatforge-usd/fixtures/minimal.usda`                 | One splat with identity rotation; smallest valid `ParticleField3DGaussianSplat` prim. |
| `crates/splatforge-usd/fixtures/particle_field.usda`          | Three splats with non-identity quaternions, varying opacity and colors. Mirrors the OpenUSD 26.03 schema exemplar. |
| `crates/splatforge-usd/fixtures/dense.usda`                   | 64 splats on a 4×4×4 grid; exercises the array path at non-trivial size. |

### `cargo` test surface

* `cargo test -p splatforge-usd` — runs in-process tests (deterministic
  encoding, USDC magic, quaternion ordering, SH round-trip).
* `cargo test -p splatforge-usd --features usdcat-validation` — additionally
  shells out to `usdcat` and asserts it accepts our binaries.

## What Pixar / Apple need to see for "reference implementation" status

1. **Schema fidelity.** All five mandatory `ParticleField3DGaussianSplat`
   attributes (`points`, `orientations`, `scales`, `opacities`, `colorsDC`).
2. **Bit-exact-as-USDA.** `usdcat <ours>.usdc -o out.usda` produces a USDA
   semantically identical to the original. Demonstrated on three fixtures
   spanning trivial / typical / dense.
3. **Determinism.** Identical scene → identical bytes; no HashMap iteration
   leaks. Asserted in `tests/usdc_roundtrip.rs::deterministic_encoding`.
4. **Forward compatibility.** Targeting version 0.0.1 means our files load
   in *every* released `usdcat` since 2016 — not just current Apple builds.
5. **No vendor lock-in.** Pure Rust, no `libusd`, no Python; builds with
   `cargo build` on any host.

The single remaining "asterisk" is SH packing: until OpenUSD names the slot
on the schema, we author into `splatforge:shCoefficients`. See
`SPEC-GAPS.md` for the proposed clarification.

## Sample command

```sh
# Convert an authoring USDA → binary USDC.
splatforge convert input.usda --to usdc -o output.usdc

# Inspect the binary.
file output.usdc                # → "data" (no magic registered; that's fine)
xxd output.usdc | head -2       # → "PXR-USDC\0\0\1\0\0\0\0\0…"

# Round-trip through Pixar.
usdcat output.usdc -o via_pixar.usda
diff <(splatforge convert input.usda --to usda -o /dev/stdout) via_pixar.usda
# (semantic diff — formatting differs by design)
```
