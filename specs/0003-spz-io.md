# SPEC-0003 — SPZ I/O

**Status:** Implemented (Phase 1, basic)
**Crate:** `splatforge-spz`

## Goal

Support SPZ as a first-class compressed input/output target.

## Wire format (v2)

This crate implements a minimal SPZ-compatible writer and reader. The on-disk layout (little-endian):

```
magic            : u32 = 0x5053_4e47   ("SNPS")
version          : u32 = 2
splat_count      : u32
sh_degree        : u8
fractional_bits  : u8
flags            : u8
reserved         : u8
positions        : u24 * 3 * splat_count   (fixed-point, fractional_bits)
scales           : u8  * 3 * splat_count   (log-quantized)
rotations        : u8  * 3 * splat_count   (sign-bit recovered, smallest-three)
alphas           : u8  * splat_count       (sigmoid)
colors           : u8  * 3 * splat_count   (DC term)
sh_coeffs        : u8  * 15 * 3 * splat_count   (if sh_degree >= 1)
```

Everything is zlib-compressed.

## Acceptance tests

```gherkin
Feature: SPZ I/O

Scenario: Decode SPZ fixture
  Given fixture "tiny/basic.spz"
  When I run "splatforge analyze tiny/basic.spz"
  Then the command exits 0
  And the report says format is "spz"

Scenario: Convert PLY to SPZ
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge convert tiny/basic_binary.ply --to spz --out out.spz"
  Then out.spz exists
  And "splatforge inspect out.spz" succeeds

Scenario: Round-trip within tolerance
  Given fixture "tiny/basic_binary.ply"
  When I convert it to SPZ and back to SplatIR
  Then positions remain within 1e-2 absolute tolerance
  And DC colors remain within 0.01 absolute tolerance
```
