# OpenUSD ParticleField3DGaussianSplat — implementation feedback + spec ambiguities

**Audience.** [OpenUSD Forum](https://forum.aousd.org/) > "Schemas & Specifications".
**Target action.** Post the body below. Link the SplatForge USDC writer + SPEC-GAPS doc.

---

## Title

`USD 26.03: production feedback on ParticleField3DGaussianSplat — 10 spec ambiguities and a reference USDC writer`

## Body

OpenUSD 26.03's `ParticleField3DGaussianSplat` schema is the first standards-aligned wire format for 3D Gaussian Splats in USD. We've been building against it for a production encoder pipeline and now have an end-to-end implementation: USDA + USDC round-trip, bit-exact-as-USDA against `usdcat` 0.25.2.

**Implementation:** [`crates/splatforge-usd`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-usd) in the SplatForge repo. Pure Rust, no Pixar/OpenUSD library dependency.

**Round-trip results** (`scripts/usdc-roundtrip.sh`):

| Fixture | Result |
|---|---|
| `minimal.usda` (1 splat, identity rotation) | PASS |
| `particle_field.usda` (3 splats, non-identity quats, varied opacity/color) | PASS |
| `dense.usda` (64 splats on a 4×4×4 grid) | PASS |

Bit-exact-as-USDA means: write USDA → SplatForge USDC → `usdcat` → USDA, with the round-tripped USDA byte-identical to the input.

### 10 spec ambiguities we hit (raised here so the WG can adjudicate or document)

**Schema gaps (most impactful first):**

1. **No `splatforge:shCoefficients` slot.** Every production 3DGS scene ships spherical harmonic coefficients for view-dependent color. The 26.03 schema omits them. We currently author into `custom float[] splatforge:shCoefficients` with a sidecar manifest, which works but isn't portable. **Ask:** define `shCoefficients` + `shDegree` slots in the next schema rev, or formally document the `custom` convention.

2. **Quaternion convention.** USDA prints quaternions as `(w, x, y, z)`; USDC stores them as `(x, y, z, w)`. Both are valid but the schema doc says only "rotation as quaternion" — production readers that don't know to swap fail silently. **Ask:** add a one-line note to the schema doc.

3. **Scale semantics.** Are the `scale` values linear units or log-space (per the original 3DGS training convention)? Both interpretations exist in the wild. **Ask:** specify which the schema means; if log-space, document the `exp` operator readers need to apply.

4. **Opacity range.** Post-sigmoid `[0, 1]` or raw logit `(-∞, ∞)`? **Ask:** specify; preferred answer is `[0, 1]` to match the rasterization convention.

**Wire-format / encoder gotchas (USDC writer authors will hit these):**

5. **Token-table sentinel `;-)\0` at index 0.** Pixar's `crateFile.cpp::StartPacking()` hard-codes this string as the first token. Files that omit it fail to load under `usdcat`, with no diagnostic. **Ask:** document this in the crate-format spec (the compressed coder uses negative indices to flag prim-property paths; an aliasing index 0 is the original reason for the sentinel).

6. **`_unused_padding_` on `Field`, `Spec_0_0_1`, `_PathItemHeader_0_0_1`.** A 4-byte padding inherited from a GCC-vs-MSVC empty-base-class layout discrepancy. Modern writers must zero this padding or silent reader misalignment ensues. **Ask:** spec the padding explicitly.

7. **`SdfFileVersion` forward-compat.** Pixar's `CanRead` permits any `minor ≤ software.minor`; we target version 0.0.1 to avoid the compressed-tokens / LZ4 path. The version-0.4.0 transition silently switches to `TfFastCompression`. **Ask:** spec the version transitions; tools that target old versions should be able to do so explicitly.

8. **Uncompressed token-stream layout (version &lt;0.4.0).** Pixar's source documents the compressed path but not the uncompressed one. We worked it out by decoding `usdcat` output byte-by-byte. **Ask:** include the uncompressed layout in the spec, or formally deprecate version &lt;0.4.0.

**Convention / interoperability:**

9. **`points`, `widths`, `velocities` inheritance.** `ParticleField3DGaussianSplat` inherits from `GeomPoints` but the prims emitted by 3DGS captures don't have widths or velocities. Are these required by the inherited schema, or optional in this concrete prim? **Ask:** document; preferred answer is "optional with reasonable defaults."

10. **Indexed vs varying primvars.** Per-splat color is currently authored as a `point3f[]` displayColor primvar. Should it be `varying` (per-point) or `vertex` (interpolated)? **Ask:** specify the interpolation convention.

### Where to look in the SplatForge repo

- Writer + reader: `crates/splatforge-usd/src/usdc.rs` (1100 LOC, version 0.0.1 target).
- Full ambiguity list: `crates/splatforge-usd/SPEC-GAPS.md`.
- Round-trip harness: `scripts/usdc-roundtrip.sh`.
- Conformance doc: `docs/openusd-conformance.md`.

We're happy to land doc-only PRs against the OpenUSD repo for any of the above clarifications, on the WG's preferred timeline.

---

## Pre-post checklist

- [ ] Confirm the post category in the OpenUSD Forum is "Schemas & Specifications" (or whichever is closest).
- [ ] Check for a recent USD release after 26.03 that may already have addressed any of the gaps; update the post accordingly.
- [ ] Link the SplatForge repo at a tagged commit, not main, so reviewers can `git checkout` a known-good state.
- [ ] Cross-post a thread reference to apple-USD-tools maintainers (dgovil et al. authored PR #3716).
