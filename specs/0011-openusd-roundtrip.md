# SPEC-0011 — OpenUSD `ParticleField3DGaussianSplat` Round-Trip

**Status:** Draft (proposal, not yet implemented)
**Crate (planned):** `catetus-usd`
**Depends on:** SPEC-0001 (IR), SPEC-0007 (streaming index — relevant for SPEC-0012)

> This spec is a design proposal for v0.2 work. It has **not** been validated against a real OpenUSD toolchain in this repo; there is currently no `catetus-usd` crate, no `usdview` dependency, and no fixture USDA files. Every attribute name, type, and prim path below MUST be confirmed against an actual OpenUSD build (≥ v26.03) before implementation lands.

## Goal

Define how `SplatIR` (SPEC-0001) maps to and from OpenUSD's native Gaussian Splat schema, `ParticleField3DGaussianSplat`, introduced in **OpenUSD v26.03**. The CLI should support:

```
catetus convert <in> --to usd --out scene.usda
catetus convert scene.usda --to gltf --out scene.gltf
```

The round-trip `IR → USD → IR` must be **lossless for the supported subset** and must produce a byte-stable USDA file for a fixed input (SPEC-0001 determinism rule).

## Source of truth

* OpenUSD release notes referencing `ParticleField3DGaussianSplat`: `https://openusd.org/release/` (release-notes page — exact anchor TBD; needs to be re-verified, see "Open questions").
* Schema reference (best-guess path, **flag for verification**): `https://openusd.org/release/api/class_usd_geom_particle_field3_d_gaussian_splat.html`
* Pixar source: `https://github.com/PixarAnimationStudios/OpenUSD` — `pxr/usd/usdGeom` (the schema may live in `usdGeom` or in a new `usdHydra`/`usdSplat` module; needs verification).

Citations above are best-guess paths based on OpenUSD's standard documentation layout. Treat as "starting point for verification," not as authoritative.

## Design

### 1. Prim hierarchy

A `SplatScene` maps to a single root `Xform` prim that owns one `ParticleField3DGaussianSplat` per **chunk** in the IR's chunked layout (see SPEC-0007). For unchunked scenes there is exactly one `ParticleField3DGaussianSplat`.

```
/World                                    # Xform (root)
    metadata: customLayerData["catetus:version"] = "0.2"
    metadata: customLayerData["catetus:coordinateSystem"] = "y_up_right_handed"

/World/Splats                             # Scope
/World/Splats/chunk_0000                  # ParticleField3DGaussianSplat
/World/Splats/chunk_0001                  # ParticleField3DGaussianSplat
...
```

**Why one prim per chunk, not one prim with sub-ranges:**

* USD's composition primitives (payloads, variant sets — SPEC-0012) operate at the prim level. One prim per chunk is what makes payload-arc streaming and LOD variants viable.
* USD does have `PrimRange`-style traversal, but no first-class concept of an attribute sub-range that can be independently payloaded.
* Renderers without streaming support load all chunks and the visual result is identical to a monolithic prim.

### 2. Attribute mapping

The mapping below assumes the v26.03 schema exposes the canonical per-splat attributes. **Names that are not yet verified are marked with `(?)`.** Where the schema does not yet define a field, Catetus uses a `catetus:` custom-attribute namespace; this is the same escape valve as `SF_spatial_streaming_index` in glTF (SPEC-0007).

| `SplatIR` field      | USD attribute (on `ParticleField3DGaussianSplat`) | Type             | Notes |
| -------------------- | ------------------------------------------------- | ---------------- | ----- |
| `position`           | `points`                                          | `point3f[]`      | World-space, post coordinate-system normalization. |
| `rotation`           | `orientations` (?)                                | `quatf[]`        | USD quaternion convention is `(real, i, j, k)` i.e. **w first**. IR is `(x, y, z, w)`. Importer/exporter MUST reorder. |
| `scale`              | `scales`                                          | `float3[]`       | IR stores linear scale. USD attribute is also linear per the v26.03 spec (verify). |
| `opacity`            | `opacities` (?)                                   | `float[]`        | Linear [0, 1]. IR is already post-sigmoid. |
| `color` (DC)         | `colorsDC` (?) or `primvars:displayColor`         | `color3f[]`      | Prefer the schema's native attribute; fall back to `primvars:displayColor` only if v26.03 didn't ship one. |
| `color` (SH degree)  | `shDegree`                                        | `int` (uniform)  | Scalar uniform on the prim. |
| `color` (SH coeffs)  | `shCoefficients` (?)                              | `float[]`        | Flat array, length = `splatCount * 3 * ((shDegree+1)^2 - 1)`. Layout MUST match glTF `_COLOR_SH` (SPEC-0004) so the same buffer can be reused. |
| per-splat ID         | `ids` (?)                                         | `int64[]`        | Optional. Used by editors for stable selection across re-exports. If schema lacks this, use `catetus:ids` custom attribute. |
| semantic labels      | `catetus:semanticLabels`                       | `int[]`          | No native USD analog. Custom attribute namespace. |
| coordinate system    | layer metadata `customLayerData`                  | dict             | `upAxis` is also written as a standard USD stage metadata. |
| temporal mode        | `catetus:temporalMode`                         | `token`          | Always `"static"` in v1. |

### 3. Quaternion convention

* IR stores quaternions as `[x, y, z, w]` (SPEC-0001).
* USD `GfQuatf` and `quatf` attributes are `(real, imaginary)` → in array layout that is `[w, x, y, z]`.
* The exporter MUST swap component order on write. The importer MUST swap on read.
* Both paths MUST renormalize quaternions and reject any with `|q| < 0.99` (consistent with SPEC-0002 PLY ingest).

### 4. Up-axis and handedness

* Stage metadata `upAxis` is written. USD supports `"Y"` or `"Z"`; Catetus writes whatever the IR carries.
* USD does not encode handedness as stage metadata. Catetus records it in `customLayerData["catetus:coordinateSystem"]`. On import, if the value is absent, the importer assumes right-handed (USD convention).

### 5. Inline attributes vs. asset references

| Data                        | Storage                             | Why |
| --------------------------- | ----------------------------------- | --- |
| Small scenes (< 50k splats) | Inline `.usda` ASCII attributes     | Human-inspectable; matches v1 PLY fixture sizes. |
| Medium scenes               | `.usdc` (Crate) single file         | USD's native binary; faster than ASCII. |
| Chunked / streaming scenes  | Payload arcs → external `.usdc`     | See SPEC-0012. Each chunk becomes a payloaded prim. |
| SH coefficient buffers      | Inline arrays on the prim           | USD has no glTF-style external buffer concept; payloading is the equivalent. |

The exporter picks the format from the output file extension: `.usda` → ASCII, `.usdc` → Crate binary, `.usdz` → zip package (no compression on the inner Crate file, per USDZ rules).

### 6. Determinism

* Per SPEC-0001, the USDA exporter must produce byte-identical output for the same IR.
* USDA attribute ordering MUST be sorted: prim children alphabetically, attributes alphabetically on each prim.
* Floating-point formatting MUST use the same canonical `f32` → string path as the rest of the workspace (no platform-dependent printf).

### 7. Out-of-scope for v0.2

* USD composition semantics beyond payloads and variant sets (SPEC-0012). No `over`s, no class-based inheritance, no `apiSchemas`.
* Time-sampled / animated splats. v1 is static; the temporal-mode field is reserved.
* Material binding. Splats carry their own color; we don't write `UsdShade` materials.
* USDZ AR Quick Look optimization. We can write `.usdz`, but we don't claim Quick Look compatibility in v0.2.

## File layout (planned)

```
crates/catetus-usd/
    src/
        lib.rs
        reader.rs        # .usda / .usdc → SplatIR
        writer.rs        # SplatIR → .usda / .usdc
        schema.rs        # ParticleField3DGaussianSplat attribute names + types
        quat.rs          # (x,y,z,w) ↔ (w,x,y,z) helpers
    tests/
        roundtrip.rs
```

Integration mechanism is TBD (see "Open questions") — likely an FFI call into the OpenUSD C++ library or a subprocess invocation of `usdcat` for the v0.2 spike, per the PRD §Phase 3.

## Acceptance tests

```gherkin
Feature: OpenUSD ParticleField3DGaussianSplat round-trip

Scenario: Export PLY to USDA
  Given fixture "tiny/basic_binary.ply"
  When I run "catetus convert tiny/basic_binary.ply --to usd --out scene.usda"
  Then scene.usda exists
  And scene.usda contains a prim of type "ParticleField3DGaussianSplat"
  And the prim has attribute "points" with 3 entries
  And the stage metadata declares upAxis "Y"

Scenario: Round-trip preserves splat count and positions
  Given fixture "tiny/basic_binary.ply"
  When I convert it to USD and back to IR
  Then the round-tripped IR has the same splat count
  And every position differs by less than 1e-6 from the original
  And every quaternion differs by less than 1e-6 (after w-first reordering)

Scenario: Quaternion component order is corrected on export
  Given an IR splat with rotation [0.1, 0.2, 0.3, 0.927]
  When I export to USDA
  Then the "orientations" attribute starts with 0.927 (w-first)

Scenario: Chunked IR produces one prim per chunk
  Given an IR with 4 chunks under "/World/Splats"
  When I export to USDA
  Then there are 4 ParticleField3DGaussianSplat prims under "/World/Splats"
  And their names are deterministic ("chunk_0000" through "chunk_0003")

Scenario: Fields with no USD analog land in catetus namespace
  Given an IR with semantic labels
  When I export to USDA
  Then the prim carries attribute "catetus:semanticLabels"
  And a viewer that ignores custom attributes still renders the splats

Scenario: USDA export is deterministic
  Given the same fixture and config
  When I export twice
  Then the two .usda files are byte-identical
  And the BLAKE3 hashes match

Scenario: Import USDA produced by another tool
  Given a USDA file that declares ParticleField3DGaussianSplat with only points, scales, orientations, opacities, colorsDC
  When I run "catetus analyze scene.usda"
  Then the command exits 0
  And the report says format is "usd"
  And missing optional attributes are reported as "absent" rather than failing
```

## Backwards compatibility

* No existing Catetus artifact format changes. USD is purely additive.
* glTF and SPZ paths are unaffected.
* The `SF_spatial_streaming_index` glTF extension (SPEC-0007) is **not** written into USD assets. Its USD counterpart is defined in SPEC-0012.

## Open questions

1. **Exact schema attribute names.** `orientations`, `opacities`, `colorsDC`, `shCoefficients`, `ids` are educated guesses based on OpenUSD's general schema naming style (compare `UsdGeomPoints` which uses `points`, `widths`, `velocities`). The real names MUST be read off the v26.03 schema definition file (`schema.usda` in the relevant USD module) before any implementation.
2. **Module location.** Is `ParticleField3DGaussianSplat` in `usdGeom`, a new module, or a `usdHydra` extension? Unverified.
3. **Integration mechanism.** OpenUSD's reference implementation is C++. Options for the Rust workspace:
   * Subprocess `usdcat` / `usdedit` (slow, no FFI complexity — PRD §795 suggests this for the spike).
   * Build OpenUSD as a static lib and call via `cxx`/`bindgen` (fast, painful).
   * Wait for `openusd-rs` or a similar binding to mature.
   Decision deferred to v0.2 implementation kickoff.
4. **USDZ AR Quick Look.** Apple's Quick Look has its own constraints on USDZ contents. We do not promise compatibility in v0.2 but should re-evaluate once a real schema validator is in CI.
5. **Color space.** glTF stores DC color as sRGB-linear (SPEC-0004). USD's color3f attribute color space is governed by stage `colorSpace` metadata. Mapping needs verification.
6. **SH coefficient layout.** glTF `_COLOR_SH` (SPEC-0004) is RGB-interleaved with a documented packing. USD's expected packing for `shCoefficients` is unverified; if it differs, the writer/reader must transpose.

## Change history

| Version | Date       | Author | Notes |
| ------- | ---------- | ------ | ----- |
| 0.1     | 2026-05-13 | Monte  | Initial draft. Not yet validated against a real OpenUSD toolchain. |
