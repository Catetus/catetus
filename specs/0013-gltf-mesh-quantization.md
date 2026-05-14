# SPEC-0013 — glTF `KHR_mesh_quantization` for Gaussian Splat attributes

**Status:** Draft
**Crate:** `splatforge-gltf`
**Depends on:** SPEC-0004 (glTF KHR_gaussian_splatting), SPEC-0007 (streaming index)
**Owner:** TBD
**Last updated:** 2026-05-14

## Goal

Shrink the glTF wire size of an optimized SplatForge scene so the glTF-only
delivery path is competitive with the SPZ-encoded equivalent. Today the
glTF buffer is f32-accessor everywhere; this spec promotes the small,
quantization-friendly attributes (POSITION, SCALE, COLOR_DC, OPACITY) to
integer accessors via the Khronos
[`KHR_mesh_quantization`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_mesh_quantization/README.md)
extension.

## Motivation

The bonsai real-world demo (see `benches/reports/bonsai-real-demo.md` §2) hits:

| Preset            | glTF buffer (raw f32) | SPZ output |
| ----------------- | --------------------- | ---------- |
| `lossless-repack` |   260.4 MB            |  34.2 MB   |
| `web-mobile`      |    56.7 MB            |  12.0 MB   |
| `size-min`        |    48.3 MB            |   8.6 MB   |

The 4.7× gap between `web-mobile` glTF and SPZ is *purely* the f32 → fixed-point
quantization SPZ applies. Promoting the glTF accessors to integer types via
`KHR_mesh_quantization` closes that gap without inventing anything proprietary
— the extension is a Khronos ratified standard supported by Three.js, Babylon,
Filament, and the glTF Sample Viewer.

A secondary benefit: many WebGL/WebGPU renderers can upload integer
attributes directly to the GPU and dequantize in the vertex shader, saving
the host-side decode pass our viewer currently does at upload time.

## Non-goals

* Re-quantizing data the user explicitly asked to preserve. `lossless-repack`
  keeps its f32 accessors; this extension is only applied automatically by
  `web-mobile`, `quest-browser`, and `size-min` presets.
* Implementing the inverse path inside the optimize pipeline. The existing
  `QuantizePosition` / `QuantizeScale` / `QuantizeRotation` passes already
  compute the quantized values in-IR; this spec is only about *serializing*
  them in integer accessors rather than re-promoting them to f32.
* Quantizing `_ROTATION` (quaternions) here. Smallest-three packing is
  better than uniform per-component quantization; that's a separate spec.

## Wire format

For each attribute we change accessor `componentType` + (optionally) add a
normalization + min/max so consumers can dequantize. We do NOT add per-prim
matrix transforms — splat attributes are unit-less, so the simpler
"normalize to [0, 1] then re-scale via accessor.min/max" form applies.

### POSITION (vec3)

```jsonc
{
  "bufferView": 0,
  "componentType": 5123,   // UNSIGNED_SHORT (16-bit)
  "count": <n>,
  "type": "VEC3",
  "normalized": true,
  "min": [bbox.min.x, bbox.min.y, bbox.min.z],
  "max": [bbox.max.x, bbox.max.y, bbox.max.z]
}
```

15-bit quantization in-IR is preserved; we emit 16-bit accessors because
glTF doesn't have a 15-bit type. The top bit is unused; consumers ignore it.

### _SCALE (vec3)

```jsonc
{
  "bufferView": 2,
  "componentType": 5121,   // UNSIGNED_BYTE (8-bit)
  "count": <n>,
  "type": "VEC3",
  "normalized": true,
  "min": [0, 0, 0],
  "max": [maxScale.x, maxScale.y, maxScale.z]
}
```

Scales are already 8-bit-quantized inside the IR (`QuantizeScale`). The
per-axis `max` is whatever the optimize pass clamped to (default 16× the
median; see `splatforge-optimize::QuantizeScale`).

### _OPACITY (scalar)

```jsonc
{
  "bufferView": 3,
  "componentType": 5121,   // UNSIGNED_BYTE (8-bit)
  "count": <n>,
  "type": "SCALAR",
  "normalized": true,
  "min": [0],
  "max": [1]
}
```

Lossless from 8-bit IR opacity to 8-bit accessor.

### _COLOR_DC (vec3)

```jsonc
{
  "bufferView": 4,
  "componentType": 5121,   // UNSIGNED_BYTE (8-bit)
  "count": <n>,
  "type": "VEC3",
  "normalized": true
}
```

DC color already lives in `[0, 1]`; 8-bit normalized accessor is sufficient.
SH-rest coefficients (`_COLOR_SH`) stay as f32 for now — they have a wide
dynamic range and don't compress well at 8-bit.

### _ROTATION

Unchanged. Stays as f32 VEC4 in v0.2. Smallest-three quaternion packing
is a separate spec.

## Size projection

For the bonsai `web-mobile` scene (1,061,636 splats):

| Attribute     | Before (f32) | After (integer) | Saved   |
| ------------- | ------------ | --------------- | ------- |
| POSITION      | 12.7 MB      |   6.4 MB        |  6.4 MB |
| _SCALE        | 12.7 MB      |   3.2 MB        |  9.5 MB |
| _OPACITY      |  4.2 MB      |   1.1 MB        |  3.2 MB |
| _COLOR_DC     | 12.7 MB      |   3.2 MB        |  9.5 MB |
| _ROTATION     | 16.9 MB      |  16.9 MB        |    —    |
| **Total**     | **59.4 MB**  | **30.9 MB**     | **28.5 MB** |

That's a 1.9× reduction in glTF buffer size, putting it within ~2.5× of the
12.0 MB SPZ payload. With `KHR_mesh_quantization` enabled, the glTF-only
delivery path covers the long-tail of viewers that don't support SPZ
without forcing partners to ship two formats.

## Compatibility

`KHR_mesh_quantization` is non-required and ratified by Khronos. Viewers
that don't understand it MUST be told via `extensionsRequired` whether the
quantization is invertible without their support:

* If we omit `extensionsRequired`, viewers can render the file with degraded
  fidelity (treat integer accessors as raw values, not dequantized).
* If we add `extensionsRequired`, viewers without the extension reject the
  asset outright.

**Decision:** add `KHR_mesh_quantization` to `extensionsUsed`, NOT to
`extensionsRequired`. Renderers that miss the extension still load the
asset; they just render with un-dequantized values. The vast majority of
glTF 2.0 renderers ship `KHR_mesh_quantization` support today.

## Acceptance criteria

```gherkin
Feature: KHR_mesh_quantization on optimize output

  Scenario: web-mobile preset emits integer POSITION accessor
    Given a 100-splat scene at the `web-mobile` preset
    When `splatforge optimize` writes the glTF
    Then accessors[POSITION].componentType is 5123 (UNSIGNED_SHORT)
    And accessors[POSITION].normalized is true
    And accessors[POSITION].min and max bracket every splat position

  Scenario: lossless-repack preserves f32 accessors
    Given a 100-splat scene at the `lossless-repack` preset
    When `splatforge optimize` writes the glTF
    Then accessors[POSITION].componentType is 5126 (FLOAT)
    And the asset does NOT list KHR_mesh_quantization in extensionsUsed

  Scenario: writer + reader round-trip preserves per-axis quantization
    Given the bonsai scene at `web-mobile`
    When the glTF is written and read back via `splatforge convert`
    Then the per-axis position error is bounded by 1 / 32768 of the bbox span
    And the per-component scale error is bounded by 1 / 256 of the scale max

  Scenario: SPZ payload remains the smallest delivery
    Given identical input and the `web-mobile` preset
    Then the SPZ output remains strictly smaller than the integer-accessor glTF
    And both round-trip to within ΔE94 < 1% under the SwiftShader fidelity gate
```

## Implementation plan

1. **Extend `splatforge_gltf::GltfAccessor`** with optional `normalized`,
   `min`, `max` fields (already partially in v0.1.1 for POSITION min/max).
2. **Add a `quantize` flag to `WriteOpts`** that defaults to `false`; preset
   presets in `splatforge-optimize` flip it on for the three quantization-
   safe ones.
3. **In `pack_chunk`**, when `quantize` is true, write the integer
   accessors instead of f32. The per-axis min/max are computed alongside
   the existing `chunk_bbox` walk.
4. **Add `KHR_mesh_quantization` to `extensionsUsed`** only when the flag is on.
5. **Update `read_gltf`** to detect normalized integer accessors and
   dequantize them back to f32 on the IR side. Today's reader assumes f32;
   the test in step 4 of "Acceptance criteria" drives this change.

## Open questions

* Per-axis vs uniform scale `max`. Per-axis is more honest about the
  data; uniform packs slightly better. Lean per-axis; revisit after
  measuring on the bonsai corpus.
* Should `_COLOR_DC` use sRGB-encoded 8-bit values (`accessor.normalized:
  true` + assumed sRGB transfer) or linear? Today the IR stores linear DC.
  Defer until we have a colour-management spec.
* Does any prominent viewer in our parity matrix (Three.js, Babylon, glTF
  Sample Viewer, Apple QuickLook) refuse `KHR_mesh_quantization` for
  `KHR_gaussian_splatting` primitives specifically? Out of scope until a
  Khronos working-group discussion.

## Change history

* 2026-05-14 — Initial draft.
