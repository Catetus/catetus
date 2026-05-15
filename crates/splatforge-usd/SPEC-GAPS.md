# SPEC-GAPS — OpenUSD 26.03 `ParticleField3DGaussianSplat`

Notes captured while implementing the SplatForge USDC writer (May 2026).
Each item is a clause of the schema we found ambiguous, under-specified,
or hostile to interoperability. Suggested resolutions in the right column;
these are intended as talking points for the OpenUSD Forum / AOUSD Core
Spec WG, not unilateral demands.

## Gaps that bit us

### 1. SH coefficient storage is unspecified

**What the schema says (26.03):** the schema includes `points`,
`orientations`, `scales`, `opacities`, and `colorsDC` (a DC-only RGB term).
There is *no* slot for higher-order spherical harmonics.

**Why this matters:** every production 3DGS implementation ships SH degree
1-3 by default. A schema that only carries the DC term throws away the
view-dependence that's the whole point of using Gaussians.

**What we did:** authored coefficients into a custom-namespaced attribute
`custom float[] splatforge:shCoefficients`. Layout: `n_splats * (degree+1)^2 * 3`
floats, interleaved per-band (RGB-major).

**Proposed clarification:** add `float[] shCoefficients` to the schema with
`degree` carried as `uniform int shDegree` metadata on the prim. Specify
the interleaving so two implementations interpret the same packed array
the same way.

### 2. Quaternion convention is asserted, not enforced

**Status:** USDA prints `quatf[]` as `(w, x, y, z)` while the on-disk
`GfQuatf` is `(x, y, z, w)` (imaginary then real). New implementers can't
tell from the schema alone whether their on-disk quaternions should be
real-first or imaginary-first; only by reading `pxr/base/gf/quatf.h` do
they discover the wire layout differs from the textual layout.

**Proposed clarification:** the schema doc should explicitly call out that
*authoring* uses `(w, x, y, z)` and *binary persistence* uses
`(x, y, z, w)` — and that the conversion is handled by the IO layer.

### 3. Scale semantics: log-space vs linear

**Status:** several upstream 3DGS papers (e.g. 3D Gaussian Splatting, Yang
et al.) author scales in log-space and exponentiate at render time.
`ParticleField3DGaussianSplat.scales` is typed as `float3[]` with no
indication of whether the values are linear or log.

**Proposed clarification:** define `scales` as **linear** per-axis radii;
require importers from log-space authoring formats (PLY with `scale_0..2`)
to exponentiate before write.

### 4. Opacity semantics: pre-sigmoid vs `[0, 1]`

**Status:** original 3DGS authors a logit-space opacity and applies
sigmoid at render. `opacities` in the schema is typed `float[]` with no
range constraint. A naive `usdcat`-driven viewer applying `floor(o)` to a
0.5-opacity splat from one toolchain vs another lands on different pixels.

**Proposed clarification:** declare `opacities` as **post-sigmoid in
`[0, 1]`**, mandate importers apply sigmoid on read.

## Gaps that did *not* bite us but probably should be fixed

### 5. Time-sampled / 4D splats

**Status:** 26.03 didn't speak to dynamic Gaussian splats. Pixar's USDC
binary supports `TimeSamples` natively; we did not exercise that path. The
schema should commit to a temporal-binding convention (per-frame
`points`/`orientations` time samples vs payload-arc'd LOD chunks per
sample) before viewers diverge.

### 6. LOD signaling

**Status:** SplatForge ships LODs as a `lod` variant set on the splat prim
(SPEC-0012). Other vendors will ship them as nested prims, or as
`SkelBindingAPI`-style references. The schema is silent on which is
"canonical." A reference implementation should pick *one*.

### 7. Coordinate-system metadata

**Status:** the schema inherits `upAxis` and `metersPerUnit` from layer
metadata. Gaussian splats produced by COLMAP / OpenSfM / nerfstudio use a
*per-asset* implicit handedness that the schema has no slot for. We
default to left-handed Y-up and warn on mismatch; another vendor will
default differently and the inter-op tax falls on viewers.

**Proposed clarification:** require `uniform token sourceCoordinateSystem`
on the prim (enum: `gl`, `vulkan`, `colmap`, `unity`, `unreal`); viewers
flip on read.

## Format gotchas worth standardizing

These aren't strictly schema gaps but bit us during the binary writer.

### 8. The `";-)"` sentinel token

**Status:** `pxr/usd/sdf/crateFile.cpp` reserves token index 0 to a
hard-coded `";-)"` to "sidestep a bug" (github issue 811) in the
compressed-path coder, where negative indices encode the property bit and
zero would alias. This is **load-bearing** for binary readers: token 0
*must* be `";-)"` or the file fails to load.

**Proposed clarification:** make this a documented requirement of the
crate format, not a comment in the C++ source. Better yet, fix the
underlying coder so the sentinel isn't needed and version-gate the
removal.

### 9. Field struct padding

**Status:** `Field`, `Spec_0_0_1`, and `_PathItemHeader_0_0_1` each carry
a `uint32_t _unused_padding_` member documented as a workaround for an
old GCC empty-base-class layout bug versus MSVC. New implementations
unaware of this end up with mis-aligned structs; the file silently won't
load because field 0's token index lives at offset 4, not 0.

**Proposed clarification:** the binary spec should specify struct *byte
layouts* directly (offsets + sizes), not C++ struct definitions. Or:
include a layout-self-test in `usdview`.

### 10. Token table padding

**Status:** `_WriteTokens` in version `< 0.4.0` writes raw
null-terminated strings; in `>= 0.4.0` it switches to `TfFastCompression`
(single-chunk LZ4 with a 1-byte chunk count prefix). The version gate
isn't called out in the schema doc; a writer that emits compressed tokens
at version 0.0.1 silently fails the reader's null-terminator check.

**Proposed clarification:** the format spec should explicitly version-gate
compression per-section. We've documented this in `usdc.rs`.

## Reception plan

Open a `discussion` on OpenUSD's GitHub against items 1, 2, 4, and 8;
file a PR for items 9 and 10 (documentation only). Defer 3, 5, 6, 7 until
the AOUSD core spec WG opens its next consultation cycle.
