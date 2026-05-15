# Changelog

## 0.2.0 — 2026-05-15

Writer + reader aligned with the `KHR_gaussian_splatting` Release Candidate
at KhronosGroup/glTF commit
[`63770cc70a3709cf101a42cece0bdf602b37e2e7`](https://github.com/KhronosGroup/glTF/commit/63770cc70a3709cf101a42cece0bdf602b37e2e7)
(2026-04-15, "Editorial review (#2567)").

### Changed

- Attribute keys are emitted at primitive-top-level (`mesh.primitive.attributes`)
  with the RC namespace prefix: `POSITION`, `KHR_gaussian_splatting:ROTATION`,
  `:SCALE`, `:OPACITY`, `:SH_DEGREE_0_COEF_0`, ... The legacy
  `_ROTATION` / `_SCALE` / `_OPACITY` / `_COLOR_DC` / `_COLOR_SH` names live
  inside the extension's own `attributes` object only when
  `WriteOpts::spec_version == SpecVersion::Pre2026`.
- Primitives carrying the extension now set `mode = 0` (POINTS).
- The extension object on every primitive carries the required `kernel` and
  `colorSpace` strings plus optional `projection` and `sortingMethod`.
- Spherical-harmonic coefficients are emitted as one VEC3 FLOAT accessor per
  coefficient (RC layout), replacing the old SCALAR-FLOAT-count-`45*N`
  layout (pre-RC layout, still emitted under `SpecVersion::Pre2026`).
- New `WriteOpts::spec_version` field (defaults to
  `SpecVersion::RcMay2026`) selects between the RC and pre-RC wire format
  for round-tripping legacy assets.
- New `WriteOpts::quantize_rotation` field, only meaningful under
  `SpecVersion::RcMay2026 + quantize`, emits ROTATION as normalized
  signed-SHORT under `KHR_mesh_quantization`.

### Reader

- Sniffs for namespaced attribute keys first, falls back to the legacy
  underscore-prefixed names so pre-RC GLBs round-trip without flags.
- Transparently decodes `KHR_gaussian_splatting_compression_spz` blobs
  (unchanged from 0.1.x).
