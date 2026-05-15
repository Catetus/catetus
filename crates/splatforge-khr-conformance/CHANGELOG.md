# Changelog

## 0.2.0 — 2026-05-15

Audited against the KHR_gaussian_splatting Release Candidate at
KhronosGroup/glTF commit
[`63770cc70a3709cf101a42cece0bdf602b37e2e7`](https://github.com/KhronosGroup/glTF/commit/63770cc70a3709cf101a42cece0bdf602b37e2e7)
(2026-04-15, "Editorial review (#2567)").

### Added (new clauses)

- `PRIM_MODE_POINTS` — RC §"Dependencies on glTF": `mesh.primitive.mode` MUST be `POINTS` (0).
- `EXT_KERNEL` — RC §"Extending Mesh Primitives": the extension object MUST declare a string `kernel`.
- `EXT_COLOR_SPACE` — RC §"Color Space": the extension object MUST declare a string `colorSpace`.
- `EXT_PROJECTION` — RC §"Projection": if `projection` is present it MUST be a string (defaults to `"perspective"`).
- `EXT_SORTING` — RC §"Sorting Method": if `sortingMethod` is present it MUST be a string (defaults to `"cameraDistance"`).
- `ATTR_SH_DC` — RC §"Spherical Harmonics Attributes": `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` is always required.
- `ACC_SH_COEF` — every `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n` accessor MUST be VEC3 FLOAT.
- `SH_DEGREES_FULL` — replaces the old `SH_DEGREE_RANGE`. Each declared SH degree MUST provide its full (2l+1) coefficient set, and using degree l requires degrees 0..l-1 to also be fully defined.

### Changed

- `ATTR_ROTATION`, `ATTR_SCALE`, `ATTR_OPACITY` now expect the RC's
  KHR-namespaced semantics (`KHR_gaussian_splatting:ROTATION` etc.) and
  search the **primitive's** attributes object (not the extension's
  attributes object — that no longer exists in the RC).
- `ACC_ROTATION` now accepts FLOAT, normalized BYTE (5120), and normalized
  SHORT (5122) per RC quaternion-quantization rules.
- `ATTRS_KNOWN_ONLY` only flags unrecognised `KHR_gaussian_splatting:*`
  namespaced keys; non-namespaced attributes (e.g. `COLOR_0` fallback)
  are no longer rejected.

### Removed

- `EXT_REQUIRED` — the RC says the extension SHOULD only land in
  `extensionsRequired` when an extending compression extension forces it,
  so listing it is no longer normative for the base extension.
- `ATTR_COLOR_DC`, `ACC_COLOR_DC`, `ACC_COLOR_SH`, `SH_DEGREE_RANGE` — the
  RC replaces the monolithic `_COLOR_DC`/`_COLOR_SH` attributes with one
  VEC3 FLOAT accessor per SH coefficient and removes the `shDegree`
  field entirely.
- `SPZ_DECLARED`, `SPZ_CONSISTENT` — the SPZ compression sub-extension is
  tracked separately and not part of the base KHR_gaussian_splatting RC
  any longer. A future crate version will reintroduce SPZ clauses once
  Khronos publishes the compression extension's RC text.

### Fixtures

All ten fixtures were regenerated from a new synthetic JSON pipeline that
does not depend on the legacy SplatForge writer (the writer still emits
pre-RC attribute names; updating it is tracked separately). The corpus
keeps the 5 positive + 5 negative split:

- `01_valid_baseline.glb` / `02_valid_baseline.gltf` — minimal valid 4-splat scene, FLOAT accessors only.
- `03_valid_quantized.glb` — `KHR_mesh_quantization` integer ROTATION / SCALE / OPACITY accessors.
- `04_valid_with_sh.glb` — adds SH degree-1 coefficient accessors (degrees 0+1 fully defined).
- `05_valid_default_methods.glb` — omits `projection` + `sortingMethod` to exercise the RC defaults.
- `06_invalid_missing_ext_used.glb` — drops `KHR_gaussian_splatting` from `extensionsUsed`.
- `07_invalid_no_rotation.gltf` — removes `KHR_gaussian_splatting:ROTATION`.
- `08_invalid_rotation_vec3.gltf` — ROTATION accessor is VEC3 instead of VEC4.
- `09_invalid_position_no_minmax.gltf` — POSITION accessor missing min/max.
- `10_invalid_count_mismatch.gltf` — per-splat counts disagree.

## 0.1.0 — Initial release (pre-RC, 2025)

23 clauses against the pre-RC draft, including the old
`_ROTATION`/`_SCALE`/`_OPACITY`/`_COLOR_DC`/`_COLOR_SH` attribute names
and the SPZ companion clauses.
