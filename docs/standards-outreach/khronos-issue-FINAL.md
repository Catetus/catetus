<!--
Polished final draft of the KHR_gaussian_splatting submission for the
KhronosGroup/glTF issue tracker. Submit verbatim with:

  gh issue create \
    --repo KhronosGroup/glTF \
    --title "KHR_gaussian_splatting: SplatForge reference implementation + conformance test suite (RC 63770cc7)" \
    --body-file docs/standards-outreach/khronos-issue-FINAL.md
-->

# KHR_gaussian_splatting: SplatForge reference implementation + conformance test suite

Hi working group — SplatForge has a candidate reference implementation and conformance test suite for `KHR_gaussian_splatting`, audited against the current RC text at commit [`63770cc7`](https://github.com/KhronosGroup/glTF/commit/63770cc70a3709cf101a42cece0bdf602b37e2e7) ("Editorial review (#2567)", 2026-04-15). Submitting now for review ahead of the Feb-2026 ratification window.

## What we're submitting

- **Implementation** in pure Rust: [`crates/splatforge-gltf`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-gltf) — reader + writer with the RC's namespaced attribute semantics (`KHR_gaussian_splatting:ROTATION`, `:SCALE`, `:OPACITY`, `:SH_DEGREE_l_COEF_n`) plus the required `kernel` / `colorSpace` extension fields. Ships in the public `splatforge` CLI.
- **Conformance test suite** at [`crates/splatforge-khr-conformance`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-khr-conformance) (v0.2.0):
  - **23 normative-clause tests** mapped 1:1 to RC paragraphs — covering primitive mode, extension fields, attribute presence/shape, SH degree completeness, bufferView bounds, and namespace hygiene.
  - **10 golden fixture glTF/GLB files** (5 positive, 5 negative). Every fixture is byte-deterministic — re-running `scripts/generate-fixtures.sh` produces the same bytes.
  - **CI workflow** (`.github/workflows/khr-conformance.yml`) runs the full matrix on every PR.
  - **Single binary** (`splatforge-khr-validate`) emits per-clause JSON for any third-party glTF/GLB.
- **Public per-clause × per-fixture report** at [splatforge.com/khr-conformance](https://splatforge.com/khr-conformance), SSR-rendered from the validator output so every push reflects the live verdict.
- **Companion documentation**: [conformance.md](https://github.com/montabano1/SplatForge/blob/main/crates/splatforge-khr-conformance/conformance.md), [CHANGELOG](https://github.com/montabano1/SplatForge/blob/main/crates/splatforge-khr-conformance/CHANGELOG.md) tracking each spec revision we audit against.

## Clauses validated (RC `63770cc7`)

| Clause ID            | Source in RC                                                |
|----------------------|-------------------------------------------------------------|
| `EXT_USED`           | "Extending Mesh Primitives" — extensionsUsed listing        |
| `ASSET_VERSION`      | glTF 2.0 §3.1                                                |
| `PRIM_EXT`           | "Extending Mesh Primitives"                                  |
| `PRIM_MODE_POINTS`   | "Dependencies on glTF" — `mode` MUST be POINTS               |
| `EXT_KERNEL`         | "Kernel" — required string property                          |
| `EXT_COLOR_SPACE`    | "Color Space" — required string property                     |
| `EXT_PROJECTION`     | "Projection" — string, default `"perspective"`               |
| `EXT_SORTING`        | "Sorting Method" — string, default `"cameraDistance"`        |
| `ATTR_POSITION`      | "Attributes" table                                           |
| `ATTR_ROTATION`      | `KHR_gaussian_splatting:ROTATION`                            |
| `ATTR_SCALE`         | `KHR_gaussian_splatting:SCALE`                               |
| `ATTR_OPACITY`       | `KHR_gaussian_splatting:OPACITY`                             |
| `ATTR_SH_DC`         | `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` (always required) |
| `ACC_POSITION`       | "Attributes" table — VEC3 FLOAT/quantized                    |
| `ACC_ROTATION`       | VEC4 FLOAT or normalized signed byte/short                   |
| `ACC_SCALE`          | VEC3 FLOAT or normalized unsigned integer                    |
| `ACC_OPACITY`        | SCALAR FLOAT or normalized unsigned integer                  |
| `ACC_SH_COEF`        | every SH coefficient accessor MUST be VEC3 FLOAT             |
| `ACC_POSITION_MINMAX`| glTF 2.0 §3.6.2.4                                            |
| `SH_DEGREES_FULL`    | "Spherical Harmonics Attributes" — no partial degrees        |
| `ACC_COUNTS_AGREE`   | implicit: all per-splat attributes are per-element           |
| `BUFFERVIEW_BOUNDS`  | glTF 2.0 §3.6.1                                              |
| `ATTRS_KNOWN_ONLY`   | "Extending the Base Extension" — namespace reservation       |

The full 23 × 10 matrix (clause × fixture) is published live at the splatforge.com/khr-conformance link above.

## Questions for the working group

1. **`COLOR_0` fallback semantics.** The RC mentions `COLOR_0` MAY be used to provide diffuse + opacity for non-splat renderers. Should the conformance suite enforce that `COLOR_0`, when present alongside `KHR_gaussian_splatting`, derives from `SH_DEGREE_0_COEF_0` per the equation in "Fallback Behavior"? Today we leave this unchecked — happy to add a clause if that's normative rather than informational.
2. **Color space enumeration.** The schema's `anyOf` permits arbitrary strings beyond `srgb_rec709_display` / `lin_rec709_display`. The conformance suite currently only validates the property is a string. Confirm whether unknown values should be rejected by a conformant reader, or treated as forward-compatible identifiers.
3. **Quantized OPACITY in `KHR_mesh_quantization`.** The attributes table lists "unsigned byte normalized / unsigned short normalized" for OPACITY but does not formally require `KHR_mesh_quantization` to be listed when those component types are used. We currently allow either; recommend the spec text say which is normative.

## Engineering attestations

- **Determinism.** All fixtures are byte-reproducible — `scripts/generate-fixtures.sh` produces identical glTF + binary buffer bytes on any host. A Rust integration test enforces this on every CI run.
- **Negative coverage.** Each negative fixture intentionally violates exactly one clause; the validator must reject each for the correct reason. A passing run on a negative fixture is treated as a regression of the validator.
- **No vendor lock-in.** The implementation requires no SplatForge-specific extension. `SF_spatial_streaming_index` (Morton-ordered LOD adjuncts shipped in our writer) is documented as removable; readers that ignore it produce a fully valid asset.
- **No proprietary fixture data.** Every byte in the corpus is generated from deterministic synthetic coefficients, so the suite is republishable under any license Khronos prefers (currently MIT alongside the SplatForge repo).

## What we're not asking for

- Brand placement on the spec page.
- Co-author credit on the spec text — the spec is the work of the working group; we're submitting an implementation against it.

## Contact

- Repo: https://github.com/montabano1/SplatForge
- Live report: https://splatforge.com/khr-conformance
- Maintainer: [@montabano1](https://github.com/montabano1)

Happy to walk through the validator on a working-group call, or land doc-only PRs against the spec repo to resolve any of the three questions above if that's the preferred path.
