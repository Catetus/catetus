# KHR_gaussian_splatting reference implementation — submission

**Audience.** Khronos `glTF` repository issue tracker.
**Target action.** Open the issue, link to the conformance suite + blog post, request acknowledgement on the spec's "Reference Implementations" section.
**File via:** `gh issue create --repo KhronosGroup/glTF --title <below> --body-file docs/standards-outreach/khronos-issue.md`

---

## Title

`KHR_gaussian_splatting + KHR_gaussian_splatting_compression_spz: SplatForge reference implementation + conformance test suite`

## Body

Hi — SplatForge has a candidate reference implementation for `KHR_gaussian_splatting` *and* a proposal for a paired SPZ-compression sub-extension (`KHR_gaussian_splatting_compression_spz`) ready for the working group's consideration. The SPZ sub-extension is the second Khronos extension this repository ships, and it is what an SPZ-compressed splat scene looks like once it is embedded inside a self-contained `.glb` — the shape Adobe's Photoshop / Substance 3D export path produces.

### What we're submitting

- **Two-extension reference implementation** in pure Rust:
  [`crates/splatforge-gltf`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-gltf)
  ships reader + writer support for both the base `KHR_gaussian_splatting`
  extension *and* the `KHR_gaussian_splatting_compression_spz`
  sub-extension we are proposing alongside it. The SPZ sub-extension
  embeds an SPZ v2 blob (Niantic's wire format, the one Adobe Photoshop
  exports today) inside a glTF `bufferView` and is fully specified in
  [`docs/standards/KHR_gaussian_splatting_compression_spz.md`](https://github.com/montabano1/SplatForge/blob/main/docs/standards/KHR_gaussian_splatting_compression_spz.md).
  The `splatforge optimize --target glb --compress spz` path produces the
  new extension form directly.
- **Conformance test suite** at [`crates/splatforge-khr-conformance`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-khr-conformance):
  - 28 normative-clause tests (23 base + 5 SPZ-compression) implemented as Rust integration tests plus a `splatforge-khr-validate` binary that emits a JSON per-clause pass/fail report.
  - 13 golden fixture glTF/GLB files: 6 valid baselines (FLOAT, KHR_mesh_quantization, SH, SPZ-stub, SPZ-compressed end-to-end), and 7 negative cases (missing-required-prop, accessor-shape mismatches, count-disagreement, SPZ-missing-extensionsUsed, SPZ-wrong-magic).
  - All fixtures byte-deterministically regenerated from `splatforge-gltf` via `scripts/generate-fixtures.sh`.
  - CI workflow runs the validator on every PR.
- **Companion documentation:** [conformance.md](https://github.com/montabano1/SplatForge/blob/main/crates/splatforge-khr-conformance/conformance.md) (per-clause coverage including the 5 new SPZ-compression clauses), [khr-conformance-submission.md](https://github.com/montabano1/SplatForge/blob/main/docs/khr-conformance-submission.md), [blog post draft](https://github.com/montabano1/SplatForge/blob/main/docs/blog/khr-reference-impl.md), and the SPZ sub-extension spec at [`docs/standards/KHR_gaussian_splatting_compression_spz.md`](https://github.com/montabano1/SplatForge/blob/main/docs/standards/KHR_gaussian_splatting_compression_spz.md).

### One spec clarification ask

The `_COLOR_SH` attribute's accessor layout is ambiguous between two readings:

1. **Strict glTF §3.6.1 reading** — `count = splat_count × 45` with 45 separate float-typed elements per splat.
2. **Per-splat record reading** — `count = splat_count` with a 180-byte stride.

Production tools (including the current SplatForge writer) emit the latter; the strict reading is more glTF-native. The validator currently enforces the strict reading and labels the loose-reading fixtures accordingly. We'd appreciate a working-group decision so the conformance suite can be definitive.

This is the only normative ambiguity we hit while building the corpus.

### Engineering attestations

- **Determinism.** All fixtures are byte-reproducible from the script: same input + same `splatforge-gltf` version produces identical glTF + binary buffer bytes. BLAKE3 hash of the canonical intermediate representation is the cache key.
- **Negative coverage.** The negative fixtures intentionally violate one clause each; the validator must reject each for the correct reason. A passing run on negative fixtures is a failure of the validator, not the spec.
- **No vendor lock-in.** We do not require any vendor-specific extension. `SF_spatial_streaming_index` (Morton-ordered LOD adjuncts) is described as removable; readers that ignore it produce a valid asset.

### What we're not asking for

- Brand placement on the spec page.
- Co-author credit on the spec text. The spec is the work of the working group; we're submitting an implementation against it.

### Contact

- Repo: https://github.com/montabano1/SplatForge
- Maintainer: @montabano1
- Project email: hello@splatforge.dev (TODO: confirm before submission)

Happy to walk through the validator on a working-group call, or land doc-only PRs against the spec repo to resolve the `_COLOR_SH` clarification if that's the preferred path.

---

## Pre-submission checklist

- [ ] Replace `hello@splatforge.dev` with a real distribution-list address (or remove if redundant).
- [ ] Confirm the `KhronosGroup/glTF` issue template requires a tag; if so, add `extension:KHR_gaussian_splatting`.
- [ ] Ensure CI green on `main` so the workflow link in the body resolves to passing.
- [ ] Verify the GitHub PR for `_COLOR_SH` clarification (if we want to lead with one).
- [ ] Cross-post the blog draft once accepted as a contributor PR.
