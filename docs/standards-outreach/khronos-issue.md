# KHR_gaussian_splatting reference implementation — submission

**Audience.** Khronos `glTF` repository issue tracker.
**Target action.** Open the issue, link to the conformance suite + blog post, request acknowledgement on the spec's "Reference Implementations" section.
**File via:** `gh issue create --repo KhronosGroup/glTF --title <below> --body-file docs/standards-outreach/khronos-issue.md`

---

## Title

`KHR_gaussian_splatting: SplatForge reference implementation + conformance test suite`

## Body

Hi — SplatForge has a candidate reference implementation for `KHR_gaussian_splatting` ready for the working group's consideration.

### What we're submitting

- **Implementation** in pure Rust: [`crates/splatforge-gltf`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-gltf) — reader + writer for `KHR_gaussian_splatting` and the `KHR_gaussian_splatting_compression_spz` sub-extension. Ships in the public `splatforge` CLI today.
- **Conformance test suite** at [`crates/splatforge-khr-conformance`](https://github.com/montabano1/SplatForge/tree/main/crates/splatforge-khr-conformance):
  - 23 normative-clause tests (Rust integration tests + a `splatforge-khr-validate` binary that emits a JSON per-clause pass/fail report).
  - 10 golden fixture glTF/GLB files (5 valid baselines covering quantized + SH + SPZ; 5 negative cases covering missing-required-prop, accessor-shape mismatches, count-disagreement).
  - All fixtures byte-deterministically regenerated from `splatforge-gltf` via `scripts/generate-fixtures.sh`.
  - CI workflow runs the validator on every PR.
- **Companion documentation:** [conformance.md](https://github.com/montabano1/SplatForge/blob/main/crates/splatforge-khr-conformance/conformance.md) (per-clause coverage), [khr-conformance-submission.md](https://github.com/montabano1/SplatForge/blob/main/docs/khr-conformance-submission.md), [blog post draft](https://github.com/montabano1/SplatForge/blob/main/docs/blog/khr-reference-impl.md).

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
