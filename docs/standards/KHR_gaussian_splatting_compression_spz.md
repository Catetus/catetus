# `KHR_gaussian_splatting_compression_spz`

**Status:** Draft / Release-Candidate (SplatForge implementation).
**Depends on:** [`KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) (base extension).
**Maintainer:** SplatForge (`crates/splatforge-gltf`).
**Last update:** 2026-05-15.

This extension lets a glTF 2.0 asset carry a Gaussian-splat point cloud as a
single SPZ-compressed binary blob embedded in a `bufferView`, instead of as
five-to-six separate per-attribute accessors. It is the second Khronos
extension SplatForge ships (alongside the lossless `KHR_gaussian_splatting`
base), and it is the explicit ask in the Adobe / SPZ partnership memo
(`docs/partnerships/adobe-spz-memo.md`): SPZ-compressed splats embeddable
inline in a `.glb` so a Creative Cloud export path can produce a single,
self-contained file.

## 1. Motivation

`KHR_gaussian_splatting` is intentionally a *lossless* container — every
splat attribute is a separate glTF accessor with its own `componentType`,
which keeps the on-the-wire shape compatible with the existing glTF tool
ecosystem (validators, glTF transform pipelines, Babylon.js readers). The
trade-off is that the wire size for a multi-million-splat scene is large
enough that real production pipelines (Photoshop, Substance 3D, Niantic
capture) already compress to the SPZ format before transport.

SPZ already has a published wire format (`specs/0003-spz-io.md`,
`crates/splatforge-spz`), a vendor-extension byte (used by Adobe), and a
flag byte (bit 0 reserved for SwVQ). What SPZ lacks is a glTF wrapping
that a Khronos-aware viewer can decode as a regular `.glb`. This
extension provides exactly that wrapping.

## 2. Scope

A primitive that declares both `KHR_gaussian_splatting` and
`KHR_gaussian_splatting_compression_spz` MUST be decoded by reading the
SPZ blob referenced by `bufferView`; the per-attribute accessors declared
on the base extension MAY be empty (`count = 0`) and SHOULD be ignored by
a reader that understands the SPZ extension. A reader that does NOT
understand this extension MAY fall back to the empty base-extension
accessors and render an empty scene; therefore implementations SHOULD
list this extension in `extensionsRequired`.

## 3. Extension JSON shape

### 3.1 Root

```json
{
  "extensionsUsed":     ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz"],
  "extensionsRequired": ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz"]
}
```

Both base + SPZ MUST be listed in `extensionsUsed`. Both SHOULD be listed
in `extensionsRequired` because the SPZ-only path produces no visible
splats for a reader that doesn't understand SPZ.

### 3.2 Primitive

A `mesh.primitives[i]` that uses the extension MUST declare both
extensions on the primitive:

```json
{
  "extensions": {
    "KHR_gaussian_splatting": {
      "attributes": {
        "POSITION":  0, "_ROTATION": 1, "_SCALE":  2,
        "_OPACITY":  3, "_COLOR_DC": 4
      },
      "shDegree": 1
    },
    "KHR_gaussian_splatting_compression_spz": {
      "version":     2,
      "bufferView":  5,
      "splatCount":  150000
    }
  }
}
```

Field definitions:

- **`version`** (integer, required) — SPZ wire-format version. Currently
  `2` (matches `SPZ_VERSION` in `crates/splatforge-spz`). Readers MUST
  reject other values rather than silently mis-decode.
- **`bufferView`** (integer, required) — index into `bufferViews[]` of the
  SPZ blob. The bufferView's `byteLength` MUST equal the entire SPZ file
  length (16-byte header + optional SwVQ extension + zlib payload). The
  underlying buffer MAY be the GLB-embedded buffer (i.e. buffer 0 with
  no `uri`) or an external `.bin`.
- **`splatCount`** (integer, optional but recommended) — splat count
  decoded from the SPZ header. When present it MUST equal the count
  decoded from the SPZ blob; a mismatch is a validator failure.

When this extension is present, the base-extension accessors MAY have
`count = 0` (empty bufferViews). A writer that wishes to keep a
fallback for non-SPZ-aware readers MAY populate them normally; the
extension is silent on that choice. SplatForge's writer emits
zero-count accessors so the wire size stays close to the SPZ blob size.

## 4. Buffer layout

Inside the `bufferView` are exactly the bytes of one SPZ v2 file:

```
offset | length | contents
-------+--------+------------------------------------------------------
0      | 4      | magic = 0x5053_4e47  ("SNPS" little-endian, "SPZ" + flags)
4      | 4      | version = 2 (u32 LE)
8      | 4      | splat_count (u32 LE)
12     | 1      | sh_degree (u8, 0 or 1)
13     | 1      | fractional_bits (u8, typically 12)
14     | 1      | flags (u8; bit 0 = SwVQ chunk follows)
15     | 1      | reserved (u8, MUST be 0)
[16]   | [...]  | optional SwVQ chunk (u32 LE payload_len + payload_len bytes)
[...]  | rest   | zlib-compressed per-splat payload
```

The blob inside the bufferView is byte-identical to what
`splatforge-spz::encode_spz` returns, and what `splatforge-spz::read_spz`
accepts. No padding, no length prefix added on top of the SPZ bytes.

## 5. Conformance clauses

The validator (`crates/splatforge-khr-conformance`) implements five
clauses scoped to this extension on top of the two SPZ clauses defined
in the base extension's conformance set (`SPZ_DECLARED`,
`SPZ_CONSISTENT`):

| ID                 | Requirement                                                                                                   |
|--------------------|---------------------------------------------------------------------------------------------------------------|
| `SPZ_EXT_PRESENT`  | A primitive that declares the SPZ extension MUST list `KHR_gaussian_splatting_compression_spz` in `extensions`. |
| `SPZ_VERSION`      | `extensions.KHR_gaussian_splatting_compression_spz.version` MUST be `2`.                                     |
| `SPZ_BUFFERVIEW`   | `bufferView` MUST be an integer index in range of `bufferViews[]`, and the referenced bufferView MUST exist within its buffer's `byteLength`. |
| `SPZ_BLOB_MAGIC`   | The bytes referenced by `bufferView` MUST begin with the four-byte SPZ magic `0x5053_4e47` ("SNPS").          |
| `SPZ_DECODED_COUNT`| The SPZ header's `splat_count` MUST equal the primitive's `_OPACITY` accessor count (or, when the base accessors are empty, MUST match the extension's `splatCount` field when present). |

Together with the two preexisting SPZ clauses, that is seven SPZ-related
checks the validator runs on every asset.

## 6. Writer behavior (SplatForge)

`splatforge-gltf` exposes the extension via `WriteOpts.compress`:

```rust
let opts = WriteOpts {
    compress: Some(SpzVariant::V2),
    ..Default::default()
};
write_glb(&scene, path, &opts)?;
```

When `compress` is `Some(SpzVariant::V2)`:

1. The scene is encoded with `splatforge-spz::encode_spz` to produce the
   blob.
2. The GLB's BIN chunk consists of (a) zero-length placeholder accessors
   for POSITION/_ROTATION/_SCALE/_OPACITY/_COLOR_DC so the base
   extension's required attributes are still present, plus (b) the SPZ
   blob itself.
3. The JSON declares both extensions in `extensionsUsed` and
   `extensionsRequired`, and the primitive declares both on its
   `extensions` object.

The output is a single self-contained `.glb` file — same shape Adobe
would ship out of a Photoshop "Export → optimize for web" path.

## 7. Reader behavior

`splatforge-gltf::read_glb` detects the SPZ extension on the primitive,
slices the SPZ blob out of buffer 0 using the declared `bufferView`,
and dispatches to `splatforge-spz::read_spz_bytes`. The result is a
`SplatScene` identical to one produced by `read_spz` against a
standalone `.spz` file.

## 8. Relationship to the base extension

This extension is **strictly additive**. An asset that declares it MUST
also declare `KHR_gaussian_splatting`. A reader that understands the
base extension but not this one will see a primitive with empty
accessors (or fall back to the base attributes if the writer chose to
populate them); that is the documented degradation path.

The two extensions are versioned independently. The base extension uses
the `version` field on the primitive-level KHR block; this extension
uses the `version` field on its own primitive-level block.

## 9. Open questions for the Khronos working group

- **Should `splatCount` in the extension block be required?** The base
  extension already derives splat count from `_OPACITY.count`. When
  base accessors are empty the SPZ block is the only place that count
  lives; a required `splatCount` makes that contract explicit.
- **Should fallback accessor population be normative?** SplatForge ships
  empty accessors; a "fallback always populated" rule would let
  non-SPZ readers always render *something*, at the cost of doubling
  the wire size (since the SPZ blob is the whole point of the extension).
- **SwVQ chunk visibility.** The SwVQ extension chunk (SPZ flag bit 0)
  is part of the SPZ wire format but is opaque to a Khronos-aware
  reader. We propose treating it as an SPZ-internal detail rather than
  a separate glTF extension, since the SwVQ payload is consumed by the
  SPZ codec itself.

---

*This document, the writer in `crates/splatforge-gltf`, the validator
clauses in `crates/splatforge-khr-conformance`, and the fixture corpus
together constitute SplatForge's reference implementation of the
extension. See `docs/standards-outreach/khronos-issue.md` for the
upstream submission story.*
