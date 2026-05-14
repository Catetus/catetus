# SPEC-0012 — OpenUSD Streaming via Payloads + Variant Sets

**Status:** Draft (proposal, not yet implemented)
**Crate (planned):** `splatforge-usd`
**Depends on:** SPEC-0001 (IR), SPEC-0007 (`SF_spatial_streaming_index` glTF extension), SPEC-0011 (USD round-trip)

> This spec is a design proposal for v0.2 work. It has not been validated against a real OpenUSD toolchain. Every USD composition behavior described below (payload load policy, variant selection semantics, asset path resolution) MUST be confirmed against `usdview` or an equivalent renderer before implementation lands. Citations are best-guess starting points.

## Goal

Define how SplatForge delivers **chunked, LOD'd Gaussian Splat assets through native OpenUSD composition primitives** — payloads, references, and variant sets — without inventing a new container format and without requiring viewer support for any SplatForge-specific extension.

This is the OpenUSD counterpart to SPEC-0007 (`SF_spatial_streaming_index` for glTF). Where SPEC-0007 adds a vendor extension on top of glTF, this spec uses USD primitives that any conforming USD renderer already implements.

## Source of truth

* USD payload arcs: `https://openusd.org/release/glossary.html#usdglossary-payload` (verify exact anchor).
* Variant sets: `https://openusd.org/release/glossary.html#usdglossary-variantset` (verify exact anchor).
* Composition arcs overview: `https://openusd.org/release/api/class_usd_stage.html` (best-guess; the canonical doc is the "Composition" chapter in the USD user guide).
* Pixar source: `https://github.com/PixarAnimationStudios/OpenUSD` — `pxr/usd/usd/payloads.h`.

## Background: USD streaming primitives

* **Payload** — a composition arc that is **not loaded by default**. The viewer calls `UsdStage::Load(path)` to bring it in. This is USD's native "lazy load this prim subtree" mechanism. Equivalent in spirit to glTF external buffers + a manifest, but built into the runtime.
* **Reference** — composition arc that **is** loaded eagerly. Cheaper than a payload but doesn't give the viewer a chance to defer.
* **Variant set** — a named switch on a prim; selecting a variant rewrites the prim's child opinions. Variants compose with payloads, so an LOD variant can switch which payload(s) are active.

These three primitives, together, give us the full set of behaviors SPEC-0007 implements with `SF_spatial_streaming_index`.

## Design

### 1. Asset layout

```
scene/
    scene.usda                     # root layer, declares prim hierarchy + variant sets + payload arcs
    chunks/
        chunk_0000.usdc            # one ParticleField3DGaussianSplat per file (LOD 0)
        chunk_0001.usdc
        ...
        chunk_0000.lod1.usdc       # downsampled variant of the same chunk
        chunk_0000.lod2.usdc
    reports/
        optimize.json              # unchanged, per SPEC-0006
```

Each `chunk_NNNN.usdc` defines exactly one `ParticleField3DGaussianSplat` prim per SPEC-0011. The root layer never inlines splat data.

### 2. Root layer structure

```usda
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    customLayerData = {
        string "splatforge:version" = "0.2"
        string "splatforge:ordering" = "morton"
        int "splatforge:chunkCount" = 64
    }
)

def Xform "World"
{
    def Scope "Splats"
    {
        def "chunk_0000" (
            payload = @./chunks/chunk_0000.usdc@</Splat>
            variantSets = "lod"
            variants = {
                string lod = "lod0"
            }
        )
        {
            variantSet "lod" = {
                "lod0" (
                    payload = @./chunks/chunk_0000.usdc@</Splat>
                ) {}
                "lod1" (
                    payload = @./chunks/chunk_0000.lod1.usdc@</Splat>
                ) {}
                "lod2" (
                    payload = @./chunks/chunk_0000.lod2.usdc@</Splat>
                ) {}
                "hidden" {
                    token visibility = "invisible"
                }
            }

            custom float3 splatforge:bboxMin = (-1.0, -1.0, -1.0)
            custom float3 splatforge:bboxMax = ( 1.0,  1.0,  1.0)
            custom int splatforge:splatCount = 12500
            custom int splatforge:loadPriority = 0
            custom string splatforge:checksum = "blake3:abcdef0123..."
        }

        def "chunk_0001" ( ... ) { ... }
    }
}
```

The example above is schematic. The exact USDA syntax for combining a `variantSet` with `payload` arcs MUST be validated against `usdview`.

### 3. Mapping to SPEC-0007

| SPEC-0007 (`SF_spatial_streaming_index`)         | USD equivalent in this spec |
| ------------------------------------------------ | --------------------------- |
| `extensions.SF_spatial_streaming_index.chunks[]` | One child prim per chunk under `/World/Splats`, each with a payload arc. |
| `chunks[i].buffer` + `byteOffset`/`byteLength`   | The payload's `@asset@` path. USD does not expose byte ranges; one chunk = one file. |
| `chunks[i].bbox`                                 | `splatforge:bboxMin` / `splatforge:bboxMax` custom attributes on the chunk prim. |
| `chunks[i].splatCount`                           | `splatforge:splatCount` custom attribute. |
| `chunks[i].checksum`                             | `splatforge:checksum` custom attribute. |
| `chunks[i].loadPriority`                         | `splatforge:loadPriority` custom attribute. The viewer-side scheduler reads it; USD itself has no priority concept. |
| `chunks[i].lod`                                  | Variant selection on the `"lod"` variant set. |
| `ordering` ("morton")                            | `customLayerData["splatforge:ordering"]` on the root layer. |
| `lods[]` array of `{ level, splatFraction }`     | Per-variant `splatforge:splatFraction` custom attribute. |

The bounding box, splat count, checksum, and load priority are not part of the USD core schema, so they live under the `splatforge:` custom-attribute namespace — mirroring the "namespace your additions" pattern from SPEC-0007. A renderer that ignores them gets correct visuals (it just can't prioritize loading).

### 4. LOD variant set scheme

* Every chunk prim carries a `"lod"` variant set with values `lod0`, `lod1`, `lod2`, and `hidden`.
* `lod0` is full quality; `lodN` for N ≥ 1 uses progressively downsampled chunk files produced by SPEC-0006 optimization passes.
* `hidden` selects no payload and sets `visibility = "invisible"`. Used for far-distance LOD or for chunks outside the view frustum on initial load.
* The default variant selection on export is `lod0` for every chunk. Viewers MAY rewrite the selection at runtime via `UsdVariantSet::SetVariantSelection()`.
* All chunks expose the **same** variant names. This is what lets a viewer drive LOD globally with a single selection map.

### 5. Payload-arc semantics

* Every chunk prim's contents (the `ParticleField3DGaussianSplat`) are reached **only** through a payload arc. The root layer never inlines splat attributes.
* On stage open, USD will not load any chunk by default. The viewer (or SplatForge's planned `splatforge-viewer-usd`) is responsible for calling `UsdStage::Load()` on chunks in load-priority order.
* Cancelling a load: the viewer calls `UsdStage::Unload()` on chunks that drop out of view. Memory is reclaimed by USD.
* Failure mode: if a payload's target file is missing, USD reports a composition error. The exporter is responsible for writing all files atomically and verifying file existence before declaring an asset complete.

### 6. Graceful degradation

* A USD renderer with **no** knowledge of `ParticleField3DGaussianSplat` will still parse the stage, follow payloads, find prims of an unknown type, and skip them. No fatal errors. (This is USD's standard behavior for unknown typed prims — verify.)
* A renderer that knows `ParticleField3DGaussianSplat` but does **not** implement priority-driven loading will load all payloads (or follow whatever default load policy is configured) and produce a correct, if non-progressive, render.
* A renderer that understands both will get full progressive streaming.

There is no SplatForge-specific extension required to render the asset; all `splatforge:` custom attributes are optional hints.

### 7. Out-of-scope for v0.2

* Animated LOD transitions (cross-fades between variants). v0.2 hard-switches.
* True byte-range streaming within a single `.usdc`. USD's smallest streaming unit is a payloaded layer.
* Server-side payload negotiation (HTTP range requests, CDN sharding). The asset just sits on a static fileserver.

## File layout (planned, in `splatforge-usd`)

```
crates/splatforge-usd/
    src/
        streaming.rs      # root-layer writer with payload + variantSet emission
        chunk_writer.rs   # one chunk → one .usdc
        checksum.rs       # BLAKE3 over the chunk file (matches SPEC-0007)
    tests/
        streaming_roundtrip.rs
        graceful_degradation.rs
```

## Acceptance tests

```gherkin
Feature: OpenUSD streaming via payloads and variant sets

Scenario: Export chunked USD
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge optimize tiny/basic_binary.ply --chunked --to usd --out scene.usda"
  Then scene.usda exists
  And it references multiple external .usdc files via payload arcs
  And no chunk file is loaded by USD on stage open by default

Scenario: Each chunk has bounding metadata
  Given a chunked USD output
  When I inspect the root layer
  Then every chunk prim carries splatforge:bboxMin and splatforge:bboxMax
  And every chunk prim carries splatforge:splatCount
  And every chunk prim carries splatforge:checksum starting with "blake3:"

Scenario: LOD variant set is present on every chunk
  Given a chunked USD output with 3 LODs
  When I inspect any chunk prim
  Then it has a variantSet named "lod"
  And the variantSet contains "lod0", "lod1", "lod2", and "hidden"
  And the default selection is "lod0"

Scenario: Switching the LOD variant changes the payload target
  Given a chunked USD output
  When I set the "lod" variant on chunk_0000 to "lod1"
  Then the active payload for chunk_0000 resolves to "chunks/chunk_0000.lod1.usdc"

Scenario: Payload arcs defer loading
  Given a chunked USD output
  When I open the stage with default load policy
  Then no ParticleField3DGaussianSplat attributes are resident in memory
  And calling Load on one chunk makes only that chunk's attributes resident

Scenario: Corrupted chunk is detected
  Given a chunked USD output
  When I flip one byte in chunks/chunk_0000.usdc
  Then "splatforge inspect scene.usda" reports a checksum mismatch for chunk_0000

Scenario: Chunk order is deterministic
  Given the same input and config
  When I export twice
  Then chunk filenames are identical
  And chunk file BLAKE3 hashes are identical
  And the root .usda is byte-identical

Scenario: Renderer without splatforge: namespace still renders
  Given a chunked USD output
  When a viewer that ignores splatforge:* custom attributes loads the stage
  Then all chunks load (no progressive streaming)
  And the rendered image is visually equivalent to the full-quality bake
```

## Backwards compatibility

* This spec is additive on top of SPEC-0011. A non-chunked USD export (SPEC-0011) is still valid; this spec just defines what a chunked export looks like.
* The `splatforge:` custom-attribute namespace is the same namespace used in SPEC-0011 for fields with no native USD analog. No new namespaces are introduced.
* No changes to glTF or SPZ paths.

## Open questions

1. **Exact USDA syntax for `payload` inside a `variantSet`.** The example in §2 is schematic. Real USD ASCII syntax may require a different placement of the payload metadata (inside the variant block vs. on the prim spec inside the variant). Needs `usdview` verification.
2. **Default load policy.** `UsdStage::Open()` accepts a load policy (`LoadAll` / `LoadNone`). The spec assumes `LoadNone` is the intended viewer behavior. Should the exporter encode a *recommended* policy somewhere (e.g., as `customLayerData`)? USD has no native field for this.
3. **Variant set inheritance.** If all chunks declare the same `"lod"` variant set, can it be hoisted to `/World/Splats` (the parent `Scope`) and inherited? That would shrink the root layer significantly. Needs verification of USD inheritance semantics for variant sets.
4. **Checksum coverage.** SPEC-0007 hashes the chunk byte range. Here we hash the entire `.usdc` file. Since `.usdc` is a binary format with potentially non-deterministic byte layout across USD versions, we may need to hash a canonical re-serialization instead. Decision deferred.
5. **`.usdz` (zip) streaming.** Payload arcs across zip boundaries are supported in USD but with caveats around in-memory cost. Whether `.usdz` is a viable target for streaming-chunked output is an open question; v0.2 targets loose `.usda` + `.usdc` files only.
6. **Interaction with USD's `instanceable` flag.** If two chunks share identical splat data (rare but possible after optimization), USD instancing could deduplicate them. Out of scope for v0.2 but worth flagging.
7. **Viewer responsibility.** This spec describes the asset layout; it does not specify a USD-aware viewer. SplatForge's current viewer SDK (SPEC-0008) is glTF-only. A `splatforge-viewer-usd` is a separate effort and is not blocked by this spec.

## Change history

| Version | Date       | Author | Notes |
| ------- | ---------- | ------ | ----- |
| 0.1     | 2026-05-13 | Monte  | Initial draft. Not yet validated against a real OpenUSD toolchain. Composition behavior described here MUST be confirmed against `usdview`. |
