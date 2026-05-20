# SPEC-0007 — Spatial Streaming Index

**Status:** Implemented (Phase 1, basic)
**Crate:** `catetus-gltf`

## Goal

Define a standards-aligned progressive-delivery convention using glTF external buffers and an optional vendor extension. **No proprietary `.sfz` package format.**

## Vendor extension: `CT_spatial_streaming_index`

Attached to the glTF root:

```jsonc
"extensions": {
  "CT_spatial_streaming_index": {
    "ordering": "morton",
    "chunkCount": 64,
    "chunks": [
      {
        "buffer": <buffer-index>,
        "byteOffset": 0,
        "byteLength": 81920,
        "splatCount": 12500,
        "bbox": { "min": [..], "max": [..] },
        "lod": 0,
        "checksum": "blake3:abcdef0123...",
        "loadPriority": 0
      }
    ],
    "lods": [
      { "level": 0, "splatFraction": 1.0 },
      { "level": 1, "splatFraction": 0.25 }
    ]
  }
}
```

## Package layout

```
scene/
  scene.gltf
  buffers/
    root.bin            <- mesh metadata, LOD 0 root chunk
    lod0_0001.bin
    lod0_0002.bin
    lod1_0001.bin
  reports/
    optimize.json
```

The asset degrades to baseline glTF + `KHR_gaussian_splatting` behavior when a viewer ignores the vendor extension: every external buffer is still a valid glTF binary buffer.

## Acceptance tests

```gherkin
Feature: Spatial streaming index

Scenario: Export chunked glTF
  Given fixture "tiny/basic_binary.ply"
  When I run "catetus optimize ... --chunked --out scene.gltf"
  Then scene.gltf references multiple external buffers
  And each chunk has bounding metadata
  And checksums validate

Scenario: Corrupted chunk is detected
  Given a chunked glTF output
  When I flip one byte in a chunk
  Then "catetus inspect" reports checksum failure

Scenario: Chunk order is deterministic
  Given the same input and config
  When I optimize twice
  Then chunk order and chunk hashes are identical
```
