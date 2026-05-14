# SPEC-0004 — glTF KHR Gaussian Splatting I/O

**Status:** Implemented (Phase 1)
**Crate:** `splatforge-gltf`

## Goal

Make glTF 2.0 + `KHR_gaussian_splatting` the primary runtime interchange target.

## Extension snapshot

The output asset declares:

```json
"extensionsUsed":     ["KHR_gaussian_splatting", "SF_spatial_streaming_index"],
"extensionsRequired": ["KHR_gaussian_splatting"]
```

Each `mesh.primitives[i]` carries the extension:

```json
"extensions": {
  "KHR_gaussian_splatting": {
    "attributes": {
      "POSITION":   <accessor>,
      "_ROTATION":  <accessor>,   // vec4 quaternion
      "_SCALE":     <accessor>,   // vec3
      "_OPACITY":   <accessor>,   // scalar
      "_COLOR_DC":  <accessor>,   // vec3 sRGB linear
      "_COLOR_SH":  <accessor>?   // optional, length = 45 floats per splat
    },
    "shDegree": 0
  }
}
```

The optional `SF_spatial_streaming_index` extension is defined in SPEC-0007.

## Acceptance tests

```gherkin
Feature: glTF KHR Gaussian Splatting I/O

Scenario: Export PLY to glTF KHR
  Given fixture "tiny/basic_binary.ply"
  When I run "splatforge convert tiny/basic_binary.ply --to gltf --out scene.gltf"
  Then scene.gltf exists
  And the glTF declares the Gaussian Splatting extension
  And "splatforge inspect scene.gltf" succeeds

Scenario: Import glTF KHR
  Given fixture "tiny/basic_khr.gltf"
  When I run "splatforge analyze tiny/basic_khr.gltf"
  Then the command exits 0
  And the report says format is "gltf"

Scenario: Unsupported extension version fails clearly
  Given fixture "invalid/unsupported_khr_version.gltf"
  When I inspect it
  Then the command exits non-zero
  And stderr includes "unsupported Gaussian Splatting extension version"
```
