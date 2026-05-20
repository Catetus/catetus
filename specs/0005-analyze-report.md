# SPEC-0005 — Analyze Report

**Status:** Implemented (Phase 1)
**Crate:** `catetus-core` + `catetus-cli`

## Goal

Generate a deterministic JSON report for any supported input.

## Report schema (v1)

```jsonc
{
  "schemaVersion": "1",
  "format": "ply" | "spz" | "gltf",
  "splatCount": 1234,
  "fileSize": 102400,
  "boundingBox": {
    "min": [x, y, z],
    "max": [x, y, z]
  },
  "coordinateSystem": { "up": "Y", "handedness": "right" },
  "attributes": {
    "position": true, "rotation": true, "scale": true,
    "opacity": true, "color_dc": true, "sh_rest": false
  },
  "opacityDistribution": {
    "min": 0.0, "max": 1.0, "mean": 0.62, "median": 0.71
  },
  "scaleDistribution": {
    "min": [..], "max": [..], "mean": [..]
  },
  "shDegree": 0,
  "estimatedMemory": { "ramMb": 64, "vramMb": 110 },
  "warnings": [
    { "code": "floater_cluster_detected", "message": "...", "severity": "warn" }
  ],
  "recommendations": [
    { "preset": "web-mobile", "rationale": "size > 100MB and target=browser" }
  ],
  "hash": "blake3:..."
}
```

## Requirements

* Byte-identical output for the same input (no timestamps; no float locale jitter).
* `--pretty` controls whitespace but not key order.
* JSON keys serialized in stable lexical order.
* Floats serialized via `ryu` so `1.0 -> "1.0"`, `0.1 -> "0.1"`.

## Acceptance tests

```gherkin
Feature: Analyze report

Scenario: Generate deterministic JSON
  Given fixture "tiny/basic_binary.ply"
  When I run analyze twice
  Then the JSON reports are byte-identical

Scenario: Detect suspicious floaters
  Given fixture "invalid/floater_cluster.ply"
  When I analyze it
  Then warnings include "floater_cluster_detected"

Scenario: Recommend web-mobile optimization
  Given a large raw PLY fixture
  When I analyze it
  Then recommendations include "web-mobile"
```
