# SPEC-0010 — Viewer Parity Matrix

**Status:** Implemented (Phase 2)
**Package:** `tests/visual`

## Goal

Measure whether the same asset renders consistently across viewer / runtime paths. This is a moat feature: the matrix is the asset, not the renderer.

## Matrix v1

| Path | Browser | Renderer | Required |
| ---- | ------- | -------- | -------- |
| `chrome-webgpu`  | Chromium | WebGPU  | yes |
| `chrome-webgl2`  | Chromium | WebGL2  | yes |
| `firefox-webgl2` | Firefox  | WebGL2  | optional |
| `webkit-webgl2`  | WebKit   | WebGL2  | optional |
| `threejs-adapter`   | Chromium | three.js + KHR_gaussian_splatting loader | future |
| `babylon-adapter`   | Chromium | Babylon.js | future |

## Report

```json
{
  "asset": "warehouse_scan",
  "matrix": {
    "chrome-webgpu": { "visualScore": 0.98, "fps": 61, "memoryMb": 421 },
    "chrome-webgl2": { "visualScore": 0.94, "fps": 47, "memoryMb": 412 },
    "webkit-webgl2": { "visualScore": 0.72, "fps": 21,
                       "warnings": ["opacity_sorting_artifacts"] }
  }
}
```

## Acceptance tests

```gherkin
Feature: Viewer parity

Scenario: Compare WebGPU and WebGL2 output
  Given an optimized glTF fixture
  When I run viewer parity benchmark
  Then the report includes WebGPU and WebGL2 visual scores

Scenario: Viewer degradation is reported
  Given a fixture known to degrade in WebGL2
  When I run viewer parity benchmark
  Then the report includes a degradation warning
```
