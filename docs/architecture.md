# SplatForge architecture

## Layered overview

```
+--------------------------------------------------------------+
|                       splatforge-cli                         |
|   analyze | inspect | convert | optimize | preview | diff    |
+----------------+------------------+--------------------------+
                 |                  |
                 v                  v
        splatforge-optimize    splatforge-bench
                 |
                 v
  +-----------------------------------------+
  |              SplatIR (core)             |
  |   Splat, SplatScene, CoordinateSystem   |
  +-----------------------------------------+
       ^             ^             ^
       |             |             |
  splatforge-ply  splatforge-spz  splatforge-gltf
```

## SplatIR

The single source of truth for splat data inside the pipeline. All importers convert to IR; all optimizers operate on IR; all exporters convert from IR.

IR is deterministic: serialization, hashing, and ordering are stable across runs and platforms.

## Optimization pipeline

A `Pipeline` is an ordered list of `Pass` impls. Passes:

* receive `&mut SplatScene` plus a `PassContext` carrying logger + RNG seed
* return `PassStats { removed, modified, duration_ms, notes }`
* must be deterministic given the same input + config + seed

Presets are named `Pipeline` configurations. See SPEC-0006.

## glTF output

Always glTF 2.0 + `KHR_gaussian_splatting`. Optional `SF_spatial_streaming_index` extension for Morton-ordered LOD streaming. The folder layout (`scene.gltf` + `buffers/*.bin`) is a **deployment layout**, not a new format: removing the vendor extension still leaves a valid glTF.

## Viewer

TypeScript + WebGPU primary, WebGL2 fallback. The renderer interface is the same — only the backend changes. Deterministic camera-path mode powers SPEC-0009 / SPEC-0010 tests.
