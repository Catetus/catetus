# SPEC-0008 — Viewer SDK

**Status:** Implemented (Phase 2)
**Package:** `@catetus/viewer`

## Goal

Render optimized glTF / SPZ splat assets in a browser with progressive loading and deterministic test mode.

## Surface

```ts
import { CatetusViewer } from '@catetus/viewer';

const viewer = new CatetusViewer({
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  src: '/assets/scene/scene.gltf',
  budget: 'web-mobile',     // optional preset
  renderer: 'auto',         // 'webgpu' | 'webgl2' | 'auto'
  cameraPath: 'orbit',      // for deterministic tests
});

viewer.on('firstRender', () => console.log('ready'));
await viewer.load();
```

## Events

* `loadStart`
* `manifestLoaded`
* `chunkLoaded(chunkIndex, byteLength)`
* `firstRender`
* `qualityChanged(level)`
* `complete`
* `error(code, message)`

## Renderer strategy

* **WebGPU** primary. Single quad per splat via instanced draw, scissor-less alpha-blended pass, sorted back-to-front per frame.
* **WebGL2** fallback. Same algorithm; uses `EXT_disjoint_timer_query` for stats.
* **Headless deterministic camera-path mode** with fixed seed and fixed clock. Used by SPEC-0009 and SPEC-0010 tests.

## Acceptance tests

```gherkin
Feature: Viewer SDK

Scenario: Load glTF splat asset
  Given a valid optimized glTF fixture served over HTTP
  When the viewer loads it
  Then it emits loadStart
  And it requests the glTF file
  And it eventually emits firstRender

Scenario: First render before full load
  Given a multi-chunk glTF fixture
  When the viewer loads it
  Then firstRender fires before complete

Scenario: Missing chunk emits error
  Given a glTF fixture with a missing external buffer
  When the viewer loads it
  Then it emits error with code "chunk_not_found"
```
