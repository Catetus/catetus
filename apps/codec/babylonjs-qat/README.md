# @splatforge/babylonjs-qat

A Babylon.js loader for the **SplatForge QAT-PLY** quantized Gaussian-splat
container. Decodes the constant-strip, int8, and int4 paths of the SplatForge
codec into a Babylon `PointsCloudSystem` mesh.

## Install

```bash
npm install @babylonjs/core @splatforge/babylonjs-qat
```

## Usage

```ts
import { Engine, Scene, ArcRotateCamera, Vector3 } from "@babylonjs/core";
import { QATPlyLoader } from "@splatforge/babylonjs-qat";

const engine = new Engine(canvas, true);
const scene = new Scene(engine);
new ArcRotateCamera("cam", Math.PI / 4, Math.PI / 3, 5, Vector3.Zero(), scene)
  .attachControl(canvas, true);

const buf = new Uint8Array(await (await fetch("/scene.qat.ply")).arrayBuffer());
const mesh = await new QATPlyLoader().loadIntoScene(scene, buf, "splats");

engine.runRenderLoop(() => scene.render());
```

For raw access to decoded buffers (e.g. to feed your own Gaussian-splat
material instead of `PointsCloudSystem`), call `loader.parse(buf)` and use
the returned `positions` / `colors` / `anchorFeat` / `offset` typed arrays.

## License

MIT. © SplatForge contributors.
