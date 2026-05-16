# @splatforge/cesium-qat

A CesiumJS `DataSource` that decodes **SplatForge QAT-PLY** and places the
splat anchors on the globe as a `PointPrimitiveCollection`. Cesium has no
native Gaussian-splat primitive yet — this package is the pragmatic mapping
for geographic apps that want SplatForge scenes alongside their 3D Tiles /
imagery layers.

## Install

```bash
npm install cesium @splatforge/cesium-qat
```

## Usage

```ts
import { Viewer, Cartesian3 } from "cesium";
import { QATPlyDataSource } from "@splatforge/cesium-qat";

const viewer = new Viewer("cesium");
const ds = new QATPlyDataSource("my-splats");
await ds.load(viewer.scene, "/scene.qat.ply", {
  origin: { longitude: -122.4194, latitude: 37.7749, height: 50 },
  scale: 50,
});
viewer.dataSources.add(ds);
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(-122.4194, 37.7749, 1000),
});
```

For apps that drive their own renderer, call `ds.decode(bytes)` to get the
raw decoded buffers (`positions`, `colors`, `anchorFeat`, `offset`) without
any Cesium scene side-effects.

## License

MIT. © SplatForge contributors.
