# @catetus/three-qat

A Three.js loader for **Catetus QAT-PLY**, a quantized PLY container for
3D Gaussian-splat scenes. Decodes the constant-strip, int8, and int4 paths
specified by the Catetus codec.

## Install

```bash
npm install three @catetus/three-qat
```

`three` is a peer dependency (>=0.160).

## Usage

```ts
import { Scene, PerspectiveCamera, WebGLRenderer } from "three";
import { QATPlyLoader } from "@catetus/three-qat";

const scene = new Scene();
const camera = new PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
camera.position.set(3, 2, 4);
const renderer = new WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const result = await new QATPlyLoader().loadAsync("/scene.qat.ply");
scene.add(result.points);
renderer.render(scene, camera);
```

`result.geometry` is a `BufferGeometry` with `position` (and `color` if the
PLY carries f_dc_{0,1,2}). For projects that already ship a Gaussian-splat
material, use `result.geometry` directly and skip the bundled `Points` object.

## Format support

| Path                                              | Status |
|---------------------------------------------------|--------|
| `comment constant_field <name> <float-as-hex>`    | yes    |
| `comment quantized_field <name> int8 channels=C scale_b64=…` | yes    |
| `comment quantized_field <name> int4 channels=C packed_per_byte=2 scale_kind=per_anchor` | yes |
| Plain binary little-endian PLY vertices           | yes    |
| ASCII / big-endian PLY                            | no     |

See [the QAT-PLY spec](https://github.com/Catetus/catetus/tree/main/specs) for the on-disk byte layout.

## License

MIT. © Catetus contributors.
