# @catetus/glb-polyfill

Pure-TS decoder for the Catetus `SF_*` glTF extensions and the
`.glb.v5tail` / `.sog.v5tail` residual sidecars. The polyfill turns a
Catetus GLB (or any 3DGS-PLY-shaped scene plus a `.v5tail` sidecar) into
a normalized per-splat `Float32Array` representation — positions, rotations,
scales, opacities, DC color, and SH-rest — without bundling the Catetus
viewer or talking to a GPU. It is the only third-party path for reading
Catetus-optimized assets end-to-end.

## Install

```sh
npm install @catetus/glb-polyfill
```

> **Pre-release note.** The package is not yet on the npm registry. Until the
> `v5.2-public` release ships you must consume it via `npm link` (or a local
> workspace install) from a checkout of the Catetus repo. Publishing is
> tracked under `v5.2-public`.

Runtime dependencies: [`fzstd`](https://www.npmjs.com/package/fzstd) (~7 kB
gz, pure JS). No DOM, no WebGL/WebGPU, no native bindings.

## Quickstart

Decode a `.glb` to flat splat arrays in three lines (no sidecar, no manual
chunk splitting):

```ts
import { decodeGlb } from '@catetus/glb-polyfill';

const glbBytes = new Uint8Array(await (await fetch('/scene.glb')).arrayBuffer());
const scene = decodeGlb(glbBytes);
console.log(scene.count, 'splats,', scene.shDegree, 'SH-rest degree');
// scene.positions : Float32Array(count*3)
// scene.dcRaw     : Float32Array(count*3)  raw SH DC (canonical public name)
// scene.opacities : Float32Array(count)    LINEAR, [0, 1]
// scene.scales    : Float32Array(count*3)  LINEAR per-axis radii
```

When the GLB carries a `.glb.shpal` palette sidecar, use
`decodeSFExtensions` directly and pass a sidecar map:

```ts
import { decodeSFExtensions } from '@catetus/glb-polyfill';
import { splitGlb } from './my-glb-splitter.js'; // your 12-line JSON+BIN splitter

const glbBytes = new Uint8Array(await (await fetch('/scene.glb')).arrayBuffer());
const { json, bin } = splitGlb(glbBytes);
const sidecars = {
  'scene.glb.shpal': await (await fetch('/scene.glb.shpal')).arrayBuffer(),
};
const scene = decodeSFExtensions(json, bin, sidecars);
```

A 30-line `splitGlb` helper is shipped in `examples/raw-canvas/main.ts`.

## API reference

All exports live in
[`src/index.ts`](./src/index.ts). The exported surface:

### `decodeGlb(bytes, zstdDecompress?) → DecodedSplats`

Ergonomic one-shot wrapper. Takes raw GLB bytes, splits the JSON + BIN
chunks internally, and returns a [`DecodedSplats`](#output-schema-decodedsplats).
Use this when you don't have (and don't need) a `.glb.shpal` palette
sidecar. Throws if the GLB declares `SF_gaussian_splatting_palette` —
that path requires the sidecar map exposed by `decodeSFExtensions`.

```ts
import { decodeGlb } from '@catetus/glb-polyfill';
const scene = decodeGlb(new Uint8Array(await (await fetch('/scene.glb')).arrayBuffer()));
```

### `decodeSFExtensions(gltfJson, binBuffer, sidecars?, zstdDecompress?) → DecodedSplats`

Lower-level entry point. Consumes a pre-split GLB (parsed JSON chunk + raw BIN
chunk bytes) plus a `{ uri → bytes }` map of sidecars referenced from the
JSON. Returns a normalized [`DecodedSplats`](#output-schema-decodedsplats).

```ts
import { decodeSFExtensions } from '@catetus/glb-polyfill';
const scene = decodeSFExtensions(json, bin, { 'scene.glb.shpal': palBytes });
```

The optional fourth argument lets you swap `fzstd.decompress` for Node 21+'s
built-in `zlib.zstdDecompressSync`, a WASM zstd, or anything else with the
`(Uint8Array) → Uint8Array` signature.

### `decodeV5TailBytes(bytes, zstdDecompress?) → DecodedV5Tail`

Parses a `.glb.v5tail` / `.sog.v5tail` residual sidecar (magic `SFV51TAL`)
and returns the per-group residuals already de-Morton-permuted into
ascending-SF order on the selected subset.

```ts
import { decodeV5TailBytes } from '@catetus/glb-polyfill';
const tailBytes = new Uint8Array(await (await fetch('/scene.glb.v5tail')).arrayBuffer());
const tail = decodeV5TailBytes(tailBytes);
console.log(tail.header.kSelected, 'of', tail.header.nSplats, 'splats touched');
```

### `applyV5TailToScene(scene, decoded) → number`

Applies a decoded V5 tail to a splat scene in place (mutates the typed
arrays). The scene shape is whatever you decoded — Catetus GLB, SOG,
plain 3DGS PLY — as long as you fill an [`ApplyTargetScene`](#applytargetscene).
Returns the number of splats actually updated.

```ts
import { decodeSFExtensions, decodeV5TailBytes, applyV5TailToScene } from '@catetus/glb-polyfill';

const base = decodeSFExtensions(json, bin, sidecars);
const tail = decodeV5TailBytes(tailBytes);

// `base.scales` / `base.opacities` are LINEAR (the polyfill eagerly applies
// `exp` / `sigmoid` when `SF_log_quant_attrs` is set), which is exactly what
// the apply path wants. No conditional needed.
applyV5TailToScene(
  {
    positions: base.positions,
    rotations: base.rotations,
    scales: base.scales,
    opacities: base.opacities,
    dcRaw: base.dcRaw,
    shRest: base.sh_rest,
    shRestCoefs: base.sh_rest ? base.sh_rest.length / base.count / 3 : 0,
  },
  tail,
);
```

### `decompressZstdSplitBuffer(compressed, ext, zstdDecompress) → Uint8Array`

Low-level decoder for `SF_zstd_split_buffer`. Use directly when you have a
loader that already understands glTF and you only need the zstd
decompression step.

```ts
import { decompressZstdSplitBuffer } from '@catetus/glb-polyfill';
import { decompress } from 'fzstd';
const bin = decompressZstdSplitBuffer(rawBin, gltfJson.extensions.SF_zstd_split_buffer, decompress);
```

### `decodeShPaletteSidecar(compressed, ext, zstdDecompress) → ShPalette`

Decodes a `.shpal` sidecar (zstd-framed 45-D VQ codebook + per-splat
indices). Pass `null` for `ext` to skip the cross-check against the JSON
extension block.

```ts
import { decodeShPaletteSidecar } from '@catetus/glb-polyfill';
import { decompress } from 'fzstd';
const palette = decodeShPaletteSidecar(new Uint8Array(palBytes), palExt, decompress);
console.log(palette.K, 'codebook entries,', palette.N, 'splats');
```

### `paletteShRestForSplat(palette, splatIndex, shDegree) → Float32Array | null`

Looks up the dequantized SH-rest vector for a single splat (length
`coefCount * 3`, interleaved per channel). Returns `null` if the requested
degree exceeds what the palette stored.

### `decodeSmallest3Quat(packed, componentBits) → [x, y, z, w]`

Decodes a single SOG-style smallest-three packed quaternion (one `u32` →
unit quaternion).

### `decodeSmallest3QuatBuffer(packed, componentBits, count?) → Float32Array`

Bulk version: takes a `Uint32Array` of packed quaternions and returns a flat
`Float32Array(count*4)` laid out `[x0,y0,z0,w0, x1,…]`. Use when you've
already pulled the `KHR_gaussian_splatting:ROTATION` buffer out of a GLB
that uses `SF_quat_smallest3` and want to bypass `decodeSFExtensions`.

### `VQ_DIM` (const = 45)

Vector dimensionality of the v1 `.shpal` codebook (covers SH-rest degrees
1..3 across the three color channels).

### Exported types

| Type | Purpose |
| --- | --- |
| `DecodedSplats` | Output of `decodeSFExtensions` — see schema below. |
| `ZstdSplitBufferExt`, `ZstdSplitView` | Shape of the `SF_zstd_split_buffer` JSON block. |
| `ShPalette`, `ShPaletteExt` | Decoded palette + its JSON block. |
| `QuatSmallest3Ext` | Shape of the `SF_quat_smallest3` JSON block. |
| `V5TailHeader`, `DecodedV5Tail` | Sidecar header + decoded per-group residuals. |
| `ApplyTargetScene` | The scene-shape contract that `applyV5TailToScene` consumes. |
| `ZstdDecompress` | `(compressed: Uint8Array) → Uint8Array` — pluggable zstd. |

## Output schema (`DecodedSplats`)

```ts
interface DecodedSplats {
  count: number;                  // = positions.length / 3
  positions: Float32Array;        // length count*3, XYZ
  rotations: Float32Array;        // length count*4, XYZW (normalized)
  scales: Float32Array;           // length count*3, per-axis, LINEAR
  opacities: Float32Array;        // length count,   LINEAR in [0, 1]
  dcRaw: Float32Array;            // length count*3, raw SH DC (no SH_C0 bake) — canonical name
  dc_color: Float32Array;         // deprecated alias for `dcRaw` (same buffer); will be removed pre-publish
  sh_rest: Float32Array | null;   // length count*coefCount*3 or null
  shDegree: number;               // 0..3 — SH-rest degree reconstructed
  bbox: { min: [number,number,number]; max: [number,number,number] } | null;
  extensionsApplied: {
    zstdSplitBuffer: boolean;
    palette: boolean;
    smallest3: boolean;
    logQuantAttrs: boolean;       // provenance only — output is already linear
  };
}
```

Important: **the polyfill eagerly de-logs scales and de-logits opacities**
when `SF_log_quant_attrs` is on, so the public `scales` / `opacities` are
ALWAYS linear regardless of source format. This matches the Rust decoder
(`crates/catetus-gltf/src/lib.rs::apply_log_quant_attrs`). The
`extensionsApplied.logQuantAttrs` flag is provenance-only — it records that
the source GLB carried the extension, not that you still owe an inverse
transform.

(Prior versions surfaced a `logQuantAttrsApplied` flag that callers had to
read and conditionally apply `exp` / `sigmoid`. Forgetting it produced the
bonsai blob render bug, task #113. The flag is gone; the inverse is built
into the decoder.)

## Format coverage matrix

| Container | Extension                          | Sidecar              | Decoder entry point                                    |
| --------- | ---------------------------------- | -------------------- | ------------------------------------------------------ |
| GLB       | `KHR_gaussian_splatting` (base)    | —                    | `decodeSFExtensions`                                    |
| GLB       | `SF_zstd_split_buffer`             | —                    | `decodeSFExtensions` (or `decompressZstdSplitBuffer`)   |
| GLB       | `SF_gaussian_splatting_palette`    | `.glb.shpal`         | `decodeSFExtensions` (or `decodeShPaletteSidecar`)      |
| GLB       | `SF_quat_smallest3`                | —                    | `decodeSFExtensions` (or `decodeSmallest3QuatBuffer`)   |
| GLB       | `SF_log_quant_attrs`               | —                    | applied eagerly inside `decodeSFExtensions` (output is linear) |
| GLB / SOG | V5.2 joint-tail residual           | `.glb.v5tail` / `.sog.v5tail` | `decodeV5TailBytes` + `applyV5TailToScene`     |
| SOG       | base container                     | —                    | **not handled by this package** — decode SOG with your existing SOG loader (SuperSplat, PlayCanvas, etc.), then apply `.sog.v5tail` via the V5 tail functions above. |

The V5 tail sidecar is container-agnostic: the same bytes apply on top of a
GLB-decoded scene or a SOG-decoded scene, as long as the splat ordering
matches the producer's ordering (the sidecar carries its own selection
mask and Morton index, but it does not realign splats).

## Browser / runtime compatibility

- Decode-only: no WebGL2 or WebGPU required.
- Needs `Uint8Array`, `Float32Array`, `Uint32Array`, `DataView`, `TextDecoder`.
- Works in any modern browser, Node ≥ 18, Deno, Bun, and Workers / Service
  Workers. Zero DOM access.
- `fzstd` is pure JS; you can swap it for Node 21+'s built-in
  `zlib.zstdDecompressSync` by passing your own `zstdDecompress` arg.

## Examples

See [`examples/`](./examples) for three minimal working integrations:

- `examples/raw-canvas/` — bare WebGL2 point-cloud viewer (~150 lines) that
  exercises the full decode pipeline.
- `examples/three.js/` — decode an SF GLB into a Three.js `BufferGeometry`
  rendered as `THREE.Points`. Drops straight into any Three app and is the
  basis for a real splat-shader integration.
- `examples/playcanvas/` — feed decoded splats into a PlayCanvas app via
  `pc.Mesh` / `pc.MeshInstance`.

Each example has its own `README.md` with run instructions.

## License

Apache-2.0.
