# @splatforge/qat-ply-wasm

WebAssembly build of the [SplatForge QAT-PLY v1](../qat-ply-c/) reference C
decoder. This is the **browser fallback** for renderers without WebGPU
(everything but the Metal-direct iOS path and the Vulkan Android path).

The wire format and dequantization arithmetic match the C decoder byte-for-byte
— verified by the cross-target conformance suite at
[`../conformance/cross-target/`](../conformance/cross-target/).

## Install

```bash
pnpm add @splatforge/qat-ply-wasm
# or
npm i @splatforge/qat-ply-wasm
```

## Build from source

```bash
# Requires emcc (Emscripten). On macOS:
brew install emscripten

make build      # -> dist/qat_ply_decode.{js,wasm}
make size       # uncompressed + gzipped sizes
pnpm test       # runs the 10-fixture conformance suite via Vitest
```

## Size

(Updated by `make size` after every `make build`.)

| File                 | Uncompressed | Gzipped (typical) |
|----------------------|--------------|-------------------|
| `qat_ply_decode.wasm`| ~16 KB       | ~7 KB             |
| `qat_ply_decode.js`  | ~40 KB       | ~12 KB            |

These are estimates: the C source is 454 LOC, the wasm payload is the
dequant kernels + base64 decode + header parser. Emscripten's value_object
bindings dominate the .js glue.

## Usage

```js
import createQatPlyModule from "@splatforge/qat-ply-wasm";

const mod = await createQatPlyModule();

// 1) Read the full PLY file (header + body) into a Uint8Array.
const bytes = new Uint8Array(await (await fetch(plyUrl)).arrayBuffer());

// 2) Find end_header to split header text from binary body.
const headerEnd = /* index of "\nend_header\n" + 12 */;
const headerStr = new TextDecoder().decode(bytes.subarray(0, headerEnd));
const body      = bytes.subarray(headerEnd);

// 3) Parse the QAT field descriptors out of the header.
const hdr   = mod.parseHeader(headerStr);
const field = hdr.fields.get(0);

// 4) For per-channel int8 fields: pull scales from the header, then pull
//    int8 columns from the body and dequantize.
const scales = mod.base64DecodeFp32(
  headerStr.substring(field.scaleB64Offset,
                      field.scaleB64Offset + field.scaleB64Len),
  field.channels
);
const out = mod.dequantInt8(q, scales, nRows, field.channels);  // Float32Array
```

For a complete example (including the property-table parser needed to
locate `<field>_q_<i>` columns), see [`examples/decode.html`](examples/decode.html)
and [`tests/wasm.test.ts`](tests/wasm.test.ts).

## License

MIT.
