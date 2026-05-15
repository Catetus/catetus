import { describe, expect, it } from 'vitest';
import { decodeGlb, manifestFromGlb } from '../glb.js';

function makeGlb(jsonText: string, bin: Uint8Array = new Uint8Array(0)): Uint8Array {
  const jsonBytes = new TextEncoder().encode(jsonText);
  const jsonPad = (4 - (jsonBytes.length & 3)) & 3;
  const jsonLen = jsonBytes.length + jsonPad;
  const binPad = (4 - (bin.byteLength & 3)) & 3;
  const binLen = bin.byteLength + binPad;
  const total = 12 + 8 + jsonLen + 8 + binLen;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonLen, true);
  dv.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(buf, 20, jsonBytes.length).set(jsonBytes);
  for (let i = 0; i < jsonPad; i++) new DataView(buf).setUint8(20 + jsonBytes.length + i, 0x20);
  dv.setUint32(20 + jsonLen, binLen, true);
  dv.setUint32(24 + jsonLen, 0x004e4942, true);
  new Uint8Array(buf, 28 + jsonLen, bin.byteLength).set(bin);
  return new Uint8Array(buf);
}

describe('decodeGlb', () => {
  it('round-trips JSON + BIN', () => {
    const json = '{"asset":{"version":"2.0"}}';
    const bin = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const glb = makeGlb(json, bin);
    const out = decodeGlb(glb);
    expect(out.json).toBe(json);
    expect(Array.from(out.bin.subarray(0, 8))).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('rejects bad magic', () => {
    const bad = new Uint8Array(12);
    expect(() => decodeGlb(bad)).toThrow(/glb_invalid/);
  });

  it('rejects bad version', () => {
    const buf = new ArrayBuffer(20);
    const dv = new DataView(buf);
    dv.setUint32(0, 0x46546c67, true);
    dv.setUint32(4, 1, true);
    dv.setUint32(8, 20, true);
    expect(() => decodeGlb(new Uint8Array(buf))).toThrow(/glb_invalid/);
  });
});

describe('manifestFromGlb — KHR_gaussian_splatting layout auto-detection', () => {
  // Five SoA bufferViews: pos(12)/rot(16)/scl(12)/op(4)/dc(12) = 56 bytes/1 splat.
  const sharedGltf = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: 56 }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 12 },
      { buffer: 0, byteOffset: 12, byteLength: 16 },
      { buffer: 0, byteOffset: 28, byteLength: 12 },
      { buffer: 0, byteOffset: 40, byteLength: 4 },
      { buffer: 0, byteOffset: 44, byteLength: 12 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 1, type: 'VEC4' },
      { bufferView: 2, componentType: 5126, count: 1, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 1, type: 'SCALAR' },
      { bufferView: 4, componentType: 5126, count: 1, type: 'VEC3' },
    ],
    extensions: {
      KHR_gaussian_splatting: { splatCount: 1, shDegree: 0 },
    },
  };

  const legacyPrim = {
    mode: 0,
    extensions: {
      KHR_gaussian_splatting: {
        attributes: {
          POSITION: 0,
          _ROTATION: 1,
          _SCALE: 2,
          _OPACITY: 3,
          _COLOR_DC: 4,
        },
      },
    },
  };

  const rcPrim = {
    mode: 0,
    attributes: {
      'KHR_gaussian_splatting:POSITION': 0,
      'KHR_gaussian_splatting:ROTATION': 1,
      'KHR_gaussian_splatting:SCALE': 2,
      'KHR_gaussian_splatting:OPACITY': 3,
      'KHR_gaussian_splatting:COLOR_DC': 4,
    },
    extensions: { KHR_gaussian_splatting: {} },
  };

  function gltfFor(prim: object): string {
    return JSON.stringify({ ...sharedGltf, meshes: [{ primitives: [prim] }] });
  }

  it('accepts legacy attribute layout', () => {
    const glb = makeGlb(gltfFor(legacyPrim), new Uint8Array(56));
    const out = decodeGlb(glb);
    const { manifest } = manifestFromGlb(out);
    expect(manifest.chunks[0]!.attributeLayout).toBeDefined();
  });

  it('accepts RC attribute layout', () => {
    const glb = makeGlb(gltfFor(rcPrim), new Uint8Array(56));
    const out = decodeGlb(glb);
    const { manifest } = manifestFromGlb(out);
    expect(manifest.chunks[0]!.attributeLayout).toBeDefined();
  });

  it('produces byte-equal layouts across both shapes', () => {
    const legacyOut = manifestFromGlb(decodeGlb(makeGlb(gltfFor(legacyPrim), new Uint8Array(56))));
    const rcOut = manifestFromGlb(decodeGlb(makeGlb(gltfFor(rcPrim), new Uint8Array(56))));
    expect(rcOut.manifest.chunks[0]!.attributeLayout).toEqual(
      legacyOut.manifest.chunks[0]!.attributeLayout,
    );
  });
});
