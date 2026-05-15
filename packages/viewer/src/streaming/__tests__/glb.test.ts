import { describe, expect, it } from 'vitest';
import { decodeGlb } from '../glb.js';

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
