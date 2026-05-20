import { describe, expect, it } from 'vitest';
import { decodeSplatsSoa } from '../renderer/base.js';
import type { SoaAttributeLayout } from '../manifest.js';

/**
 * Build a synthetic SoA chunk that mirrors what catetus-gltf emits for the
 * `KHR_mesh_quantization` (SPEC-0013) path: POSITION u16 normalized,
 * _SCALE / _OPACITY / _COLOR_DC u8 normalized, _ROTATION f32. Lays the
 * attributes out one after another with 4-byte padding between sections, the
 * same way `pack_chunk_with` does in Rust.
 */
function buildQuantizedChunk(
  posMin: [number, number, number],
  posMax: [number, number, number],
  scaleMin: [number, number, number],
  scaleMax: [number, number, number],
  splats: Array<{
    p: [number, number, number];
    r: [number, number, number, number];
    s: [number, number, number];
    op: number;
    dc: [number, number, number];
  }>,
): { bytes: Uint8Array; layout: SoaAttributeLayout } {
  const n = splats.length;

  const align4 = (x: number): number => (x + 3) & ~3;
  const posBytes = align4(n * 6); // u16 vec3
  const rotBytes = n * 16; // f32 vec4
  const scaleBytes = align4(n * 3); // u8 vec3
  const opBytes = align4(n); // u8 scalar
  const dcBytes = align4(n * 3); // u8 vec3
  const total = posBytes + rotBytes + scaleBytes + opBytes + dcBytes;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);

  const quantU16 = (v: number, lo: number, hi: number): number =>
    Math.round(((v - lo) / (hi - lo)) * 65535);
  const quantU8 = (v: number, lo: number, hi: number): number =>
    Math.round(((v - lo) / (hi - lo)) * 255);

  // POSITION
  let o = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      dv.setUint16(o, quantU16(splats[i]!.p[k]!, posMin[k]!, posMax[k]!), true);
      o += 2;
    }
  }
  o = posBytes;

  // ROTATION
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      dv.setFloat32(o, splats[i]!.r[k]!, true);
      o += 4;
    }
  }

  // SCALE
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      dv.setUint8(o, quantU8(splats[i]!.s[k]!, scaleMin[k]!, scaleMax[k]!));
      o += 1;
    }
  }
  o = posBytes + rotBytes + scaleBytes;

  // OPACITY
  for (let i = 0; i < n; i++) {
    dv.setUint8(o, quantU8(splats[i]!.op, 0, 1));
    o += 1;
  }
  o = posBytes + rotBytes + scaleBytes + opBytes;

  // COLOR_DC
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      dv.setUint8(o, quantU8(splats[i]!.dc[k]!, 0, 1));
      o += 1;
    }
  }

  const layout: SoaAttributeLayout = {
    positions: {
      byteOffset: 0,
      byteLength: n * 6,
      componentType: 5123,
      normalized: true,
      min: posMin,
      max: posMax,
    },
    rotations: {
      byteOffset: posBytes,
      byteLength: rotBytes,
      componentType: 5126,
    },
    scales: {
      byteOffset: posBytes + rotBytes,
      byteLength: n * 3,
      componentType: 5121,
      normalized: true,
      min: scaleMin,
      max: scaleMax,
    },
    opacities: {
      byteOffset: posBytes + rotBytes + scaleBytes,
      byteLength: n,
      componentType: 5121,
      normalized: true,
      min: [0],
      max: [1],
    },
    colorDC: {
      byteOffset: posBytes + rotBytes + scaleBytes + opBytes,
      byteLength: n * 3,
      componentType: 5121,
      normalized: true,
      min: [0, 0, 0],
      max: [1, 1, 1],
    },
  };

  return { bytes: new Uint8Array(buf), layout };
}

describe('decodeSplatsSoa — KHR_mesh_quantization', () => {
  it('round-trips quantized integer accessors to within one quantization step', () => {
    const splats = [
      {
        p: [-0.5, 0, 0.25] as [number, number, number],
        r: [0, 0, 0, 1] as [number, number, number, number],
        s: [0.05, 0.06, 0.07] as [number, number, number],
        op: 0.5,
        dc: [0.2, 0.4, 0.8] as [number, number, number],
      },
      {
        p: [0.4, 0.25, 0] as [number, number, number],
        r: [0.1, 0.2, 0.3, 0.9] as [number, number, number, number],
        s: [0.05, 0.05, 0.05] as [number, number, number],
        op: 0.9,
        dc: [1, 0, 0.5] as [number, number, number],
      },
    ];
    const { bytes, layout } = buildQuantizedChunk(
      [-0.5, 0, 0],
      [0.4, 0.25, 0.25],
      [0.05, 0.05, 0.05],
      [0.05, 0.06, 0.07],
      splats,
    );

    const decoded = decodeSplatsSoa(bytes, layout, splats.length);
    expect(decoded.length).toBe(2);
    const posStep = 0.9 / 65535;
    const opStep = 1 / 255;
    const dcStep = 1 / 255;
    for (let i = 0; i < splats.length; i++) {
      for (let k = 0; k < 3; k++) {
        expect(Math.abs(decoded[i]!.position[k]! - splats[i]!.p[k]!)).toBeLessThan(posStep * 2);
        expect(Math.abs(decoded[i]!.colorDC[k]! - splats[i]!.dc[k]!)).toBeLessThan(dcStep * 2);
      }
      // rotation passes through as f32 (round-trip preserves to f32 precision).
      for (let k = 0; k < 4; k++) {
        expect(Math.abs(decoded[i]!.rotation[k]! - splats[i]!.r[k]!)).toBeLessThan(1e-6);
      }
      expect(Math.abs(decoded[i]!.opacity - splats[i]!.op)).toBeLessThan(opStep * 2);
    }
  });
});
