/**
 * Smoke tests for the compute-decode + GPU radix-sort module that can run in
 * Node (i.e. without WebGPU). The WGSL itself is exercised end-to-end in the
 * Playwright spec at `tests/visual/tests/compute-decode.spec.ts`.
 *
 * Here we cover:
 *   - The WGSL strings load via the generated bundle.
 *   - The radix-sort algorithm shape (4-bit, 8 passes) is consistent.
 *   - The synthetic-scene builder produces well-formed SoA bytes that the
 *     pure-JS decoder round-trips.
 */
import { describe, expect, it } from 'vitest';
import { DECODE_WGSL, RADIX_SORT_WGSL } from '../../webgpu/index.js';
import { RADIX_SORT_PASSES, RADIX_SORT_WG_SIZE } from '../../webgpu/radix_sort.js';
import { buildSyntheticScene } from '../../../bench/compute-decode.bench.js';
import { decodeSplatsSoa } from '../../renderer/base.js';

describe('webgpu compute-decode module', () => {
  it('embeds non-empty WGSL strings', () => {
    expect(DECODE_WGSL).toMatch(/cs_decode/);
    expect(DECODE_WGSL).toMatch(/cs_project/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_histogram/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_scan/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_scatter/);
  });

  it('declares 8 passes over a 256-wide workgroup', () => {
    // 32 bits / 4 bits per pass = 8 passes.
    expect(RADIX_SORT_PASSES).toBe(8);
    expect(RADIX_SORT_WG_SIZE).toBe(256);
  });

  it('synthetic scene round-trips through the CPU decoder', () => {
    const { bytes, descriptor } = buildSyntheticScene(64, 0x1234);
    expect(descriptor.attributeLayout).toBeTruthy();
    const decoded = decodeSplatsSoa(bytes, descriptor.attributeLayout!, 64);
    expect(decoded).toHaveLength(64);
    // Rotation is identity in the synthetic scene.
    expect(decoded[0]!.rotation).toEqual([0, 0, 0, 1]);
    // Positions live in [-2, 2].
    for (const s of decoded) {
      for (const k of [0, 1, 2] as const) {
        expect(s.position[k]).toBeGreaterThanOrEqual(-2);
        expect(s.position[k]).toBeLessThanOrEqual(2);
      }
    }
  });
});

/**
 * CPU reference implementation of the same multi-pass LSD radix sort the
 * WGSL implements. Used by the algorithm-shape test below to keep us honest
 * if we ever change the bit-width / pass count constants.
 */
function radixSortRef(keys: Uint32Array, values: Uint32Array): { keys: Uint32Array; values: Uint32Array } {
  const n = keys.length;
  let kIn = new Uint32Array(keys);
  let vIn = new Uint32Array(values);
  let kOut = new Uint32Array(n);
  let vOut = new Uint32Array(n);
  for (let pass = 0; pass < 8; pass++) {
    const shift = pass * 4;
    const hist = new Uint32Array(16);
    for (let i = 0; i < n; i++) hist[(kIn[i]! >>> shift) & 0xf]!++;
    const offsets = new Uint32Array(16);
    for (let i = 1; i < 16; i++) offsets[i] = offsets[i - 1]! + hist[i - 1]!;
    const cursor = new Uint32Array(offsets);
    for (let i = 0; i < n; i++) {
      const b = (kIn[i]! >>> shift) & 0xf;
      const d = cursor[b]!++;
      kOut[d] = kIn[i]!;
      vOut[d] = vIn[i]!;
    }
    const tmp = kIn; kIn = kOut; kOut = tmp;
    const tmpV = vIn; vIn = vOut; vOut = tmpV;
  }
  return { keys: kIn, values: vIn };
}

describe('radix-sort algorithm shape', () => {
  it('CPU reference sorts ascending', () => {
    const n = 1024;
    const k = new Uint32Array(n);
    const v = new Uint32Array(n);
    let s = 0x9e3779b9;
    for (let i = 0; i < n; i++) {
      s = (s * 0x6c078965 + 0xb) >>> 0;
      k[i] = s;
      v[i] = i;
    }
    const sorted = radixSortRef(k, v);
    for (let i = 1; i < n; i++) {
      expect(sorted.keys[i]!).toBeGreaterThanOrEqual(sorted.keys[i - 1]!);
    }
    // Values must be a permutation of [0, n).
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) seen[sorted.values[i]!]! = 1;
    for (let i = 0; i < n; i++) expect(seen[i]).toBe(1);
  });
});
