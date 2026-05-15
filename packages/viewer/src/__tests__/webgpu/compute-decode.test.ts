/**
 * Smoke tests for the compute-decode + GPU radix-sort module that can run in
 * Node (i.e. without WebGPU). The WGSL itself is exercised end-to-end in the
 * Playwright spec at `tests/visual/tests/compute-decode.spec.ts`.
 *
 * Here we cover:
 *   - The WGSL strings load via the generated bundle.
 *   - The radix-sort algorithm shape (8-bit, 4 passes) is consistent.
 *   - The synthetic-scene builder produces well-formed SoA bytes that the
 *     pure-JS decoder round-trips.
 *   - The histogram_subgroup.wgsl module is embedded with the right entry
 *     point + `enable subgroups;` directive.
 */
import { describe, expect, it } from 'vitest';
import {
  DECODE_WGSL,
  RADIX_SORT_WGSL,
  SCAN_MULTIBLOCK_WGSL,
  HISTOGRAM_SUBGROUP_WGSL,
} from '../../webgpu/shaders.generated.js';
import {
  RADIX_SORT_BITS_PER_PASS,
  RADIX_SORT_PASSES,
  RADIX_SORT_RADIX,
  RADIX_SORT_WG_SIZE,
} from '../../webgpu/radix_sort.js';
import { buildSyntheticScene } from '../../../bench/compute-decode.bench.js';
import { decodeSplatsSoa } from '../../renderer/base.js';

describe('webgpu compute-decode module', () => {
  it('embeds non-empty WGSL strings', () => {
    expect(DECODE_WGSL).toMatch(/cs_decode/);
    expect(DECODE_WGSL).toMatch(/cs_project/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_histogram/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_scan/);
    expect(RADIX_SORT_WGSL).toMatch(/cs_scatter/);
    expect(SCAN_MULTIBLOCK_WGSL).toMatch(/cs_scan_per_wg/);
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/cs_histogram_subgroup/);
  });

  it('declares 4 passes of 8-bit radix over a 256-wide workgroup', () => {
    // 32 bits / 8 bits per pass = 4 passes; 2^8 = 256 bins.
    expect(RADIX_SORT_PASSES).toBe(4);
    expect(RADIX_SORT_BITS_PER_PASS).toBe(8);
    expect(RADIX_SORT_RADIX).toBe(256);
    expect(RADIX_SORT_WG_SIZE).toBe(256);
    // PASSES must be even so the final ping-pong lands on the A buffers.
    expect(RADIX_SORT_PASSES % 2).toBe(0);
    // The WGSL must agree on the constants - otherwise the TS dispatch
    // sizes the histogram array wrong.
    expect(RADIX_SORT_WGSL).toMatch(/const\s+RADIX\s*:\s*u32\s*=\s*256u/);
    expect(RADIX_SORT_WGSL).toMatch(/const\s+PASSES\s*:\s*u32\s*=\s*4u/);
  });

  it('subgroup histogram has the correct enable directive and entry point', () => {
    // `enable subgroups;` MUST appear before any other top-level statement
    // - this is what gates the shader on the WebGPU 1.1 'subgroups' feature
    // at module-creation time.
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/enable subgroups;/);
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/fn cs_histogram_subgroup/);
    // Shares the same RADIX constant as the atomic kernel.
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/const\s+RADIX\s*:\s*u32\s*=\s*256u/);
    // Uses the conservative "all-lanes-agree" coalesce, not a per-bin
    // ballot loop (256 iterations would be a perf regression).
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/subgroupBroadcastFirst/);
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/subgroupAll/);
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
 * if we ever change the bit-width / pass count constants. Parameterized on
 * `bitsPerPass` so the same reference covers both the old 4-bit / 8-pass
 * and the new 8-bit / 4-pass shapes.
 */
function radixSortRef(
  keys: Uint32Array,
  values: Uint32Array,
  bitsPerPass: number,
): { keys: Uint32Array; values: Uint32Array } {
  const n = keys.length;
  const radix = 1 << bitsPerPass;
  const mask = radix - 1;
  const passes = 32 / bitsPerPass;
  let kIn = new Uint32Array(keys);
  let vIn = new Uint32Array(values);
  let kOut = new Uint32Array(n);
  let vOut = new Uint32Array(n);
  for (let pass = 0; pass < passes; pass++) {
    const shift = pass * bitsPerPass;
    const hist = new Uint32Array(radix);
    for (let i = 0; i < n; i++) hist[(kIn[i]! >>> shift) & mask]!++;
    const offsets = new Uint32Array(radix);
    for (let i = 1; i < radix; i++) offsets[i] = offsets[i - 1]! + hist[i - 1]!;
    const cursor = new Uint32Array(offsets);
    for (let i = 0; i < n; i++) {
      const b = (kIn[i]! >>> shift) & mask;
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
  it('CPU 8-bit / 4-pass reference sorts ascending (matches new WGSL)', () => {
    const n = 1024;
    const k = new Uint32Array(n);
    const v = new Uint32Array(n);
    let s = 0x9e3779b9;
    for (let i = 0; i < n; i++) {
      s = (s * 0x6c078965 + 0xb) >>> 0;
      k[i] = s;
      v[i] = i;
    }
    const sorted = radixSortRef(k, v, RADIX_SORT_BITS_PER_PASS);
    for (let i = 1; i < n; i++) {
      expect(sorted.keys[i]!).toBeGreaterThanOrEqual(sorted.keys[i - 1]!);
    }
    // Values must be a permutation of [0, n).
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) seen[sorted.values[i]!]! = 1;
    for (let i = 0; i < n; i++) expect(seen[i]).toBe(1);
  });

  it('CPU 8-bit / 4-pass reference equals CPU 4-bit / 8-pass reference', () => {
    // Both bit-widths must produce the same total order over u32 keys
    // (LSD radix sort is bit-width agnostic at the API surface). This
    // catches regressions where the bit-shift calculation drifts from
    // the radix.
    const n = 512;
    const k = new Uint32Array(n);
    const v = new Uint32Array(n);
    let s = 0xdeadbeef;
    for (let i = 0; i < n; i++) {
      s = (s * 0x6c078965 + 0xb) >>> 0;
      k[i] = s;
      v[i] = i;
    }
    const a = radixSortRef(k, v, 4);
    const b = radixSortRef(k, v, 8);
    expect(Array.from(a.keys)).toEqual(Array.from(b.keys));
    expect(Array.from(a.values)).toEqual(Array.from(b.values));
  });
});
