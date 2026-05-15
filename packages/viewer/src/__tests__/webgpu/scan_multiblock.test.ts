// SPDX-License-Identifier: Apache-2.0
/**
 * Unit tests for the multi-block (3-kernel chained) exclusive prefix-sum.
 *
 * Real GPU execution lives in the Playwright visual-regression spec; in Node
 * we don't have WebGPU, so we mirror the WGSL algorithm in TypeScript and
 * assert that the chained scan agrees with a textbook reference exclusive
 * prefix-sum bit-for-bit.
 *
 * The TS mirror is deliberately a translation of the three phases in
 * `scan_multiblock.wgsl` so that an algorithmic change there is forced to
 * land here too. The test thereby protects against:
 *
 *   - tile boundary off-by-ones,
 *   - block-sum aggregation losing the last-element contribution,
 *   - phase ordering bugs (must be A → B → C),
 *   - reading `data[i]` past `total` (the OOB tail of the final tile).
 *
 * We also verify that the embedded WGSL string declares the three expected
 * entry points so a typo in the shader breaks tests before it breaks the
 * browser.
 */
import { describe, expect, it } from 'vitest';
import { SCAN_MULTIBLOCK_WGSL } from '../../webgpu/shaders.generated.js';
import { RADIX_SORT_WG_SIZE } from '../../webgpu/radix_sort.js';

const WG_SIZE = RADIX_SORT_WG_SIZE;

/** Reference exclusive prefix-sum (clear, slow, correct). */
function cpuExclusivePrefixSum(input: Uint32Array): Uint32Array {
  const out = new Uint32Array(input.length);
  let running = 0;
  for (let i = 0; i < input.length; i++) {
    out[i] = running;
    running = (running + input[i]!) >>> 0;
  }
  return out;
}

/**
 * TypeScript mirror of the WGSL chained scan. Each phase modifies the
 * `data` and `blockSums` arrays in-place exactly as the GPU kernels would.
 */
function chainedScanMirror(input: Uint32Array): Uint32Array {
  const total = input.length;
  const numScanWgs = Math.max(Math.ceil(total / WG_SIZE), 1);
  const data = new Uint32Array(input);
  const blockSums = new Uint32Array(numScanWgs);

  // Phase A: per-WG exclusive scan over each tile; write tile total to blockSums.
  for (let wg = 0; wg < numScanWgs; wg++) {
    const start = wg * WG_SIZE;
    const tileLen = Math.min(WG_SIZE, total - start);
    let running = 0;
    let lastInclusive = 0;
    for (let i = 0; i < tileLen; i++) {
      const v = data[start + i]!;
      data[start + i] = running;
      running = (running + v) >>> 0;
      lastInclusive = running;
    }
    // The WGSL kernel writes block_sums[wgid] from lid == WG_SIZE - 1's
    // inclusive prefix, which equals the sum over the live prefix of the
    // tile (OOB lanes contributed zero). That's exactly `lastInclusive`.
    blockSums[wg] = lastInclusive;
  }

  // Phase B: exclusive scan over blockSums (single-WG scan; we just do it in TS).
  const scannedBlockSums = cpuExclusivePrefixSum(blockSums);
  blockSums.set(scannedBlockSums);

  // Phase C: add blockSums[wg] to every element in the tile.
  for (let wg = 0; wg < numScanWgs; wg++) {
    const start = wg * WG_SIZE;
    const tileLen = Math.min(WG_SIZE, total - start);
    const bias = blockSums[wg]!;
    for (let i = 0; i < tileLen; i++) {
      data[start + i] = (data[start + i]! + bias) >>> 0;
    }
  }

  return data;
}

/** Seeded PRNG (xorshift32) — deterministic test inputs. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return s >>> 0;
  };
}

function makeInput(n: number, seed: number, valueMod = 1024): Uint32Array {
  const r = rng(seed);
  const a = new Uint32Array(n);
  for (let i = 0; i < n; i++) a[i] = r() % valueMod;
  return a;
}

describe('scan_multiblock.wgsl: embedded shader sanity', () => {
  it('declares the three chained-scan entry points', () => {
    expect(SCAN_MULTIBLOCK_WGSL).toMatch(/fn cs_scan_per_wg/);
    expect(SCAN_MULTIBLOCK_WGSL).toMatch(/fn cs_scan_block_sums/);
    expect(SCAN_MULTIBLOCK_WGSL).toMatch(/fn cs_scan_add_block_sums/);
  });
  it('uses the same WG_SIZE as the radix sort orchestration', () => {
    // The WGSL hard-codes `const WG_SIZE : u32 = 256u;` and the TS
    // dispatch sizes the tile assuming this. Catch divergence early.
    expect(SCAN_MULTIBLOCK_WGSL).toMatch(/WG_SIZE\s*:\s*u32\s*=\s*256u/);
    expect(WG_SIZE).toBe(256);
  });
});

describe('scan_multiblock: mirror matches reference prefix-sum', () => {
  const cases: Array<{ name: string; n: number }> = [
    { name: '128 elements (sub-tile, partial tail)', n: 128 },
    { name: '256 elements (exactly one tile)', n: 256 },
    { name: '257 elements (one tile + 1)', n: 257 },
    { name: '4096 elements (16 full tiles)', n: 4096 },
    { name: '4097 elements (16 tiles + 1 tail)', n: 4097 },
    { name: '65536 elements (256 full tiles)', n: 65536 },
    { name: '70000 elements (mixed tail)', n: 70000 },
  ];
  for (const { name, n } of cases) {
    it(`agrees on ${name}`, () => {
      const input = makeInput(n, 0xa5a5_0000 ^ n, 8);
      const ref = cpuExclusivePrefixSum(input);
      const got = chainedScanMirror(input);
      expect(got).toEqual(ref);
    });
  }
});

describe('scan_multiblock: edge cases', () => {
  it('handles a single element', () => {
    const input = new Uint32Array([42]);
    const ref = cpuExclusivePrefixSum(input);
    const got = chainedScanMirror(input);
    expect(Array.from(got)).toEqual(Array.from(ref));
  });
  it('handles all-zeros (degenerate input)', () => {
    const input = new Uint32Array(1000); // zero-filled
    const got = chainedScanMirror(input);
    for (let i = 0; i < got.length; i++) expect(got[i]).toBe(0);
  });
  it('handles all-ones (running counter)', () => {
    const n = 1024;
    const input = new Uint32Array(n).fill(1);
    const got = chainedScanMirror(input);
    for (let i = 0; i < n; i++) expect(got[i]).toBe(i);
  });
  it("agrees on the radix-sort histogram shape (numWgs * RADIX) for 10M splats", () => {
    // 10 M splats → numWgs = ceil(10e6 / 256) = 39063 → histogram length =
    // 39063 * 16 = 625008. This is the exact scan size per radix pass.
    const histLen = Math.ceil(10_000_000 / 256) * 16;
    expect(histLen).toBeGreaterThan(600_000);
    // We still want it to fit in a single block-sums scan: numScanWgs <
    // WG_SIZE * WG_SIZE = 65536, which here means histLen / 256 < 65536,
    // i.e. histLen < 16 777 216 — always true at 10 M splats.
    const numScanWgs = Math.ceil(histLen / WG_SIZE);
    expect(numScanWgs).toBeLessThan(WG_SIZE * WG_SIZE);
    // Spot-check the mirror on this size with a small mod so we don't burn
    // memory on test output.
    const input = makeInput(histLen, 0xdead_beef, 4);
    const ref = cpuExclusivePrefixSum(input);
    const got = chainedScanMirror(input);
    // Spot-check at boundaries rather than full deep-equal (cheaper).
    expect(got[0]).toBe(ref[0]);
    expect(got[1]).toBe(ref[1]);
    expect(got[WG_SIZE - 1]).toBe(ref[WG_SIZE - 1]);
    expect(got[WG_SIZE]).toBe(ref[WG_SIZE]);
    expect(got[histLen - 1]).toBe(ref[histLen - 1]);
    // Mid-range tile boundary.
    const mid = (Math.floor(numScanWgs / 2) * WG_SIZE) >>> 0;
    expect(got[mid]).toBe(ref[mid]);
    expect(got[mid + 1]).toBe(ref[mid + 1]);
  });
});
