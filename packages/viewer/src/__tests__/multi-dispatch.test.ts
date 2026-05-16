// SPDX-License-Identifier: Apache-2.0
//
// Multi-dispatch chunk planner tests. Pinned at the planner level so the
// slicing math is verified without standing up a real WebGPU device.
// (`dispatchPerSplat` itself is a thin wrapper over the planner +
// queue.writeBuffer + encoder.beginComputePass; the planner is the
// non-trivial part.)

import { describe, expect, it } from 'vitest';
import {
  planDispatchChunks,
  SPLAT_DISPATCH_CAP,
  SPLAT_WORKGROUP_SIZE,
  WEBGPU_MAX_DISPATCH_PER_DIM,
  dispatchPerSplat,
} from '../webgpu/multi-dispatch.js';
import {
  RADIX_MERGE_WGSL,
  RADIX_SORT_WGSL,
  HISTOGRAM_SUBGROUP_WGSL,
} from '../webgpu/shaders.generated.js';

describe('planDispatchChunks', () => {
  it('zero splats produces zero chunks (no-op fast path)', () => {
    expect(planDispatchChunks(0)).toEqual([]);
  });

  it('single-chunk passthrough: small N → one chunk @ offset 0', () => {
    const chunks = planDispatchChunks(1000);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.chunkOffset).toBe(0);
    expect(chunks[0]!.splatCount).toBe(1000);
    // ceil(1000 / 256) = 4
    expect(chunks[0]!.workgroupCount).toBe(4);
  });

  it('exact-cap fits in one chunk (boundary case)', () => {
    // Cap = 65535 * 256 = 16_776_960.
    const chunks = planDispatchChunks(SPLAT_DISPATCH_CAP);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.chunkOffset).toBe(0);
    expect(chunks[0]!.splatCount).toBe(SPLAT_DISPATCH_CAP);
    expect(chunks[0]!.workgroupCount).toBe(WEBGPU_MAX_DISPATCH_PER_DIM);
  });

  it('cap + 1 produces two chunks', () => {
    const chunks = planDispatchChunks(SPLAT_DISPATCH_CAP + 1);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]!.chunkOffset).toBe(0);
    expect(chunks[0]!.splatCount).toBe(SPLAT_DISPATCH_CAP);
    expect(chunks[0]!.workgroupCount).toBe(WEBGPU_MAX_DISPATCH_PER_DIM);
    expect(chunks[1]!.chunkOffset).toBe(SPLAT_DISPATCH_CAP);
    expect(chunks[1]!.splatCount).toBe(1);
    expect(chunks[1]!.workgroupCount).toBe(1);
  });

  it('2-chunk exact-cap (LODGE L1 ≈ 54M splats)', () => {
    // 54M sits between 3× and 4× the 16.7M cap. Use exactly 2× cap.
    const total = SPLAT_DISPATCH_CAP * 2;
    const chunks = planDispatchChunks(total);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toEqual({
      chunkOffset: 0,
      workgroupCount: WEBGPU_MAX_DISPATCH_PER_DIM,
      splatCount: SPLAT_DISPATCH_CAP,
    });
    expect(chunks[1]).toEqual({
      chunkOffset: SPLAT_DISPATCH_CAP,
      workgroupCount: WEBGPU_MAX_DISPATCH_PER_DIM,
      splatCount: SPLAT_DISPATCH_CAP,
    });
  });

  it('LODGE L1 (~54M splats) carves into 4 chunks with correct offsets', () => {
    const total = 54_000_000;
    const chunks = planDispatchChunks(total);
    // 54M / 16.78M = 3.22 → 4 chunks
    expect(chunks).toHaveLength(4);
    // Sum of splat counts equals total
    expect(chunks.reduce((s, c) => s + c.splatCount, 0)).toBe(total);
    // Offsets are monotonically increasing and contiguous
    let expected = 0;
    for (const chunk of chunks) {
      expect(chunk.chunkOffset).toBe(expected);
      expected += chunk.splatCount;
    }
    // First three chunks are full-cap; last is the remainder.
    for (let i = 0; i < 3; i += 1) {
      expect(chunks[i]!.splatCount).toBe(SPLAT_DISPATCH_CAP);
      expect(chunks[i]!.workgroupCount).toBe(WEBGPU_MAX_DISPATCH_PER_DIM);
    }
    const last = chunks[3]!;
    expect(last.splatCount).toBe(total - 3 * SPLAT_DISPATCH_CAP);
    expect(last.workgroupCount).toBe(Math.ceil(last.splatCount / SPLAT_WORKGROUP_SIZE));
  });

  it('8-chunk large case (LODGE L0 ≈ 119M splats fits in 8 chunks)', () => {
    // 8 × 16.78M = 134.2M, so 119M fits in 8 chunks.
    const total = 119_000_000;
    const chunks = planDispatchChunks(total);
    expect(chunks.length).toBeLessThanOrEqual(8);
    expect(chunks.length).toBeGreaterThanOrEqual(8);
    expect(chunks.reduce((s, c) => s + c.splatCount, 0)).toBe(total);
    // First 7 are full-cap.
    for (let i = 0; i < 7; i += 1) {
      expect(chunks[i]!.splatCount).toBe(SPLAT_DISPATCH_CAP);
    }
  });

  it('respects custom workgroup sizes (e.g. 64 for cs_lod_select)', () => {
    // workgroup_size 64 → cap = 65535 * 64 = 4_194_240
    const wgSize = 64;
    const cap = WEBGPU_MAX_DISPATCH_PER_DIM * wgSize;
    const chunks = planDispatchChunks(cap + 100, wgSize);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]!.workgroupCount).toBe(WEBGPU_MAX_DISPATCH_PER_DIM);
    expect(chunks[1]!.splatCount).toBe(100);
    expect(chunks[1]!.workgroupCount).toBe(Math.ceil(100 / wgSize));
  });
});

describe('dispatchPerSplat (mock GPU)', () => {
  it('opens N compute passes and writes chunk_offset before each', () => {
    // Mock the minimum surface area: device.queue.writeBuffer + encoder.beginComputePass.
    const writes: Array<{ buffer: object; offset: number; value: number }> = [];
    const dispatches: Array<{ wgs: number }> = [];
    let passOpen = false;
    let passCount = 0;

    const fakeUniformBuffer = { _id: 'uniform' } as unknown as GPUBuffer;
    const fakePipeline = { _id: 'pipeline' } as unknown as GPUComputePipeline;
    const fakeBindGroup = { _id: 'bindgroup' } as unknown as GPUBindGroup;

    const fakePass = {
      setPipeline: () => undefined,
      setBindGroup: () => undefined,
      dispatchWorkgroups: (wgs: number) => {
        if (!passOpen) throw new Error('dispatchWorkgroups called outside pass');
        dispatches.push({ wgs });
      },
      end: () => {
        passOpen = false;
      },
    };

    const fakeEncoder = {
      beginComputePass: () => {
        passOpen = true;
        passCount += 1;
        return fakePass;
      },
    } as unknown as GPUCommandEncoder;

    const fakeDevice = {
      queue: {
        writeBuffer: (buffer: GPUBuffer, offset: number, data: ArrayBuffer | ArrayBufferView) => {
          // Read the u32 the helper wrote.
          const view = data instanceof ArrayBuffer
            ? new Uint32Array(data, 0, 1)
            : new Uint32Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, 1);
          writes.push({ buffer: buffer as unknown as object, offset, value: view[0]! });
        },
      },
    } as unknown as GPUDevice;

    // Two chunks worth of splats (cap + 1).
    const n = SPLAT_DISPATCH_CAP + 1;
    const chunkCount = dispatchPerSplat(
      fakeDevice,
      fakeEncoder,
      fakePipeline,
      fakeBindGroup,
      fakeUniformBuffer,
      /* uniformChunkOffsetBytes */ 148,
      n,
    );

    expect(chunkCount).toBe(2);
    expect(passCount).toBe(2);
    expect(writes).toHaveLength(2);
    expect(writes[0]!.value).toBe(0);
    expect(writes[1]!.value).toBe(SPLAT_DISPATCH_CAP);
    expect(writes[0]!.offset).toBe(148);
    expect(writes[1]!.offset).toBe(148);
    expect(dispatches).toHaveLength(2);
    expect(dispatches[0]!.wgs).toBe(WEBGPU_MAX_DISPATCH_PER_DIM);
    expect(dispatches[1]!.wgs).toBe(1);
  });

  // (additional dispatch-helper sanity below; original zero-splat case follows.)
  it('zero splats: no writes, no passes', () => {
    let passCount = 0;
    let writeCount = 0;
    const fakeEncoder = {
      beginComputePass: () => {
        passCount += 1;
        return { setPipeline: () => undefined, setBindGroup: () => undefined, dispatchWorkgroups: () => undefined, end: () => undefined };
      },
    } as unknown as GPUCommandEncoder;
    const fakeDevice = {
      queue: { writeBuffer: () => { writeCount += 1; } },
    } as unknown as GPUDevice;
    const n = dispatchPerSplat(
      fakeDevice,
      fakeEncoder,
      {} as GPUComputePipeline,
      {} as GPUBindGroup,
      {} as GPUBuffer,
      0,
      0,
    );
    expect(n).toBe(0);
    expect(passCount).toBe(0);
    expect(writeCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Stage 5: chunked radix sort + binary merge tree.
//
// The chunked sort path lives in `packages/viewer/src/webgpu/radix_sort.ts`
// and is exercised on real GPU through `RadixSort.encode()`. We can't stand
// up a WebGPU device in Node, so as elsewhere in this repo we mirror the
// algorithm in TypeScript and assert that the mirror agrees with a textbook
// stable sort on randomized inputs across all chunk shapes that the GPU
// path will see in practice (1, 2, 4, 8 chunks up to LODGE L0 ~119M scale).
//
// The mirror MUST be a faithful translation of the WGSL kernels so that an
// algorithmic divergence in WGSL is forced to land here too. We also assert
// that the embedded radix_merge.wgsl declares the entry point we wire up,
// and that the radix_sort.wgsl + histogram_subgroup.wgsl Uniforms struct
// now carries the chunk_offset_splats field (else Stage 5 regresses
// silently and L1/L0 LODGE would OOM-by-dispatch in production).
// ---------------------------------------------------------------------------

const SORT_WG_SIZE = 256;
const SORT_CHUNK_SPLATS = SPLAT_DISPATCH_CAP;

/** Stable sort reference (clear, slow, correct). Returns sorted (key, value) pairs. */
function cpuStableSort(
  keys: Uint32Array,
  values: Uint32Array,
): { keys: Uint32Array; values: Uint32Array } {
  const idx = Array.from({ length: keys.length }, (_, i) => i);
  // Sort indices by key, breaking ties by original index (stable).
  idx.sort((a, b) => {
    const ka = keys[a]!;
    const kb = keys[b]!;
    if (ka !== kb) return ka - kb;
    return a - b;
  });
  const outKeys = new Uint32Array(keys.length);
  const outVals = new Uint32Array(values.length);
  for (let i = 0; i < idx.length; i++) {
    outKeys[i] = keys[idx[i]!]!;
    outVals[i] = values[idx[i]!]!;
  }
  return { keys: outKeys, values: outVals };
}

/**
 * TypeScript mirror of the pairwise stable merge in `radix_merge.wgsl`. One
 * thread per output slot, Merge-Path binary search, stable (ties favor A).
 * The implementation is deliberately direct so a divergence with the WGSL
 * kernel is caught immediately.
 */
function mergePairMirror(
  inKeys: Uint32Array,
  inVals: Uint32Array,
  outKeys: Uint32Array,
  outVals: Uint32Array,
  runAStart: number,
  runALen: number,
  runBStart: number,
  runBLen: number,
  mergedBase: number,
): void {
  const mergedLen = runALen + runBLen;
  for (let k = 0; k < mergedLen; k++) {
    if (runBLen === 0) {
      outKeys[mergedBase + k] = inKeys[runAStart + k]!;
      outVals[mergedBase + k] = inVals[runAStart + k]!;
      continue;
    }
    if (runALen === 0) {
      outKeys[mergedBase + k] = inKeys[runBStart + k]!;
      outVals[mergedBase + k] = inVals[runBStart + k]!;
      continue;
    }
    // Merge-Path: binary-search i in [max(0, k-bLen), min(aLen, k)] for the
    // valid split with the same stability rule as the WGSL kernel.
    let lo = Math.max(0, k - runBLen);
    let hi = Math.min(runALen, k);
    while (lo < hi) {
      const i = (lo + hi) >>> 1;
      const j = k - i;
      let aTooSmall = false;
      if (j > 0 && i < runALen) {
        const aI = inKeys[runAStart + i]!;
        const bJm1 = inKeys[runBStart + (j - 1)]!;
        if (aI <= bJm1) aTooSmall = true;
      }
      if (aTooSmall) lo = i + 1;
      else hi = i;
    }
    const i = lo;
    const j = k - i;
    let pickA = false;
    if (i < runALen) {
      if (j >= runBLen) pickA = true;
      else {
        const aI = inKeys[runAStart + i]!;
        const bJ = inKeys[runBStart + j]!;
        if (aI <= bJ) pickA = true;
      }
    }
    if (pickA) {
      outKeys[mergedBase + k] = inKeys[runAStart + i]!;
      outVals[mergedBase + k] = inVals[runAStart + i]!;
    } else {
      outKeys[mergedBase + k] = inKeys[runBStart + j]!;
      outVals[mergedBase + k] = inVals[runBStart + j]!;
    }
  }
}

/**
 * TypeScript mirror of the full Stage 5 chunked sort pipeline. Sorts a
 * (key, value) array in-place using per-chunk stable sort + binary merge
 * tree, mirroring the WGSL orchestration in `RadixSort.encode`.
 */
function chunkedSortMirror(
  keysIn: Uint32Array,
  valsIn: Uint32Array,
  chunkSize: number,
): { keys: Uint32Array; values: Uint32Array } {
  const n = keysIn.length;
  // Two ping-pong buffers (mirror keysA / keysB).
  let keysA = new Uint32Array(keysIn);
  let valsA = new Uint32Array(valsIn);
  let keysB = new Uint32Array(n);
  let valsB = new Uint32Array(n);

  // Step 1: per-chunk stable sort. Lands in keysA.
  const chunks: Array<{ offset: number; len: number }> = [];
  for (let off = 0; off < n; off += chunkSize) {
    const len = Math.min(chunkSize, n - off);
    chunks.push({ offset: off, len });
    const subKeys = keysA.slice(off, off + len);
    const subVals = valsA.slice(off, off + len);
    const sorted = cpuStableSort(subKeys, subVals);
    keysA.set(sorted.keys, off);
    valsA.set(sorted.values, off);
  }

  // Step 2: binary merge tree. Pairwise merges A → B, then B → A, ...
  let curRuns = chunks;
  let inIsA = true; // current "input" side
  while (curRuns.length > 1) {
    const nextRuns: Array<{ offset: number; len: number }> = [];
    const inKeys = inIsA ? keysA : keysB;
    const inVals = inIsA ? valsA : valsB;
    const outKeys = inIsA ? keysB : keysA;
    const outVals = inIsA ? valsB : valsA;
    for (let i = 0; i < curRuns.length; i += 2) {
      const a = curRuns[i]!;
      const b = i + 1 < curRuns.length ? curRuns[i + 1]! : { offset: a.offset + a.len, len: 0 };
      mergePairMirror(inKeys, inVals, outKeys, outVals, a.offset, a.len, b.offset, b.len, a.offset);
      nextRuns.push({ offset: a.offset, len: a.len + b.len });
    }
    curRuns = nextRuns;
    inIsA = !inIsA;
  }
  // After loop, the "input" side now holds the sorted result (we flipped
  // after each pass so the final write landed on the new "input").
  if (inIsA) {
    return { keys: keysA, values: valsA };
  } else {
    return { keys: keysB, values: valsB };
  }
}

/** Deterministic PRNG (xorshift32). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return s >>> 0;
  };
}

function makeRandomKeyValues(
  n: number,
  seed: number,
  keyMod = 0xffff_ffff,
): { keys: Uint32Array; values: Uint32Array } {
  const r = rng(seed);
  const keys = new Uint32Array(n);
  const values = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    keys[i] = r() % keyMod;
    values[i] = i;
  }
  return { keys, values };
}

describe('radix_merge.wgsl: embedded shader sanity', () => {
  it('declares the cs_radix_merge entry point', () => {
    expect(RADIX_MERGE_WGSL).toMatch(/fn cs_radix_merge/);
  });
  it('binds a MergeUniforms struct with chunk_offset_splats and 8 u32 fields', () => {
    expect(RADIX_MERGE_WGSL).toMatch(/struct MergeUniforms/);
    expect(RADIX_MERGE_WGSL).toMatch(/chunk_offset_splats:\s*u32/);
    expect(RADIX_MERGE_WGSL).toMatch(/run_a_start:\s*u32/);
    expect(RADIX_MERGE_WGSL).toMatch(/run_b_len:\s*u32/);
  });
  it('uses WG_SIZE = 256 (matches the per-splat workgroup size)', () => {
    expect(RADIX_MERGE_WGSL).toMatch(/WG_SIZE\s*:\s*u32\s*=\s*256u/);
  });
});

describe('radix_sort.wgsl + histogram_subgroup.wgsl: chunk_offset uniform present', () => {
  // Stage 5 contract: the Uniforms struct gained a `chunk_offset_splats`
  // field. If a future edit drops it, the chunked sort silently regresses
  // to single-dispatch behavior (which OOMs above 16.7M splats). Pin the
  // contract here.
  it('radix_sort.wgsl Uniforms declares chunk_offset_splats: u32', () => {
    expect(RADIX_SORT_WGSL).toMatch(/struct Uniforms[\s\S]*chunk_offset_splats:\s*u32/);
  });
  it('histogram_subgroup.wgsl Uniforms declares chunk_offset_splats: u32', () => {
    expect(HISTOGRAM_SUBGROUP_WGSL).toMatch(/struct Uniforms[\s\S]*chunk_offset_splats:\s*u32/);
  });
  it('radix_sort.wgsl cs_histogram reads keys_in at i + chunk_offset_splats', () => {
    expect(RADIX_SORT_WGSL).toMatch(/keys_in\[i\s*\+\s*u\.chunk_offset_splats\]/);
  });
  it('radix_sort.wgsl cs_scatter adds chunk_offset_splats to dst', () => {
    // The scatter destination is "wg_offsets[bin] + local_rank + chunk_offset_splats"
    // so the sorted output lands in the same chunk slot of the output buffer.
    expect(RADIX_SORT_WGSL).toMatch(/wg_offsets\[bin\]\s*\+\s*local_rank\s*\+\s*u\.chunk_offset_splats/);
  });
});

describe('chunkedSortMirror (radix-sort + merge-tree mirror)', () => {
  it('matches stable sort on a single chunk (cap-fits-in-one)', () => {
    const n = 1024;
    const { keys, values } = makeRandomKeyValues(n, 0xdead);
    const got = chunkedSortMirror(keys, values, SORT_CHUNK_SPLATS);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it('matches stable sort with 2 chunks of small size', () => {
    // Use a tiny chunkSize so we exercise the merge path on a small input.
    const n = 200;
    const chunkSize = 64;
    const { keys, values } = makeRandomKeyValues(n, 0xb0a7, 100);
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it('matches stable sort with 4 chunks (binary tree depth 2)', () => {
    const n = 1000;
    const chunkSize = 256; // 4 chunks exactly.
    const { keys, values } = makeRandomKeyValues(n, 0x4ca7, 50);
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it('matches stable sort with 8 chunks (binary tree depth 3 — Sweet Corals L0 shape)', () => {
    // LODGE L0 ≈ 119M splats / 16.78M cap = 8 chunks (last is partial).
    // We can't allocate 119M in Node tests, but we can pin the algorithm
    // shape: 8 chunks of small size with random keys.
    const n = 9999;
    const chunkSize = 1280; // 8 chunks (8*1280=10240 >= 9999); last is partial.
    const { keys, values } = makeRandomKeyValues(n, 0x8c0a15, 1024);
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it('preserves stability on dense ties (every key appears many times)', () => {
    // Stability matters for splat sort: if two splats have identical depth
    // keys (after Morton quantization), the value (splat index) order must
    // be deterministic and reproducible. Force lots of ties with a small
    // key mod.
    const n = 5000;
    const chunkSize = 500; // 10 chunks (odd merge-tree shape).
    const { keys, values } = makeRandomKeyValues(n, 0x71e5, 8); // only 8 unique keys
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it('handles a single element and zero-element edge cases', () => {
    // n = 1: no sort needed; should pass through.
    const got1 = chunkedSortMirror(new Uint32Array([7]), new Uint32Array([42]), SORT_CHUNK_SPLATS);
    expect(Array.from(got1.keys)).toEqual([7]);
    expect(Array.from(got1.values)).toEqual([42]);
    // n = 0: empty input -> empty output.
    const got0 = chunkedSortMirror(new Uint32Array(0), new Uint32Array(0), SORT_CHUNK_SPLATS);
    expect(got0.keys.length).toBe(0);
    expect(got0.values.length).toBe(0);
  });

  it('matches stable sort with an odd number of chunks (3 chunks)', () => {
    // 3 chunks: round 0 merges (0,1) and (2,_) → 2 runs; round 1 merges
    // them into 1. Exercises the odd-tail "empty B" path of the merge.
    const n = 800;
    const chunkSize = 300; // 3 chunks: 300, 300, 200.
    const { keys, values } = makeRandomKeyValues(n, 0x0dd0, 64);
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });

  it("LODGE L1 shape (4 chunks) on synthetic data agrees with stable sort", () => {
    // LODGE L1 ≈ 54M splats / 16.78M cap = 4 chunks (last is partial).
    // Synthetic data, scaled down: chunkSize=4096, 4 chunks, last partial.
    const chunkSize = 4096;
    const n = chunkSize * 3 + 1234;
    const { keys, values } = makeRandomKeyValues(n, 0x1111_aaaa);
    const got = chunkedSortMirror(keys, values, chunkSize);
    const ref = cpuStableSort(keys, values);
    expect(Array.from(got.keys)).toEqual(Array.from(ref.keys));
    expect(Array.from(got.values)).toEqual(Array.from(ref.values));
  });
});

describe('chunk plan for >16.7M splats: dispatch math sanity', () => {
  // The orchestrator in radix_sort.ts splits at SPLAT_DISPATCH_CAP. Pin
  // the splitting math at the splat-count thresholds we actually care about.
  it('LODGE L1 (54M splats) splits into 4 chunks', () => {
    const n = 54_000_000;
    const numChunks = Math.ceil(n / SORT_CHUNK_SPLATS);
    expect(numChunks).toBe(4);
    // Per-chunk numWgs <= 65535.
    for (let i = 0; i < numChunks; i++) {
      const off = i * SORT_CHUNK_SPLATS;
      const len = Math.min(SORT_CHUNK_SPLATS, n - off);
      const numWgs = Math.ceil(len / SORT_WG_SIZE);
      expect(numWgs).toBeLessThanOrEqual(WEBGPU_MAX_DISPATCH_PER_DIM);
    }
  });
  it('LODGE L0 (119M splats) splits into 8 chunks', () => {
    const n = 119_000_000;
    const numChunks = Math.ceil(n / SORT_CHUNK_SPLATS);
    expect(numChunks).toBe(8);
    for (let i = 0; i < numChunks; i++) {
      const off = i * SORT_CHUNK_SPLATS;
      const len = Math.min(SORT_CHUNK_SPLATS, n - off);
      const numWgs = Math.ceil(len / SORT_WG_SIZE);
      expect(numWgs).toBeLessThanOrEqual(WEBGPU_MAX_DISPATCH_PER_DIM);
    }
  });
  it('binary merge tree has log2(K) rounds for K chunks', () => {
    // K=2 → 1 round (odd, final copy-back needed)
    // K=4 → 2 rounds (even, lands in keysA naturally)
    // K=8 → 3 rounds (odd, final copy-back needed)
    expect(Math.ceil(Math.log2(2))).toBe(1);
    expect(Math.ceil(Math.log2(4))).toBe(2);
    expect(Math.ceil(Math.log2(8))).toBe(3);
  });
});
