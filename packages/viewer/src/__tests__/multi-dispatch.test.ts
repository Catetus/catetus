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
