// SPDX-License-Identifier: Apache-2.0
//
// Multi-dispatch helper. WebGPU 1.0 limits `dispatchWorkgroups(x, y, z)` to
// 65535 per dimension. With workgroup_size=256 (the splat-keyed default),
// that caps a single dispatch at 16,776,960 splats. At LODGE L1 (~54M) or
// L0 (~119M) we exceed the cap.
//
// This helper carves a per-splat dispatch into N chunks of <= dispatchCap
// workgroups. Between chunks it rewrites a `chunk_offset: u32` field inside
// the kernel's uniform buffer (WebGPU has no push constants, so per-chunk
// offsets must travel via writeBuffer + a fresh compute pass).
//
// Each WGSL kernel that participates in chunked dispatch follows the
// uniform contract:
//
//   * Slot at byte offset `uniformChunkOffsetBytes` holds a u32 = chunk_offset.
//   * The shader reads `splats[gid + chunk_offset]` and guards
//     `if (gid + chunk_offset >= splat_count) { return; }`.
//
// At call sites the splat-count uniform is set ONCE (the global N). Only
// chunk_offset moves between chunks.
//
// This file is the single source of truth for the dispatch carving math
// so the unit test (multi-dispatch.test.ts) can pin the slicing without
// reaching into individual kernel call sites.

/** Maximum workgroups dispatchable per dimension by WebGPU 1.0. */
export const WEBGPU_MAX_DISPATCH_PER_DIM = 65535;

/** Default per-splat workgroup size (splats / WG). Matches all 1D splat kernels today. */
export const SPLAT_WORKGROUP_SIZE = 256;

/** Largest splat count a single dispatch can cover at workgroup_size=256. */
export const SPLAT_DISPATCH_CAP =
  WEBGPU_MAX_DISPATCH_PER_DIM * SPLAT_WORKGROUP_SIZE; // 16,776,960

export interface DispatchChunk {
  /** Splat-index offset for this chunk (0 for the first). */
  chunkOffset: number;
  /** Number of workgroups to dispatch for this chunk (<= 65535). */
  workgroupCount: number;
  /** Number of splats this chunk's dispatch covers. */
  splatCount: number;
}

/**
 * Carve `splatCount` total splats into 1..N chunks of <= `dispatchCap`
 * workgroups (default 65535). Each chunk's workgroup count is `ceil(slice/wg)`.
 *
 * Pure function. Used both by the dispatch wrapper and the unit test.
 */
export function planDispatchChunks(
  splatCount: number,
  workgroupSize: number = SPLAT_WORKGROUP_SIZE,
  dispatchCap: number = WEBGPU_MAX_DISPATCH_PER_DIM,
): DispatchChunk[] {
  if (splatCount <= 0) return [];
  if (!Number.isFinite(splatCount) || splatCount < 0) {
    throw new Error(`planDispatchChunks: invalid splatCount ${splatCount}`);
  }
  const splatsPerChunk = dispatchCap * workgroupSize;
  const chunks: DispatchChunk[] = [];
  let remaining = splatCount;
  let offset = 0;
  while (remaining > 0) {
    const slice = Math.min(remaining, splatsPerChunk);
    const wgs = Math.ceil(slice / workgroupSize);
    chunks.push({ chunkOffset: offset, workgroupCount: wgs, splatCount: slice });
    offset += slice;
    remaining -= slice;
  }
  return chunks;
}

/**
 * Per-chunk dispatch wrapper. The caller owns the queue and uniform buffer;
 * we update the chunk-offset slot between chunks (queue.writeBuffer) and
 * begin a fresh compute pass per chunk so the uniform write is observable
 * by the next dispatch (WebGPU requires queue writes to be ordered with
 * encoder boundaries — the safe pattern is one pass per uniform mutation).
 *
 * @param device GPUDevice owning `uniformBuffer`.
 * @param encoder current command encoder; we open & close one compute
 *               pass per chunk inside it (so all chunks land on the same
 *               submission boundary).
 * @param pipeline the compute pipeline to dispatch.
 * @param bindGroup the bind group (must include `uniformBuffer` at
 *                  whichever binding the kernel expects).
 * @param uniformBuffer the buffer containing the kernel's uniform struct.
 * @param uniformChunkOffsetBytes byte offset of the `chunk_offset` u32
 *                                slot inside the uniform struct.
 * @param splatCount total splats to cover (the SAME `splat_count` the
 *                   kernel uniform was packed with).
 * @param workgroupSize splats per workgroup. Defaults to 256.
 *
 * Behaviour notes:
 *   - splatCount === 0 ⇒ no-op (no pass opened).
 *   - splatCount fits in one dispatch ⇒ a single pass with
 *     chunk_offset = 0 (the kernel's existing one-dispatch behaviour).
 *   - We always rewrite chunk_offset BEFORE the pass that uses it,
 *     even for chunk 0, to guarantee it isn't stale from a prior frame.
 */
export function dispatchPerSplat(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  uniformBuffer: GPUBuffer,
  uniformChunkOffsetBytes: number,
  splatCount: number,
  workgroupSize: number = SPLAT_WORKGROUP_SIZE,
): number {
  const chunks = planDispatchChunks(splatCount, workgroupSize);
  if (chunks.length === 0) return 0;
  const scratch = new Uint32Array(1);
  for (const chunk of chunks) {
    scratch[0] = chunk.chunkOffset >>> 0;
    device.queue.writeBuffer(uniformBuffer, uniformChunkOffsetBytes, scratch.buffer, 0, 4);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(chunk.workgroupCount);
    pass.end();
  }
  return chunks.length;
}
