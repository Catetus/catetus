// SPDX-License-Identifier: Apache-2.0
/**
 * GPU compute-decode + radix-sort pipeline for `@splatforge/viewer`.
 *
 * Replaces the CPU decode + JS sort path used by {@link WebGPURenderer}
 * when the viewer is constructed with `useComputeDecode: true`.
 *
 * Pipeline (all in WGSL, all on the GPU):
 *
 *   raw chunk bytes                 (storage<read>)
 *           │
 *           ▼  cs_decode  — SoA → canonical DecodedSplat (position, scale,
 *           │              rotation, opacity, colorDC). 1 thread per splat.
 *           ▼
 *      decoded splats               (storage<read_write>)
 *           │
 *           ▼  cs_project — Canonical splats + camera → per-instance vertex
 *           │              attributes (clipPos, 2D covariance, color) and
 *           │              depth-sort keys + indirection indices.
 *           ▼
 *      instance buffer (unsorted)   (storage)
 *      keys + indices                (storage)
 *           │
 *           ▼  radix sort (8×4-bit passes, three kernels per pass).
 *           ▼
 *      sorted indices                (storage)
 *           │
 *           ▼  cs_gather — write the final instance buffer in sorted order.
 *           ▼
 *      sorted instance buffer        (vertex)
 *
 * The final buffer is bound as a vertex buffer to the existing render
 * pipeline in `renderer/webgpu.ts` — the rasterizer stays untouched.
 *
 * Determinism: every step is data-parallel without atomic order-dependence on
 * storage buffers. The only atomics are workgroup-shared counters (mandatory
 * in WebGPU 1.0). Equal-depth splats fall back to splat-index by virtue of
 * the radix sort packing the splat index into the value buffer; ties in the
 * key resolve in scatter order, which is deterministic for a fixed dispatch.
 */

import { DECODE_WGSL, RADIX_SORT_WGSL } from './shaders.generated.js';
import { createRadixSortPipelines, RadixSort, type RadixSortPipelines } from './radix_sort.js';
import type { ChunkDescriptor, SoaAttributeLayout } from '../manifest.js';

/** Floats per per-instance render record. Mirrors `FLOATS_PER_INSTANCE` in webgpu.ts. */
export const FLOATS_PER_INSTANCE = 12;

/** Bytes per canonical decoded splat (4×vec4 = 64 bytes). Matches `DecodedSplat` in WGSL. */
export const BYTES_PER_DECODED_SPLAT = 64;

const FLOAT_CT = 5126;
const UBYTE_CT = 5121;
const USHORT_CT = 5123;

/* --------------------------------------------------------------------------- */
/* Gather shader (sorted indices → final instance buffer).                     */
/*                                                                             */
/* Kept tiny + inline because it doesn't need to live in its own .wgsl file.   */
/* --------------------------------------------------------------------------- */
const GATHER_WGSL = /* wgsl */ `
struct Instance {
  clip_pos: vec4<f32>,
  cov:      vec4<f32>,
  color:    vec4<f32>,
};
struct Uniforms { count: u32, _pad: vec3<u32> };
@group(0) @binding(0) var<storage, read>       src    : array<Instance>;
@group(0) @binding(1) var<storage, read>       order  : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst    : array<Instance>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;
@compute @workgroup_size(256)
fn cs_gather(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  dst[i] = src[order[i]];
}
`;

/* --------------------------------------------------------------------------- */
/* Decode + project pipelines.                                                 */
/* --------------------------------------------------------------------------- */

interface DecodePipelines {
  decode: GPUComputePipeline;
  project: GPUComputePipeline;
  gather: GPUComputePipeline;
  decodeBgl: GPUBindGroupLayout;
  projectBgl: GPUBindGroupLayout;
  gatherBgl: GPUBindGroupLayout;
}

function createDecodePipelines(device: GPUDevice): DecodePipelines {
  const decodeMod = device.createShaderModule({ code: DECODE_WGSL });
  const decodeBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const projectBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const decode = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [decodeBgl] }),
    compute: { module: decodeMod, entryPoint: 'cs_decode' },
  });
  const project = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [projectBgl] }),
    compute: { module: decodeMod, entryPoint: 'cs_project' },
  });
  const gatherMod = device.createShaderModule({ code: GATHER_WGSL });
  const gatherBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const gather = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [gatherBgl] }),
    compute: { module: gatherMod, entryPoint: 'cs_gather' },
  });
  return { decode, project, gather, decodeBgl, projectBgl, gatherBgl };
}

/* --------------------------------------------------------------------------- */
/* Per-chunk decode state.                                                     */
/* --------------------------------------------------------------------------- */

interface DecodedChunk {
  splatCount: number;
  bytesBuffer: GPUBuffer;
  splatsBuffer: GPUBuffer;          // canonical decoded splats (64 B each)
  decodeUniforms: GPUBuffer;
  decodeBindGroup: GPUBindGroup;
}

function componentTypeId(t: number | undefined): number {
  // WGSL only knows about three component types. Default to f32 if absent.
  if (t === USHORT_CT) return USHORT_CT;
  if (t === UBYTE_CT) return UBYTE_CT;
  return FLOAT_CT;
}

/**
 * Pack the per-slice decode uniforms (5 attribute slices + count) into a
 * 4-byte-aligned buffer that matches the WGSL `DecodeUniforms` struct.
 */
function buildDecodeUniforms(
  device: GPUDevice,
  layout: SoaAttributeLayout,
  splatCount: number,
): GPUBuffer {
  const SLICE_FLOATS = 12; // 4 ints + vec4 min + vec4 max
  const FLOATS = 4 + 5 * SLICE_FLOATS;
  const ab = new ArrayBuffer(FLOATS * 4);
  const i32 = new Int32Array(ab);
  const f32 = new Float32Array(ab);
  let o = 0;
  i32[o++] = splatCount;
  i32[o++] = 0;
  i32[o++] = 0;
  i32[o++] = 0;
  for (const slice of [layout.positions, layout.rotations, layout.scales, layout.opacities, layout.colorDC]) {
    i32[o++] = slice.byteOffset;
    i32[o++] = componentTypeId(slice.componentType);
    i32[o++] = slice.normalized ? 1 : 0;
    i32[o++] = 0; // _pad
    // vmin
    for (let k = 0; k < 4; k++) f32[o + k] = slice.min?.[k] ?? 0;
    o += 4;
    // vmax
    for (let k = 0; k < 4; k++) f32[o + k] = slice.max?.[k] ?? 0;
    o += 4;
  }
  const buf = device.createBuffer({
    size: ab.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

/* --------------------------------------------------------------------------- */
/* Public compute-decode pipeline.                                             */
/* --------------------------------------------------------------------------- */

export interface ComputeDecodePipelineInit {
  device: GPUDevice;
  /** Maximum total splats across all chunks. Sizes the output buffers. */
  capacity: number;
}

/**
 * GPU compute-decode + radix-sort pipeline. Holds device-lifetime state
 * (compiled pipelines, sorter, transient buffers).
 *
 * Usage:
 *   1. Construct once after `device.requestDevice()`.
 *   2. For each chunk, call `uploadChunk(descriptor, bytes)`.
 *   3. Each frame, call `encode(encoder, view, viewProj, focal, viewport)`
 *      and bind the resulting `instanceBuffer` to the render pipeline.
 */
export class ComputeDecodePipeline {
  readonly device: GPUDevice;
  readonly capacity: number;
  private readonly pipes: DecodePipelines;
  private readonly radixPipes: RadixSortPipelines;
  /** Canonical decoded-splat buffer. One per-splat record across all chunks. */
  private readonly splatsBuffer: GPUBuffer;
  /** Unsorted instance buffer (project shader output). */
  private readonly instUnsorted: GPUBuffer;
  /** Sorted final instance buffer. Used as the vertex buffer by the renderer. */
  readonly instanceBuffer: GPUBuffer;
  /** Radix-sort runner. `keysA`/`valuesA` are scratch we write into in cs_project. */
  private readonly sorter: RadixSort;
  /** Project pass uniform buffer (view + viewProj + viewport + focal + count). */
  private readonly projectUniforms: GPUBuffer;
  private readonly projectBindGroup: GPUBindGroup;
  /** Gather pass uniform (count). */
  private readonly gatherUniforms: GPUBuffer;
  private readonly gatherBindGroup: GPUBindGroup;
  private readonly chunks: DecodedChunk[] = [];
  /** Splats already decoded (offset into `splatsBuffer`). */
  private decodedSplats = 0;

  constructor(init: ComputeDecodePipelineInit) {
    this.device = init.device;
    this.capacity = init.capacity;
    this.pipes = createDecodePipelines(this.device);
    this.radixPipes = createRadixSortPipelines(this.device, RADIX_SORT_WGSL);

    const decodedSize = Math.max(this.capacity * BYTES_PER_DECODED_SPLAT, BYTES_PER_DECODED_SPLAT);
    this.splatsBuffer = this.device.createBuffer({
      size: decodedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const instSize = Math.max(this.capacity * FLOATS_PER_INSTANCE * 4, FLOATS_PER_INSTANCE * 4);
    this.instUnsorted = this.device.createBuffer({
      size: instSize,
      usage: GPUBufferUsage.STORAGE,
    });
    this.instanceBuffer = this.device.createBuffer({
      size: instSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.sorter = new RadixSort(this.device, this.capacity, this.radixPipes);

    // Project uniforms: 2 mat4 (32 floats) + viewport vec2 + focal vec2 + count u32 + pads = 40 floats; round to 48 for alignment.
    this.projectUniforms = this.device.createBuffer({
      size: 4 * (16 + 16 + 2 + 2 + 4), // 160 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.projectBindGroup = this.device.createBindGroup({
      layout: this.pipes.projectBgl,
      entries: [
        { binding: 0, resource: { buffer: this.splatsBuffer } },
        { binding: 1, resource: { buffer: this.instUnsorted } },
        { binding: 2, resource: { buffer: this.sorter.keysA } },
        { binding: 3, resource: { buffer: this.sorter.valuesA } },
        { binding: 4, resource: { buffer: this.projectUniforms } },
      ],
    });

    // 32 B minimum: WGSL Uniforms { count: u32, _pad: vec3<u32> } occupies
    // 16 B but WebGPU pads uniform-buffer bindings up to the next power-of-2
    // ≥ the struct size, with a 32 B minimum on most adapters.
    this.gatherUniforms = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gatherBindGroup = this.device.createBindGroup({
      layout: this.pipes.gatherBgl,
      entries: [
        { binding: 0, resource: { buffer: this.instUnsorted } },
        { binding: 1, resource: { buffer: this.sorter.valuesA } },
        { binding: 2, resource: { buffer: this.instanceBuffer } },
        { binding: 3, resource: { buffer: this.gatherUniforms } },
      ],
    });
  }

  /**
   * Stage one chunk's raw bytes onto the GPU and dispatch the decode shader.
   * The decoded splats land at `[decodedSplats, decodedSplats + chunkCount)`
   * inside `splatsBuffer`.
   *
   * Returns immediately after queueing; the decode runs on the GPU before
   * the next frame's project pass.
   */
  uploadChunk(descriptor: ChunkDescriptor, bytes: Uint8Array): void {
    if (!descriptor.attributeLayout) {
      throw new Error('compute-decode: chunk has no attributeLayout (legacy AoS not supported on GPU path)');
    }
    if (descriptor.splatCount === 0) return;
    if (this.decodedSplats + descriptor.splatCount > this.capacity) {
      throw new Error(`compute-decode: capacity exceeded (${this.decodedSplats + descriptor.splatCount} > ${this.capacity})`);
    }
    // Round up to a multiple of 4 for u32-aligned storage reads.
    const padBytes = (bytes.byteLength + 3) & ~3;
    const bytesBuffer = this.device.createBuffer({
      size: Math.max(padBytes, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // writeBuffer requires a 4-byte-aligned source size and offset. Pad with zeros.
    if (padBytes === bytes.byteLength) {
      this.device.queue.writeBuffer(bytesBuffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
    } else {
      const padded = new Uint8Array(padBytes);
      padded.set(bytes);
      this.device.queue.writeBuffer(bytesBuffer, 0, padded.buffer, 0, padBytes);
    }

    const decodeUniforms = buildDecodeUniforms(this.device, descriptor.attributeLayout, descriptor.splatCount);

    // A per-chunk "splats slice" view — since WebGPU doesn't have offset
    // bindings for storage buffers without dynamic offsets, we bind the full
    // buffer and pass the destination offset via a *separate* uniform. The
    // shader writes at `dst_splats[i]`, so we need a second tiny shader OR a
    // per-chunk dst buffer. We pick the simpler latter: per-chunk dst slice
    // expressed via `binding.offset` of the bind group (which IS supported).
    const dstView: GPUBindingResource = {
      buffer: this.splatsBuffer,
      offset: this.decodedSplats * BYTES_PER_DECODED_SPLAT,
      size: descriptor.splatCount * BYTES_PER_DECODED_SPLAT,
    };
    const decodeBindGroup = this.device.createBindGroup({
      layout: this.pipes.decodeBgl,
      entries: [
        { binding: 0, resource: { buffer: bytesBuffer } },
        { binding: 1, resource: dstView },
        { binding: 2, resource: { buffer: decodeUniforms } },
      ],
    });

    // Dispatch decode immediately. Subsequent project passes will see the
    // decoded splats.
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipes.decode);
    pass.setBindGroup(0, decodeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(descriptor.splatCount / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.chunks.push({
      splatCount: descriptor.splatCount,
      bytesBuffer,
      splatsBuffer: this.splatsBuffer,
      decodeUniforms,
      decodeBindGroup,
    });
    this.decodedSplats += descriptor.splatCount;
  }

  /** Number of splats decoded so far. */
  get splatCount(): number {
    return this.decodedSplats;
  }

  /**
   * Record project + sort + gather dispatches for the current frame.
   *
   * @param encoder command encoder owned by the caller (typically the renderer).
   * @param view column-major view matrix.
   * @param viewProj column-major view-projection matrix.
   * @param focal `[focalX, focalY]` in pixels.
   * @param viewport `[width, height]` in pixels.
   */
  encode(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
  ): void {
    const count = this.decodedSplats;
    if (count === 0) return;

    // Pack project uniforms: 2 mat4 + viewport + focal + count.
    const ab = new ArrayBuffer(160);
    const f32 = new Float32Array(ab);
    const i32 = new Int32Array(ab);
    f32.set(view, 0);
    f32.set(viewProj, 16);
    f32[32] = viewport[0]; f32[33] = viewport[1];
    f32[34] = focal[0];    f32[35] = focal[1];
    i32[36] = count;
    this.device.queue.writeBuffer(this.projectUniforms, 0, ab);

    // Project pass.
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.project);
      pass.setBindGroup(0, this.projectBindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 256));
      pass.end();
    }

    // Radix sort over (key, value) = (depth-bits, splat-index). After this
    // call, `sorter.valuesA` holds indices in back-to-front order.
    this.sorter.encode(encoder, count);

    // Gather pass: write `instanceBuffer[i] = instUnsorted[sorted_indices[i]]`.
    {
      const u = new Uint32Array(8); // 32 bytes
      u[0] = count;
      this.device.queue.writeBuffer(this.gatherUniforms, 0, u.buffer);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 256));
      pass.end();
    }
  }

  /** Tear down. Idempotent. */
  destroy(): void {
    for (const c of this.chunks) {
      c.bytesBuffer.destroy();
      c.decodeUniforms.destroy();
    }
    this.chunks.length = 0;
    this.splatsBuffer.destroy();
    this.instUnsorted.destroy();
    this.instanceBuffer.destroy();
    this.projectUniforms.destroy();
    this.gatherUniforms.destroy();
    this.sorter.destroy();
  }
}

export { createRadixSortPipelines, RadixSort } from './radix_sort.js';
export { DECODE_WGSL, RADIX_SORT_WGSL } from './shaders.generated.js';
