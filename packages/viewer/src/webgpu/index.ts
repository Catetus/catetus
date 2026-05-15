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

import { DECODE_WGSL, RADIX_SORT_WGSL, PROJECT_GATHER_WGSL } from './shaders.generated.js';
import { createRadixSortPipelines, RadixSort, type RadixSortPipelines } from './radix_sort.js';
import { createCullPipelines, CullPipeline } from './cull.js';
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
  /** Optional fused project+gather path (depth-only keygen + sorted-order project). */
  keygen?: GPUComputePipeline;
  projectGather?: GPUComputePipeline;
  keygenBgl?: GPUBindGroupLayout;
  projectGatherBgl?: GPUBindGroupLayout;
}

/**
 * Replace the EWA-anti-aliasing 2D-covariance dilation floor `let reg = 0.3;`
 * (marked with the `SF_EWA_DILATION` token) with `let reg = <dilation>;` so
 * a caller can shrink or disable the screen-space sigma_min^2 = 0.3 floor
 * that's inherited from the Inria CUDA rasterizer. See research log entry
 * for 2026-05-15 (novel-2-renderer) for why this matters: trained scenes
 * have median sigma 0.55-1.3 px so the radius cull never fires, but dropping
 * the floor to ~0.05-0.1 px brings cull rate from <1% up to 10-30%+ on
 * already-trained PLYs with negligible visible-pixel impact.
 */
function applyDilationOverride(wgsl: string, dilation: number): string {
  if (dilation === 0.3) return wgsl;
  // Render a clean WGSL literal: small but non-zero floor stays positive,
  // exact 0.0 emits "0.0" to keep WGSL happy with type inference.
  const lit = dilation === 0 ? '0.0' : dilation.toFixed(6);
  return wgsl.replace(/let reg = 0\.3; \/\/ SF_EWA_DILATION/g, `let reg = ${lit}; // SF_EWA_DILATION(override=${dilation})`);
}

function createDecodePipelines(device: GPUDevice, includeFused: boolean, dilation: number = 0.3): DecodePipelines {
  const decodeMod = device.createShaderModule({ code: applyDilationOverride(DECODE_WGSL, dilation) });
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

  if (!includeFused) {
    return { decode, project, gather, decodeBgl, projectBgl, gatherBgl };
  }

  // -----------------------------------------------------------------
  // Fused project+gather pipeline. See cs_project_gather.wgsl.
  //
  //   cs_keygen:           splats + camera → keys[] + indices[].
  //                        Bindings: (0 splats, 1 keys, 2 indices, 3 uniforms).
  //   cs_project_gather:   splats + sorted_indices[] + camera → instanceBuffer[].
  //                        Bindings: (0 splats, 1 indices, 2 inst_out, 3 uniforms).
  // -----------------------------------------------------------------
  const fusedMod = device.createShaderModule({ code: applyDilationOverride(PROJECT_GATHER_WGSL, dilation) });
  const keygenBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const projectGatherBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const keygen = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [keygenBgl] }),
    compute: { module: fusedMod, entryPoint: 'cs_keygen' },
  });
  const projectGather = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [projectGatherBgl] }),
    compute: { module: fusedMod, entryPoint: 'cs_project_gather' },
  });

  return {
    decode, project, gather, decodeBgl, projectBgl, gatherBgl,
    keygen, projectGather, keygenBgl, projectGatherBgl,
  };
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
  /**
   * Use the fused project+gather path: depth-only keygen → radix sort →
   * direct-to-vertex projection in sorted order. Eliminates the per-frame
   * 640 MB read+write of the `instUnsorted` scratch buffer at 10 M splats.
   *
   * Default: `true`. Set `false` to use the original separate cs_project +
   * cs_gather path (kept as a deterministic fallback and reference for the
   * parity test).
   */
  useFusedProject?: boolean;
  /**
   * Enable opacity-radius pre-sort cull. When true, `encodeWithCull` runs
   * the cull/compact stages before keygen so the radix sort + gather only
   * see the survivors. Default: false (cull pipeline is allocated lazily
   * but inactive). The bench harness opts in explicitly.
   */
  useCull?: boolean;
  /**
   * EWA anti-aliasing 2D-covariance dilation floor (the `sigma_min^2` added
   * to the diagonal of the projected screen-space covariance). Inherited
   * from the Inria CUDA rasterizer where it's hard-coded to 0.3 — the
   * default here matches that behaviour. Lowering it (e.g. 0.05 - 0.1)
   * tightens sub-pixel Gaussians and lets the opacity-radius pre-sort cull
   * (see `useCull`) actually fire on production-trained PLYs at the cost
   * of some screen-space anti-aliasing fidelity. See research log
   * 2026-05-15 (novel-2-renderer) for the sweep + decision criteria.
   * Default: 0.3.
   */
  dilation?: number;
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
  /** True when the fused project+gather path is active. */
  readonly useFusedProject: boolean;
  /** EWA dilation floor applied to projected 2D covariance. See `ComputeDecodePipelineInit.dilation`. */
  readonly dilation: number;
  private readonly pipes: DecodePipelines;
  private readonly radixPipes: RadixSortPipelines;
  /** Canonical decoded-splat buffer. One per-splat record across all chunks. */
  private readonly splatsBuffer: GPUBuffer;
  /** Unsorted instance buffer (project shader output). Only allocated in the non-fused path. */
  private readonly instUnsorted: GPUBuffer | null;
  /** Sorted final instance buffer. Used as the vertex buffer by the renderer. */
  readonly instanceBuffer: GPUBuffer;
  /** Radix-sort runner. `keysA`/`valuesA` are scratch we write into in cs_project. */
  private readonly sorter: RadixSort;
  /** Project pass uniform buffer (view + viewProj + viewport + focal + count). */
  private readonly projectUniforms: GPUBuffer;
  /** Bind group for cs_project (non-fused only). */
  private readonly projectBindGroup: GPUBindGroup | null;
  /** Gather pass uniform (count). Non-fused only. */
  private readonly gatherUniforms: GPUBuffer | null;
  private readonly gatherBindGroup: GPUBindGroup | null;
  /** Bind groups for the fused path (keygen + project_gather). */
  private readonly keygenBindGroup: GPUBindGroup | null;
  private readonly projectGatherBindGroup: GPUBindGroup | null;
  /** Optional opacity-radius pre-sort cull. Allocated when useCull=true. */
  readonly cull: CullPipeline | null;
  /** True when useCull was requested at construction. */
  readonly useCull: boolean;
  private readonly chunks: DecodedChunk[] = [];
  /** Splats already decoded (offset into `splatsBuffer`). */
  private decodedSplats = 0;

  constructor(init: ComputeDecodePipelineInit) {
    this.device = init.device;
    this.capacity = init.capacity;
    // PERFORMANCE NOTE: the fused `cs_project_gather` path (true) eliminates
    // a 640 MB unsorted-scratch pass at 10 M splats but pays a 2.3× cost in
    // per-frame GPU time on the laptop 4090 — projection is recomputed in
    // sorted order with cache-hostile reads back into `splats[]`, vs the
    // non-fused path which projects once into a coherent scratch buffer and
    // the per-frame gather is a 64-byte memcpy. We default to the non-fused
    // path (false) until the fuse is made bandwidth-efficient. Real per-stage
    // numbers in `docs/perf/webgpu-10m-profile.md`.
    this.useFusedProject = init.useFusedProject ?? false;
    this.dilation = init.dilation ?? 0.3;
    this.pipes = createDecodePipelines(this.device, this.useFusedProject, this.dilation);
    this.radixPipes = createRadixSortPipelines(this.device, RADIX_SORT_WGSL);

    const decodedSize = Math.max(this.capacity * BYTES_PER_DECODED_SPLAT, BYTES_PER_DECODED_SPLAT);
    this.splatsBuffer = this.device.createBuffer({
      size: decodedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const instSize = Math.max(this.capacity * FLOATS_PER_INSTANCE * 4, FLOATS_PER_INSTANCE * 4);
    // The 640-MB-at-10M scratch buffer is only needed for the non-fused path.
    // In the fused path we write the final instance record directly from the
    // project_gather kernel, so we skip the allocation entirely.
    this.instUnsorted = this.useFusedProject
      ? null
      : this.device.createBuffer({ size: instSize, usage: GPUBufferUsage.STORAGE });
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

    if (!this.useFusedProject) {
      this.projectBindGroup = this.device.createBindGroup({
        layout: this.pipes.projectBgl,
        entries: [
          { binding: 0, resource: { buffer: this.splatsBuffer } },
          { binding: 1, resource: { buffer: this.instUnsorted! } },
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
          { binding: 0, resource: { buffer: this.instUnsorted! } },
          { binding: 1, resource: { buffer: this.sorter.valuesA } },
          { binding: 2, resource: { buffer: this.instanceBuffer } },
          { binding: 3, resource: { buffer: this.gatherUniforms } },
        ],
      });
      this.keygenBindGroup = null;
      this.projectGatherBindGroup = null;
    } else {
      this.projectBindGroup = null;
      this.gatherUniforms = null;
      this.gatherBindGroup = null;
      // Fused path: reuse the same projectUniforms buffer (matching struct).
      this.keygenBindGroup = this.device.createBindGroup({
        layout: this.pipes.keygenBgl!,
        entries: [
          { binding: 0, resource: { buffer: this.splatsBuffer } },
          { binding: 1, resource: { buffer: this.sorter.keysA } },
          { binding: 2, resource: { buffer: this.sorter.valuesA } },
          { binding: 3, resource: { buffer: this.projectUniforms } },
        ],
      });
      this.projectGatherBindGroup = this.device.createBindGroup({
        layout: this.pipes.projectGatherBgl!,
        entries: [
          { binding: 0, resource: { buffer: this.splatsBuffer } },
          { binding: 1, resource: { buffer: this.sorter.valuesA } },
          { binding: 2, resource: { buffer: this.instanceBuffer } },
          { binding: 3, resource: { buffer: this.projectUniforms } },
        ],
      });
    }

    // -----------------------------------------------------------------
    // Optional opacity-radius cull. Allocated only when requested so the
    // existing tests + production renderer (which don't yet integrate the
    // cull's survivor-count readback) stay byte-identical.
    // -----------------------------------------------------------------
    this.useCull = init.useCull ?? false;
    if (this.useCull) {
      const cullPipes = createCullPipelines(this.device, this.dilation);
      // For the cull path we write Instance records straight into the
      // unsorted scratch buffer (non-fused only — fused-path integration
      // is a follow-up because the fused path skips the scratch buffer
      // entirely). Bench harness uses the non-fused path by default per
      // commit 354cd8e.
      if (this.useFusedProject) {
        throw new Error('useCull is only supported on the non-fused project path');
      }
      this.cull = new CullPipeline({
        device: this.device,
        capacity: this.capacity,
        pipes: cullPipes,
        splatsBuffer: this.splatsBuffer,
        keysBuffer: this.sorter.keysA,
        valuesBuffer: this.sorter.valuesA,
        instBuffer: this.instUnsorted!,
        projectUniforms: this.projectUniforms,
      });
    } else {
      this.cull = null;
    }
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

    const wgs = Math.ceil(count / 256);

    if (this.useFusedProject) {
      // Fused path:
      //   1. cs_keygen        — depth-only key + identity index.
      //   2. radix sort       — sorts (key, index) ascending by key.
      //   3. cs_project_gather — re-projects in sorted order, writes
      //                          instanceBuffer[i] directly. No 640 MB
      //                          unsorted-scratch buffer touched.
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipes.keygen!);
        pass.setBindGroup(0, this.keygenBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      this.sorter.encode(encoder, count);
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipes.projectGather!);
        pass.setBindGroup(0, this.projectGatherBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      return;
    }

    // Non-fused (legacy) path: cs_project → radix sort → cs_gather.
    // Kept as a fallback / parity reference.
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.project);
      pass.setBindGroup(0, this.projectBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
    this.sorter.encode(encoder, count);
    {
      const u = new Uint32Array(8); // 32 bytes
      u[0] = count;
      this.device.queue.writeBuffer(this.gatherUniforms!, 0, u.buffer);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
  }

  /**
   * Cull-enabled production encode. Runs:
   *   1. cs_cull + scan + cs_compact + cs_project_cmpct  (writes keysA /
   *      valuesA / instUnsorted at the first `survivors` slots).
   *   2. Radix sort over `survivors`.
   *   3. cs_gather over `survivors`.
   *
   * Requires the cull's survivor count to have been refreshed via
   * `cull.readSurvivorCount()` after a prior `encodeWithCull` was
   * submitted. On the very first frame, the count is 0 and the sort /
   * gather are skipped — the caller is expected to do a warm-up
   * submit + readback before the timed loop begins.
   *
   * `tau` is the opacity floor (default 1/255). The bench overrides it
   * if 1/255 prunes too many splats on the synthetic scene.
   */
  async encodeWithCull(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    tau: number = 1 / 255,
  ): Promise<void> {
    if (!this.cull) throw new Error('encodeWithCull requires useCull=true');
    const count = this.decodedSplats;
    if (count === 0) return;

    // Pack project uniforms — same packing as encode(), but the `splat_count`
    // slot will be overwritten by the host to `survivors` for cs_project_cmpct.
    const ab = new ArrayBuffer(160);
    const f32 = new Float32Array(ab);
    const i32 = new Int32Array(ab);
    f32.set(view, 0);
    f32.set(viewProj, 16);
    f32[32] = viewport[0]; f32[33] = viewport[1];
    f32[34] = focal[0];    f32[35] = focal[1];
    i32[36] = this.cull.cachedSurvivors > 0 ? this.cull.cachedSurvivors : count;
    this.device.queue.writeBuffer(this.projectUniforms, 0, ab);

    // 1. Cull + compact + project (project only fires if cachedSurvivors > 0).
    this.cull.encode(encoder, view, viewProj, focal, viewport, count, tau);

    // 2. Radix sort + gather over survivors. On the very first frame,
    //    cachedSurvivors is 0; we skip these and let the readback populate
    //    the count for the next frame.
    const survivors = this.cull.cachedSurvivors;
    if (survivors > 0) {
      this.sorter.encode(encoder, survivors);
      const u = new Uint32Array(8);
      u[0] = survivors;
      this.device.queue.writeBuffer(this.gatherUniforms!, 0, u.buffer);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup!);
      pass.dispatchWorkgroups(Math.ceil(survivors / 256));
      pass.end();
    }
  }

  /**
   * Timestamp-instrumented variant of `encodeWithCull`. Writes 8 timestamps:
   *   [0..1) cull+scan+compact   (combined; cull internals not drilled)
   *   [2..3) project_cmpct
   *   [4..5) sort_full
   *   [6..7) gather
   * Returns the number of timestamps written. Requires `timestamp-query`.
   */
  encodeWithCullTimed(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    querySet: GPUQuerySet,
    baseIndex: number,
    tau: number = 1 / 255,
  ): number {
    if (!this.cull) throw new Error('encodeWithCullTimed requires useCull=true');
    const count = this.decodedSplats;
    if (count === 0) return 0;

    const ab = new ArrayBuffer(160);
    const f32 = new Float32Array(ab);
    const i32 = new Int32Array(ab);
    f32.set(view, 0);
    f32.set(viewProj, 16);
    f32[32] = viewport[0]; f32[33] = viewport[1];
    f32[34] = focal[0];    f32[35] = focal[1];
    i32[36] = this.cull.cachedSurvivors > 0 ? this.cull.cachedSurvivors : count;
    this.device.queue.writeBuffer(this.projectUniforms, 0, ab);

    // Re-pack the cull's own uniforms.
    const cull = this.cull;
    {
      const cb = new ArrayBuffer(cull.cullUniforms.size);
      const cf = new Float32Array(cb);
      const cu = new Uint32Array(cb);
      cf.set(view, 0);
      cf.set(viewProj, 16);
      cf[32] = viewport[0]; cf[33] = viewport[1];
      cf[34] = focal[0];    cf[35] = focal[1];
      cu[36] = count;
      cf[37] = tau;
      this.device.queue.writeBuffer(cull.cullUniforms, 0, cb);
    }
    const numScanWgs = Math.ceil(count / 256);
    {
      const u = new Uint32Array(8);
      u[0] = count;
      u[1] = numScanWgs;
      this.device.queue.writeBuffer(cull.scanUniforms, 0, u.buffer);
    }
    {
      const u = new Uint32Array(8);
      u[0] = count;
      this.device.queue.writeBuffer(cull.compactUniforms, 0, u.buffer);
    }
    const wgs = Math.ceil(count / 256);

    // Window 1: cull + scan + compact.
    {
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: baseIndex + 0,
          endOfPassWriteIndex: baseIndex + 1,
        },
      });
      pass.setPipeline(cull.pipes.cull);
      pass.setBindGroup(0, (cull as unknown as { cullBindGroup: GPUBindGroup }).cullBindGroup);
      pass.dispatchWorkgroups(wgs);
      pass.setPipeline(cull.pipes.scanPerWg);
      pass.setBindGroup(0, (cull as unknown as { scanBindGroup: GPUBindGroup }).scanBindGroup);
      pass.dispatchWorkgroups(numScanWgs);
      pass.setPipeline(cull.pipes.scanBlockSums);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(cull.pipes.scanAddBlockSums);
      pass.dispatchWorkgroups(numScanWgs);
      pass.setPipeline(cull.pipes.compact);
      pass.setBindGroup(0, (cull as unknown as { compactBindGroup: GPUBindGroup }).compactBindGroup);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }

    // Copy the tail readback (after compact's pass closed).
    const tailOff = (count - 1) * 4;
    encoder.copyBufferToBuffer(cull.prefixBuffer, tailOff, cull.readbackTail, 0, 4);
    encoder.copyBufferToBuffer(cull.flagBuffer,   tailOff, cull.readbackTail, 4, 4);

    const survivors = cull.cachedSurvivors;
    if (survivors === 0) {
      // No survivors yet — skip the downstream stages but emit zero-width
      // timestamp windows so the resolveQuerySet copy stays valid.
      for (let k = 2; k < 8; k += 2) {
        const p = encoder.beginComputePass({
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: baseIndex + k,
            endOfPassWriteIndex: baseIndex + k + 1,
          },
        });
        p.end();
      }
      return 8;
    }

    // Window 2: project_cmpct.
    {
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: baseIndex + 2,
          endOfPassWriteIndex: baseIndex + 3,
        },
      });
      pass.setPipeline(cull.pipes.projectCmpct);
      pass.setBindGroup(0, (cull as unknown as { projectCmpctBindGroup: GPUBindGroup }).projectCmpctBindGroup);
      pass.dispatchWorkgroups(Math.ceil(survivors / 256));
      pass.end();
    }

    // Window 3: sort_full.
    this.sorter.encodeTimed(encoder, survivors, querySet, baseIndex + 4);

    // Window 4: gather.
    {
      const u = new Uint32Array(8);
      u[0] = survivors;
      this.device.queue.writeBuffer(this.gatherUniforms!, 0, u.buffer);
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: baseIndex + 6,
          endOfPassWriteIndex: baseIndex + 7,
        },
      });
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup!);
      pass.dispatchWorkgroups(Math.ceil(survivors / 256));
      pass.end();
    }
    return 8;
  }

    /**
   * Bench-only variant of `encode` that writes timestamps around each top-
   * level compute pass. Returns the number of timestamps written, so the
   * caller knows how to slice the resolve buffer.
   *
   * Layout, fused path (the only path we ship by default):
   *   [0..1) keygen          [1..2) sort_full          [2..3) project_gather
   * Non-fused fallback:
   *   [0..1) project_unsorted [1..2) sort_full         [2..3) gather
   *
   * Requires the `timestamp-query` feature on the device. The bench owns
   * the GPUQuerySet so we keep the dependency one-way (production code does
   * not import any timestamp plumbing).
   */
  encodeTimed(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    querySet: GPUQuerySet,
    baseIndex: number,
  ): number {
    const count = this.decodedSplats;
    if (count === 0) return 0;

    const ab = new ArrayBuffer(160);
    const f32 = new Float32Array(ab);
    const i32 = new Int32Array(ab);
    f32.set(view, 0);
    f32.set(viewProj, 16);
    f32[32] = viewport[0]; f32[33] = viewport[1];
    f32[34] = focal[0];    f32[35] = focal[1];
    i32[36] = count;
    this.device.queue.writeBuffer(this.projectUniforms, 0, ab);

    const wgs = Math.ceil(count / 256);

    if (this.useFusedProject) {
      // keygen
      {
        const pass = encoder.beginComputePass({
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: baseIndex + 0,
            endOfPassWriteIndex: baseIndex + 1,
          },
        });
        pass.setPipeline(this.pipes.keygen!);
        pass.setBindGroup(0, this.keygenBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      // sort_full — radix sort orchestrates its own 4 passes; we wrap the
      // *whole* sort in one timestamp window. Drilling into histogram/scan/
      // scatter requires `timestamp-query-inside-passes`, which we'll add
      // in a follow-up if sort_full proves to dominate frame time.
      this.sorter.encodeTimed(encoder, count, querySet, baseIndex + 2);
      // project_gather
      {
        const pass = encoder.beginComputePass({
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: baseIndex + 4,
            endOfPassWriteIndex: baseIndex + 5,
          },
        });
        pass.setPipeline(this.pipes.projectGather!);
        pass.setBindGroup(0, this.projectGatherBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      return 6;
    }

    // Non-fused (legacy) path.
    {
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: baseIndex + 0,
          endOfPassWriteIndex: baseIndex + 1,
        },
      });
      pass.setPipeline(this.pipes.project);
      pass.setBindGroup(0, this.projectBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
    this.sorter.encodeTimed(encoder, count, querySet, baseIndex + 2);
    {
      const u = new Uint32Array(8);
      u[0] = count;
      this.device.queue.writeBuffer(this.gatherUniforms!, 0, u.buffer);
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: baseIndex + 4,
          endOfPassWriteIndex: baseIndex + 5,
        },
      });
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
    return 6;
  }

  /**
   * Bench-only variant that drills the radix sort into per-sub-stage
   * timestamps. Writes 20 timestamps total:
   *   [0..1)   keygen / project
   *   [2..3)   pass0_histogram
   *   [4..5)   pass0_scan_per_wg
   *   [6..7)   pass0_scan_block_sums
   *   [8..9)   pass0_scan_add_block_sums
   *   [10..11) pass0_scatter
   *   [12..13) pass1_full
   *   [14..15) pass2_full
   *   [16..17) pass3_full
   *   [18..19) project_gather / gather
   * Caller must supply a querySet of capacity >= 20.
   */
  encodeTimedDrilled(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    querySet: GPUQuerySet,
    baseIndex: number,
  ): number {
    const count = this.decodedSplats;
    if (count === 0) return 0;

    const ab = new ArrayBuffer(160);
    const f32 = new Float32Array(ab);
    const i32 = new Int32Array(ab);
    f32.set(view, 0);
    f32.set(viewProj, 16);
    f32[32] = viewport[0]; f32[33] = viewport[1];
    f32[34] = focal[0];    f32[35] = focal[1];
    i32[36] = count;
    this.device.queue.writeBuffer(this.projectUniforms, 0, ab);

    const wgs = Math.ceil(count / 256);
    const ts = (begin: number, end: number): GPUComputePassDescriptor => ({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: begin,
        endOfPassWriteIndex: end,
      },
    });

    if (this.useFusedProject) {
      {
        const pass = encoder.beginComputePass(ts(baseIndex + 0, baseIndex + 1));
        pass.setPipeline(this.pipes.keygen!);
        pass.setBindGroup(0, this.keygenBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      this.sorter.encodeTimedDrilled(encoder, count, querySet, baseIndex + 2);
      {
        const pass = encoder.beginComputePass(ts(baseIndex + 18, baseIndex + 19));
        pass.setPipeline(this.pipes.projectGather!);
        pass.setBindGroup(0, this.projectGatherBindGroup!);
        pass.dispatchWorkgroups(wgs);
        pass.end();
      }
      return 20;
    }

    {
      const pass = encoder.beginComputePass(ts(baseIndex + 0, baseIndex + 1));
      pass.setPipeline(this.pipes.project);
      pass.setBindGroup(0, this.projectBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
    this.sorter.encodeTimedDrilled(encoder, count, querySet, baseIndex + 2);
    {
      const u = new Uint32Array(8);
      u[0] = count;
      this.device.queue.writeBuffer(this.gatherUniforms!, 0, u.buffer);
      const pass = encoder.beginComputePass(ts(baseIndex + 18, baseIndex + 19));
      pass.setPipeline(this.pipes.gather);
      pass.setBindGroup(0, this.gatherBindGroup!);
      pass.dispatchWorkgroups(wgs);
      pass.end();
    }
    return 20;
  }

  /** Tear down. Idempotent. */
  destroy(): void {
    for (const c of this.chunks) {
      c.bytesBuffer.destroy();
      c.decodeUniforms.destroy();
    }
    this.chunks.length = 0;
    this.splatsBuffer.destroy();
    this.instUnsorted?.destroy();
    this.instanceBuffer.destroy();
    this.projectUniforms.destroy();
    this.gatherUniforms?.destroy();
    this.sorter.destroy();
    this.cull?.destroy();
  }
}

export { createRadixSortPipelines, RadixSort } from './radix_sort.js';
export { DECODE_WGSL, RADIX_SORT_WGSL, PROJECT_GATHER_WGSL, CULL_WGSL } from './shaders.generated.js';
export { CullPipeline, createCullPipelines } from './cull.js';
