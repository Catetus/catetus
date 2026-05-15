// SPDX-License-Identifier: Apache-2.0
/**
 * Weighted Sum Rendering (WSR) pipeline — PR1.
 *
 * Order-independent rendering via the Hou et al. (arXiv:2410.18931, ICLR 2025)
 * weighted-sum equation:
 *
 *     C_px = (w_B · c_B + Σ_i  α_i · w(d_i) · c_i)  /  (w_B + Σ_i  α_i · w(d_i))
 *
 * with LC-WSR (Linear Clamped) depth weight `w(d_i) = max(0, 1 - d_i/σ - v_i)`.
 * PR1 ships with the heuristic `σ = 2 × mean_scene_depth`, `v_i = 0`
 * everywhere; PR3 adds per-splat learned `v_i[]` and a manifest-baked σ.
 *
 * This class encapsulates the three new WGSL kernels:
 *
 *   1. `cs_wsr_clear`      — zeros the numerator + denominator accumulators.
 *   2. `cs_wsr_accumulate` — one thread per splat, scatter-adds the weighted
 *                            Gaussian footprint into per-pixel accumulators
 *                            via CAS-loop atomic float-add on u32-bitcasted
 *                            f32 storage buffers (B7.1 portability path).
 *   3. `cs_wsr_resolve`    — combines numerator + denominator into final
 *                            rgba8unorm via the WSR equation; writes to a
 *                            u32 buffer matched to `rgba8unorm` byte order.
 *
 * The `ComputeDecodePipeline` owns the splats buffer and chooses whether to
 * call `encodeWSR()` (this module) or the legacy `cs_keygen → radix_sort →
 * cs_project_gather` path. PR5 flips the default to WSR; until then both
 * paths coexist and the unit-test suite (`wsr.test.ts`) keeps the WSR-path
 * code surface live.
 *
 * Design rationale (called out by the spec § 5 "Two-Pass vs One-Pass"):
 *
 *   We chose **Option B — compute scatter with CAS-loop atomic float-add**
 *   over Option A (fragment-shader ROP additive blending) for PR1 portability.
 *   The spec recommends Option A on a per-frame-cost basis, but Option A
 *   requires the `float32-blendable` WebGPU device feature for rgba32float +
 *   r32float MRT, which is not in any shipped WebGPU 1.0 implementation as of
 *   2026-05 (Chrome canary only). Option B works on any conformant WebGPU 1.0
 *   adapter. The B7.1 EXECUTION-LOG entry (2026-05-15) established that on
 *   the laptop 4090 the scatter path is **DRAM-write-bound, not atomic-bound**
 *   at 10 M splats — atomic-free B7.1 was +0.27 fps vs atomic-add, within
 *   noise. So we incur no measurable cost from the CAS-loop while gaining
 *   portability across Chrome stable, Safari, and Firefox WebGPU. If the
 *   `float32-blendable` feature ships in stable channels before PR5 we
 *   re-evaluate; until then Option B is the deliverable.
 */

import { WSR_CLEAR_WGSL, WSR_ACCUMULATE_WGSL, WSR_RESOLVE_WGSL } from './shaders.generated.js';

/** Workgroup size for the per-pixel kernels (16 × 16). */
export const WSR_TILE = 16;

/** Workgroup size for the per-splat accumulate kernel. */
export const WSR_WG = 256;

/** Bytes per pixel of the numerator buffer (4 × u32, vec4-aligned). */
const NUMERATOR_BYTES_PER_PX = 16;

/** Bytes per pixel of the denominator buffer (1 × u32). */
const DENOMINATOR_BYTES_PER_PX = 4;

/** Bytes per pixel of the resolve output (rgba8unorm packed into 1 × u32). */
const OUTPUT_BYTES_PER_PX = 4;

/** Default background weight (Σ-denominator floor) to prevent div-by-zero. */
export const WSR_DEFAULT_BG_WEIGHT = 1e-4;

export interface WSRPipelines {
  clear: GPUComputePipeline;
  accumulate: GPUComputePipeline;
  resolve: GPUComputePipeline;
  clearBgl: GPUBindGroupLayout;
  accumulateBgl: GPUBindGroupLayout;
  resolveBgl: GPUBindGroupLayout;
}

export function createWSRPipelines(device: GPUDevice): WSRPipelines {
  const clearMod = device.createShaderModule({ code: WSR_CLEAR_WGSL });
  const accumMod = device.createShaderModule({ code: WSR_ACCUMULATE_WGSL });
  const resolveMod = device.createShaderModule({ code: WSR_RESOLVE_WGSL });

  // ---- cs_wsr_clear: (numerator rw, denominator rw, uniforms). ----
  const clearBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  // ---- cs_wsr_accumulate: (splats ro, numerator rw, denominator rw, uniforms). ----
  const accumulateBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  // ---- cs_wsr_resolve: (numerator ro, denominator ro, output rw, uniforms). ----
  const resolveBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const mk = (mod: GPUShaderModule, bgl: GPUBindGroupLayout, entry: string): GPUComputePipeline =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: mod, entryPoint: entry },
    });

  return {
    clear:      mk(clearMod,   clearBgl,      'cs_wsr_clear'),
    accumulate: mk(accumMod,   accumulateBgl, 'cs_wsr_accumulate'),
    resolve:    mk(resolveMod, resolveBgl,    'cs_wsr_resolve'),
    clearBgl, accumulateBgl, resolveBgl,
  };
}

export interface WSRPipelineInit {
  device: GPUDevice;
  /** Maximum viewport width × height in pixels. Sizes the accumulators. */
  maxWidth: number;
  maxHeight: number;
  /** Splats storage buffer (owned by ComputeDecodePipeline). */
  splatsBuffer: GPUBuffer;
  pipes?: WSRPipelines;
}

/**
 * `WSRPipeline` — manages the three WSR kernels and their per-frame state.
 *
 * Lifecycle mirrors `CullPipeline`: device-lifetime resources (pipelines,
 * accumulator buffers, output buffer, uniform buffers, bind groups) are
 * created in the constructor; `encode()` records all three dispatches into
 * the caller's `GPUCommandEncoder`; `destroy()` releases the buffers.
 *
 * The accumulator + output buffers are sized for the worst-case viewport
 * (`maxWidth × maxHeight`) at construction so per-frame resizes don't
 * trigger an allocation. The `encode()` method takes the actual per-frame
 * `width × height` and only clears / accumulates / resolves over that
 * sub-region.
 */
export class WSRPipeline {
  readonly device: GPUDevice;
  readonly maxWidth: number;
  readonly maxHeight: number;
  readonly pipes: WSRPipelines;
  /**
   * Numerator accumulator. Storage buffer of `4 × u32` per pixel (RGB sum +
   * unused alpha slot kept for 16B vec4 alignment). f32 values are stored
   * via `bitcast<u32>` and updated via CAS-loop atomic-add.
   */
  readonly numeratorBuffer: GPUBuffer;
  /** Denominator accumulator. One `u32` per pixel, same bitcast scheme. */
  readonly denominatorBuffer: GPUBuffer;
  /**
   * Resolve output. Storage buffer of `1 × u32` per pixel, packed as
   * rgba8unorm (R in the low byte). Sized to match the output `GPUTexture`
   * that the renderer creates and presents; the buffer can be copied into
   * any rgba8unorm texture via `copyBufferToTexture`.
   */
  readonly outputBuffer: GPUBuffer;
  readonly clearUniforms: GPUBuffer;
  readonly accumulateUniforms: GPUBuffer;
  readonly resolveUniforms: GPUBuffer;

  private readonly clearBindGroup: GPUBindGroup;
  private readonly accumulateBindGroup: GPUBindGroup;
  private readonly resolveBindGroup: GPUBindGroup;

  /** Last-encoded viewport size, exposed for the renderer's blit logic. */
  lastWidth = 0;
  lastHeight = 0;

  constructor(init: WSRPipelineInit) {
    this.device = init.device;
    this.maxWidth = init.maxWidth;
    this.maxHeight = init.maxHeight;
    this.pipes = init.pipes ?? createWSRPipelines(this.device);

    const pxMax = this.maxWidth * this.maxHeight;
    const stUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.numeratorBuffer = this.device.createBuffer({
      size: Math.max(pxMax * NUMERATOR_BYTES_PER_PX, NUMERATOR_BYTES_PER_PX),
      usage: stUsage,
    });
    this.denominatorBuffer = this.device.createBuffer({
      size: Math.max(pxMax * DENOMINATOR_BYTES_PER_PX, DENOMINATOR_BYTES_PER_PX),
      usage: stUsage,
    });
    this.outputBuffer = this.device.createBuffer({
      size: Math.max(pxMax * OUTPUT_BYTES_PER_PX, OUTPUT_BYTES_PER_PX),
      usage: stUsage,
    });

    // Clear uniforms: viewport(u32x2) + pad(u32x2) = 16B.
    this.clearUniforms = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Accumulate uniforms: 2 mat4 + viewport(f32x2) + focal(f32x2) + count(u32) +
    // _pad(u32) + sigma(f32) + v_default(f32) + viewport_u(u32x2) = 152B; round to 160 (40 floats).
    this.accumulateUniforms = this.device.createBuffer({
      size: 4 * (16 + 16 + 2 + 2 + 4 + 4),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Resolve uniforms: viewport(u32x2) + pad(u32x2) + bg_color(f32x4) = 32B.
    this.resolveUniforms = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.clearBindGroup = this.device.createBindGroup({
      layout: this.pipes.clearBgl,
      entries: [
        { binding: 0, resource: { buffer: this.numeratorBuffer } },
        { binding: 1, resource: { buffer: this.denominatorBuffer } },
        { binding: 2, resource: { buffer: this.clearUniforms } },
      ],
    });
    this.accumulateBindGroup = this.device.createBindGroup({
      layout: this.pipes.accumulateBgl,
      entries: [
        { binding: 0, resource: { buffer: init.splatsBuffer } },
        { binding: 1, resource: { buffer: this.numeratorBuffer } },
        { binding: 2, resource: { buffer: this.denominatorBuffer } },
        { binding: 3, resource: { buffer: this.accumulateUniforms } },
      ],
    });
    this.resolveBindGroup = this.device.createBindGroup({
      layout: this.pipes.resolveBgl,
      entries: [
        { binding: 0, resource: { buffer: this.numeratorBuffer } },
        { binding: 1, resource: { buffer: this.denominatorBuffer } },
        { binding: 2, resource: { buffer: this.outputBuffer } },
        { binding: 3, resource: { buffer: this.resolveUniforms } },
      ],
    });
  }

  /**
   * Record the three WSR dispatches (`clear → accumulate → resolve`) into
   * the caller's command encoder.
   *
   * @param encoder caller-owned command encoder.
   * @param view column-major view matrix.
   * @param viewProj column-major view-projection matrix.
   * @param focal `[focalX, focalY]` in pixels.
   * @param viewport `[width, height]` in pixels. Must fit within
   *                 `[maxWidth, maxHeight]`.
   * @param splatCount number of splats currently resident in the splats
   *                   storage buffer.
   * @param sigma scene-wide WSR depth scale (PR1 heuristic:
   *              `2 × mean_scene_depth`).
   * @param bgColor optional background RGB and weight `w_B`. Default:
   *              `[0, 0, 0, 1e-4]`. The fourth component is `w_B`, not alpha.
   * @param vDefault optional per-splat-bias default. PR1 has no per-splat
   *                 storage so all splats share this single value.
   */
  encode(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    splatCount: number,
    sigma: number,
    bgColor: [number, number, number, number] = [0, 0, 0, WSR_DEFAULT_BG_WEIGHT],
    vDefault: number = 0,
  ): void {
    const w = viewport[0];
    const h = viewport[1];
    if (w <= 0 || h <= 0) return;
    if (w > this.maxWidth || h > this.maxHeight) {
      throw new Error(
        `WSRPipeline: viewport ${w}×${h} exceeds capacity ${this.maxWidth}×${this.maxHeight}`,
      );
    }
    this.lastWidth = w;
    this.lastHeight = h;

    // ---- Clear uniforms. ----
    {
      const u = new Uint32Array(4);
      u[0] = w;
      u[1] = h;
      this.device.queue.writeBuffer(this.clearUniforms, 0, u.buffer);
    }

    // ---- Accumulate uniforms. ----
    {
      const ab = new ArrayBuffer(this.accumulateUniforms.size);
      const f = new Float32Array(ab);
      const u = new Uint32Array(ab);
      f.set(view, 0);
      f.set(viewProj, 16);
      f[32] = w;        f[33] = h;
      f[34] = focal[0]; f[35] = focal[1];
      u[36] = splatCount;
      u[37] = 0;
      f[38] = sigma;
      f[39] = vDefault;
      u[40] = w;
      u[41] = h;
      this.device.queue.writeBuffer(this.accumulateUniforms, 0, ab);
    }

    // ---- Resolve uniforms. ----
    {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab);
      const f = new Float32Array(ab);
      u[0] = w; u[1] = h;
      u[2] = 0; u[3] = 0;
      f[4] = bgColor[0];
      f[5] = bgColor[1];
      f[6] = bgColor[2];
      f[7] = bgColor[3];
      this.device.queue.writeBuffer(this.resolveUniforms, 0, ab);
    }

    const pxWgsX = Math.ceil(w / WSR_TILE);
    const pxWgsY = Math.ceil(h / WSR_TILE);
    const splatWgs = Math.ceil(splatCount / WSR_WG);

    // ---- cs_wsr_clear ----
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.clear);
      pass.setBindGroup(0, this.clearBindGroup);
      pass.dispatchWorkgroups(pxWgsX, pxWgsY);
      pass.end();
    }

    // ---- cs_wsr_accumulate ----
    if (splatCount > 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.accumulate);
      pass.setBindGroup(0, this.accumulateBindGroup);
      pass.dispatchWorkgroups(splatWgs);
      pass.end();
    }

    // ---- cs_wsr_resolve ----
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.resolve);
      pass.setBindGroup(0, this.resolveBindGroup);
      pass.dispatchWorkgroups(pxWgsX, pxWgsY);
      pass.end();
    }
  }

  destroy(): void {
    this.numeratorBuffer.destroy();
    this.denominatorBuffer.destroy();
    this.outputBuffer.destroy();
    this.clearUniforms.destroy();
    this.accumulateUniforms.destroy();
    this.resolveUniforms.destroy();
  }
}
