// SPDX-License-Identifier: Apache-2.0
/**
 * Weighted Sum Rendering (WSR) — tile-prefix-sum scatter pipeline.
 *
 * Comeback path from the B8 PR2 KILL (2026-05-15). The PR2 CAS-loop
 * atomic-add scatter (`cs_wsr_accumulate.wgsl`) measured 0.29 fps + 17 dB
 * PSNR on bonsai because many splats land on the same hot pixels and the
 * 256-iter CAS retry cap drops contributions. The tile-prefix-sum path
 * eliminates per-pixel atomic contention with two stages:
 *
 *   1. `cs_tile_bin`              — one thread per splat. Computes the
 *      splat's projected 2D bounding-box, atomicAdd into per-tile counters
 *      (the contention domain is ~256 tiles at 1080p, ~10^4× lower than
 *      per-pixel scatter), writes the splat index into the tile's slot.
 *      Per-tile lists are capped at `maxPerTile`; overflow is silently
 *      dropped (rare on real-scene captures).
 *   2. `cs_wsr_tile_accumulate`   — one workgroup per tile, 16×16 = 256
 *      threads (one per pixel). Each thread accumulates the per-pixel
 *      `(α·w·c, α·w)` weighted sums in thread-private REGISTERS while
 *      iterating the tile's splat list (cooperatively prefetched in
 *      256-index batches via workgroup-shared memory). Final write:
 *      one coalesced write per pixel to global `numerator`/`denominator`.
 *      NO atomics inside the workgroup.
 *   3. `cs_wsr_resolve`           — same as the PR1 path: rational evaluate
 *      `C = (w_B·c_B + N) / (w_B + D)` and pack to rgba8unorm.
 *
 * Buffer layout matches `WSRPipeline`'s storage (the resolve pass is shared),
 * so the two pipelines can in principle coexist behind their respective
 * feature flags on the same `ComputeDecodePipeline`.
 */

import {
  TILE_BIN_WGSL,
  WSR_TILE_ACCUMULATE_WGSL,
  WSR_RESOLVE_WGSL,
} from './shaders.generated.js';
import { dispatchPerSplat } from './multi-dispatch.js';

/** Byte offset of `chunk_offset: u32` inside `cs_tile_bin.wgsl::TileBinUniforms`.
 *  Layout: 2×mat4(128) + viewport(8) + focal(8) + splat_count(4) + tile_size(4)
 *  + tiles_x(4) + tiles_y(4) + max_per_tile(4) = 164. */
const TILE_BIN_UNIFORM_CHUNK_OFFSET_BYTES = 164;

/** Tile edge (pixels). 16×16 matches the workgroup size of the accumulate kernel. */
export const WSR_TILE_SIZE = 16;

/** Default per-tile splat-list capacity. Tunable via `WSRTilePipelineInit.maxPerTile`. */
export const WSR_TILE_DEFAULT_MAX_PER_TILE = 16384;

/** Workgroup size for the binning kernel (1D, one thread per splat). */
export const WSR_TILE_BIN_WG = 256;

/** Workgroup size for the bin-clear kernel (1D, one thread per tile). */
export const WSR_TILE_CLEAR_WG = 64;

const NUMERATOR_BYTES_PER_PX = 16; // 4 × u32
const DENOMINATOR_BYTES_PER_PX = 4; // 1 × u32
const OUTPUT_BYTES_PER_PX = 4;      // packed rgba8unorm

/** Default background weight (denominator floor) to prevent div-by-zero. */
export const WSR_TILE_DEFAULT_BG_WEIGHT = 1e-4;

export interface WSRTilePipelines {
  bin: GPUComputePipeline;
  binClear: GPUComputePipeline;
  accumulate: GPUComputePipeline;
  resolve: GPUComputePipeline;
  binBgl: GPUBindGroupLayout;
  accumulateBgl: GPUBindGroupLayout;
  resolveBgl: GPUBindGroupLayout;
}

export function createWSRTilePipelines(device: GPUDevice): WSRTilePipelines {
  const binMod = device.createShaderModule({ code: TILE_BIN_WGSL });
  const accumMod = device.createShaderModule({ code: WSR_TILE_ACCUMULATE_WGSL });
  const resolveMod = device.createShaderModule({ code: WSR_RESOLVE_WGSL });

  // cs_tile_bin + cs_tile_bin_clear share one bind group layout
  // (splats RO, tile_count RW atomic, tile_lists RW, uniforms).
  const binBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  // cs_wsr_tile_accumulate: splats RO, tile_count RO, tile_lists RO,
  // numerator RW, denominator RW, uniforms.
  const accumulateBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  // cs_wsr_resolve: numerator RO, denominator RO, output RW, uniforms.
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
    bin:        mk(binMod,     binBgl,        'cs_tile_bin'),
    binClear:   mk(binMod,     binBgl,        'cs_tile_bin_clear'),
    accumulate: mk(accumMod,   accumulateBgl, 'cs_wsr_tile_accumulate'),
    resolve:    mk(resolveMod, resolveBgl,    'cs_wsr_resolve'),
    binBgl, accumulateBgl, resolveBgl,
  };
}

export interface WSRTilePipelineInit {
  device: GPUDevice;
  /** Maximum viewport width × height in pixels. Sizes the accumulators + tile grid. */
  maxWidth: number;
  maxHeight: number;
  /** Splats storage buffer (owned by ComputeDecodePipeline). */
  splatsBuffer: GPUBuffer;
  /**
   * Per-tile splat-list capacity. Overflow is silently dropped in the
   * binning kernel. Default {@link WSR_TILE_DEFAULT_MAX_PER_TILE}.
   */
  maxPerTile?: number;
  pipes?: WSRTilePipelines;
}

/**
 * `WSRTilePipeline` — manages the tile-prefix-sum scatter kernels.
 *
 * Lifetime parallels `WSRPipeline`. The accumulator buffers (`numerator`,
 * `denominator`, `output`) and tile-binning buffers (`tileCount`,
 * `tileLists`) are sized for the worst-case viewport at construction.
 *
 * `encode()` records four dispatches:
 *   1. tile-count clear
 *   2. tile binning (one thread per splat)
 *   3. per-tile accumulate (one workgroup per tile)
 *   4. resolve (per-pixel, rational evaluate + rgba8unorm pack)
 *
 * The output `rgba8unorm`-packed frame lands in `outputBuffer`, suitable
 * for `copyBufferToTexture(rgba8unorm, ...)` by the renderer, or staging
 * readback by the unit test.
 */
export class WSRTilePipeline {
  readonly device: GPUDevice;
  readonly maxWidth: number;
  readonly maxHeight: number;
  readonly maxPerTile: number;
  readonly pipes: WSRTilePipelines;

  /** Numerator (4 u32-bitcast-of-f32 per pixel). Non-atomic — written once per pixel. */
  readonly numeratorBuffer: GPUBuffer;
  /** Denominator (1 u32-bitcast-of-f32 per pixel). Non-atomic — written once per pixel. */
  readonly denominatorBuffer: GPUBuffer;
  /** Resolve output (rgba8unorm packed into u32 per pixel). */
  readonly outputBuffer: GPUBuffer;
  /** Per-tile splat count (1 atomic u32 per tile). */
  readonly tileCountBuffer: GPUBuffer;
  /** Per-tile splat-index list (`maxPerTile` u32 per tile). */
  readonly tileListsBuffer: GPUBuffer;
  readonly binUniforms: GPUBuffer;
  readonly accumulateUniforms: GPUBuffer;
  readonly resolveUniforms: GPUBuffer;

  private readonly binBindGroup: GPUBindGroup;
  private readonly accumulateBindGroup: GPUBindGroup;
  private readonly resolveBindGroup: GPUBindGroup;

  lastWidth = 0;
  lastHeight = 0;
  lastTilesX = 0;
  lastTilesY = 0;

  constructor(init: WSRTilePipelineInit) {
    this.device = init.device;
    this.maxWidth = init.maxWidth;
    this.maxHeight = init.maxHeight;
    this.maxPerTile = init.maxPerTile ?? WSR_TILE_DEFAULT_MAX_PER_TILE;
    this.pipes = init.pipes ?? createWSRTilePipelines(this.device);

    const pxMax = this.maxWidth * this.maxHeight;
    const tilesX = Math.ceil(this.maxWidth / WSR_TILE_SIZE);
    const tilesY = Math.ceil(this.maxHeight / WSR_TILE_SIZE);
    const totalTiles = tilesX * tilesY;

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
    this.tileCountBuffer = this.device.createBuffer({
      size: Math.max(totalTiles * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.tileListsBuffer = this.device.createBuffer({
      size: Math.max(totalTiles * this.maxPerTile * 4, 4),
      usage: GPUBufferUsage.STORAGE,
    });

    // Binning uniforms: 2 mat4 (32 floats) + vec2 viewport + vec2 focal +
    // u32 splat_count + u32 tile_size + u32 tiles_x + u32 tiles_y +
    // u32 max_per_tile + u32 _pad0 + f32 sigma + f32 v_default = 46 floats
    // → round to 48 floats = 192 bytes.
    this.binUniforms = this.device.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Accumulate uniforms: 2 mat4 + vec2 viewport + vec2 focal +
    // u32 splat_count + u32 tile_size + u32 tiles_x + u32 tiles_y +
    // u32 max_per_tile + u32 viewport_x + u32 viewport_y + u32 _pad0 +
    // f32 sigma + f32 v_default + u32 _pad1 + u32 _pad2 = 52 floats
    // → round to 56 = 224 bytes.
    this.accumulateUniforms = this.device.createBuffer({
      size: 224,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Resolve uniforms: viewport(u32x2) + pad(u32x2) + bg_color(f32x4) = 32B.
    this.resolveUniforms = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.binBindGroup = this.device.createBindGroup({
      layout: this.pipes.binBgl,
      entries: [
        { binding: 0, resource: { buffer: init.splatsBuffer } },
        { binding: 1, resource: { buffer: this.tileCountBuffer } },
        { binding: 2, resource: { buffer: this.tileListsBuffer } },
        { binding: 3, resource: { buffer: this.binUniforms } },
      ],
    });
    this.accumulateBindGroup = this.device.createBindGroup({
      layout: this.pipes.accumulateBgl,
      entries: [
        { binding: 0, resource: { buffer: init.splatsBuffer } },
        { binding: 1, resource: { buffer: this.tileCountBuffer } },
        { binding: 2, resource: { buffer: this.tileListsBuffer } },
        { binding: 3, resource: { buffer: this.numeratorBuffer } },
        { binding: 4, resource: { buffer: this.denominatorBuffer } },
        { binding: 5, resource: { buffer: this.accumulateUniforms } },
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
   * Record the four-stage tile-prefix-sum render pass into the caller's
   * command encoder.
   *
   * @param encoder caller-owned command encoder.
   * @param view column-major view matrix.
   * @param viewProj column-major view-projection matrix.
   * @param focal `[focalX, focalY]` in pixels.
   * @param viewport `[width, height]` in pixels (must fit within
   *                 `[maxWidth, maxHeight]`).
   * @param splatCount number of splats currently resident.
   * @param sigma scene-wide WSR depth scale.
   * @param bgColor optional `(R, G, B, w_B)` background. Default
   *                `(0, 0, 0, 1e-4)`.
   * @param vDefault per-splat-bias default. PR1 keeps `v_i = 0`.
   */
  encode(
    encoder: GPUCommandEncoder,
    view: Float32Array,
    viewProj: Float32Array,
    focal: [number, number],
    viewport: [number, number],
    splatCount: number,
    sigma: number,
    bgColor: [number, number, number, number] = [0, 0, 0, WSR_TILE_DEFAULT_BG_WEIGHT],
    vDefault: number = 0,
  ): void {
    const w = viewport[0];
    const h = viewport[1];
    if (w <= 0 || h <= 0) return;
    if (w > this.maxWidth || h > this.maxHeight) {
      throw new Error(
        `WSRTilePipeline: viewport ${w}×${h} exceeds capacity ${this.maxWidth}×${this.maxHeight}`,
      );
    }
    this.lastWidth = w;
    this.lastHeight = h;
    const tilesX = Math.ceil(w / WSR_TILE_SIZE);
    const tilesY = Math.ceil(h / WSR_TILE_SIZE);
    this.lastTilesX = tilesX;
    this.lastTilesY = tilesY;

    // ---- Bin uniforms (also used by the bin-clear pass). ----
    {
      const ab = new ArrayBuffer(this.binUniforms.size);
      const f = new Float32Array(ab);
      const u = new Uint32Array(ab);
      f.set(view, 0);
      f.set(viewProj, 16);
      f[32] = w;        f[33] = h;
      f[34] = focal[0]; f[35] = focal[1];
      u[36] = splatCount;
      u[37] = WSR_TILE_SIZE;
      u[38] = tilesX;
      u[39] = tilesY;
      u[40] = this.maxPerTile;
      u[41] = 0; // _pad0
      f[42] = sigma;
      f[43] = vDefault;
      this.device.queue.writeBuffer(this.binUniforms, 0, ab);
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
      u[37] = WSR_TILE_SIZE;
      u[38] = tilesX;
      u[39] = tilesY;
      u[40] = this.maxPerTile;
      u[41] = w;
      u[42] = h;
      u[43] = 0; // _pad0
      f[44] = sigma;
      f[45] = vDefault;
      u[46] = 0; // _pad1
      u[47] = 0; // _pad2
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

    const totalTiles = tilesX * tilesY;
    const tileClearWgs = Math.ceil(totalTiles / WSR_TILE_CLEAR_WG);
    const splatWgs = Math.ceil(Math.max(splatCount, 1) / WSR_TILE_BIN_WG);

    // ---- Pass 1a: clear tile counts. ----
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.binClear);
      pass.setBindGroup(0, this.binBindGroup);
      pass.dispatchWorkgroups(tileClearWgs);
      pass.end();
    }

    // ---- Pass 1b: bin splats into tile lists. ----
    // Multi-dispatch over splat_count so > 16.7M splats clear the WebGPU
    // dispatchWorkgroups cap. chunk_offset slot in TileBinUniforms is at
    // byte 164: 2×mat4(128) + viewport(8) + focal(8) + splat_count(4) +
    // tile_size(4) + tiles_x(4) + tiles_y(4) + max_per_tile(4) = 164.
    if (splatCount > 0) {
      void splatWgs;
      dispatchPerSplat(
        this.device,
        encoder,
        this.pipes.bin,
        this.binBindGroup,
        this.binUniforms,
        TILE_BIN_UNIFORM_CHUNK_OFFSET_BYTES,
        splatCount,
        WSR_TILE_BIN_WG,
      );
    }

    // ---- Pass 2: per-tile accumulate (one workgroup per tile). ----
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipes.accumulate);
      pass.setBindGroup(0, this.accumulateBindGroup);
      pass.dispatchWorkgroups(tilesX, tilesY);
      pass.end();
    }

    // ---- Pass 3: resolve (numerator + denominator → rgba8unorm). ----
    {
      const pxWgsX = Math.ceil(w / WSR_TILE_SIZE);
      const pxWgsY = Math.ceil(h / WSR_TILE_SIZE);
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
    this.tileCountBuffer.destroy();
    this.tileListsBuffer.destroy();
    this.binUniforms.destroy();
    this.accumulateUniforms.destroy();
    this.resolveUniforms.destroy();
  }
}
