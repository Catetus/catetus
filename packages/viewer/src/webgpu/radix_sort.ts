// SPDX-License-Identifier: Apache-2.0
/**
 * GPU radix sort over u32 keys and u32 values.
 *
 * Wraps `radix_sort.wgsl` (histogram + scan + scatter) and, when the
 * multi-block scan path is enabled, the chained 3-kernel exclusive prefix
 * sum in `scan_multiblock.wgsl`.
 *
 * Public API:
 *   - `createRadixSort(device, capacity)` allocates buffers sized for up to
 *     `capacity` elements.
 *   - `sorter.encode(encoder, count)` records the dispatch calls into the
 *     given command encoder.
 *   - `sorter.keysA` / `sorter.valuesA` are the input buffers callers write
 *     into; after `encode`, the sorted output ends up in `keysA` / `valuesA`
 *     too (we make sure the final ping-pong lands on A).
 *
 * Two scan strategies live side-by-side:
 *
 *   1. **Single-WG scan** (legacy). The original `cs_scan` in
 *      `radix_sort.wgsl` runs as one workgroup of 256 threads striding
 *      through the entire `histograms` array (`num_wgs * RADIX` elements).
 *      For 10 M splats that's ~625 K elements per pass × 8 passes — the
 *      dominant cost in the sort.
 *
 *   2. **Multi-block chained scan** (default — `useMultiBlockScan: true`).
 *      Three kernels per scan, parallelized over many workgroups in phases
 *      (A) per-WG tile scan and (C) per-WG add-block-sums. Phase (B), the
 *      block-sums scan, still runs in a single workgroup because the
 *      block-sums array is tiny (≤ a few thousand entries even at 10 M
 *      splats). This is the architecturally-correct change regardless of
 *      the absolute fps target; the single-WG path is kept for A/B
 *      comparison and as a fallback.
 *
 * Bind groups are created lazily per (numWgs) shape; the implementation
 * caches them since `numWgs` is a function of `count` and changes rarely.
 */

/**
 * `tsc` (the only bundler in this package) doesn't support `?raw` imports,
 * so the WGSL source for the decode/project and radix-sort pipelines is
 * embedded as TypeScript string constants in `shaders.generated.ts`. The
 * caller passes the appropriate string into `createRadixSortPipelines`.
 */

const WG_SIZE = 256;
const RADIX = 16;
const PASSES = 8;
const BITS_PER_PASS = 4;

export interface RadixSortPipelines {
  histogram: GPUComputePipeline;
  scan: GPUComputePipeline;
  scatter: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  /** Optional multi-block scan pipelines (chained 3-kernel scan). */
  scanMb?: MultiBlockScanPipelines;
}

/** Multi-block exclusive prefix-sum pipelines (see `scan_multiblock.wgsl`). */
export interface MultiBlockScanPipelines {
  perWg: GPUComputePipeline;
  blockSums: GPUComputePipeline;
  addBlockSums: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Compile the radix-sort compute pipelines from the WGSL source. Done once
 * per device.
 *
 * @param multiBlockScanWgsl optional WGSL for the 3-kernel chained scan. When
 *   provided, the orchestration in `RadixSort.encode` will replace the
 *   single-workgroup `cs_scan` with the multi-block path. Pass an empty
 *   string to opt out (legacy behavior).
 */
export function createRadixSortPipelines(
  device: GPUDevice,
  wgslSource: string,
  multiBlockScanWgsl: string = '',
): RadixSortPipelines {
  const module = device.createShaderModule({ code: wgslSource });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const mk = (entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({ layout, compute: { module, entryPoint } });

  let scanMb: MultiBlockScanPipelines | undefined;
  if (multiBlockScanWgsl.length > 0) {
    const mbModule = device.createShaderModule({ code: multiBlockScanWgsl });
    const mbBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const mbLayout = device.createPipelineLayout({ bindGroupLayouts: [mbBgl] });
    const mbMk = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({ layout: mbLayout, compute: { module: mbModule, entryPoint } });
    scanMb = {
      perWg: mbMk('cs_scan_per_wg'),
      blockSums: mbMk('cs_scan_block_sums'),
      addBlockSums: mbMk('cs_scan_add_block_sums'),
      bindGroupLayout: mbBgl,
    };
  }

  return {
    histogram: mk('cs_histogram'),
    scan: mk('cs_scan'),
    scatter: mk('cs_scatter'),
    bindGroupLayout,
    ...(scanMb ? { scanMb } : {}),
  };
}

/** Options for the radix sorter. */
export interface RadixSortOptions {
  /**
   * If true (default), use the multi-block chained scan when available.
   * When the supplied `RadixSortPipelines` was constructed without
   * `multiBlockScanWgsl`, this flag is silently ignored.
   */
  useMultiBlockScan?: boolean;
}

/**
 * A reusable GPU sorter. Allocates two ping-pong (key,value) pairs and a
 * histogram scratch buffer sized for the worst-case `capacity`.
 */
export class RadixSort {
  readonly device: GPUDevice;
  readonly capacity: number;
  /** Caller-visible keys input/output (final sorted lands here). */
  readonly keysA: GPUBuffer;
  /** Caller-visible values input/output. */
  readonly valuesA: GPUBuffer;
  private readonly keysB: GPUBuffer;
  private readonly valuesB: GPUBuffer;
  private readonly histograms: GPUBuffer;
  private readonly uniformBuffers: GPUBuffer[] = [];
  private readonly bindGroups: GPUBindGroup[] = [];
  private readonly pipes: RadixSortPipelines;
  private readonly maxWgs: number;
  private readonly useMultiBlockScan: boolean;

  /** Multi-block scan state (only allocated when the scan is enabled). */
  private readonly mbBlockSums?: GPUBuffer;
  private readonly mbUniform?: GPUBuffer;
  private readonly mbBindGroup?: GPUBindGroup;

  constructor(
    device: GPUDevice,
    capacity: number,
    pipes: RadixSortPipelines,
    options: RadixSortOptions = {},
  ) {
    this.device = device;
    this.capacity = capacity;
    this.pipes = pipes;
    this.maxWgs = Math.ceil(capacity / WG_SIZE);
    this.useMultiBlockScan = (options.useMultiBlockScan ?? true) && pipes.scanMb !== undefined;
    const bufSize = Math.max(capacity, 1) * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.keysA = device.createBuffer({ size: bufSize, usage });
    this.valuesA = device.createBuffer({ size: bufSize, usage });
    this.keysB = device.createBuffer({ size: bufSize, usage });
    this.valuesB = device.createBuffer({ size: bufSize, usage });
    const histSize = Math.max(this.maxWgs * RADIX * 4, 64);
    this.histograms = device.createBuffer({
      size: histSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Pre-allocate 8 uniform buffers (one per pass) and matching bind groups.
    // The bind groups depend on which ping-pong direction each pass uses.
    for (let pass = 0; pass < PASSES; pass++) {
      const ub = device.createBuffer({
        // 16-byte struct; some adapters require ≥32 B uniform bindings.
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.uniformBuffers.push(ub);
      const evenPass = (pass & 1) === 0;
      const keysIn = evenPass ? this.keysA : this.keysB;
      const valuesIn = evenPass ? this.valuesA : this.valuesB;
      const keysOut = evenPass ? this.keysB : this.keysA;
      const valuesOut = evenPass ? this.valuesB : this.valuesA;
      this.bindGroups.push(
        device.createBindGroup({
          layout: pipes.bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: keysIn } },
            { binding: 1, resource: { buffer: valuesIn } },
            { binding: 2, resource: { buffer: keysOut } },
            { binding: 3, resource: { buffer: valuesOut } },
            { binding: 4, resource: { buffer: this.histograms } },
            { binding: 5, resource: { buffer: ub } },
          ],
        }),
      );
    }

    // Multi-block scan scratch + bind group.
    if (this.useMultiBlockScan && pipes.scanMb) {
      // total histogram entries scanned per pass = maxWgs * RADIX.
      // Number of scan-tile workgroups = ceil(total / WG_SIZE).
      const maxTotal = this.maxWgs * RADIX;
      const maxScanWgs = Math.max(Math.ceil(maxTotal / WG_SIZE), 1);
      this.mbBlockSums = device.createBuffer({
        size: maxScanWgs * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.mbUniform = device.createBuffer({
        size: 16, // ScanUniforms: 4×u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.mbBindGroup = device.createBindGroup({
        layout: pipes.scanMb.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.histograms } },
          { binding: 1, resource: { buffer: this.mbBlockSums } },
          { binding: 2, resource: { buffer: this.mbUniform } },
        ],
      });
    }
  }

  /**
   * Record dispatches for sorting `count` elements. After `encoder.finish()`
   * + `queue.submit()`, the sorted keys/values live in `keysA` / `valuesA`.
   *
   * PASSES is even (8) so we always end on the A buffers.
   */
  encode(encoder: GPUCommandEncoder, count: number): void {
    if (count <= 1) return;
    if (count > this.capacity) {
      throw new Error(`RadixSort: count ${count} exceeds capacity ${this.capacity}`);
    }
    const numWgs = Math.ceil(count / WG_SIZE);
    // Update the per-pass uniform buffers. 32 B each (only the first 16 B
    // is meaningful; the trailing 16 B is required by the binding-size
    // floor enforced by the bind-group validation).
    for (let pass = 0; pass < PASSES; pass++) {
      const u = new Uint32Array(8);
      u[0] = count;
      u[1] = pass * BITS_PER_PASS;
      u[2] = numWgs;
      this.device.queue.writeBuffer(this.uniformBuffers[pass]!, 0, u.buffer);
    }

    // Multi-block scan uniforms. `total` = numWgs * RADIX (the histogram
    // length we want an exclusive prefix sum over). `num_scan_wgs` = number
    // of scan-tile workgroups = ceil(total / WG_SIZE).
    let numScanWgs = 0;
    if (this.useMultiBlockScan) {
      const total = numWgs * RADIX;
      numScanWgs = Math.max(Math.ceil(total / WG_SIZE), 1);
      const su = new Uint32Array(4);
      su[0] = total;
      su[1] = numScanWgs;
      this.device.queue.writeBuffer(this.mbUniform!, 0, su.buffer);
    }

    for (let pass = 0; pass < PASSES; pass++) {
      const pp = encoder.beginComputePass();
      pp.setBindGroup(0, this.bindGroups[pass]!);
      pp.setPipeline(this.pipes.histogram);
      pp.dispatchWorkgroups(numWgs);
      if (this.useMultiBlockScan && this.pipes.scanMb) {
        // Switch bind groups to the scan layout (separate layout).
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.perWg);
        pp.dispatchWorkgroups(numScanWgs);
        pp.setPipeline(this.pipes.scanMb.blockSums);
        pp.dispatchWorkgroups(1);
        pp.setPipeline(this.pipes.scanMb.addBlockSums);
        pp.dispatchWorkgroups(numScanWgs);
        // Restore radix bind group for scatter.
        pp.setBindGroup(0, this.bindGroups[pass]!);
      } else {
        pp.setPipeline(this.pipes.scan);
        pp.dispatchWorkgroups(1);
      }
      pp.setPipeline(this.pipes.scatter);
      pp.dispatchWorkgroups(numWgs);
      pp.end();
    }
  }

  destroy(): void {
    this.keysA.destroy();
    this.keysB.destroy();
    this.valuesA.destroy();
    this.valuesB.destroy();
    this.histograms.destroy();
    for (const u of this.uniformBuffers) u.destroy();
    this.mbBlockSums?.destroy();
    this.mbUniform?.destroy();
  }
}

/** Exported constants so other modules don't redeclare them. */
export const RADIX_SORT_WG_SIZE = WG_SIZE;
export const RADIX_SORT_PASSES = PASSES;
export const RADIX_SORT_RADIX = RADIX;
