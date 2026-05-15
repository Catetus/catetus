// SPDX-License-Identifier: Apache-2.0
/**
 * GPU radix sort over u32 keys and u32 values.
 *
 * Wraps `radix_sort.wgsl` (histogram + scan + scatter), the multi-block
 * chained scan in `scan_multiblock.wgsl`, and an optional subgroup-aware
 * histogram kernel in `histogram_subgroup.wgsl`.
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
 * Algorithm: classic LSD radix sort, **8-bit radix (256 bins) -> 4 passes**
 * over a 32-bit key. The 4-bit / 16-bin / 8-pass variant lived here before;
 * the 8-bit change halves the dispatch count for the sort and is the second
 * lever (after the multi-block scan) on the path to 60 fps @ 10 M splats.
 *
 * Two scan strategies are supported:
 *
 *   1. **Single-WG scan** (legacy). The original `cs_scan` in
 *      `radix_sort.wgsl` runs as one workgroup of 256 threads striding
 *      through the entire `histograms` array. With 8-bit radix the
 *      histogram has `num_wgs * 256` entries per pass — ~10 M u32s at
 *      10 M splats — which is far too large for a single workgroup. This
 *      path is retained for completeness only; production callers must
 *      use the multi-block scan.
 *
 *   2. **Multi-block chained scan** (default — `useMultiBlockScan: true`).
 *      Three kernels per scan, parallelized over many workgroups in phases
 *      (A) per-WG tile scan and (C) per-WG add-block-sums. Phase (B), the
 *      block-sums scan, still runs in a single workgroup because the
 *      block-sums array is small (~40 K entries for 10 M splats x 256 bins).
 *
 * Two histogram kernels are supported:
 *
 *   1. **Atomic histogram** (mandatory). `cs_histogram` from
 *      `radix_sort.wgsl`. Each lane does one workgroup-shared
 *      `atomicAdd(&wg_hist[my_bin], 1u)`. Workgroup atomics are mandatory
 *      in WebGPU 1.0 so this path always works.
 *   2. **Subgroup-aware histogram** (optional, WebGPU 1.1 `'subgroups'`
 *      feature). `cs_histogram_subgroup` from `histogram_subgroup.wgsl`.
 *      When every live lane in a subgroup shares the same bin (very common
 *      after the first pass on partially-sorted data), one atomicAdd of
 *      `subgroup_size` replaces N. Falls back to per-lane atomicAdd on
 *      mixed-bin subgroups.
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
const RADIX = 256;
const PASSES = 4;
const BITS_PER_PASS = 8;

export interface RadixSortPipelines {
  histogram: GPUComputePipeline;
  scan: GPUComputePipeline;
  scatter: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  /** Optional multi-block scan pipelines (chained 3-kernel scan). */
  scanMb?: MultiBlockScanPipelines;
  /**
   * Optional subgroup-aware histogram kernel. When present, the encode path
   * uses it instead of `histogram`. Created only when the caller passes
   * `histogramSubgroupWgsl` AND has confirmed the device supports subgroups.
   */
  histogramSubgroup?: GPUComputePipeline;
}

/** Multi-block exclusive prefix-sum pipelines (see `scan_multiblock.wgsl`). */
export interface MultiBlockScanPipelines {
  perWg: GPUComputePipeline;
  blockSums: GPUComputePipeline;
  addBlockSums: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Feature-detect WebGPU subgroups. Subgroups is an optional feature; the
 * adapter advertises it, and the device must be requested with it in
 * `requiredFeatures` for the shader's `enable subgroups;` directive to
 * compile.
 *
 * Callers who want the subgroup histogram should:
 *   1. Check `adapterSupportsSubgroups(adapter)`.
 *   2. Pass `requiredFeatures: ['subgroups']` to `adapter.requestDevice()`.
 *   3. Pass `histogramSubgroupWgsl` into `createRadixSortPipelines`.
 *
 * Returns false (instead of throwing) if the adapter doesn't advertise the
 * feature - the caller silently falls back to the atomic-add path.
 */
export function adapterSupportsSubgroups(adapter: GPUAdapter): boolean {
  // `'subgroups'` is the spec-mandated feature name (WebGPU 1.1 / Dawn /
  // wgpu). Cast through `unknown` because `GPUFeatureName` in @webgpu/types
  // is a narrow string-literal union that doesn't yet include it.
  return adapter.features.has('subgroups' as unknown as GPUFeatureName);
}

/**
 * Compile the radix-sort compute pipelines from the WGSL source. Done once
 * per device.
 *
 * @param multiBlockScanWgsl optional WGSL for the 3-kernel chained scan. When
 *   provided, the orchestration in `RadixSort.encode` will replace the
 *   single-workgroup `cs_scan` with the multi-block path. Pass an empty
 *   string to opt out (legacy behavior).
 * @param histogramSubgroupWgsl optional WGSL for the subgroup-aware
 *   histogram kernel. Caller is responsible for ensuring the device was
 *   created with `requiredFeatures: ['subgroups']`; otherwise the shader
 *   module will fail to compile. Pass an empty string to disable.
 */
export function createRadixSortPipelines(
  device: GPUDevice,
  wgslSource: string,
  multiBlockScanWgsl: string = '',
  histogramSubgroupWgsl: string = '',
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

  // Optional subgroup histogram. Compiles the WGSL with `enable subgroups;`
  // - the device MUST have been requested with the 'subgroups' feature or
  // shader module creation will surface a validation error. The caller
  // owns that decision; see `adapterSupportsSubgroups`.
  let histogramSubgroup: GPUComputePipeline | undefined;
  if (histogramSubgroupWgsl.length > 0) {
    const sgModule = device.createShaderModule({ code: histogramSubgroupWgsl });
    histogramSubgroup = device.createComputePipeline({
      layout,
      compute: { module: sgModule, entryPoint: 'cs_histogram_subgroup' },
    });
  }

  return {
    histogram: mk('cs_histogram'),
    scan: mk('cs_scan'),
    scatter: mk('cs_scatter'),
    bindGroupLayout,
    ...(scanMb ? { scanMb } : {}),
    ...(histogramSubgroup ? { histogramSubgroup } : {}),
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
  /**
   * If true (default), use the subgroup-aware histogram kernel when
   * compiled. When the supplied `RadixSortPipelines` was constructed
   * without `histogramSubgroupWgsl` (or the device doesn't expose the
   * feature), this flag is silently ignored.
   */
  useSubgroupHistogram?: boolean;
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
  private readonly useSubgroupHistogram: boolean;

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
    this.useSubgroupHistogram =
      (options.useSubgroupHistogram ?? true) && pipes.histogramSubgroup !== undefined;
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
    // Pre-allocate PASSES uniform buffers (one per pass) and matching bind
    // groups. The bind groups depend on which ping-pong direction each pass
    // uses. PASSES is even (4) so the final ping-pong lands on the A
    // buffers.
    for (let pass = 0; pass < PASSES; pass++) {
      const ub = device.createBuffer({
        // 16-byte struct; some adapters require >= 32 B uniform bindings.
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
        size: 16, // ScanUniforms: 4xu32
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
   * PASSES is even (4) so we always end on the A buffers.
   */
  /**
   * Bench-only variant of `encode`. Wraps the entire 4-pass radix sort in
   * one timestamp window [baseIndex .. baseIndex+1]. We do not drill into
   * histogram/scan/scatter here — that needs `timestamp-query-inside-
   * passes`, added in a follow-up if the top-level sort window dominates.
   */
  encodeTimed(
    encoder: GPUCommandEncoder,
    count: number,
    querySet: GPUQuerySet,
    baseIndex: number,
  ): void {
    if (count <= 1) return;
    if (count > this.capacity) {
      throw new Error(`RadixSort: count ${count} exceeds capacity ${this.capacity}`);
    }
    const numWgs = Math.ceil(count / WG_SIZE);
    for (let pass = 0; pass < PASSES; pass++) {
      const u = new Uint32Array(8);
      u[0] = count;
      u[1] = pass * BITS_PER_PASS;
      u[2] = numWgs;
      this.device.queue.writeBuffer(this.uniformBuffers[pass]!, 0, u.buffer);
    }
    let numScanWgs = 0;
    if (this.useMultiBlockScan) {
      const total = numWgs * RADIX;
      numScanWgs = Math.max(Math.ceil(total / WG_SIZE), 1);
      const su = new Uint32Array(4);
      su[0] = total;
      su[1] = numScanWgs;
      this.device.queue.writeBuffer(this.mbUniform!, 0, su.buffer);
    }
    const histogramPipe = this.useSubgroupHistogram
      ? this.pipes.histogramSubgroup!
      : this.pipes.histogram;
    for (let pass = 0; pass < PASSES; pass++) {
      const descr: GPUComputePassDescriptor =
        pass === 0
          ? {
              timestampWrites: {
                querySet,
                beginningOfPassWriteIndex: baseIndex + 0,
              },
            }
          : pass === PASSES - 1
            ? {
                timestampWrites: {
                  querySet,
                  endOfPassWriteIndex: baseIndex + 1,
                },
              }
            : {};
      const pp = encoder.beginComputePass(descr);
      pp.setBindGroup(0, this.bindGroups[pass]!);
      pp.setPipeline(histogramPipe);
      pp.dispatchWorkgroups(numWgs);
      if (this.useMultiBlockScan && this.pipes.scanMb) {
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.perWg);
        pp.dispatchWorkgroups(numScanWgs);
        pp.setPipeline(this.pipes.scanMb.blockSums);
        pp.dispatchWorkgroups(1);
        pp.setPipeline(this.pipes.scanMb.addBlockSums);
        pp.dispatchWorkgroups(numScanWgs);
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

  /**
   * Bench-only variant of `encode` that drills into per-sub-stage timing.
   * Writes 10 timestamps:
   *   [base+0..1)  pass0 histogram
   *   [base+2..3)  pass0 scan_per_wg
   *   [base+4..5)  pass0 scan_block_sums
   *   [base+6..7)  pass0 scan_add_block_sums
   *   [base+8..9)  pass0 scatter
   * Plus 4 timestamps wrapping passes 1-3 as a bundle ([base+10..11),
   * [base+12..13), [base+14..15)) so we know if pass-to-pass cost is
   * symmetric. Total: 16 timestamps. Caller must provide a querySet of
   * capacity at least baseIndex + 16.
   *
   * Each sub-stage gets its OWN beginComputePass() so we can use the basic
   * `timestamp-query` feature (not `timestamp-query-inside-passes`).
   * Pass-boundary cost is on the order of 50 µs each; we eat ~0.4 ms of
   * extra overhead per frame in exchange for sub-µs sub-stage timing.
   */
  encodeTimedDrilled(
    encoder: GPUCommandEncoder,
    count: number,
    querySet: GPUQuerySet,
    baseIndex: number,
  ): void {
    if (count <= 1) return;
    if (count > this.capacity) {
      throw new Error(`RadixSort: count ${count} exceeds capacity ${this.capacity}`);
    }
    const numWgs = Math.ceil(count / WG_SIZE);
    for (let pass = 0; pass < PASSES; pass++) {
      const u = new Uint32Array(8);
      u[0] = count;
      u[1] = pass * BITS_PER_PASS;
      u[2] = numWgs;
      this.device.queue.writeBuffer(this.uniformBuffers[pass]!, 0, u.buffer);
    }
    let numScanWgs = 0;
    if (this.useMultiBlockScan) {
      const total = numWgs * RADIX;
      numScanWgs = Math.max(Math.ceil(total / WG_SIZE), 1);
      const su = new Uint32Array(4);
      su[0] = total;
      su[1] = numScanWgs;
      this.device.queue.writeBuffer(this.mbUniform!, 0, su.buffer);
    }
    const histogramPipe = this.useSubgroupHistogram
      ? this.pipes.histogramSubgroup!
      : this.pipes.histogram;

    const ts = (begin: number, end: number): GPUComputePassDescriptor => ({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: begin,
        endOfPassWriteIndex: end,
      },
    });

    // Pass 0: drill into 5 sub-stages.
    {
      const pp = encoder.beginComputePass(ts(baseIndex + 0, baseIndex + 1));
      pp.setBindGroup(0, this.bindGroups[0]!);
      pp.setPipeline(histogramPipe);
      pp.dispatchWorkgroups(numWgs);
      pp.end();
    }
    if (this.useMultiBlockScan && this.pipes.scanMb) {
      {
        const pp = encoder.beginComputePass(ts(baseIndex + 2, baseIndex + 3));
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.perWg);
        pp.dispatchWorkgroups(numScanWgs);
        pp.end();
      }
      {
        const pp = encoder.beginComputePass(ts(baseIndex + 4, baseIndex + 5));
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.blockSums);
        pp.dispatchWorkgroups(1);
        pp.end();
      }
      {
        const pp = encoder.beginComputePass(ts(baseIndex + 6, baseIndex + 7));
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.addBlockSums);
        pp.dispatchWorkgroups(numScanWgs);
        pp.end();
      }
    } else {
      // Single-WG scan (legacy) — bundle into the block_sums slot for
      // schema stability.
      const pp = encoder.beginComputePass(ts(baseIndex + 4, baseIndex + 5));
      pp.setBindGroup(0, this.bindGroups[0]!);
      pp.setPipeline(this.pipes.scan);
      pp.dispatchWorkgroups(1);
      pp.end();
    }
    {
      const pp = encoder.beginComputePass(ts(baseIndex + 8, baseIndex + 9));
      pp.setBindGroup(0, this.bindGroups[0]!);
      pp.setPipeline(this.pipes.scatter);
      pp.dispatchWorkgroups(numWgs);
      pp.end();
    }

    // Passes 1-3: bundle each as one timestamp window.
    for (let pass = 1; pass < PASSES; pass++) {
      const slot = baseIndex + 10 + (pass - 1) * 2;
      const pp = encoder.beginComputePass(ts(slot, slot + 1));
      pp.setBindGroup(0, this.bindGroups[pass]!);
      pp.setPipeline(histogramPipe);
      pp.dispatchWorkgroups(numWgs);
      if (this.useMultiBlockScan && this.pipes.scanMb) {
        pp.setBindGroup(0, this.mbBindGroup!);
        pp.setPipeline(this.pipes.scanMb.perWg);
        pp.dispatchWorkgroups(numScanWgs);
        pp.setPipeline(this.pipes.scanMb.blockSums);
        pp.dispatchWorkgroups(1);
        pp.setPipeline(this.pipes.scanMb.addBlockSums);
        pp.dispatchWorkgroups(numScanWgs);
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

    // Pick the histogram kernel once per encode. Bind-group layout is
    // identical between the atomic and subgroup variants.
    const histogramPipe = this.useSubgroupHistogram
      ? this.pipes.histogramSubgroup!
      : this.pipes.histogram;

    for (let pass = 0; pass < PASSES; pass++) {
      const pp = encoder.beginComputePass();
      pp.setBindGroup(0, this.bindGroups[pass]!);
      pp.setPipeline(histogramPipe);
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
export const RADIX_SORT_BITS_PER_PASS = BITS_PER_PASS;
