// SPDX-License-Identifier: Apache-2.0
/**
 * GPU radix sort over u32 keys and u32 values.
 *
 * Wraps `radix_sort.wgsl`. The shader sorts 4 bits per pass; we run 8 passes
 * to cover a full 32-bit key, ping-ponging between two pairs of buffers.
 *
 * Public API:
 *   - `createRadixSort(device, capacity)` allocates buffers sized for up to
 *     `capacity` elements.
 *   - `sorter.encode(encoder, count)` records the 24 dispatch calls (3 per
 *     pass × 8 passes) into the given command encoder.
 *   - `sorter.keysA` / `sorter.valuesA` are the input buffers callers write
 *     into; after `encode`, the sorted output ends up in `keysA` / `valuesA`
 *     too (we make sure the final ping-pong lands on A).
 *
 * Bind groups are created lazily per (numWgs) shape; the implementation
 * caches them since `numWgs` is a function of `count` and changes rarely.
 */
/**
 * `tsc` (the only bundler in this package) doesn't support `?raw` imports,
 * so the WGSL source for the decode/project and radix-sort pipelines is
 * embedded as TypeScript string constants in `shaders.ts`. The caller
 * passes the appropriate string into `createRadixSortPipelines`.
 */
const WG_SIZE = 256;
const RADIX = 16;
const PASSES = 8;
const BITS_PER_PASS = 4;
/**
 * Compile the radix-sort compute pipelines from the WGSL source. Done once
 * per device.
 */
export function createRadixSortPipelines(device, wgslSource) {
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
    const mk = (entryPoint) => device.createComputePipeline({ layout, compute: { module, entryPoint } });
    return {
        histogram: mk('cs_histogram'),
        scan: mk('cs_scan'),
        scatter: mk('cs_scatter'),
        bindGroupLayout,
    };
}
/**
 * A reusable GPU sorter. Allocates two ping-pong (key,value) pairs and a
 * histogram scratch buffer sized for the worst-case `capacity`.
 */
export class RadixSort {
    device;
    capacity;
    /** Caller-visible keys input/output (final sorted lands here). */
    keysA;
    /** Caller-visible values input/output. */
    valuesA;
    keysB;
    valuesB;
    histograms;
    uniformBuffers = [];
    bindGroups = [];
    pipes;
    maxWgs;
    constructor(device, capacity, pipes) {
        this.device = device;
        this.capacity = capacity;
        this.pipes = pipes;
        this.maxWgs = Math.ceil(capacity / WG_SIZE);
        const bufSize = Math.max(capacity, 1) * 4;
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        this.keysA = device.createBuffer({ size: bufSize, usage });
        this.valuesA = device.createBuffer({ size: bufSize, usage });
        this.keysB = device.createBuffer({ size: bufSize, usage });
        this.valuesB = device.createBuffer({ size: bufSize, usage });
        this.histograms = device.createBuffer({
            size: Math.max(this.maxWgs * RADIX * 4, 64),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
            this.bindGroups.push(device.createBindGroup({
                layout: pipes.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: keysIn } },
                    { binding: 1, resource: { buffer: valuesIn } },
                    { binding: 2, resource: { buffer: keysOut } },
                    { binding: 3, resource: { buffer: valuesOut } },
                    { binding: 4, resource: { buffer: this.histograms } },
                    { binding: 5, resource: { buffer: ub } },
                ],
            }));
        }
    }
    /**
     * Record dispatches for sorting `count` elements. After `encoder.finish()`
     * + `queue.submit()`, the sorted keys/values live in `keysA` / `valuesA`.
     *
     * PASSES is even (8) so we always end on the A buffers.
     */
    encode(encoder, count) {
        if (count <= 1)
            return;
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
            this.device.queue.writeBuffer(this.uniformBuffers[pass], 0, u.buffer);
        }
        for (let pass = 0; pass < PASSES; pass++) {
            const pp = encoder.beginComputePass();
            pp.setBindGroup(0, this.bindGroups[pass]);
            pp.setPipeline(this.pipes.histogram);
            pp.dispatchWorkgroups(numWgs);
            pp.setPipeline(this.pipes.scan);
            pp.dispatchWorkgroups(1);
            pp.setPipeline(this.pipes.scatter);
            pp.dispatchWorkgroups(numWgs);
            pp.end();
        }
    }
    destroy() {
        this.keysA.destroy();
        this.keysB.destroy();
        this.valuesA.destroy();
        this.valuesB.destroy();
        this.histograms.destroy();
        for (const u of this.uniformBuffers)
            u.destroy();
    }
}
/** Exported constants so other modules don't redeclare them. */
export const RADIX_SORT_WG_SIZE = WG_SIZE;
export const RADIX_SORT_PASSES = PASSES;
