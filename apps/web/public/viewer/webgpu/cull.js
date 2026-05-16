// SPDX-License-Identifier: Apache-2.0
/**
 * Opacity-radius pre-sort cull. Wraps the three kernels in `cs_cull.wgsl`
 * (cs_cull, cs_compact, cs_project_cmpct) plus a re-use of the existing
 * scan_multiblock.wgsl prefix-sum.
 *
 * Pipeline:
 *
 *   splats[]
 *      |
 *      v   cs_cull            -> flag_buffer[] (0/1)
 *                                prefix_buffer[] (0/1)
 *      v   scan_multiblock     -> prefix_buffer[] (exclusive prefix sum)
 *      v   cs_compact          -> compact_indices[]
 *      v   cs_project_cmpct    -> inst[], keys[], values[]
 *      v   radix sort (count=survivors)
 *      v   cs_gather  (count=survivors)
 *
 * Survivor count is computed via a one-shot mapAsync readback of
 * prefix_buffer[N-1] + flag_buffer[N-1]. The bench harness does this once
 * after a warm-up frame and reuses the count for the timed iterations.
 */
import { CULL_WGSL, SCAN_MULTIBLOCK_WGSL } from './shaders.generated.js';
/** Workgroup size used by every kernel in cs_cull.wgsl + scan_multiblock.wgsl. */
const WG = 256;
/**
 * Override the EWA dilation floor in the cull-pipeline WGSL. Mirrors the
 * same `SF_EWA_DILATION` token-replace in `index.ts::applyDilationOverride`.
 * Keeps cs_cull's radius prediction and cs_project_cmpct's c00/c11 in lock-
 * step with the main project pass.
 */
function applyDilationToCullWgsl(wgsl, dilation) {
    if (dilation === 0.3)
        return wgsl;
    const lit = dilation === 0 ? '0.0' : dilation.toFixed(6);
    return wgsl.replace(/let reg = 0\.3; \/\/ SF_EWA_DILATION/g, `let reg = ${lit}; // SF_EWA_DILATION(override=${dilation})`);
}
export function createCullPipelines(device, dilation = 0.3) {
    const cullMod = device.createShaderModule({ code: applyDilationToCullWgsl(CULL_WGSL, dilation) });
    const scanMod = device.createShaderModule({ code: SCAN_MULTIBLOCK_WGSL });
    const cullBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const compactBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const projectCmpctBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
    });
    const scanBgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const mk = (mod, bgl, entryPoint) => device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint },
    });
    return {
        cull: mk(cullMod, cullBgl, 'cs_cull'),
        compact: mk(cullMod, compactBgl, 'cs_compact'),
        projectCmpct: mk(cullMod, projectCmpctBgl, 'cs_project_cmpct'),
        scanPerWg: mk(scanMod, scanBgl, 'cs_scan_per_wg'),
        scanBlockSums: mk(scanMod, scanBgl, 'cs_scan_block_sums'),
        scanAddBlockSums: mk(scanMod, scanBgl, 'cs_scan_add_block_sums'),
        cullBgl, compactBgl, projectCmpctBgl, scanBgl,
    };
}
export class CullPipeline {
    device;
    capacity;
    pipes;
    flagBuffer;
    prefixBuffer;
    compactBuffer;
    blockSums;
    scanUniforms;
    cullUniforms;
    compactUniforms;
    /** 2-element u32 readback: [prefix[N-1], flag[N-1]]. Sum = survivor count. */
    readbackTail;
    readbackStaging;
    cullBindGroup;
    compactBindGroup;
    projectCmpctBindGroup;
    scanBindGroup;
    /** Cached survivor count from the last readSurvivorCount(). */
    cachedSurvivors = 0;
    /** Last splatCount that was cull-encoded — used to size the readback copy offset. */
    lastSplatCount = 0;
    constructor(init) {
        this.device = init.device;
        this.capacity = init.capacity;
        this.pipes = init.pipes;
        const u32Size = Math.max(this.capacity * 4, 4);
        const stUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        this.flagBuffer = this.device.createBuffer({ size: u32Size, usage: stUsage });
        this.prefixBuffer = this.device.createBuffer({ size: u32Size, usage: stUsage });
        this.compactBuffer = this.device.createBuffer({ size: u32Size, usage: stUsage });
        const numScanWgs = Math.ceil(this.capacity / WG);
        this.blockSums = this.device.createBuffer({
            size: Math.max(numScanWgs * 4, 4),
            usage: stUsage,
        });
        this.scanUniforms = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Cull uniforms: 2 mat4 + viewport + focal + count + tau + pad = 40 floats; round to 48 for alignment.
        this.cullUniforms = this.device.createBuffer({
            size: 4 * (16 + 16 + 2 + 2 + 4),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.compactUniforms = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.readbackTail = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.readbackStaging = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.cullBindGroup = this.device.createBindGroup({
            layout: this.pipes.cullBgl,
            entries: [
                { binding: 0, resource: { buffer: init.splatsBuffer } },
                { binding: 1, resource: { buffer: this.flagBuffer } },
                { binding: 2, resource: { buffer: this.prefixBuffer } },
                { binding: 3, resource: { buffer: this.cullUniforms } },
            ],
        });
        this.scanBindGroup = this.device.createBindGroup({
            layout: this.pipes.scanBgl,
            entries: [
                { binding: 0, resource: { buffer: this.prefixBuffer } },
                { binding: 1, resource: { buffer: this.blockSums } },
                { binding: 2, resource: { buffer: this.scanUniforms } },
            ],
        });
        this.compactBindGroup = this.device.createBindGroup({
            layout: this.pipes.compactBgl,
            entries: [
                { binding: 0, resource: { buffer: this.flagBuffer } },
                { binding: 1, resource: { buffer: this.prefixBuffer } },
                { binding: 2, resource: { buffer: this.compactBuffer } },
                { binding: 3, resource: { buffer: this.compactUniforms } },
            ],
        });
        this.projectCmpctBindGroup = this.device.createBindGroup({
            layout: this.pipes.projectCmpctBgl,
            entries: [
                { binding: 0, resource: { buffer: init.splatsBuffer } },
                { binding: 1, resource: { buffer: init.instBuffer } },
                { binding: 2, resource: { buffer: init.keysBuffer } },
                { binding: 3, resource: { buffer: init.valuesBuffer } },
                { binding: 4, resource: { buffer: init.projectUniforms } },
                { binding: 5, resource: { buffer: this.compactBuffer } },
            ],
        });
    }
    /**
     * Encode the cull + scan + compact + project_cmpct dispatches into the
     * caller's command encoder. The caller is responsible for dispatching the
     * downstream radix sort + gather at `cachedSurvivors` workgroups.
     *
     * `tau` defaults to 1/255 (the visible-pixel floor). The bench can pass
     * 1/1024 or 1/4096 if 1/255 prunes too aggressively on the synthetic
     * scene.
     *
     * Also queues a 2-element readback copy from (prefix[N-1], flag[N-1])
     * into `readbackStaging`. Call `readSurvivorCount()` after the encoder
     * has been submitted to refresh `cachedSurvivors`.
     */
    encode(encoder, view, viewProj, focal, viewport, splatCount, tau) {
        if (splatCount === 0)
            return;
        this.lastSplatCount = splatCount;
        // 1. Cull uniforms.
        {
            const ab = new ArrayBuffer(this.cullUniforms.size);
            const f = new Float32Array(ab);
            const u = new Uint32Array(ab);
            f.set(view, 0);
            f.set(viewProj, 16);
            f[32] = viewport[0];
            f[33] = viewport[1];
            f[34] = focal[0];
            f[35] = focal[1];
            u[36] = splatCount;
            f[37] = tau;
            this.device.queue.writeBuffer(this.cullUniforms, 0, ab);
        }
        // 2. Scan uniforms (total = splatCount; num_scan_wgs = ceil(splatCount/WG)).
        const numScanWgs = Math.ceil(splatCount / WG);
        {
            const u = new Uint32Array(8);
            u[0] = splatCount;
            u[1] = numScanWgs;
            this.device.queue.writeBuffer(this.scanUniforms, 0, u.buffer);
        }
        // 3. Compact uniforms.
        {
            const u = new Uint32Array(8);
            u[0] = splatCount;
            this.device.queue.writeBuffer(this.compactUniforms, 0, u.buffer);
        }
        const wgs = Math.ceil(splatCount / WG);
        // ---- cs_cull ----
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipes.cull);
            pass.setBindGroup(0, this.cullBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        // ---- scan_multiblock (3 phases) on prefixBuffer in-place ----
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipes.scanPerWg);
            pass.setBindGroup(0, this.scanBindGroup);
            pass.dispatchWorkgroups(numScanWgs);
            pass.setPipeline(this.pipes.scanBlockSums);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(this.pipes.scanAddBlockSums);
            pass.dispatchWorkgroups(numScanWgs);
            pass.end();
        }
        // ---- cs_compact ----
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipes.compact);
            pass.setBindGroup(0, this.compactBindGroup);
            pass.dispatchWorkgroups(wgs);
            pass.end();
        }
        // ---- copy (prefix[N-1], flag[N-1]) into the 8B readback staging ----
        // Survivor count = prefix[N-1] + flag[N-1].
        const tailOff = (splatCount - 1) * 4;
        encoder.copyBufferToBuffer(this.prefixBuffer, tailOff, this.readbackTail, 0, 4);
        encoder.copyBufferToBuffer(this.flagBuffer, tailOff, this.readbackTail, 4, 4);
        encoder.copyBufferToBuffer(this.readbackTail, 0, this.readbackStaging, 0, 8);
        // ---- cs_project_cmpct ----
        // Dispatched at `cachedSurvivors`. On the very first frame this is 0;
        // the caller is expected to no-op the downstream sort+gather in that
        // case (or to do a warm-up readback before the timed loop).
        if (this.cachedSurvivors > 0) {
            // Repurpose projectUniforms.splat_count = survivors. The host wrote
            // the per-frame uniforms earlier (in the orchestrator); we
            // overwrite the count field (offset 144 bytes = 36 floats).
            const u = new Uint32Array(1);
            u[0] = this.cachedSurvivors;
            // 36 floats * 4 = 144B offset to `splat_count`.
            // (This matches the existing ProjectUniforms struct layout.)
            // We rely on the outer host having written the rest of the bytes.
            // To keep things robust, we just write the count slot.
            // NOTE: the actual write happens via a host-side helper because the
            // bench harness owns the projectUniforms. The cull module exposes
            // the survivor count via `cachedSurvivors`; the orchestrator decides
            // whether to dispatch at splat_count or survivors.
            // Here we ONLY dispatch the project_cmpct kernel; the orchestrator
            // is responsible for the uniform layout.
            void u;
            const survWgs = Math.ceil(this.cachedSurvivors / WG);
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipes.projectCmpct);
            pass.setBindGroup(0, this.projectCmpctBindGroup);
            pass.dispatchWorkgroups(survWgs);
            pass.end();
        }
    }
    /** Read back the survivor count. Updates `cachedSurvivors`. */
    async readSurvivorCount() {
        if (this.lastSplatCount === 0)
            return 0;
        await this.readbackStaging.mapAsync(GPUMapMode.READ);
        const u = new Uint32Array(this.readbackStaging.getMappedRange().slice(0));
        this.readbackStaging.unmap();
        const survivors = u[0] + u[1];
        this.cachedSurvivors = survivors;
        return survivors;
    }
    destroy() {
        this.flagBuffer.destroy();
        this.prefixBuffer.destroy();
        this.compactBuffer.destroy();
        this.blockSums.destroy();
        this.scanUniforms.destroy();
        this.cullUniforms.destroy();
        this.compactUniforms.destroy();
        this.readbackTail.destroy();
        this.readbackStaging.destroy();
    }
}
//# sourceMappingURL=cull.js.map