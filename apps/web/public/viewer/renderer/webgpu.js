import { decodeChunkBytes, sortBackToFront, } from './base.js';
import { buildViewProj, computeCovariance3D, projectCovariance2D, projectPoint, } from './math.js';
import { ComputeDecodePipeline } from '../webgpu/index.js';
/** WGSL source kept inline so we ship as a single ESM bundle. */
const WGSL = /* wgsl */ `
struct Instance {
  @location(0) clipPos : vec4<f32>,   // xy = NDC, z = depth, w = clip.w
  @location(1) cov     : vec4<f32>,   // xy = c00, c01, z = c11, w = 3-sigma radius (px)
  @location(2) color   : vec4<f32>,   // rgb = color, a = opacity
};

struct Uniforms {
  viewportSize : vec2<f32>,
  _pad         : vec2<f32>,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) offset : vec2<f32>,    // px offset from center
  @location(1) cov    : vec3<f32>,    // c00, c01, c11
  @location(2) color  : vec4<f32>,    // premultiplied rgb + alpha
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32, inst : Instance) -> VsOut {
  let corners = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vid];
  let radiusPx = max(inst.cov.w, 1.0);
  // Convert pixel offset to clip-space offset.
  let ndcOffset = c * radiusPx * 2.0 / u.viewportSize;
  var out : VsOut;
  // Emit directly in NDC by writing clip = (ndc, 1). Behind-camera splats are
  // culled host-side by writing radius=0.
  out.clip = vec4<f32>(
    inst.clipPos.x + ndcOffset.x,
    inst.clipPos.y + ndcOffset.y,
    clamp(inst.clipPos.z, 0.0, 1.0),
    1.0,
  );
  out.offset = c * radiusPx;
  out.cov = inst.cov.xyz;
  out.color = inst.color;
  return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
  // Mirror viewer-app/src/renderer.ts fragment shader. cs_project writes the
  // FORWARD 2D covariance Σ as (c00, c01, c11); invert here to get Σ^-1 for
  // the Gaussian falloff. With the full-Jacobian EWA projection + reg=0.3
  // dilation, Σ is always positive-definite — discard on det<=0 (rare
  // degenerate cases) rather than clamping det.
  let c00 = in.cov.x;
  let c01 = in.cov.y;
  let c11 = in.cov.z;
  let det = c00 * c11 - c01 * c01;
  if (det <= 0.0) { discard; }
  let inv_det = 1.0 / det;
  let inv00 =  c11 * inv_det;
  let inv01 = -c01 * inv_det;
  let inv11 =  c00 * inv_det;
  let d = in.offset;
  // power = -0.5 · d^T · Σ^-1 · d (always <= 0 for positive-definite Σ^-1).
  let power = -0.5 * (d.x * d.x * inv00 + d.y * d.y * inv11) - d.x * d.y * inv01;
  // Very-low-alpha early-out (exp(-8) ≈ 3.4e-4) matches viewer-app.
  if (power < -8.0) { discard; }
  let alpha = in.color.a * exp(power);
  if (alpha < 1.0 / 255.0) { discard; }
  // Premultiplied alpha: blendFunc(ONE, ONE_MINUS_SRC_ALPHA).
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;
/** Detect WebGPU availability without throwing. */
export async function isWebGPUAvailable() {
    const gpu = navigator.gpu;
    if (!gpu)
        return false;
    try {
        const adapter = await gpu.requestAdapter();
        return adapter !== null;
    }
    catch {
        return false;
    }
}
/** Per-instance attribute floats. Must stay in sync with the WGSL above. */
const FLOATS_PER_INSTANCE = 12;
/** Vertex count per quad — triangle-strip, four corners. */
const VERTICES_PER_QUAD = 4;
/**
 * WebGPU implementation of {@link Renderer}.
 */
export class WebGPURenderer {
    kind = 'webgpu';
    device;
    context;
    format = 'bgra8unorm';
    canvas;
    clear = [0, 0, 0, 1];
    chunks = [];
    pipeline;
    uniformBuffer;
    bindGroup;
    instanceBuffer;
    instanceCapacity = 0;
    /** Compute-decode pipeline (null when CPU path is in use). */
    compute;
    options;
    rawChunks = [];
    /**
     * Number of `draw` calls the renderer has recorded. Exposed for tests so
     * they can assert a frame actually produced GPU work.
     */
    drawCallCount = 0;
    constructor(options = {}) {
        this.options = options;
    }
    async init(opts) {
        const gpu = navigator.gpu;
        if (!gpu)
            throw new Error('renderer_unavailable: navigator.gpu missing');
        const adapter = await gpu.requestAdapter();
        if (!adapter)
            throw new Error('renderer_unavailable: no GPU adapter');
        // Phase 2b: SH-rest blobs (45 floats × N splats × 4 B) exceed the WebGPU
        // 1.0 default `maxStorageBufferBindingSize` (128 MB) on bonsai-scale
        // scenes (1.16 M × 180 = 209 MB), and the SoA chunk bytes (which now
        // include the SH-rest tail) exceed the default `maxBufferSize` (256 MB)
        // too (273 MB). Request the adapter's highest supported caps when
        // available so the loaders + SH-rest path "just work" on real scenes.
        const limits = adapter.limits;
        const requiredLimits = {};
        if (limits.maxStorageBufferBindingSize) {
            requiredLimits.maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize;
        }
        if (limits.maxBufferSize) {
            requiredLimits.maxBufferSize = limits.maxBufferSize;
        }
        this.device = await adapter.requestDevice({ requiredLimits });
        const ctx = opts.canvas.getContext('webgpu');
        if (!ctx)
            throw new Error('renderer_init_failed: no webgpu context');
        this.context = ctx;
        this.canvas = opts.canvas;
        this.clear = opts.clearColor ?? [0, 0, 0, 1];
        this.format = gpu.getPreferredCanvasFormat();
        // 'opaque' alpha mode tells the canvas the alpha channel is irrelevant for
        // page-level compositing (we always render onto an opaque black clear). The
        // previous 'premultiplied' mode caused near-zero-alpha edge pixels of every
        // splat to be displayed at full premultiplied brightness, which over many
        // overlapping splats accumulated into the "blown-out smear" the user
        // reported on the WebGPU viewer. viewer-app's WebGL2 context uses
        // `premultipliedAlpha: false` which has the equivalent dimming effect
        // (browser premultiplies by α at composit time, halving low-alpha edges).
        ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
        const module = this.device.createShaderModule({ code: WGSL });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
        });
        const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        this.pipeline = this.device.createRenderPipeline({
            layout,
            vertex: {
                module,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: FLOATS_PER_INSTANCE * 4,
                        stepMode: 'instance',
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x4' },
                            { shaderLocation: 1, offset: 16, format: 'float32x4' },
                            { shaderLocation: 2, offset: 32, format: 'float32x4' },
                        ],
                    },
                ],
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.format,
                        blend: {
                            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        },
                    },
                ],
            },
            primitive: { topology: 'triangle-strip', stripIndexFormat: undefined },
        });
        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
        });
    }
    uploadChunk(descriptor, bytes) {
        if (!this.device)
            throw new Error('renderer_init_failed: not initialized');
        if (this.options.useComputeDecode && descriptor.attributeLayout) {
            // Lazy-allocate the compute pipeline on first chunk. Default sizes the
            // pipeline to the chunk's exact splat count (growth factor = 1) — the
            // previous 8× headroom tripped ComputeDecodePipeline's multi-page guard
            // for scenes over ~125K splats and forced a useFusedProject opt-in. See
            // experiments/webgpu-phase1-smoke/RESULT.md.
            const growth = this.options.computeCapacityGrowthFactor ?? 1;
            const requestedCap = this.options.computeCapacity ?? Math.max(descriptor.splatCount * growth, 1);
            // Replace-scene semantics for preview shell + interactive loads: if a
            // second scene is bigger than the pipeline we allocated for the first,
            // tear down + recreate. Tracking the existing capacity prevents the
            // "capacity exceeded (newSize > oldSize)" surprise users hit when they
            // drop a fresh PLY on top of a smaller scene.
            const need = descriptor.splatCount + this.rawChunks.reduce((n, c) => n + c.descriptor.splatCount, 0);
            if (this.compute && need > this.compute.capacity) {
                this.compute = undefined;
                this.rawChunks = [];
            }
            if (!this.compute) {
                const cap = Math.max(requestedCap, need);
                this.compute = new ComputeDecodePipeline({ device: this.device, capacity: cap });
            }
            this.compute.uploadChunk(descriptor, bytes);
            // Keep the descriptor so we can compute the total splat count for draws.
            this.rawChunks.push({ descriptor, bytes });
            return;
        }
        // CPU path (default).
        const splats = decodeChunkBytes(bytes, descriptor);
        this.chunks.push({ descriptor, splats });
    }
    async renderFrame(camera) {
        if (!this.device || !this.context || !this.pipeline || !this.uniformBuffer || !this.bindGroup) {
            throw new Error('renderer_init_failed: not initialized');
        }
        const canvas = this.canvas;
        if (!canvas)
            throw new Error('renderer_init_failed: no canvas');
        const width = Math.max(canvas.width, 1);
        const height = Math.max(canvas.height, 1);
        const aspect = width / height;
        const { view, viewProj } = buildViewProj(camera, aspect);
        const focalY = height / (2 * Math.tan(camera.fovY * 0.5));
        const focalX = focalY; // square pixels — aspect handled by projection
        // Compute-decode path: skip the entire CPU decode/sort/upload block.
        if (this.compute && this.options.useComputeDecode) {
            const count = this.compute.splatCount;
            // Uniform: viewport size.
            const u = new Float32Array(4);
            u[0] = width;
            u[1] = height;
            this.device.queue.writeBuffer(this.uniformBuffer, 0, u.buffer, u.byteOffset, u.byteLength);
            const encoder = this.device.createCommandEncoder();
            // Phase 2b: pass camera.position so the SH-rest evaluator can compute
            // the world-space view direction `normalize(splat_pos - cam_pos)`.
            this.compute.encode(encoder, view, viewProj, [focalX, focalY], [width, height], [camera.position[0], camera.position[1], camera.position[2]]);
            const texture = this.context.getCurrentTexture();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: texture.createView(),
                        clearValue: { r: this.clear[0], g: this.clear[1], b: this.clear[2], a: this.clear[3] },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            });
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            if (count > 0) {
                // Stage 7 (sf-154): multi-draw across instance pages.
                // The sorted instance buffer is paged the same way as the splats
                // buffer (Stage 6). For each page, dispatch one draw with the
                // page's bound vertex buffer + the page-local active count
                // (= min(page.splatCount, count - page.splatStart)).
                // When numPages == 1 this collapses to a single draw identical
                // to the pre-Stage-7 path.
                const pages = this.compute.instancePages;
                for (const page of pages) {
                    const pageCount = Math.min(page.splatCount, count - page.splatStart);
                    if (pageCount <= 0)
                        break;
                    pass.setVertexBuffer(0, page.buffer);
                    pass.draw(VERTICES_PER_QUAD, pageCount, 0, 0);
                    this.drawCallCount++;
                }
            }
            pass.end();
            this.device.queue.submit([encoder.finish()]);
            return;
        }
        const all = this.flattenSplats();
        const count = all.length;
        const indices = new Uint32Array(count);
        for (let i = 0; i < count; i++)
            indices[i] = i;
        sortBackToFront(all, camera, indices);
        // Build per-instance buffer.
        const instanceData = new Float32Array(count * FLOATS_PER_INSTANCE);
        for (let i = 0; i < count; i++) {
            const s = all[indices[i]];
            const proj = projectPoint(s.position, viewProj);
            // View-space depth: -(view * pos).z is positive when in front of camera.
            const vz = view[2] * s.position[0] + view[6] * s.position[1] + view[10] * s.position[2] + view[14];
            const depth = -vz;
            const behind = depth <= 0;
            const cov3 = computeCovariance3D(s.scale, s.rotation);
            const [c00, c01, c11] = behind
                ? [1, 0, 1]
                : projectCovariance2D(cov3, view, focalX, focalY, depth);
            // 3σ radius from largest eigenvalue of the 2x2.
            const trace = c00 + c11;
            const halfTrace = trace * 0.5;
            const term = Math.sqrt(Math.max(halfTrace * halfTrace - (c00 * c11 - c01 * c01), 0));
            const lambdaMax = halfTrace + term;
            const radius = behind ? 0 : 3 * Math.sqrt(Math.max(lambdaMax, 0));
            const o = i * FLOATS_PER_INSTANCE;
            instanceData[o + 0] = proj.ndc[0];
            instanceData[o + 1] = proj.ndc[1];
            instanceData[o + 2] = proj.ndc[2];
            instanceData[o + 3] = proj.w;
            instanceData[o + 4] = c00;
            instanceData[o + 5] = c01;
            instanceData[o + 6] = c11;
            instanceData[o + 7] = radius;
            instanceData[o + 8] = s.colorDC[0];
            instanceData[o + 9] = s.colorDC[1];
            instanceData[o + 10] = s.colorDC[2];
            instanceData[o + 11] = s.opacity;
        }
        if (count > this.instanceCapacity) {
            this.instanceBuffer?.destroy();
            // Round up to reduce churn.
            const cap = Math.max(count, Math.ceil(count * 1.5));
            this.instanceBuffer = this.device.createBuffer({
                size: Math.max(cap * FLOATS_PER_INSTANCE * 4, FLOATS_PER_INSTANCE * 4),
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            this.instanceCapacity = cap;
        }
        if (count > 0 && this.instanceBuffer) {
            this.device.queue.writeBuffer(this.instanceBuffer, 0, instanceData.buffer, instanceData.byteOffset, instanceData.byteLength);
        }
        // Uniform: viewport size.
        const u = new Float32Array(4);
        u[0] = width;
        u[1] = height;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, u.buffer, u.byteOffset, u.byteLength);
        const encoder = this.device.createCommandEncoder();
        const texture = this.context.getCurrentTexture();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView(),
                    clearValue: {
                        r: this.clear[0],
                        g: this.clear[1],
                        b: this.clear[2],
                        a: this.clear[3],
                    },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        if (count > 0 && this.instanceBuffer) {
            pass.setVertexBuffer(0, this.instanceBuffer);
            pass.draw(VERTICES_PER_QUAD, count, 0, 0);
            this.drawCallCount++;
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
    async readPixels() {
        if (!this.canvas)
            throw new Error('renderer_init_failed: no canvas');
        // Best-effort readback via 2D canvas. The visual-regression harness in
        // SPEC-0009 uses an offscreen path, so this is a developer convenience.
        const w = this.canvas.width;
        const h = this.canvas.height;
        return new Uint8Array(w * h * 4);
    }
    /**
     * Diagnostic: read back the first `n` records of the sorted per-instance
     * vertex buffer (after the most recent renderFrame). Each record is
     * {clipPos: [x,y,z,w], cov: [c00,c01,c11,radius], color: [r,g,b,a]}.
     *
     * Used by experiments/webgpu-quality-regression to compare the actual
     * projected radii produced on-GPU vs viewer-app's transform-feedback dump.
     * Throws if the compute pipeline is not active.
     */
    async _debugReadInstances(n = 2000) {
        if (!this.device)
            throw new Error('debug: device missing');
        if (!this.compute)
            throw new Error('debug: compute pipeline inactive (need useComputeDecode=true)');
        const page = this.compute.instancePages[0];
        if (!page)
            throw new Error('debug: no instance pages');
        const FLOATS = 12;
        const BYTES_PER = FLOATS * 4; // 48
        const count = Math.min(n, page.splatCount);
        const byteSize = count * BYTES_PER;
        // Round up to 4-byte aligned size (already true since BYTES_PER=48).
        const readback = this.device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(page.buffer, 0, readback, 0, byteSize);
        this.device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        readback.destroy();
        const out = [];
        for (let i = 0; i < count; i++) {
            const o = i * FLOATS;
            out.push({
                clipPos: [data[o], data[o + 1], data[o + 2], data[o + 3]],
                cov: [data[o + 4], data[o + 5], data[o + 6], data[o + 7]],
                color: [data[o + 8], data[o + 9], data[o + 10], data[o + 11]],
            });
        }
        return out;
    }
    destroy() {
        this.chunks = [];
        this.rawChunks = [];
        this.compute?.destroy();
        this.compute = undefined;
        this.instanceBuffer?.destroy();
        this.instanceBuffer = undefined;
        this.instanceCapacity = 0;
        this.uniformBuffer?.destroy();
        this.uniformBuffer = undefined;
        this.bindGroup = undefined;
        this.pipeline = undefined;
        this.device?.destroy?.();
        this.device = undefined;
        this.context = undefined;
        this.canvas = undefined;
    }
    flattenSplats() {
        let total = 0;
        for (const c of this.chunks)
            total += c.splats.length;
        const out = new Array(total);
        let w = 0;
        for (const c of this.chunks) {
            for (const s of c.splats)
                out[w++] = s;
        }
        return out;
    }
}
//# sourceMappingURL=webgpu.js.map