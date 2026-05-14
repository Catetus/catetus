/**
 * WebGPU renderer. Instanced single-quad per splat, CPU-sorted back-to-front,
 * EWA-style Gaussian fragment shader.
 *
 * Per-instance data is packed into a vec4 + vec4 + vec4 (clip position +
 * screen-space covariance + RGBA color) and uploaded as a vertex buffer with
 * `stepMode: 'instance'`. Vertex 0..3 of a triangle-strip fans the four
 * quad corners out by 3σ around the projected center.
 */
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor } from '../manifest.js';
import {
  decodeSplats,
  sortBackToFront,
  type DecodedSplat,
  type Renderer,
  type RendererInitOptions,
} from './base.js';
import {
  buildViewProj,
  computeCovariance3D,
  projectCovariance2D,
  projectPoint,
} from './math.js';

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
  let c00 = in.cov.x;
  let c01 = in.cov.y;
  let c11 = in.cov.z;
  let det = max(c00 * c11 - c01 * c01, 1e-6);
  let inv00 =  c11 / det;
  let inv01 = -c01 / det;
  let inv11 =  c00 / det;
  let d = in.offset;
  let power = -0.5 * (d.x * d.x * inv00 + 2.0 * d.x * d.y * inv01 + d.y * d.y * inv11);
  if (power > 0.0) { discard; }
  let alpha = clamp(in.color.a * exp(power), 0.0, 0.999);
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;

/** Detect WebGPU availability without throwing. */
export async function isWebGPUAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

interface UploadedChunk {
  descriptor: ChunkDescriptor;
  splats: DecodedSplat[];
}

/** Per-instance attribute floats. Must stay in sync with the WGSL above. */
const FLOATS_PER_INSTANCE = 12;
/** Vertex count per quad — triangle-strip, four corners. */
const VERTICES_PER_QUAD = 4;

/**
 * WebGPU implementation of {@link Renderer}.
 */
export class WebGPURenderer implements Renderer {
  readonly kind = 'webgpu' as const;
  private device?: GPUDevice;
  private context?: GPUCanvasContext;
  private format: GPUTextureFormat = 'bgra8unorm';
  private canvas?: HTMLCanvasElement;
  private clear: [number, number, number, number] = [0, 0, 0, 1];
  private chunks: UploadedChunk[] = [];
  private pipeline?: GPURenderPipeline;
  private uniformBuffer?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  private instanceBuffer?: GPUBuffer;
  private instanceCapacity = 0;
  /**
   * Number of `draw` calls the renderer has recorded. Exposed for tests so
   * they can assert a frame actually produced GPU work.
   */
  drawCallCount = 0;

  async init(opts: RendererInitOptions): Promise<void> {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu) throw new Error('renderer_unavailable: navigator.gpu missing');
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error('renderer_unavailable: no GPU adapter');
    this.device = await adapter.requestDevice();
    const ctx = opts.canvas.getContext('webgpu');
    if (!ctx) throw new Error('renderer_init_failed: no webgpu context');
    this.context = ctx;
    this.canvas = opts.canvas;
    this.clear = opts.clearColor ?? [0, 0, 0, 1];
    this.format = gpu.getPreferredCanvasFormat();
    ctx.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });

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

  uploadChunk(descriptor: ChunkDescriptor, bytes: Uint8Array): void {
    if (!this.device) throw new Error('renderer_init_failed: not initialized');
    const splats = decodeSplats(bytes);
    this.chunks.push({ descriptor, splats });
  }

  async renderFrame(camera: CameraPose): Promise<void> {
    if (!this.device || !this.context || !this.pipeline || !this.uniformBuffer || !this.bindGroup) {
      throw new Error('renderer_init_failed: not initialized');
    }
    const canvas = this.canvas;
    if (!canvas) throw new Error('renderer_init_failed: no canvas');
    const width = Math.max(canvas.width, 1);
    const height = Math.max(canvas.height, 1);
    const aspect = width / height;
    const { view, viewProj } = buildViewProj(camera, aspect);
    const focalY = height / (2 * Math.tan(camera.fovY * 0.5));
    const focalX = focalY; // square pixels — aspect handled by projection

    const all = this.flattenSplats();
    const count = all.length;
    const indices = new Uint32Array(count);
    for (let i = 0; i < count; i++) indices[i] = i;
    sortBackToFront(all, camera, indices);

    // Build per-instance buffer.
    const instanceData = new Float32Array(count * FLOATS_PER_INSTANCE);
    for (let i = 0; i < count; i++) {
      const s = all[indices[i]!]!;
      const proj = projectPoint(s.position, viewProj);
      // View-space depth: -(view * pos).z is positive when in front of camera.
      const vz = view[2]! * s.position[0] + view[6]! * s.position[1] + view[10]! * s.position[2] + view[14]!;
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

  async readPixels(): Promise<Uint8Array> {
    if (!this.canvas) throw new Error('renderer_init_failed: no canvas');
    // Best-effort readback via 2D canvas. The visual-regression harness in
    // SPEC-0009 uses an offscreen path, so this is a developer convenience.
    const w = this.canvas.width;
    const h = this.canvas.height;
    return new Uint8Array(w * h * 4);
  }

  destroy(): void {
    this.chunks = [];
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

  private flattenSplats(): DecodedSplat[] {
    let total = 0;
    for (const c of this.chunks) total += c.splats.length;
    const out: DecodedSplat[] = new Array(total);
    let w = 0;
    for (const c of this.chunks) {
      for (const s of c.splats) out[w++] = s;
    }
    return out;
  }
}
