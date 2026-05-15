// SPDX-License-Identifier: Apache-2.0
/**
 * EWA-dilation sweep bench. Companion to `real-scene.bench.ts`.
 *
 * For each scene listed at `/scenes/index.json`, runs the WebGPU compute-
 * decode pipeline FIVE times with different `dilation` floors (the EWA
 * anti-aliasing 2D-covariance regulariser; default 0.3 inherited from the
 * Inria CUDA rasteriser). Per dilation, measures:
 *
 *   * `fpsCullOff` — vanilla `encode()` path fps at 1080p, median of 3
 *     viewpoints (front / orbit45 / orbit90).
 *   * `fpsCullOn`  — `encodeWithCull(tau=1/255)` fps at 1080p, same 3
 *     viewpoints, after a warm-up survivor-readback.
 *   * `cullRate`    — `1 - survivors/totalSplats` at the front view.
 *   * `renderImage` — a 512x512 off-screen RGBA8 render of the front view
 *     (the production WGSL shader from `renderer/webgpu.ts` re-embedded in
 *     this file) that the host turns into a PSNR vs the dilation=0.3
 *     baseline image of the same scene/view.
 *
 * Output:
 *   window.__benchDilation = { results: [{ scene, splatCount, dilations:
 *     [{ dilation, fpsCullOff, fpsCullOn, cullRate, psnrVsBaseline }] }],
 *     ... }
 */

import {
  ComputeDecodePipeline,
  FLOATS_PER_INSTANCE,
} from '../src/webgpu/index.js';
import type { ChunkDescriptor, SoaAttributeLayout } from '../src/manifest.js';
import { buildViewProj } from '../src/renderer/math.js';
import type { CameraPose } from '../src/camera.js';

interface SceneMeta {
  splatCount: number;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  byteLength: number;
  source: string;
  layout: SoaAttributeLayout;
}

const VIEWPORT: [number, number] = [1920, 1080];
const RENDER_SIZE: [number, number] = [512, 512]; // smaller for readback bandwidth
const DILATIONS: number[] = [0.3, 0.2, 0.1, 0.05, 0.0];
const TAU = 1 / 255;

const RENDER_WGSL = /* wgsl */ `
struct Instance {
  @location(0) clipPos : vec4<f32>,
  @location(1) cov     : vec4<f32>,
  @location(2) color   : vec4<f32>,
};
struct Uniforms {
  viewportSize : vec2<f32>,
  _pad         : vec2<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;
struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) offset : vec2<f32>,
  @location(1) cov    : vec3<f32>,
  @location(2) color  : vec4<f32>,
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
  let ndcOffset = c * radiusPx * 2.0 / u.viewportSize;
  var out : VsOut;
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

async function loadScene(name: string): Promise<{ bytes: Uint8Array; meta: SceneMeta }> {
  const metaRes = await fetch(`/scenes/${name}.meta.json`);
  if (!metaRes.ok) throw new Error(`scene meta load failed: ${name} (${metaRes.status})`);
  const meta = (await metaRes.json()) as SceneMeta;
  const binRes = await fetch(`/scenes/${name}.bin`);
  if (!binRes.ok) throw new Error(`scene bin load failed: ${name} (${binRes.status})`);
  const buf = await binRes.arrayBuffer();
  return { bytes: new Uint8Array(buf), meta };
}

function buildDescriptorFromMeta(name: string, meta: SceneMeta): ChunkDescriptor {
  return {
    uri: `bench:real-dilation:${name}`,
    byteOffset: 0,
    byteLength: meta.byteLength,
    splatCount: meta.splatCount,
    bbox: meta.bbox,
    lod: 0,
    checksum: '',
    loadPriority: 0,
    attributeLayout: meta.layout,
  };
}

function buildCameras(meta: SceneMeta, aspect: number): Array<{ name: 'front' | 'orbit45' | 'orbit90'; cam: CameraPose }> {
  const cx = (meta.bbox.min[0] + meta.bbox.max[0]) * 0.5;
  const cy = (meta.bbox.min[1] + meta.bbox.max[1]) * 0.5;
  const cz = (meta.bbox.min[2] + meta.bbox.max[2]) * 0.5;
  const dx = meta.bbox.max[0] - meta.bbox.min[0];
  const dy = meta.bbox.max[1] - meta.bbox.min[1];
  const dz = meta.bbox.max[2] - meta.bbox.min[2];
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const radius = Math.max(diag * 0.7, 1.0);
  const up: [number, number, number] = [0, 1, 0];
  const target: [number, number, number] = [cx, cy, cz];
  const make = (angleDeg: number): CameraPose => {
    const a = (angleDeg * Math.PI) / 180;
    const px = cx + radius * Math.sin(a);
    const pz = cz + radius * Math.cos(a);
    return {
      position: [px, cy, pz],
      target,
      up,
      fovY: Math.PI / 3,
      near: Math.max(diag * 0.005, 0.05),
      far: Math.max(diag * 10, 100),
      aspect,
    };
  };
  return [
    { name: 'front', cam: make(0) },
    { name: 'orbit45', cam: make(45) },
    { name: 'orbit90', cam: make(90) },
  ];
}

function focalFor(viewport: [number, number], fovY: number): [number, number] {
  const fy = viewport[1] / (2 * Math.tan(fovY / 2));
  return [fy, fy];
}

function median(xs: number[]): number {
  const sorted = xs.slice().sort((p, q) => p - q);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Setup an off-screen render target + bind group + pipeline reusing the production WGSL render. */
function makeRenderTarget(device: GPUDevice, viewport: [number, number]): {
  pipeline: GPURenderPipeline;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  colorTex: GPUTexture;
  depthTex: GPUTexture;
  readbackBuf: GPUBuffer;
} {
  const module = device.createShaderModule({ code: RENDER_WGSL });
  const bgl = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
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
          format: 'rgba8unorm',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip', stripIndexFormat: undefined },
  });
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([viewport[0], viewport[1], 0, 0]).buffer);
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });
  const colorTex = device.createTexture({
    size: [viewport[0], viewport[1], 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depthTex = device.createTexture({
    size: [viewport[0], viewport[1], 1],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // 256-byte-aligned bytesPerRow for COPY_TEXTURE_TO_BUFFER.
  const bytesPerRow = Math.ceil(viewport[0] * 4 / 256) * 256;
  const readbackBuf = device.createBuffer({
    size: bytesPerRow * viewport[1],
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return { pipeline, uniformBuffer, bindGroup, colorTex, depthTex, readbackBuf };
}

async function renderAndReadback(
  device: GPUDevice,
  pipeline: ComputeDecodePipeline,
  rt: ReturnType<typeof makeRenderTarget>,
  cam: CameraPose,
  splatCount: number,
): Promise<Uint8Array> {
  const aspect = RENDER_SIZE[0] / RENDER_SIZE[1];
  const camWithAspect: CameraPose = { ...cam, aspect };
  const { view, viewProj } = buildViewProj(camWithAspect, aspect);
  const focal = focalFor(RENDER_SIZE, camWithAspect.fovY);

  const enc = device.createCommandEncoder();
  // 1. Compute path: project + sort into instanceBuffer.
  pipeline.encode(enc, view, viewProj, focal, RENDER_SIZE);
  // 2. Rasterise into off-screen texture.
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: rt.colorTex.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(rt.pipeline);
  pass.setBindGroup(0, rt.bindGroup);
  pass.setVertexBuffer(0, pipeline.instanceBuffer);
  pass.draw(4, splatCount, 0, 0);
  pass.end();
  // 3. Copy texture → readback buffer.
  const bytesPerRow = Math.ceil(RENDER_SIZE[0] * 4 / 256) * 256;
  enc.copyTextureToBuffer(
    { texture: rt.colorTex },
    { buffer: rt.readbackBuf, bytesPerRow, rowsPerImage: RENDER_SIZE[1] },
    { width: RENDER_SIZE[0], height: RENDER_SIZE[1], depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await rt.readbackBuf.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(rt.readbackBuf.getMappedRange().slice(0));
  rt.readbackBuf.unmap();
  // De-pad rows.
  const out = new Uint8Array(RENDER_SIZE[0] * RENDER_SIZE[1] * 4);
  for (let y = 0; y < RENDER_SIZE[1]; y++) {
    out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + RENDER_SIZE[0] * 4), y * RENDER_SIZE[0] * 4);
  }
  return out;
}

/** PSNR in dB of `a` vs `b` (both unsigned-byte RGBA, alpha ignored). */
function psnrRgb(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) throw new Error('psnr: length mismatch');
  let sse = 0;
  let n = 0;
  for (let i = 0; i + 3 < a.length; i += 4) {
    const dr = a[i] - b[i];
    const dg = a[i + 1] - b[i + 1];
    const db = a[i + 2] - b[i + 2];
    sse += dr * dr + dg * dg + db * db;
    n += 3;
  }
  if (sse === 0) return Infinity;
  const mse = sse / n;
  return 10 * Math.log10((255 * 255) / mse);
}

interface DilationResult {
  dilation: number;
  fpsCullOff: number;
  fpsCullOn: number;
  cullRate: number;
  survivors: number;
  psnrVsBaseline: number;
  perStageMsCullOff: { keygenOrProject: number; sortFull: number; projectGatherOrGather: number; totalGpuMs: number };
}

interface SceneDilationReport {
  scene: string;
  splatCount: number;
  viewport: [number, number];
  iterations: number;
  baselineImageMean: number;
  dilations: DilationResult[];
}

async function fpsLoop(
  device: GPUDevice,
  pipeline: ComputeDecodePipeline,
  cam: CameraPose,
  iterations: number,
  cullOn: boolean,
): Promise<{ perFrameMs: number; fps: number }> {
  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const camWithAspect: CameraPose = { ...cam, aspect };
  const { view, viewProj } = buildViewProj(camWithAspect, aspect);
  const focal = focalFor(VIEWPORT, camWithAspect.fovY);

  // Warm-up.
  {
    const e = device.createCommandEncoder();
    if (cullOn) await pipeline.encodeWithCull(e, view, viewProj, focal, VIEWPORT, TAU);
    else pipeline.encode(e, view, viewProj, focal, VIEWPORT);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  if (cullOn && pipeline.cull) {
    await pipeline.cull.readSurvivorCount();
    // Re-issue once so cs_project_cmpct sees the updated count.
    const e = device.createCommandEncoder();
    await pipeline.encodeWithCull(e, view, viewProj, focal, VIEWPORT, TAU);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    const e = device.createCommandEncoder();
    if (cullOn) await pipeline.encodeWithCull(e, view, viewProj, focal, VIEWPORT, TAU);
    else pipeline.encode(e, view, viewProj, focal, VIEWPORT);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const perFrameMs = totalMs / iterations;
  return { perFrameMs, fps: 1000 / perFrameMs };
}

async function timedStages(
  device: GPUDevice,
  pipeline: ComputeDecodePipeline,
  cam: CameraPose,
): Promise<{ keygenOrProject: number; sortFull: number; projectGatherOrGather: number; totalGpuMs: number } | undefined> {
  if (!device.features.has('timestamp-query')) return undefined;
  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const camWithAspect: CameraPose = { ...cam, aspect };
  const { view, viewProj } = buildViewProj(camWithAspect, aspect);
  const focal = focalFor(VIEWPORT, camWithAspect.fovY);
  const TS_COUNT = 6;
  const querySet = device.createQuerySet({ type: 'timestamp', count: TS_COUNT });
  const resolveBuf = device.createBuffer({ size: TS_COUNT * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: TS_COUNT * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const N = 7;
  const sK: number[] = []; const sS: number[] = []; const sG: number[] = []; const sT: number[] = [];
  for (let i = 0; i < N; i++) {
    const enc = device.createCommandEncoder();
    pipeline.encodeTimed(enc, view, viewProj, focal, VIEWPORT, querySet, 0);
    enc.resolveQuerySet(querySet, 0, TS_COUNT, resolveBuf, 0);
    enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, TS_COUNT * 8);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const tsBuf = new BigInt64Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    const a = Number(tsBuf[1] - tsBuf[0]) / 1e6;
    const b = Number(tsBuf[3] - tsBuf[2]) / 1e6;
    const c = Number(tsBuf[5] - tsBuf[4]) / 1e6;
    sK.push(a); sS.push(b); sG.push(c); sT.push(a + b + c);
  }
  querySet.destroy(); resolveBuf.destroy(); readBuf.destroy();
  return { keygenOrProject: median(sK), sortFull: median(sS), projectGatherOrGather: median(sG), totalGpuMs: median(sT) };
}

async function runOneSceneOneDilation(
  device: GPUDevice,
  sceneName: string,
  bytes: Uint8Array,
  meta: SceneMeta,
  dilation: number,
  iterations: number,
): Promise<{
  result: Omit<DilationResult, 'psnrVsBaseline'>;
  frontFrame: Uint8Array;
}> {
  const descriptor = buildDescriptorFromMeta(sceneName, meta);
  // We need cull, so must use non-fused. fused-project is the production
  // path but it currently throws on useCull=true; non-fused mirrors the
  // existing cull harness exactly.
  const pipeline = new ComputeDecodePipeline({
    device,
    capacity: meta.splatCount,
    useFusedProject: false,
    useCull: true,
    dilation,
  });
  pipeline.uploadChunk(descriptor, bytes);
  await device.queue.onSubmittedWorkDone();

  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const cams = buildCameras(meta, aspect);

  // FPS sweep (3 viewpoints, cull off and on).
  const cullOffFps: number[] = [];
  const cullOnFps: number[] = [];
  for (const { cam } of cams) {
    cullOffFps.push((await fpsLoop(device, pipeline, cam, iterations, false)).fps);
  }
  // Cull-rate readback at front view.
  let frontSurvivors = 0;
  {
    const cam = cams[0].cam;
    const cwa: CameraPose = { ...cam, aspect };
    const { view, viewProj } = buildViewProj(cwa, aspect);
    const focal = focalFor(VIEWPORT, cwa.fovY);
    const e = device.createCommandEncoder();
    await pipeline.encodeWithCull(e, view, viewProj, focal, VIEWPORT, TAU);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (pipeline.cull) frontSurvivors = await pipeline.cull.readSurvivorCount();
  }
  for (const { cam } of cams) {
    cullOnFps.push((await fpsLoop(device, pipeline, cam, iterations, true)).fps);
  }

  // Stage breakdown at front view (cull-off).
  const stages = await timedStages(device, pipeline, cams[0].cam);

  // Render front view to off-screen 512x512 RGBA + readback for PSNR.
  const rt = makeRenderTarget(device, RENDER_SIZE);
  // Re-issue compute encode for the render frame. Use fpsLoop's last camera
  // state (front) so the instanceBuffer reflects that viewpoint.
  const frontFrame = await renderAndReadback(device, pipeline, rt, cams[0].cam, meta.splatCount);
  rt.colorTex.destroy(); rt.depthTex.destroy(); rt.readbackBuf.destroy(); rt.uniformBuffer.destroy();

  pipeline.destroy();

  return {
    result: {
      dilation,
      fpsCullOff: median(cullOffFps),
      fpsCullOn: median(cullOnFps),
      cullRate: 1 - frontSurvivors / meta.splatCount,
      survivors: frontSurvivors,
      perStageMsCullOff: stages ?? { keygenOrProject: 0, sortFull: 0, projectGatherOrGather: 0, totalGpuMs: 0 },
    },
    frontFrame,
  };
}

export async function runDilationSweepForScene(
  device: GPUDevice,
  sceneName: string,
  iterations = 20,
): Promise<SceneDilationReport> {
  const { bytes, meta } = await loadScene(sceneName);

  const dilationResults: DilationResult[] = [];
  let baselineFrame: Uint8Array | null = null;
  let baselineMean = 0;
  for (const dilation of DILATIONS) {
    const { result, frontFrame } = await runOneSceneOneDilation(device, sceneName, bytes, meta, dilation, iterations);
    let psnr: number;
    if (baselineFrame === null) {
      baselineFrame = frontFrame;
      // Track mean pixel value to confirm the frame wasn't black.
      let s = 0; let n = 0;
      for (let i = 0; i + 3 < baselineFrame.length; i += 4) { s += baselineFrame[i] + baselineFrame[i + 1] + baselineFrame[i + 2]; n += 3; }
      baselineMean = s / n;
      psnr = Infinity;
    } else {
      psnr = psnrRgb(frontFrame, baselineFrame);
    }
    dilationResults.push({ ...result, psnrVsBaseline: psnr });
  }
  return {
    scene: sceneName,
    splatCount: meta.splatCount,
    viewport: VIEWPORT,
    iterations,
    baselineImageMean: baselineMean,
    dilations: dilationResults,
  };
}

export async function main(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) { (window as unknown as { __benchDilation: unknown }).__benchDilation = { error: 'no_webgpu' }; return; }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { (window as unknown as { __benchDilation: unknown }).__benchDilation = { error: 'no_adapter' }; return; }
  const want = {
    maxStorageBufferBindingSize: Math.min((adapter.limits.maxStorageBufferBindingSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
    maxBufferSize: Math.min((adapter.limits.maxBufferSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
  };
  const tsFeature: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({ requiredFeatures: tsFeature, requiredLimits: want });
  const adapterInfo = (adapter as unknown as { info?: { vendor?: string; architecture?: string; device?: string } }).info ?? {};

  const log = (msg: string): void => {
    const el = document.getElementById('log');
    if (el) el.textContent += `\n${msg}`;
    // eslint-disable-next-line no-console
    console.log(msg);
  };

  let scenes: string[] = [];
  try {
    const idx = await fetch('/scenes/index.json');
    if (!idx.ok) throw new Error(`index.json: ${idx.status}`);
    scenes = (await idx.json()) as string[];
  } catch (err) {
    (window as unknown as { __benchDilation: unknown }).__benchDilation = {
      error: 'no_scenes',
      message: String((err as Error)?.message ?? err),
    };
    return;
  }

  const results: SceneDilationReport[] = [];
  for (const name of scenes) {
    try {
      log(`dilation sweep: starting ${name}`);
      const r = await runDilationSweepForScene(device, name, 20);
      for (const d of r.dilations) {
        log(`  d=${d.dilation.toFixed(3)} cullRate=${(d.cullRate * 100).toFixed(2)}% fpsOff=${d.fpsCullOff.toFixed(2)} fpsOn=${d.fpsCullOn.toFixed(2)} psnr=${d.psnrVsBaseline === Infinity ? 'inf' : d.psnrVsBaseline.toFixed(2)}`);
      }
      results.push(r);
    } catch (err) {
      (window as unknown as { __benchDilation: unknown }).__benchDilation = {
        error: `dilation_bench_failed_${name}`,
        message: String((err as Error)?.message ?? err),
        stack: String((err as Error)?.stack ?? ''),
        results,
      };
      return;
    }
  }

  (window as unknown as { __benchDilation: unknown }).__benchDilation = {
    results,
    dilations: DILATIONS,
    tau: TAU,
    adapter: adapterInfo,
    limits: want,
    timestamp: new Date().toISOString(),
  };
}
