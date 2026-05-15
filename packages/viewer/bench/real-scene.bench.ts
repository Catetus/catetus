// SPDX-License-Identifier: Apache-2.0
/**
 * Real-scene WebGPU bench harness.
 *
 * Loads a pre-packed SoA chunk-bytes file (produced by
 * `bench/scripts/ply-to-soa.py` from an Inria-format 3DGS PLY) plus its
 * sidecar `.meta.json`, then runs the same fps + per-stage timestamp-query
 * measurement loop as `compute-decode.bench.ts::runBench`, but at production-
 * shape 1080p and from three real camera viewpoints derived from the scene's
 * bounding-box centroid.
 *
 * This file is loaded by `bench/index.html` only when the page is hit with a
 * `?scene=<name>` query — the harness fetches `/scenes/<name>.bin` and
 * `/scenes/<name>.meta.json` from the bench server.
 *
 * Why this exists: the synthetic random-cube bench mis-leadingly reported
 * 41-73% culled and 2.5x fps for B4b opacity cull, but real Inria-trained
 * scenes have <1% cull rate and essentially no fps win. Every future render
 * optimization (LODGE, atomic-free scatter on mobile, progressive bitstream
 * viewer, WebSplatter culling) is gated on this harness reporting a real-
 * scene improvement.
 */

import {
  ComputeDecodePipeline,
  FLOATS_PER_INSTANCE,
  BYTES_PER_DECODED_SPLAT,
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

export interface RealSceneStageTimings {
  keygenOrProject: number;
  sortFull: number;
  projectGatherOrGather: number;
  totalGpuMs: number;
  path: 'fused' | 'legacy';
}

export interface RealSceneViewResult {
  view: 'front' | 'orbit45' | 'orbit90';
  perFrameMs: number;
  framesPerSecond: number;
  perStageMsTimestamp?: RealSceneStageTimings;
}

export interface RealSceneResult {
  scene: string;
  splatCount: number;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  decodeMs: number;
  viewport: [number, number];
  iterations: number;
  views: RealSceneViewResult[];
  /** Median fps across the three viewpoints — headline number. */
  medianFps: number;
  /** Median per-stage timestamps across the three viewpoints. */
  medianPerStageMs?: RealSceneStageTimings;
}

/** Fetch + parse a scene from the bench server. */
async function loadScene(name: string): Promise<{ bytes: Uint8Array; meta: SceneMeta }> {
  const metaRes = await fetch(`/scenes/${name}.meta.json`);
  if (!metaRes.ok) throw new Error(`scene meta load failed: ${name} (${metaRes.status})`);
  const meta = (await metaRes.json()) as SceneMeta;
  const binRes = await fetch(`/scenes/${name}.bin`);
  if (!binRes.ok) throw new Error(`scene bin load failed: ${name} (${binRes.status})`);
  const buf = await binRes.arrayBuffer();
  if (buf.byteLength !== meta.byteLength) {
    throw new Error(`scene ${name}: byteLength mismatch meta=${meta.byteLength} got=${buf.byteLength}`);
  }
  return { bytes: new Uint8Array(buf), meta };
}

function buildDescriptorFromMeta(name: string, meta: SceneMeta): ChunkDescriptor {
  return {
    uri: `bench:real-scene:${name}`,
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

/** Produce three camera poses for a scene: front, +45 orbit, +90 orbit. */
function buildCameras(meta: SceneMeta, aspect: number): Array<{ name: RealSceneViewResult['view']; cam: CameraPose }> {
  const cx = (meta.bbox.min[0] + meta.bbox.max[0]) * 0.5;
  const cy = (meta.bbox.min[1] + meta.bbox.max[1]) * 0.5;
  const cz = (meta.bbox.min[2] + meta.bbox.max[2]) * 0.5;
  const dx = meta.bbox.max[0] - meta.bbox.min[0];
  const dy = meta.bbox.max[1] - meta.bbox.min[1];
  const dz = meta.bbox.max[2] - meta.bbox.min[2];
  // Distance: ~1.4x the diagonal so the scene fills the frame.
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const radius = Math.max(diag * 0.7, 1.0);
  // Inria scenes use a roughly +Y-up world.
  const up: [number, number, number] = [0, 1, 0];
  const target: [number, number, number] = [cx, cy, cz];
  const make = (angleDeg: number): CameraPose => {
    const a = (angleDeg * Math.PI) / 180;
    const px = cx + radius * Math.sin(a);
    const pz = cz + radius * Math.cos(a);
    const py = cy; // keep camera at scene-centre height; simpler than chasing y
    // Far plane: 10x the diagonal so we don't clip; near 0.5% of diagonal.
    return {
      position: [px, py, pz],
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

const VIEWPORT: [number, number] = [1920, 1080];

function focalFor(viewport: [number, number], fovY: number): [number, number] {
  const fy = viewport[1] / (2 * Math.tan(fovY / 2));
  // square-pixel assumption: fx == fy.
  return [fy, fy];
}

function median(xs: number[]): number {
  const sorted = xs.slice().sort((p, q) => p - q);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Run one camera viewpoint: fps loop + optional timestamp-query loop. */
async function runOneView(
  device: GPUDevice,
  pipeline: ComputeDecodePipeline,
  cam: CameraPose,
  iterations: number,
): Promise<{ perFrameMs: number; fps: number; ts?: RealSceneStageTimings }> {
  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const camWithAspect: CameraPose = { ...cam, aspect };
  const { view, viewProj } = buildViewProj(camWithAspect, aspect);
  const focal = focalFor(VIEWPORT, camWithAspect.fovY);

  // Warm-up.
  {
    const e = device.createCommandEncoder();
    pipeline.encode(e, view, viewProj, focal, VIEWPORT);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // fps loop.
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    const e = device.createCommandEncoder();
    pipeline.encode(e, view, viewProj, focal, VIEWPORT);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const perFrameMs = totalMs / iterations;
  const fps = 1000 / perFrameMs;

  // Optional timestamp loop (fused path: 6 timestamps).
  let ts: RealSceneStageTimings | undefined;
  if (device.features.has('timestamp-query')) {
    const tsIterations = 11;
    const TS_COUNT = 6;
    const querySet = device.createQuerySet({ type: 'timestamp', count: TS_COUNT });
    const resolveBuf = device.createBuffer({
      size: TS_COUNT * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = device.createBuffer({
      size: TS_COUNT * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const samplesKeygen: number[] = [];
    const samplesSort: number[] = [];
    const samplesGather: number[] = [];
    const samplesTotal: number[] = [];
    for (let i = 0; i < tsIterations; i++) {
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
      samplesKeygen.push(a);
      samplesSort.push(b);
      samplesGather.push(c);
      samplesTotal.push(a + b + c);
    }
    ts = {
      keygenOrProject: median(samplesKeygen),
      sortFull: median(samplesSort),
      projectGatherOrGather: median(samplesGather),
      totalGpuMs: median(samplesTotal),
      path: 'fused',
    };
    querySet.destroy();
    resolveBuf.destroy();
    readBuf.destroy();
  }

  return { perFrameMs, fps, ts };
}

/** Bench one real scene through three viewpoints. */
export async function runRealSceneBench(
  device: GPUDevice,
  sceneName: string,
  iterations = 30,
): Promise<RealSceneResult> {
  const { bytes, meta } = await loadScene(sceneName);
  const descriptor = buildDescriptorFromMeta(sceneName, meta);
  const pipeline = new ComputeDecodePipeline({ device, capacity: meta.splatCount });

  const decodeStart = performance.now();
  pipeline.uploadChunk(descriptor, bytes);
  await device.queue.onSubmittedWorkDone();
  const decodeMs = performance.now() - decodeStart;

  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const cams = buildCameras(meta, aspect);

  const views: RealSceneViewResult[] = [];
  for (const { name, cam } of cams) {
    const r = await runOneView(device, pipeline, cam, iterations);
    views.push({
      view: name,
      perFrameMs: r.perFrameMs,
      framesPerSecond: r.fps,
      perStageMsTimestamp: r.ts,
    });
  }

  pipeline.destroy();

  const medianFps = median(views.map((v) => v.framesPerSecond));
  let medianPerStage: RealSceneStageTimings | undefined;
  if (views.every((v) => v.perStageMsTimestamp)) {
    medianPerStage = {
      keygenOrProject: median(views.map((v) => v.perStageMsTimestamp!.keygenOrProject)),
      sortFull: median(views.map((v) => v.perStageMsTimestamp!.sortFull)),
      projectGatherOrGather: median(views.map((v) => v.perStageMsTimestamp!.projectGatherOrGather)),
      totalGpuMs: median(views.map((v) => v.perStageMsTimestamp!.totalGpuMs)),
      path: 'fused',
    };
  }

  return {
    scene: sceneName,
    splatCount: meta.splatCount,
    bbox: meta.bbox,
    decodeMs,
    viewport: VIEWPORT,
    iterations,
    views,
    medianFps,
    medianPerStageMs: medianPerStage,
  };
}

/**
 * Entry point. Bench server provides a scene list via `/scenes/index.json`
 * (an array of names). Each scene is fetched + benched. Populates
 * `window.__benchRealScene` with the full report.
 */
export async function main(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    (window as unknown as { __benchRealScene: unknown }).__benchRealScene = { error: 'no_webgpu' };
    return;
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    (window as unknown as { __benchRealScene: unknown }).__benchRealScene = { error: 'no_adapter' };
    return;
  }
  const want = {
    maxStorageBufferBindingSize: Math.min((adapter.limits.maxStorageBufferBindingSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
    maxBufferSize: Math.min((adapter.limits.maxBufferSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
  };
  const tsFeature: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({
    requiredFeatures: tsFeature,
    requiredLimits: want,
  });
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
    (window as unknown as { __benchRealScene: unknown }).__benchRealScene = {
      error: 'no_scenes',
      message: String((err as Error)?.message ?? err),
    };
    return;
  }

  const results: RealSceneResult[] = [];
  for (const name of scenes) {
    try {
      log(`real-scene bench: starting ${name}`);
      const r = await runRealSceneBench(device, name, 30);
      log(`real-scene bench: ${name} n=${r.splatCount} medianFps=${r.medianFps.toFixed(2)}`);
      results.push(r);
    } catch (err) {
      (window as unknown as { __benchRealScene: unknown }).__benchRealScene = {
        error: `real_scene_bench_failed_${name}`,
        message: String((err as Error)?.message ?? err),
        results,
      };
      return;
    }
  }

  (window as unknown as { __benchRealScene: unknown }).__benchRealScene = {
    results,
    sizes: { bytes_per_decoded_splat: BYTES_PER_DECODED_SPLAT, floats_per_instance: FLOATS_PER_INSTANCE },
    adapter: adapterInfo,
    limits: want,
    timestamp: new Date().toISOString(),
  };
}
