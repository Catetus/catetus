// SPDX-License-Identifier: Apache-2.0
/**
 * Real-scene LODGE WebGPU bench harness.
 *
 * Drives the Phase A.3 `LodgeLODPipeline` against a real `.lodge`
 * directory (the Phase A.1 chunker's output). Measures fps from three
 * camera viewpoints, varying the LOD level selected by the CPU heuristic
 * so the bench captures the speedup curve "L0 (finest) → L_N (coarsest)".
 *
 * The headline number reported in `results.json::realSceneLodge` is the
 * median fps **at the LOD level the on-camera heuristic would pick** —
 * i.e. the "production" fps the LOD system would deliver. The
 * `levelBreakdown[]` field exposes per-level fps for diagnosis.
 *
 * Triggered by the 4090 schtask `sf_bench_real` via run-bench-windows.mjs
 * when the bench server discovers a `.lodge/` directory in
 * `SF_BENCH_PLY_DIR`. Mirrors `real-scene.bench.ts` for the underlying
 * compute-decode + radix-sort + project-gather pipeline; the difference
 * is the streaming + LOD layer wrapped around it.
 */

import {
  ComputeDecodePipeline,
  FLOATS_PER_INSTANCE,
  BYTES_PER_DECODED_SPLAT,
} from '../src/webgpu/index.js';
import { buildViewProj } from '../src/renderer/math.js';
import type { CameraPose } from '../src/camera.js';
import {
  LodgeChunkLoader,
  LodgeLODPipeline,
  parseLodgeManifest,
  sceneBboxCenter,
  sceneBboxRadius,
  type LodgeManifest,
} from '../src/lodge/index.js';

const VIEWPORT: [number, number] = [1920, 1080];

export interface LodgeLevelResult {
  level: number;
  splatCount: number;
  /** Median ms-per-frame over the three viewpoints. */
  perFrameMs: number;
  /** Median fps over the three viewpoints. */
  framesPerSecond: number;
  /** Per-viewpoint fps for diagnosis. */
  viewFps: { front: number; orbit45: number; orbit90: number };
  /** Wall-clock ms to upload + decode this level into the GPU pipeline. */
  uploadMs: number;
  /** Bytes resident in the GPU decoded-splat buffer for this level. */
  decodedBytes: number;
}

export interface LodgeSceneResult {
  scene: string;
  manifestSource: string;
  originalSplatCount: number;
  numLevels: number;
  viewport: [number, number];
  iterations: number;
  levelBreakdown: LodgeLevelResult[];
  /** fps at the level the CPU LOD heuristic picks for the centre camera. */
  selectedLevel: number;
  selectedFps: number;
  /** fps if we forced L0 (full representation) — the "no-LOD baseline". */
  baselineFps: number | null;
  /** Speedup factor `selectedFps / baselineFps`. Null when baseline OOM. */
  speedup: number | null;
}

function focalFor(viewport: [number, number], fovY: number): [number, number] {
  const fy = viewport[1] / (2 * Math.tan(fovY / 2));
  return [fy, fy];
}

function median(xs: number[]): number {
  if (xs.length === 0) return Number.NaN;
  const sorted = xs.slice().sort((p, q) => p - q);
  return sorted[Math.floor(sorted.length / 2)]!;
}

function buildCameras(manifest: LodgeManifest, aspect: number): Array<{ name: 'front' | 'orbit45' | 'orbit90'; cam: CameraPose }> {
  const c = sceneBboxCenter(manifest);
  const r = sceneBboxRadius(manifest);
  const radius = Math.max(r * 1.4, 1.0);
  const up: [number, number, number] = [0, 1, 0];
  const target: [number, number, number] = [c[0], c[1], c[2]];
  const make = (angleDeg: number): CameraPose => {
    const a = (angleDeg * Math.PI) / 180;
    return {
      position: [c[0] + radius * Math.sin(a), c[1], c[2] + radius * Math.cos(a)],
      target,
      up,
      fovY: Math.PI / 3,
      near: Math.max(r * 0.01, 0.05),
      far: Math.max(r * 20, 100),
      aspect,
    };
  };
  return [
    { name: 'front', cam: make(0) },
    { name: 'orbit45', cam: make(45) },
    { name: 'orbit90', cam: make(90) },
  ];
}

/** Measure fps for an already-warmed-up pipeline at one camera pose. */
async function runOneView(
  device: GPUDevice,
  pipeline: ComputeDecodePipeline,
  cam: CameraPose,
  iterations: number,
): Promise<{ perFrameMs: number; fps: number }> {
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

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    const e = device.createCommandEncoder();
    pipeline.encode(e, view, viewProj, focal, VIEWPORT);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const perFrameMs = totalMs / iterations;
  return { perFrameMs, fps: 1000 / perFrameMs };
}

/** Run the per-level fps sweep for one .lodge directory. */
export async function runLodgeBench(
  device: GPUDevice,
  sceneName: string,
  iterations = 20,
  capacityCap = 50_000_000,
): Promise<LodgeSceneResult> {
  // Manifest is at /lodges/<name>/manifest.json. The bench server (see
  // run-bench-windows.mjs) maps the SF_BENCH_PLY_DIR/<name>.lodge to
  // /lodges/<name>/ for browser fetches.
  const manifestRes = await fetch(`/lodges/${sceneName}/manifest.json`);
  if (!manifestRes.ok) {
    throw new Error(`lodge manifest load failed: ${sceneName} (${manifestRes.status})`);
  }
  const manifest = parseLodgeManifest(await manifestRes.text());

  const aspect = VIEWPORT[0] / VIEWPORT[1];
  const cams = buildCameras(manifest, aspect);

  const breakdown: LodgeLevelResult[] = [];
  let baselineFps: number | null = null;

  // Walk from coarsest → finest so the device sees the smallest
  // pipelines first (fastest to recover when L0 doesn't fit).
  for (let li = manifest.levels.length - 1; li >= 0; li--) {
    const lvl = manifest.levels[li]!;
    if (lvl.splatCount > capacityCap) {
      // Skip levels too large for this device's VRAM budget. Capture an
      // explicit "OOM" record so the report shows what was skipped.
      breakdown.push({
        level: li,
        splatCount: lvl.splatCount,
        perFrameMs: Number.POSITIVE_INFINITY,
        framesPerSecond: 0,
        viewFps: { front: 0, orbit45: 0, orbit90: 0 },
        uploadMs: -1,
        decodedBytes: lvl.splatCount * BYTES_PER_DECODED_SPLAT,
      });
      continue;
    }

    // Fresh pipeline per level — the pipeline's `capacity` is fixed at
    // construction time and we don't want a 100M-capacity arena when
    // we're only rendering 100k splats at L_N (cheaper per-frame because
    // the compute-shader dispatch count uses `splatCount`, not
    // capacity).
    //
    // Worst-case padding per chunk: 3 splats (so the per-chunk
    // decoded-splat offset lands on a 256-byte boundary — see
    // alignDecodedSplats in chunk-loader.ts). Bump the capacity by
    // `numChunks * 3` so the final padded count still fits.
    const numChunks = lvl.chunks.length;
    const pipeline = new ComputeDecodePipeline({
      device,
      capacity: lvl.splatCount + numChunks * 3,
    });

    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: `/lodges/${sceneName}/`,
      pipeline,
    });
    const lod = new LodgeLODPipeline(loader, {});

    const uploadStart = performance.now();
    await lod.warmLevel(li);
    await device.queue.onSubmittedWorkDone();
    const uploadMs = performance.now() - uploadStart;

    // Run each viewpoint once, capture (perFrameMs, fps).
    const viewResults: Array<{ fps: number; perFrameMs: number }> = [];
    for (const { cam } of cams) {
      viewResults.push(await runOneView(device, pipeline, cam, iterations));
    }
    const fps = viewResults.map((v) => v.fps);
    const perFrameMs = viewResults.map((v) => v.perFrameMs);

    const result: LodgeLevelResult = {
      level: li,
      splatCount: lvl.splatCount,
      perFrameMs: median(perFrameMs),
      framesPerSecond: median(fps),
      viewFps: {
        front: fps[0] ?? 0,
        orbit45: fps[1] ?? 0,
        orbit90: fps[2] ?? 0,
      },
      uploadMs,
      decodedBytes: pipeline.splatCount * BYTES_PER_DECODED_SPLAT,
    };
    breakdown.push(result);

    if (li === 0) baselineFps = result.framesPerSecond;

    pipeline.destroy();
  }

  breakdown.sort((a, b) => a.level - b.level);

  // Pick the level the on-camera heuristic would choose for the centre
  // camera. That's the "production" fps the LOD path actually delivers.
  const ctrCam = cams[0]!.cam.position; // 'front' = camera at scene-center +radius
  let selectedLevel = 0;
  for (let l = 0; l < manifest.levels.length; l++) {
    const lvl = manifest.levels[l]!;
    // Distance from front camera to scene centre.
    const c = sceneBboxCenter(manifest);
    const d = Math.hypot(ctrCam[0] - c[0], ctrCam[1] - c[1], ctrCam[2] - c[2]);
    if (d >= lvl.depthThreshold) selectedLevel = l;
  }
  const selectedEntry = breakdown.find((b) => b.level === selectedLevel);
  const selectedFps = selectedEntry?.framesPerSecond ?? Number.NaN;
  const speedup = baselineFps !== null && Number.isFinite(baselineFps) && baselineFps > 0
    ? selectedFps / baselineFps
    : null;

  return {
    scene: sceneName,
    manifestSource: manifest.source,
    originalSplatCount: manifest.originalSplatCount,
    numLevels: manifest.levels.length,
    viewport: VIEWPORT,
    iterations,
    levelBreakdown: breakdown,
    selectedLevel,
    selectedFps,
    baselineFps,
    speedup,
  };
}

/**
 * Entry point — the harness page (`real-scene-lodge.html`) calls this
 * after the WebGPU device is ready. Fetches the scene list from
 * `/lodges/index.json` (emitted by the bench server) and benches each
 * one.
 */
export async function main(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    (window as unknown as { __benchRealSceneLodge: unknown }).__benchRealSceneLodge = { error: 'no_webgpu' };
    return;
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    (window as unknown as { __benchRealSceneLodge: unknown }).__benchRealSceneLodge = { error: 'no_adapter' };
    return;
  }
  // 119M Sweet Corals at 64 B/splat = 7.6 GB. The adapter's
  // `maxStorageBufferBindingSize` caps individual buffers — typically
  // 2 GB on D3D12. We can't fit all of L0 in one buffer; instead we
  // bench each level's chunks as a single contiguous splat array up to
  // ~30 M (max single-buffer size at 64 B/splat × 0.5 ≈ 16 M-floor).
  //
  // For the headline number we report a coarser level that fits in
  // ≤ 2 GB / 64 B = ~32 M splats. The bench harness skips levels above
  // `capacityCap`.
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
    const idx = await fetch('/lodges/index.json');
    if (!idx.ok) throw new Error(`lodges/index.json: ${idx.status}`);
    scenes = (await idx.json()) as string[];
  } catch (err) {
    (window as unknown as { __benchRealSceneLodge: unknown }).__benchRealSceneLodge = {
      error: 'no_lodges',
      message: String((err as Error)?.message ?? err),
    };
    return;
  }

  const capacityCap = Math.floor(want.maxBufferSize / BYTES_PER_DECODED_SPLAT) - 1;

  const results: LodgeSceneResult[] = [];
  for (const name of scenes) {
    try {
      log(`lodge bench: starting ${name}`);
      const r = await runLodgeBench(device, name, 20, capacityCap);
      log(`lodge bench: ${name} n=${r.originalSplatCount} selectedLevel=${r.selectedLevel} selectedFps=${r.selectedFps.toFixed(2)} speedup=${r.speedup?.toFixed(2) ?? 'n/a'}`);
      results.push(r);
    } catch (err) {
      (window as unknown as { __benchRealSceneLodge: unknown }).__benchRealSceneLodge = {
        error: `lodge_bench_failed_${name}`,
        message: String((err as Error)?.message ?? err),
        results,
      };
      return;
    }
  }

  (window as unknown as { __benchRealSceneLodge: unknown }).__benchRealSceneLodge = {
    results,
    sizes: { bytes_per_decoded_splat: BYTES_PER_DECODED_SPLAT, floats_per_instance: FLOATS_PER_INSTANCE },
    adapter: adapterInfo,
    limits: want,
    timestamp: new Date().toISOString(),
  };
}
