// SPDX-License-Identifier: Apache-2.0
/**
 * Streaming-tile viewer bench harness.
 *
 * Loaded by `bench/streaming-tileset.html` (served from the bench runner).
 * Drives `StreamingTileset` against the committed `geospatial-sample`
 * fixture and measures:
 *
 *   1. Cold-start: time from `loadTilesetJson` resolve to first
 *      `requestFrame` returning a non-empty render set.
 *   2. Sustained FPS over a 60-frame orbit.
 *   3. Peak resident bytes (against the 512 MB budget).
 *
 * Results land on `window.__streamingBench` so the runner can scrape them.
 *
 * The bench is GPU-light by design: we exercise the streamer + selector
 * end-to-end and call the WebGPU renderer's `uploadChunk` + `renderFrame`
 * paths through a real `<canvas>`. The fixture is intentionally tiny (4
 * tiles, ~450 splats total) so we measure pipeline overhead, not splat
 * count.
 */

import { SplatForgeViewer } from '../src/viewer.js';
import { orbitFrames, orbitPose } from '../src/camera.js';
import type { CameraPose } from '../src/camera.js';

/** Fixture URL relative to the harness page. */
const TILESET_URL = '/fixtures/geospatial-sample/tileset.json';

interface BenchResult {
  coldStartMs: number;
  frames: number;
  totalMs: number;
  fps: number;
  peakResidentBytes: number;
  evictions: number;
  cacheHits: number;
  cacheMisses: number;
  residentTiles: number;
}

/** Build an orbit camera pose that frames the sample's bbox. */
function poseForYaw(yaw: number, aspect: number): CameraPose {
  // Sample bbox from README.json: roughly ±3.7 on x, ±3.5 on y, ±0.79 on z.
  const bbox = { min: [-3.7, -3.5, -0.8] as [number, number, number],
                 max: [ 3.7,  3.5,  0.8] as [number, number, number] };
  return orbitPose(bbox, yaw, aspect);
}

async function runBench(canvas: HTMLCanvasElement): Promise<BenchResult> {
  const viewer = new SplatForgeViewer({
    canvas,
    src: TILESET_URL,
    renderer: 'webgpu',
    deterministic: true,
    seed: 0xc0ffee,
    useComputeDecode: true,
  });

  const t0 = performance.now();
  await viewer.loadTileset(TILESET_URL, {
    maximumScreenSpaceError: 16,
    maxBytes: 512 * 1024 * 1024,
  });
  const coldStartMs = performance.now() - t0;

  // 60-frame sustained orbit.
  const FRAMES = 60;
  const aspect = canvas.width / canvas.height;
  const yaws = orbitFrames(FRAMES);
  let peak = 0;
  let evictions = 0;
  const start = performance.now();
  for (const yaw of yaws) {
    const pose = poseForYaw(yaw, aspect);
    const report = await viewer.streamingRenderFrame(pose);
    if (report) {
      if (report.stats.residentBytes > peak) peak = report.stats.residentBytes;
      evictions = report.stats.evictions;
    }
  }
  const totalMs = performance.now() - start;
  const fps = (FRAMES * 1000) / totalMs;
  const stats = viewer.streamingTileset?.streamer.stats();

  viewer.dispose();

  return {
    coldStartMs,
    frames: FRAMES,
    totalMs,
    fps,
    peakResidentBytes: peak,
    evictions,
    cacheHits: stats?.cacheHits ?? 0,
    cacheMisses: stats?.cacheMisses ?? 0,
    residentTiles: stats?.residentTiles ?? 0,
  };
}

export async function main(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    (window as unknown as { __streamingBench: unknown }).__streamingBench = { error: 'no_webgpu' };
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  document.body.appendChild(canvas);

  try {
    const result = await runBench(canvas);
    (window as unknown as { __streamingBench: unknown }).__streamingBench = {
      result,
      timestamp: new Date().toISOString(),
    };
  } catch (err) {
    (window as unknown as { __streamingBench: unknown }).__streamingBench = {
      error: 'bench_failed',
      message: String((err as Error).message ?? err),
    };
  }
}
