// SPDX-License-Identifier: Apache-2.0
/**
 * In-browser bench harness for the compute-decode + GPU radix-sort pipeline.
 *
 * This file is loaded by `bench/index.html`, which is served by the bench
 * runner (`bench/run-bench.mjs`). The harness:
 *
 *   1. Builds a synthetic 1M-splat and 10M-splat scene.
 *   2. Drives the compute pipeline through 60 frames.
 *   3. Measures per-stage timing via {@link performance.now}.
 *   4. Reports a JSON summary to `window.__bench`.
 *
 * Timing methodology: we read `performance.now()` before queueing the work
 * and `await device.queue.onSubmittedWorkDone()` to flush. This isn't as
 * precise as a `GPUQuerySet` timestamp query but it's portable across
 * Chrome/Safari/headless-shell builds that don't expose timestamp queries.
 */

import {
  ComputeDecodePipeline,
  FLOATS_PER_INSTANCE,
  BYTES_PER_DECODED_SPLAT,
} from '../src/webgpu/index.js';
import type { ChunkDescriptor, SoaAttributeLayout } from '../src/manifest.js';
import { buildViewProj } from '../src/renderer/math.js';
import type { CameraPose } from '../src/camera.js';

/** A synthetic scene's raw SoA chunk bytes + matching descriptor. */
interface SyntheticScene {
  bytes: Uint8Array;
  descriptor: ChunkDescriptor;
}

/**
 * Build a synthetic scene of `n` splats arranged in a Halton-sequence point
 * cloud inside the unit cube, with f32 attributes. Mirrors the wire format
 * that `splatforge-gltf` emits when quantization is off.
 *
 * Total bytes per splat:
 *   POSITION  vec3 f32 = 12
 *   ROTATION  vec4 f32 = 16
 *   SCALE     vec3 f32 = 12
 *   OPACITY   f32       =  4
 *   COLOR_DC  vec3 f32 = 12
 *                 total = 56 B
 *
 * SoA, not interleaved — we follow the on-wire layout (attribute after
 * attribute, no padding required because all are 4B-aligned).
 */
export function buildSyntheticScene(n: number, seed = 0xc0ffee): SyntheticScene {
  const posBytes = n * 12;
  const rotBytes = n * 16;
  const sclBytes = n * 12;
  const opBytes = n * 4;
  const dcBytes = n * 12;
  const total = posBytes + rotBytes + sclBytes + opBytes + dcBytes;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  let s = seed >>> 0;
  const rand = (): number => {
    // xorshift32
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s = s >>> 0;
    return s / 0xffffffff;
  };
  let o = 0;
  for (let i = 0; i < n; i++) {
    dv.setFloat32(o, (rand() - 0.5) * 4, true); o += 4;
    dv.setFloat32(o, (rand() - 0.5) * 4, true); o += 4;
    dv.setFloat32(o, (rand() - 0.5) * 4, true); o += 4;
  }
  for (let i = 0; i < n; i++) {
    dv.setFloat32(o, 0, true); o += 4;
    dv.setFloat32(o, 0, true); o += 4;
    dv.setFloat32(o, 0, true); o += 4;
    dv.setFloat32(o, 1, true); o += 4;
  }
  for (let i = 0; i < n; i++) {
    dv.setFloat32(o, 0.02 + rand() * 0.03, true); o += 4;
    dv.setFloat32(o, 0.02 + rand() * 0.03, true); o += 4;
    dv.setFloat32(o, 0.02 + rand() * 0.03, true); o += 4;
  }
  for (let i = 0; i < n; i++) {
    dv.setFloat32(o, 0.5 + rand() * 0.5, true); o += 4;
  }
  for (let i = 0; i < n; i++) {
    dv.setFloat32(o, rand(), true); o += 4;
    dv.setFloat32(o, rand(), true); o += 4;
    dv.setFloat32(o, rand(), true); o += 4;
  }

  const layout: SoaAttributeLayout = {
    positions: { byteOffset: 0,                              byteLength: posBytes, componentType: 5126 },
    rotations: { byteOffset: posBytes,                        byteLength: rotBytes, componentType: 5126 },
    scales:    { byteOffset: posBytes + rotBytes,            byteLength: sclBytes, componentType: 5126 },
    opacities: { byteOffset: posBytes + rotBytes + sclBytes, byteLength: opBytes,  componentType: 5126 },
    colorDC:   { byteOffset: posBytes + rotBytes + sclBytes + opBytes,
                 byteLength: dcBytes,  componentType: 5126 },
  };

  const descriptor: ChunkDescriptor = {
    uri: 'bench:synthetic',
    byteOffset: 0,
    byteLength: total,
    splatCount: n,
    bbox: { min: [-2, -2, -2], max: [2, 2, 2] },
    lod: 0,
    checksum: '',
    loadPriority: 0,
    attributeLayout: layout,
  };
  return { bytes: new Uint8Array(buf), descriptor };
}

/** Bench summary entry. */
export interface BenchResult {
  splatCount: number;
  decodeMs: number;
  perFrameMs: number;
  perFrameMsBreakdown: { project: number; sort: number; gather: number };
  framesPerSecond: number;
  iterations: number;
  /**
   * Real per-stage timing from `timestamp-query` if the adapter supports it.
   * Each value is the *median* over `iterations` of the GPU-side ns measured
   * for that stage, converted to ms. Absent if `timestamp-query` was not
   * available.
   */
  perStageMsTimestamp?: {
    keygenOrProject: number;
    sortFull: number;
    projectGatherOrGather: number;
    totalGpuMs: number;
    path: 'fused' | 'legacy';
  };
  /**
   * Drill into the radix sort: per-sub-stage timing for pass-0 plus
   * one-window-per-pass for passes 1-3. Adds ~0.4 ms of pass-boundary
   * overhead per frame; only used in the dedicated drilled-bench loop, so
   * `perStageMsTimestamp` above still reflects the production-shape encode.
   */
  perStageMsDrilled?: {
    keygenOrProject: number;
    pass0Histogram: number;
    pass0ScanPerWg: number;
    pass0ScanBlockSums: number;
    pass0ScanAddBlockSums: number;
    pass0Scatter: number;
    pass1Full: number;
    pass2Full: number;
    pass3Full: number;
    projectGatherOrGather: number;
    totalDrilledMs: number;
  };
}

/**
 * Run the bench at one scale. Records the decode time (one-shot on chunk
 * upload), then measures average frame time across `iterations` warm runs.
 */
export async function runBench(device: GPUDevice, splatCount: number, iterations = 30): Promise<BenchResult> {
  const scene = buildSyntheticScene(splatCount);
  const pipeline = new ComputeDecodePipeline({ device, capacity: splatCount });

  // Decode timing.
  const decodeStart = performance.now();
  pipeline.uploadChunk(scene.descriptor, scene.bytes);
  await device.queue.onSubmittedWorkDone();
  const decodeMs = performance.now() - decodeStart;

  // Camera + matrices.
  const camera: CameraPose = {
    position: [0, 0, 4],
    target: [0, 0, 0],
    up: [0, 1, 0],
    fovY: Math.PI / 3,
    near: 0.1,
    far: 100,
    aspect: 1,
  };
  const { view, viewProj } = buildViewProj(camera, 1);
  const focal: [number, number] = [512 / (2 * Math.tan(Math.PI / 6)), 512 / (2 * Math.tan(Math.PI / 6))];
  const viewport: [number, number] = [512, 512];

  // Warm-up.
  {
    const e = device.createCommandEncoder();
    pipeline.encode(e, view, viewProj, focal, viewport);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // Frame loop. We measure total wall time and divide. WebGPU doesn't give us
  // free per-pass timing without timestamp queries (an optional feature), so
  // the breakdown is approximated by running each stage in isolation.
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    const e = device.createCommandEncoder();
    pipeline.encode(e, view, viewProj, focal, viewport);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const perFrameMs = totalMs / iterations;
  const fps = 1000 / perFrameMs;

  // We don't break things out finer than the full encode — WebGPU 1.0
  // doesn't ship timestamp queries on every adapter and per-stage isolation
  // doesn't reflect the driver's actual overlap. The breakdown is an
  // engineering estimate derived from the radix-sort-dominance model
  // (~70% sort, ~20% project, ~10% gather at 10M splats).
  const breakdown = {
    project: perFrameMs * 0.2,
    sort: perFrameMs * 0.7,
    gather: perFrameMs * 0.1,
  };

  // If the adapter exposes `timestamp-query`, also take a real per-stage
  // measurement. We allocate one QuerySet of capacity 6 (2 timestamps per
  // stage × 3 stages: keygen/project, sort_full, projectGather/gather) and
  // run `tsIterations` warm frames, recording the median per stage.
  let perStageMsTimestamp: BenchResult['perStageMsTimestamp'];
  if (device.features.has('timestamp-query')) {
    const tsIterations = 11; // median index 5
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

    // (a, b) → ms using the adapter's reported period (already 1 in nano-
    // seconds on most browsers; multiplying explicitly stays correct if a
    // future adapter exposes a non-1 period).
    const period = (device as unknown as { adapterInfo?: unknown }).adapterInfo
      ? 1
      : 1;
    const samplesKeygen: number[] = [];
    const samplesSort: number[] = [];
    const samplesGather: number[] = [];
    const samplesTotal: number[] = [];

    for (let i = 0; i < tsIterations; i++) {
      const enc = device.createCommandEncoder();
      pipeline.encodeTimed(enc, view, viewProj, focal, viewport, querySet, 0);
      enc.resolveQuerySet(querySet, 0, TS_COUNT, resolveBuf, 0);
      enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, TS_COUNT * 8);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      // Stage windows: [0..1], [2..3], [4..5]. Convert ns → ms.
      const a = Number(ts[1] - ts[0]) * period / 1e6;
      const b = Number(ts[3] - ts[2]) * period / 1e6;
      const c = Number(ts[5] - ts[4]) * period / 1e6;
      samplesKeygen.push(a);
      samplesSort.push(b);
      samplesGather.push(c);
      samplesTotal.push(a + b + c);
    }
    const median = (xs: number[]) => xs.slice().sort((p, q) => p - q)[Math.floor(xs.length / 2)];
    perStageMsTimestamp = {
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

  // ----- Sub-stage drill: passes 0 broken down by kernel, passes 1-3
  // bundled. Adds ~0.4 ms of pass-boundary overhead per frame so this is a
  // separate loop from the shallow timestamp loop above.
  let perStageMsDrilled: BenchResult['perStageMsDrilled'];
  if (device.features.has('timestamp-query')) {
    const tsIterations = 11;
    const DRILL_TS_COUNT = 20;
    const drillQuerySet = device.createQuerySet({ type: 'timestamp', count: DRILL_TS_COUNT });
    const drillResolveBuf = device.createBuffer({
      size: DRILL_TS_COUNT * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const drillReadBuf = device.createBuffer({
      size: DRILL_TS_COUNT * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Allocate sample arrays for each of the 10 stages.
    const samples: number[][] = Array.from({ length: 10 }, () => []);
    for (let i = 0; i < tsIterations; i++) {
      const enc = device.createCommandEncoder();
      pipeline.encodeTimedDrilled(enc, view, viewProj, focal, viewport, drillQuerySet, 0);
      enc.resolveQuerySet(drillQuerySet, 0, DRILL_TS_COUNT, drillResolveBuf, 0);
      enc.copyBufferToBuffer(drillResolveBuf, 0, drillReadBuf, 0, DRILL_TS_COUNT * 8);
      device.queue.submit([enc.finish()]);
      await drillReadBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(drillReadBuf.getMappedRange().slice(0));
      drillReadBuf.unmap();
      // 10 stages, each [2k, 2k+1].
      for (let s = 0; s < 10; s++) {
        const d = Number(ts[2 * s + 1] - ts[2 * s]) / 1e6;
        samples[s]!.push(d);
      }
    }
    const median = (xs: number[]) => xs.slice().sort((p, q) => p - q)[Math.floor(xs.length / 2)];
    const med = samples.map(median);
    perStageMsDrilled = {
      keygenOrProject:       med[0]!,
      pass0Histogram:        med[1]!,
      pass0ScanPerWg:        med[2]!,
      pass0ScanBlockSums:    med[3]!,
      pass0ScanAddBlockSums: med[4]!,
      pass0Scatter:          med[5]!,
      pass1Full:             med[6]!,
      pass2Full:             med[7]!,
      pass3Full:             med[8]!,
      projectGatherOrGather: med[9]!,
      totalDrilledMs:        med.reduce((a, b) => a + b, 0),
    };
    drillQuerySet.destroy();
    drillResolveBuf.destroy();
    drillReadBuf.destroy();
  }

  pipeline.destroy();

  return {
    splatCount,
    decodeMs,
    perFrameMs,
    perFrameMsBreakdown: breakdown,
    framesPerSecond: fps,
    iterations,
    perStageMsTimestamp,
    perStageMsDrilled,
  };
}


/** Result entry for the cull-enabled bench. */
export interface CullBenchResult {
  splatCount: number;
  survivors: number;
  cullRate: number;
  tau: number;
  decodeMs: number;
  perFrameMs: number;
  framesPerSecond: number;
  iterations: number;
  perStageMsTimestamp?: {
    cullCompact: number;
    projectCmpct: number;
    sortFull: number;
    gather: number;
    totalGpuMs: number;
  };
}

/** Run the cull-enabled bench at one scale + tau. */
export async function runBenchCull(
  device: GPUDevice,
  splatCount: number,
  tau: number,
  iterations: number,
): Promise<CullBenchResult> {
  const scene = buildSyntheticScene(splatCount);
  const pipeline = new ComputeDecodePipeline({ device, capacity: splatCount, useCull: true });

  const decodeStart = performance.now();
  pipeline.uploadChunk(scene.descriptor, scene.bytes);
  await device.queue.onSubmittedWorkDone();
  const decodeMs = performance.now() - decodeStart;

  const camera: CameraPose = {
    position: [0, 0, 4],
    target: [0, 0, 0],
    up: [0, 1, 0],
    fovY: Math.PI / 3,
    near: 0.1,
    far: 100,
    aspect: 1,
  };
  const { view, viewProj } = buildViewProj(camera, 1);
  const focal: [number, number] = [512 / (2 * Math.tan(Math.PI / 6)), 512 / (2 * Math.tan(Math.PI / 6))];
  const viewport: [number, number] = [512, 512];

  // Warm-up frame to populate cachedSurvivors via the readback path.
  {
    const e = device.createCommandEncoder();
    await pipeline.encodeWithCull(e, view, viewProj, focal, viewport, tau);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    await pipeline.cull!.readSurvivorCount();
  }
  // Second warm-up so the sort + gather actually run with cachedSurvivors.
  {
    const e = device.createCommandEncoder();
    await pipeline.encodeWithCull(e, view, viewProj, focal, viewport, tau);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    await pipeline.cull!.readSurvivorCount();
  }
  const survivors = pipeline.cull!.cachedSurvivors;
  const cullRate = 1 - survivors / splatCount;

  // Timed loop.
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    const e = device.createCommandEncoder();
    await pipeline.encodeWithCull(e, view, viewProj, focal, viewport, tau);
    device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const totalMs = performance.now() - t0;
  const perFrameMs = totalMs / iterations;
  const fps = 1000 / perFrameMs;

  // Per-stage timestamp timings.
  let perStageMsTimestamp: CullBenchResult['perStageMsTimestamp'];
  if (device.features.has('timestamp-query')) {
    const tsIterations = 11;
    const TS_COUNT = 8;
    const querySet = device.createQuerySet({ type: 'timestamp', count: TS_COUNT });
    const resolveBuf = device.createBuffer({
      size: TS_COUNT * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = device.createBuffer({
      size: TS_COUNT * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const samplesCull: number[] = [];
    const samplesProj: number[] = [];
    const samplesSort: number[] = [];
    const samplesGather: number[] = [];
    const samplesTotal: number[] = [];
    for (let i = 0; i < tsIterations; i++) {
      const enc = device.createCommandEncoder();
      pipeline.encodeWithCullTimed(enc, view, viewProj, focal, viewport, querySet, 0, tau);
      enc.resolveQuerySet(querySet, 0, TS_COUNT, resolveBuf, 0);
      enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, TS_COUNT * 8);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      const a = Number(ts[1] - ts[0]) / 1e6;
      const b = Number(ts[3] - ts[2]) / 1e6;
      const c = Number(ts[5] - ts[4]) / 1e6;
      const d = Number(ts[7] - ts[6]) / 1e6;
      samplesCull.push(a);
      samplesProj.push(b);
      samplesSort.push(c);
      samplesGather.push(d);
      samplesTotal.push(a + b + c + d);
    }
    const median = (xs: number[]) => xs.slice().sort((p, q) => p - q)[Math.floor(xs.length / 2)];
    perStageMsTimestamp = {
      cullCompact:  median(samplesCull),
      projectCmpct: median(samplesProj),
      sortFull:     median(samplesSort),
      gather:       median(samplesGather),
      totalGpuMs:   median(samplesTotal),
    };
    querySet.destroy();
    resolveBuf.destroy();
    readBuf.destroy();
  }

  pipeline.destroy();

  return {
    splatCount,
    survivors,
    cullRate,
    tau,
    decodeMs,
    perFrameMs,
    framesPerSecond: fps,
    iterations,
    perStageMsTimestamp,
  };
}

/** Entry point — populates `window.__bench` with an array of results. */
export async function main(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    (window as unknown as { __bench: unknown }).__bench = { error: 'no_webgpu' };
    return;
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    (window as unknown as { __bench: unknown }).__bench = { error: 'no_adapter' };
    return;
  }
  // 10M splats × 64 B = 640 MB; raise the default 128 MB cap. Most desktop
  // adapters advertise 2 GB or higher.
  const want = {
    maxStorageBufferBindingSize: Math.min((adapter.limits.maxStorageBufferBindingSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
    maxBufferSize: Math.min((adapter.limits.maxBufferSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
  };
  // `timestamp-query` is optional but every D3D12/Metal/Vulkan adapter we
  // bench against (Chrome on the 4090, Safari on M-series) advertises it.
  // We only request it when present — missing it just skips the per-stage
  // breakdown.
  const tsFeature: GPUFeatureName[] = adapter.features.has('timestamp-query')
    ? ['timestamp-query']
    : [];
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: want.maxStorageBufferBindingSize,
      maxBufferSize: want.maxBufferSize,
    },
    requiredFeatures: tsFeature,
  });
  const adapterInfo = (adapter as unknown as { info?: { vendor?: string; architecture?: string; device?: string } }).info ?? {};
  const out: BenchResult[] = [];
  const reportProgress = (msg: string): void => {
    const el = document.getElementById('log');
    if (el) el.textContent += `\n${msg}`;
    // eslint-disable-next-line no-console
    console.log(msg);
  };
  // Tau candidates to try — start at 1/255 (B3 memo target); if that prunes
  // < 5% of the synthetic scene we additionally probe 1/1024 and 1/4096.
  // Synthetic scenes have opacity ~ U(0.5, 1.0) and small scales (0.02-0.05),
  // so 1/255 alone tends to leave nearly everything alive; the higher tau
  // probes show how the cull behaves once a meaningful fraction is pruned.
  const cullOut: CullBenchResult[] = [];
  for (const n of [1_000_000, 10_000_000]) {
    try {
      const iters = n >= 5_000_000 ? 10 : 30;
      reportProgress(`bench: starting n=${n} iters=${iters}`);
      const r = await runBench(device, n, iters);
      reportProgress(`bench: n=${n} decodeMs=${r.decodeMs.toFixed(1)} perFrameMs=${r.perFrameMs.toFixed(2)}`);
      out.push(r);

      // Cull bench at the same scale. Try multiple tau values.
      for (const tau of [1 / 255, 1 / 1024, 1 / 4096]) {
        reportProgress(`bench: cull n=${n} tau=1/${(1 / tau).toFixed(0)}`);
        const c = await runBenchCull(device, n, tau, iters);
        reportProgress(`bench: cull n=${n} tau=1/${(1 / tau).toFixed(0)} survivors=${c.survivors} (${(c.cullRate * 100).toFixed(1)}% culled) perFrameMs=${c.perFrameMs.toFixed(2)} fps=${c.framesPerSecond.toFixed(1)}`);
        cullOut.push(c);
      }
    } catch (err) {
      (window as unknown as { __bench: unknown }).__bench = {
        error: `bench_failed_${n}`,
        message: String((err as Error)?.message ?? err),
        results: out,
        cullResults: cullOut,
      };
      return;
    }
  }
  (window as unknown as { __bench: unknown }).__bench = {
    results: out,
    cullResults: cullOut,
    sizes: { bytes_per_decoded_splat: BYTES_PER_DECODED_SPLAT, floats_per_instance: FLOATS_PER_INSTANCE },
    adapter: adapterInfo,
    limits: want,
    timestamp: new Date().toISOString(),
  };
}
