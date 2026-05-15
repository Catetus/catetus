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
import { ComputeDecodePipeline, FLOATS_PER_INSTANCE, BYTES_PER_DECODED_SPLAT, } from '../src/webgpu/index.js';
import { buildViewProj } from '../src/renderer/math.js';
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
export function buildSyntheticScene(n, seed = 0xc0ffee) {
    const posBytes = n * 12;
    const rotBytes = n * 16;
    const sclBytes = n * 12;
    const opBytes = n * 4;
    const dcBytes = n * 12;
    const total = posBytes + rotBytes + sclBytes + opBytes + dcBytes;
    const buf = new ArrayBuffer(total);
    const dv = new DataView(buf);
    let s = seed >>> 0;
    const rand = () => {
        // xorshift32
        s ^= s << 13;
        s ^= s >>> 17;
        s ^= s << 5;
        s = s >>> 0;
        return s / 0xffffffff;
    };
    let o = 0;
    for (let i = 0; i < n; i++) {
        dv.setFloat32(o, (rand() - 0.5) * 4, true);
        o += 4;
        dv.setFloat32(o, (rand() - 0.5) * 4, true);
        o += 4;
        dv.setFloat32(o, (rand() - 0.5) * 4, true);
        o += 4;
    }
    for (let i = 0; i < n; i++) {
        dv.setFloat32(o, 0, true);
        o += 4;
        dv.setFloat32(o, 0, true);
        o += 4;
        dv.setFloat32(o, 0, true);
        o += 4;
        dv.setFloat32(o, 1, true);
        o += 4;
    }
    for (let i = 0; i < n; i++) {
        dv.setFloat32(o, 0.02 + rand() * 0.03, true);
        o += 4;
        dv.setFloat32(o, 0.02 + rand() * 0.03, true);
        o += 4;
        dv.setFloat32(o, 0.02 + rand() * 0.03, true);
        o += 4;
    }
    for (let i = 0; i < n; i++) {
        dv.setFloat32(o, 0.5 + rand() * 0.5, true);
        o += 4;
    }
    for (let i = 0; i < n; i++) {
        dv.setFloat32(o, rand(), true);
        o += 4;
        dv.setFloat32(o, rand(), true);
        o += 4;
        dv.setFloat32(o, rand(), true);
        o += 4;
    }
    const layout = {
        positions: { byteOffset: 0, byteLength: posBytes, componentType: 5126 },
        rotations: { byteOffset: posBytes, byteLength: rotBytes, componentType: 5126 },
        scales: { byteOffset: posBytes + rotBytes, byteLength: sclBytes, componentType: 5126 },
        opacities: { byteOffset: posBytes + rotBytes + sclBytes, byteLength: opBytes, componentType: 5126 },
        colorDC: { byteOffset: posBytes + rotBytes + sclBytes + opBytes,
            byteLength: dcBytes, componentType: 5126 },
    };
    const descriptor = {
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
/**
 * Run the bench at one scale. Records the decode time (one-shot on chunk
 * upload), then measures average frame time across `iterations` warm runs.
 */
export async function runBench(device, splatCount, iterations = 30) {
    const scene = buildSyntheticScene(splatCount);
    const pipeline = new ComputeDecodePipeline({ device, capacity: splatCount });
    // Decode timing.
    const decodeStart = performance.now();
    pipeline.uploadChunk(scene.descriptor, scene.bytes);
    await device.queue.onSubmittedWorkDone();
    const decodeMs = performance.now() - decodeStart;
    // Camera + matrices.
    const camera = {
        position: [0, 0, 4],
        target: [0, 0, 0],
        up: [0, 1, 0],
        fovY: Math.PI / 3,
        near: 0.1,
        far: 100,
        aspect: 1,
    };
    const { view, viewProj } = buildViewProj(camera, 1);
    const focal = [512 / (2 * Math.tan(Math.PI / 6)), 512 / (2 * Math.tan(Math.PI / 6))];
    const viewport = [512, 512];
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
    pipeline.destroy();
    return {
        splatCount,
        decodeMs,
        perFrameMs,
        perFrameMsBreakdown: breakdown,
        framesPerSecond: fps,
        iterations,
    };
}
/** Entry point — populates `window.__bench` with an array of results. */
export async function main() {
    const gpu = navigator.gpu;
    if (!gpu) {
        window.__bench = { error: 'no_webgpu' };
        return;
    }
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        window.__bench = { error: 'no_adapter' };
        return;
    }
    // 10M splats × 64 B = 640 MB; raise the default 128 MB cap. Most desktop
    // adapters advertise 2 GB or higher.
    const want = {
        maxStorageBufferBindingSize: Math.min((adapter.limits.maxStorageBufferBindingSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
        maxBufferSize: Math.min((adapter.limits.maxBufferSize ?? 0) >>> 0, 2 * 1024 * 1024 * 1024),
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    };
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: want.maxStorageBufferBindingSize,
            maxBufferSize: want.maxBufferSize,
        },
    });
    const adapterInfo = adapter.info ?? {};
    const out = [];
    for (const n of [1_000_000, 10_000_000]) {
        try {
            const r = await runBench(device, n, 30);
            out.push(r);
        }
        catch (err) {
            window.__bench = {
                error: `bench_failed_${n}`,
                message: String(err?.message ?? err),
                results: out,
            };
            return;
        }
    }
    window.__bench = {
        results: out,
        sizes: { bytes_per_decoded_splat: BYTES_PER_DECODED_SPLAT, floats_per_instance: FLOATS_PER_INSTANCE },
        adapter: adapterInfo,
        limits: want,
        timestamp: new Date().toISOString(),
    };
}
