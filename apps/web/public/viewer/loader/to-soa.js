// SPDX-License-Identifier: Apache-2.0
/**
 * Adapter: {@link SplatScene} → SoA chunk + {@link ChunkDescriptor}.
 *
 * The WebGPU compute-decode pipeline in `packages/viewer/src/webgpu/`
 * consumes splats attribute-by-attribute via the canonical SoA layout
 * (POSITION | ROTATION | SCALE | OPACITY | COLOR_DC = 56 B per splat) that
 * `progressive/uploader.ts` emits. This module converts a `SplatScene`
 * (produced by the format loaders in this directory) into that same byte
 * layout so the preview shell can do:
 *
 *     const scene = await loadFromFiles(files);
 *     const { descriptor, bytes } = splatSceneToSoaChunk(scene, 'scene:bag');
 *     renderer.uploadChunk(descriptor, bytes);
 *
 * Encoding semantics match the progressive uploader exactly so SoA chunks
 * built here are bit-identical to those produced from raw PLY records:
 *   - scales : log → exp (we apply `Math.exp` because SplatScene carries
 *              log-space scales and the WebGPU decode kernel expects LINEAR).
 *   - rotations: re-normalize defensively.
 *   - colorDC slot is RAW `f_dc` (`scene.dcRaw`), NOT the baked color — the
 *     decode shader applies the `0.5 + SH_C0 * f_dc` bake itself.
 */
import { FLOAT_COMPONENT, } from '../manifest.js';
import { SOA_BYTES_PER_SPLAT } from '../progressive/uploader.js';
import { shRestCoefCount } from './splat-scene.js';
/** Per-splat byte stride of each SoA slice. Mirrors `progressive/uploader.ts`. */
const STRIDE_POS = 12;
const STRIDE_ROT = 16;
const STRIDE_SCL = 12;
const STRIDE_OP = 4;
const STRIDE_DC = 12;
/**
 * Pack a {@link SplatScene} into a single SoA chunk + {@link ChunkDescriptor}
 * the {@link Renderer.uploadChunk} interface accepts. Total byte cost is
 * `scene.count * 56`.
 */
export function splatSceneToSoaChunk(scene, uri = 'scene:in-memory') {
    const N = scene.count;
    const posOff = 0;
    const rotOff = posOff + N * STRIDE_POS;
    const sclOff = rotOff + N * STRIDE_ROT;
    const opOff = sclOff + N * STRIDE_SCL;
    const dcOff = opOff + N * STRIDE_OP;
    const dcEnd = dcOff + N * STRIDE_DC;
    // Phase 2b: optional SH-rest blob appended after the 56-B canonical slice.
    // The pre-Phase-2b SOA_BYTES_PER_SPLAT == 56 invariant holds only when
    // there is no SH-rest; otherwise the total grows by coefCount * 3 * 4.
    const shDegree = scene.shRest && scene.shDegree && scene.shDegree > 0 ? scene.shDegree : 0;
    const coefCount = shRestCoefCount(shDegree); // 0/3/8/15
    const shStride = coefCount * 3 * 4; // bytes per splat
    const shOff = dcEnd;
    const shLen = N * shStride;
    const total = dcEnd + shLen;
    // The 56-byte stride sanity-check only applies when no SH-rest is present;
    // with SH-rest, the trailing blob extends the total beyond N * 56.
    if (shDegree === 0 && dcEnd !== N * SOA_BYTES_PER_SPLAT) {
        throw new Error(`to-soa: stride bookkeeping mismatch (computed ${dcEnd} B, expected ${N * SOA_BYTES_PER_SPLAT} B)`);
    }
    const out = new Uint8Array(total);
    const dv = new DataView(out.buffer);
    // SH-rest blob (if present): straight passthrough of the canonical
    // splat-major / k-major / channel-minor float32 layout that ply.ts /
    // sog-container.ts / sf-glb.ts already produce.
    if (shDegree > 0 && scene.shRest) {
        // Use Float32Array view for a single block copy (fast path) instead of
        // a per-float DataView loop.
        const sh32 = new Float32Array(out.buffer, shOff, N * coefCount * 3);
        sh32.set(scene.shRest.subarray(0, N * coefCount * 3));
    }
    for (let i = 0; i < N; i++) {
        // Position.
        dv.setFloat32(posOff + i * STRIDE_POS + 0, scene.positions[i * 3 + 0], true);
        dv.setFloat32(posOff + i * STRIDE_POS + 4, scene.positions[i * 3 + 1], true);
        dv.setFloat32(posOff + i * STRIDE_POS + 8, scene.positions[i * 3 + 2], true);
        // Rotation (XYZW, re-normalize).
        const rx = scene.rotations[i * 4 + 0];
        const ry = scene.rotations[i * 4 + 1];
        const rz = scene.rotations[i * 4 + 2];
        const rw = scene.rotations[i * 4 + 3];
        const rnorm = Math.hypot(rx, ry, rz, rw) || 1.0;
        const inv = 1.0 / rnorm;
        dv.setFloat32(rotOff + i * STRIDE_ROT + 0, rx * inv, true);
        dv.setFloat32(rotOff + i * STRIDE_ROT + 4, ry * inv, true);
        dv.setFloat32(rotOff + i * STRIDE_ROT + 8, rz * inv, true);
        dv.setFloat32(rotOff + i * STRIDE_ROT + 12, rw * inv, true);
        // Scale (log → linear) — matches the progressive uploader's `exp(scale_n)`.
        dv.setFloat32(sclOff + i * STRIDE_SCL + 0, Math.exp(scene.scales[i * 3 + 0]), true);
        dv.setFloat32(sclOff + i * STRIDE_SCL + 4, Math.exp(scene.scales[i * 3 + 1]), true);
        dv.setFloat32(sclOff + i * STRIDE_SCL + 8, Math.exp(scene.scales[i * 3 + 2]), true);
        // Opacity (already linear [0,1] in the SplatScene contract).
        dv.setFloat32(opOff + i * STRIDE_OP, scene.opacity[i], true);
        // Color DC: write RAW f_dc — the decode kernel re-bakes via `0.5 + SH_C0 * f`.
        dv.setFloat32(dcOff + i * STRIDE_DC + 0, scene.dcRaw[i * 3 + 0], true);
        dv.setFloat32(dcOff + i * STRIDE_DC + 4, scene.dcRaw[i * 3 + 1], true);
        dv.setFloat32(dcOff + i * STRIDE_DC + 8, scene.dcRaw[i * 3 + 2], true);
    }
    const bbox = {
        min: [scene.bbox.min[0], scene.bbox.min[1], scene.bbox.min[2]],
        max: [scene.bbox.max[0], scene.bbox.max[1], scene.bbox.max[2]],
    };
    const layout = {
        positions: {
            byteOffset: posOff,
            byteLength: N * STRIDE_POS,
            componentType: FLOAT_COMPONENT,
            min: bbox.min,
            max: bbox.max,
        },
        rotations: {
            byteOffset: rotOff,
            byteLength: N * STRIDE_ROT,
            componentType: FLOAT_COMPONENT,
        },
        scales: {
            byteOffset: sclOff,
            byteLength: N * STRIDE_SCL,
            componentType: FLOAT_COMPONENT,
        },
        opacities: {
            byteOffset: opOff,
            byteLength: N * STRIDE_OP,
            componentType: FLOAT_COMPONENT,
        },
        colorDC: {
            byteOffset: dcOff,
            byteLength: N * STRIDE_DC,
            componentType: FLOAT_COMPONENT,
        },
        ...(shDegree > 0
            ? {
                shRest: {
                    byteOffset: shOff,
                    byteLength: shLen,
                    componentType: FLOAT_COMPONENT,
                },
                shDegree,
            }
            : {}),
    };
    const descriptor = {
        uri,
        byteOffset: 0,
        byteLength: total,
        splatCount: N,
        bbox,
        lod: 0,
        checksum: '',
        loadPriority: 0,
        attributeLayout: layout,
    };
    return { descriptor, bytes: out };
}
//# sourceMappingURL=to-soa.js.map