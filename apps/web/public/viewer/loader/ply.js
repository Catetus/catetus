// SPDX-License-Identifier: Apache-2.0
/**
 * Inria 3DGS PLY loader.
 *
 * Accepts binary little-endian PLY whose `vertex` element carries the
 * canonical 3DGS properties:
 *
 *   x, y, z, opacity,
 *   scale_0, scale_1, scale_2,
 *   rot_0, rot_1, rot_2, rot_3,
 *   f_dc_0, f_dc_1, f_dc_2
 *
 * Optional: `f_rest_0..f_rest_44` (SH degree 3) — when all 45 are present
 * (or 24 for SH2, 9 for SH1) we carry them through as `shRest`. Inria PLY
 * stores SH-rest channel-major: `f_rest_0..(C-1)` = R, then G, then B.
 * We transpose into the canonical splat-major then coef-major then
 * channel-major layout: `shRest[i*C*3 + k*3 + c]`.
 *
 * `nx/ny/nz` are tolerated and ignored.
 *
 * Returns a fully normalized {@link SplatScene}. Unlike viewer-app's
 * `loaders/ply.ts`, this version ALWAYS populates `dcRaw` (even when SH-rest
 * is absent) so the WebGPU SoA path can read raw `f_dc` uniformly.
 *
 * Source-of-truth parity: keep this loader's signatures and field names in
 * lock-step with `packages/viewer-app/src/loaders/ply.ts`. Phase 3 will
 * dedupe.
 */
import { clamp01, computeBbox, normalizeQuatInto, SH_C0, shRestCoefCount, sigmoid, } from './splat-scene.js';
const TYPE_SIZE = {
    float: 4, double: 8, uchar: 1, char: 1,
    ushort: 2, short: 2, uint: 4, int: 4,
};
function normalizeType(t) {
    switch (t) {
        case 'float32':
        case 'float': return 'float';
        case 'float64':
        case 'double': return 'double';
        case 'uint8':
        case 'uchar': return 'uchar';
        case 'int8':
        case 'char': return 'char';
        case 'uint16':
        case 'ushort': return 'ushort';
        case 'int16':
        case 'short': return 'short';
        case 'uint32':
        case 'uint': return 'uint';
        case 'int32':
        case 'int': return 'int';
        default: throw new Error(`ply: unsupported property type "${t}"`);
    }
}
function parseHeader(buf) {
    const text = new TextDecoder('ascii').decode(buf.subarray(0, Math.min(buf.length, 1 << 16)));
    const endMarker = '\nend_header\n';
    const endIdx = text.indexOf(endMarker);
    if (endIdx < 0)
        throw new Error('ply: missing end_header');
    const headerStr = text.substring(0, endIdx);
    const bodyOffset = endIdx + endMarker.length;
    const lines = headerStr.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    if (lines[0] !== 'ply')
        throw new Error('ply: missing magic');
    let format = null;
    let inVertex = false;
    let vertexCount = 0;
    const props = [];
    let stride = 0;
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith('comment '))
            continue;
        if (line.startsWith('format ')) {
            const parts = line.split(/\s+/);
            if (parts[1] === 'binary_little_endian')
                format = 'binary_little_endian';
            else if (parts[1] === 'binary_big_endian')
                format = 'binary_big_endian';
            else if (parts[1] === 'ascii')
                format = 'ascii';
            else
                throw new Error(`ply: unknown format "${parts[1]}"`);
            continue;
        }
        if (line.startsWith('element ')) {
            const parts = line.split(/\s+/);
            inVertex = parts[1] === 'vertex';
            if (inVertex)
                vertexCount = parseInt(parts[2], 10);
            continue;
        }
        if (line.startsWith('property ')) {
            if (!inVertex)
                continue;
            const parts = line.split(/\s+/);
            if (parts[1] === 'list') {
                // List properties not supported in 3DGS PLY.
                throw new Error('ply: list properties unsupported');
            }
            const type = normalizeType(parts[1]);
            const name = parts[2];
            const size = TYPE_SIZE[type];
            props.push({ name, type, byteOffset: stride, size });
            stride += size;
            continue;
        }
    }
    if (format !== 'binary_little_endian') {
        throw new Error(`ply: only binary_little_endian supported (got ${format})`);
    }
    return { vertexCount, props, vertexStride: stride, bodyOffset, littleEndian: true };
}
function getter(type) {
    switch (type) {
        case 'float': return (dv, o) => dv.getFloat32(o, true);
        case 'double': return (dv, o) => dv.getFloat64(o, true);
        case 'uchar': return (dv, o) => dv.getUint8(o);
        case 'char': return (dv, o) => dv.getInt8(o);
        case 'ushort': return (dv, o) => dv.getUint16(o, true);
        case 'short': return (dv, o) => dv.getInt16(o, true);
        case 'uint': return (dv, o) => dv.getUint32(o, true);
        case 'int': return (dv, o) => dv.getInt32(o, true);
    }
}
/**
 * Detect whether a PLY blob looks like a 3DGS scene (vs a plain point-cloud).
 * Cheap: ASCII-scan the first 16 KiB for `f_dc_0` (the most distinctive token).
 */
export function isLikely3DGSPly(buf) {
    const slice = buf.subarray(0, Math.min(buf.length, 16 * 1024));
    const text = new TextDecoder('ascii', { fatal: false }).decode(slice);
    return text.includes('f_dc_0') && text.includes('opacity') && text.includes('scale_0');
}
export function loadPly(buf, sourceName) {
    const header = parseHeader(buf);
    const required = ['x', 'y', 'z', 'opacity',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'f_dc_0', 'f_dc_1', 'f_dc_2'];
    const byName = new Map(header.props.map((p) => [p.name, p]));
    for (const r of required) {
        if (!byName.has(r))
            throw new Error(`ply: missing required property "${r}"`);
    }
    const N = header.vertexCount;
    const bodyBytes = N * header.vertexStride;
    if (header.bodyOffset + bodyBytes > buf.byteLength) {
        throw new Error(`ply: body truncated (${header.bodyOffset + bodyBytes - buf.byteLength} bytes short)`);
    }
    const dv = new DataView(buf.buffer, buf.byteOffset + header.bodyOffset, bodyBytes);
    const positions = new Float32Array(N * 3);
    const rotations = new Float32Array(N * 4);
    const scales = new Float32Array(N * 3);
    const opacity = new Float32Array(N);
    const colorDC = new Float32Array(N * 3);
    const dcRaw = new Float32Array(N * 3);
    const stride = header.vertexStride;
    // Resolve each property's getter + offset once.
    const r = (name) => {
        const p = byName.get(name);
        return { get: getter(p.type), off: p.byteOffset };
    };
    const X = r('x'), Y = r('y'), Z = r('z');
    const OP = r('opacity');
    const S0 = r('scale_0'), S1 = r('scale_1'), S2 = r('scale_2');
    const R0 = r('rot_0'), R1 = r('rot_1'), R2 = r('rot_2'), R3 = r('rot_3');
    const D0 = r('f_dc_0'), D1 = r('f_dc_1'), D2 = r('f_dc_2');
    // Detect SH-rest. Inria PLY stores 45 of them at SH3, 24 at SH2, 9 at SH1.
    let shDegree = 0;
    if (byName.has('f_rest_44'))
        shDegree = 3;
    else if (byName.has('f_rest_23'))
        shDegree = 2;
    else if (byName.has('f_rest_8'))
        shDegree = 1;
    const coefCount = shRestCoefCount(shDegree);
    let restAccessors = null;
    let shRest = undefined;
    if (shDegree > 0) {
        // Total count = coefCount * 3. PLY layout is channel-major (R coefs 0..coefCount-1,
        // then G coefs 0..coefCount-1, then B coefs 0..coefCount-1).
        const total = coefCount * 3;
        restAccessors = [];
        for (let k = 0; k < total; k++) {
            const name = `f_rest_${k}`;
            if (!byName.has(name)) {
                // Missing some entries — bail on SH-rest, fall back to DC only.
                restAccessors = null;
                shDegree = 0;
                break;
            }
            restAccessors.push(r(name));
        }
        if (restAccessors)
            shRest = new Float32Array(N * total);
    }
    for (let i = 0; i < N; i++) {
        const base = i * stride;
        positions[i * 3 + 0] = X.get(dv, base + X.off);
        positions[i * 3 + 1] = Y.get(dv, base + Y.off);
        positions[i * 3 + 2] = Z.get(dv, base + Z.off);
        opacity[i] = sigmoid(OP.get(dv, base + OP.off));
        scales[i * 3 + 0] = S0.get(dv, base + S0.off);
        scales[i * 3 + 1] = S1.get(dv, base + S1.off);
        scales[i * 3 + 2] = S2.get(dv, base + S2.off);
        // Inria PLY rot order is rot_0=w, rot_1=x, rot_2=y, rot_3=z. We carry XYZW.
        const qw = R0.get(dv, base + R0.off);
        const qx = R1.get(dv, base + R1.off);
        const qy = R2.get(dv, base + R2.off);
        const qz = R3.get(dv, base + R3.off);
        normalizeQuatInto(rotations, i * 4, qx, qy, qz, qw);
        // DC color = 0.5 + SH_C0 * f_dc, clamped. Store BOTH raw + baked so
        // downstream consumers (WebGPU SoA = raw, WebGL2 quad shader = baked) pick.
        const d0 = D0.get(dv, base + D0.off);
        const d1 = D1.get(dv, base + D1.off);
        const d2 = D2.get(dv, base + D2.off);
        dcRaw[i * 3 + 0] = d0;
        dcRaw[i * 3 + 1] = d1;
        dcRaw[i * 3 + 2] = d2;
        colorDC[i * 3 + 0] = clamp01(0.5 + SH_C0 * d0);
        colorDC[i * 3 + 1] = clamp01(0.5 + SH_C0 * d1);
        colorDC[i * 3 + 2] = clamp01(0.5 + SH_C0 * d2);
        if (restAccessors && shRest) {
            // PLY layout: f_rest_[c*coefCount + k]. Canonical: shRest[i*coefCount*3 + k*3 + c].
            const dst = i * coefCount * 3;
            for (let c = 0; c < 3; c++) {
                for (let k = 0; k < coefCount; k++) {
                    const acc = restAccessors[c * coefCount + k];
                    shRest[dst + k * 3 + c] = acc.get(dv, base + acc.off);
                }
            }
        }
    }
    return {
        count: N,
        positions, rotations, scales, opacity, colorDC,
        shRest,
        shDegree: shDegree > 0 ? shDegree : undefined,
        dcRaw,
        bbox: computeBbox(positions),
        meta: { source: sourceName, format: 'ply' },
    };
}
//# sourceMappingURL=ply.js.map