/**
 * Decode a chunk based on its descriptor: SoA when `attributeLayout` is set
 * (the wire format emitted by `catetus convert`/`optimize`), otherwise the
 * legacy interleaved AoS layout used by hand-crafted test fixtures.
 */
export function decodeChunkBytes(bytes, descriptor) {
    if (descriptor.attributeLayout) {
        return decodeSplatsSoa(bytes, descriptor.attributeLayout, descriptor.splatCount);
    }
    return decodeSplats(bytes);
}
/**
 * Decode splats from a structure-of-arrays binary chunk. Attribute order on
 * disk (POSITION, _ROTATION, _SCALE, _OPACITY, _COLOR_DC) follows
 * `KHR_gaussian_splatting`; we re-interleave to `DecodedSplat`.
 *
 * Each attribute is decoded according to its accessor `componentType` /
 * `normalized` / `min` / `max` — SPEC-0013 (`KHR_mesh_quantization`) permits
 * POSITION → u16, _SCALE/_OPACITY/_COLOR_DC → u8 with normalized=true.
 *
 * Quaternion convention on the wire is (x, y, z, w) which already matches our
 * runtime convention — no axis flip needed.
 */
export function decodeSplatsSoa(bytes, layout, splatCount) {
    if (splatCount === 0)
        return [];
    const pos = decodeAttribute(bytes, layout.positions, splatCount, 3);
    const rot = decodeAttribute(bytes, layout.rotations, splatCount, 4);
    const scl = decodeAttribute(bytes, layout.scales, splatCount, 3);
    const op = decodeAttribute(bytes, layout.opacities, splatCount, 1);
    const dc = decodeAttribute(bytes, layout.colorDC, splatCount, 3);
    const out = new Array(splatCount);
    for (let i = 0; i < splatCount; i++) {
        out[i] = {
            position: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
            rotation: [rot[i * 4], rot[i * 4 + 1], rot[i * 4 + 2], rot[i * 4 + 3]],
            scale: [scl[i * 3], scl[i * 3 + 1], scl[i * 3 + 2]],
            opacity: op[i],
            colorDC: [dc[i * 3], dc[i * 3 + 1], dc[i * 3 + 2]],
        };
    }
    return out;
}
const FLOAT_CT = 5126;
const UBYTE_CT = 5121;
const USHORT_CT = 5123;
/**
 * Decode one SoA attribute into a flat `Float32Array` of length
 * `splatCount * comps`. Handles FLOAT pass-through plus the
 * `KHR_mesh_quantization` integer variants (UNSIGNED_BYTE, UNSIGNED_SHORT),
 * dequantizing against the accessor's per-component min/max when normalized.
 */
function decodeAttribute(bytes, slice, splatCount, comps) {
    const total = splatCount * comps;
    const buf = bytes.buffer;
    const base = bytes.byteOffset + slice.byteOffset;
    const ct = slice.componentType ?? FLOAT_CT;
    if (ct === FLOAT_CT) {
        // Zero-copy float view when the byte offset is 4-aligned; copy otherwise.
        if ((base & 3) === 0) {
            return new Float32Array(buf, base, total);
        }
        const dv = new DataView(buf, base, total * 4);
        const out = new Float32Array(total);
        for (let i = 0; i < total; i++)
            out[i] = dv.getFloat32(i * 4, true);
        return out;
    }
    if (ct === USHORT_CT) {
        const src = new Uint16Array(buf, base, total);
        const out = new Float32Array(total);
        if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
            for (let i = 0; i < total; i++) {
                const k = i % comps;
                const lo = slice.min[k];
                const hi = slice.max[k];
                out[i] = lo + (src[i] / 65535) * (hi - lo);
            }
        }
        else if (slice.normalized) {
            for (let i = 0; i < total; i++)
                out[i] = src[i] / 65535;
        }
        else {
            for (let i = 0; i < total; i++)
                out[i] = src[i];
        }
        return out;
    }
    if (ct === UBYTE_CT) {
        const src = new Uint8Array(buf, base, total);
        const out = new Float32Array(total);
        if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
            for (let i = 0; i < total; i++) {
                const k = i % comps;
                const lo = slice.min[k];
                const hi = slice.max[k];
                out[i] = lo + (src[i] / 255) * (hi - lo);
            }
        }
        else if (slice.normalized) {
            for (let i = 0; i < total; i++)
                out[i] = src[i] / 255;
        }
        else {
            for (let i = 0; i < total; i++)
                out[i] = src[i];
        }
        return out;
    }
    throw new Error(`unsupported componentType: ${ct}`);
}
/**
 * Naive bytes-to-splats decoder. The optimized packer produces a 32-byte
 * fixed-point layout we'll wire in once SPEC-0007 lands; for now we parse a
 * float32 layout so unit tests can round-trip easily.
 *
 * Layout (per splat, 14 float32 = 56 bytes):
 *   px py pz  sx sy sz  qx qy qz qw  opacity  cr cg cb
 */
export function decodeSplats(bytes) {
    const stride = 14 * 4;
    const count = Math.floor(bytes.byteLength / stride);
    if (count === 0)
        return [];
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Array(count);
    for (let i = 0; i < count; i++) {
        const o = i * stride;
        out[i] = {
            position: [dv.getFloat32(o + 0, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)],
            scale: [dv.getFloat32(o + 12, true), dv.getFloat32(o + 16, true), dv.getFloat32(o + 20, true)],
            rotation: [
                dv.getFloat32(o + 24, true),
                dv.getFloat32(o + 28, true),
                dv.getFloat32(o + 32, true),
                dv.getFloat32(o + 36, true),
            ],
            opacity: dv.getFloat32(o + 40, true),
            colorDC: [dv.getFloat32(o + 44, true), dv.getFloat32(o + 48, true), dv.getFloat32(o + 52, true)],
        };
    }
    return out;
}
/**
 * Sort `indices` so splats are drawn back-to-front (largest depth first).
 *
 * Determinism note: equal depths fall back to ascending splat index so two
 * runs on the same inputs produce the same draw order. The sort is in-place
 * via a paired index/depth array so it remains O(n log n) on big scenes.
 */
export function sortBackToFront(splats, cam, indices) {
    const n = indices.length;
    if (n <= 1)
        return;
    const depths = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const s = splats[indices[i]];
        if (!s) {
            depths[i] = 0;
            continue;
        }
        const dx = s.position[0] - cam.position[0];
        const dy = s.position[1] - cam.position[1];
        const dz = s.position[2] - cam.position[2];
        depths[i] = dx * dx + dy * dy + dz * dz;
    }
    // Build paired array, sort, copy back. Pair-based sort avoids the O(n²)
    // insertion sort that doesn't scale past ~10K splats.
    const pairs = new Array(n);
    for (let i = 0; i < n; i++)
        pairs[i] = [depths[i], indices[i]];
    pairs.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]));
    for (let i = 0; i < n; i++)
        indices[i] = pairs[i][1];
}
//# sourceMappingURL=base.js.map