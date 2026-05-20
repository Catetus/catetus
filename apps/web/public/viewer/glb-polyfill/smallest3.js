/**
 * Decode `SF_quat_smallest3`: SOG-style smallest-3 quaternion packing.
 *
 * Encoding (see `crates/catetus-gltf/src/lib.rs::RotationSmallest3Table`):
 *   - Normalize quaternion q.
 *   - tag = argmax_i |q[i]|. The largest component is *dropped*.
 *   - Multiply remaining 3 components by `sign(q[tag])` so the dropped
 *     component is non-negative; this removes the q ≡ -q ambiguity.
 *   - Each stored component is in `[-1/sqrt(2), +1/sqrt(2)]`; map to
 *     `[0,1]` via `t = v / sqrt(2) + 0.5`, then quantize to
 *     `round(t * (2^bits - 1))`.
 *   - Pack `[q0:bits | q1:bits | q2:bits | tag:2]` LE into a u32.
 *
 * Decoding inverts that. At 10 bits the layout fits exactly in 32 bits with
 * the 2 tag bits at the top (positions 30..31).
 */
/** Decoder for a single packed u32 quaternion. */
export function decodeSmallest3Quat(packed, componentBits) {
    const bits = Math.max(6, Math.min(10, componentBits | 0));
    const mask = (1 << bits) - 1;
    const levels = (1 << bits) - 1;
    // Unpack three components and tag (top 2 bits at position 30).
    const q0 = packed & mask;
    const q1 = (packed >>> bits) & mask;
    const q2 = (packed >>> (2 * bits)) & mask;
    const tag = (packed >>> 30) & 0x3;
    const sqrt2 = Math.SQRT2;
    const dq = (q) => (q / levels - 0.5) * sqrt2;
    const a = dq(q0);
    const b = dq(q1);
    const c = dq(q2);
    // Recover dropped component magnitude from unit-length constraint.
    const sum = a * a + b * b + c * c;
    const dropped = Math.sqrt(Math.max(0, 1 - sum));
    // Re-insert at slot `tag`; the other three slots take a, b, c in order.
    const out = [0, 0, 0, 0];
    const others = [a, b, c];
    let k = 0;
    for (let i = 0; i < 4; i++) {
        if (i === tag) {
            out[i] = dropped;
        }
        else {
            out[i] = others[k++];
        }
    }
    return out;
}
/**
 * Decode N packed-u32 quaternions into a flat `Float32Array` (length `N*4`,
 * laid out `[x0,y0,z0,w0, x1,y1,z1,w1, ...]` to match the rest of the splat
 * SoA returned by `decodeSFExtensions`).
 *
 * The accessor for SF_quat_smallest3 ROTATION is SCALAR / UNSIGNED_INT (5125)
 * per the producer, so the caller passes the raw u32 buffer (`Uint32Array`).
 */
export function decodeSmallest3QuatBuffer(packed, componentBits, count) {
    const N = count ?? packed.length;
    const out = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
        const q = decodeSmallest3Quat(packed[i], componentBits);
        out[i * 4 + 0] = q[0];
        out[i * 4 + 1] = q[1];
        out[i * 4 + 2] = q[2];
        out[i * 4 + 3] = q[3];
    }
    return out;
}
//# sourceMappingURL=smallest3.js.map