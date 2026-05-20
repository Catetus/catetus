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
export declare function decodeSmallest3Quat(packed: number, componentBits: number): [number, number, number, number];
/**
 * Decode N packed-u32 quaternions into a flat `Float32Array` (length `N*4`,
 * laid out `[x0,y0,z0,w0, x1,y1,z1,w1, ...]` to match the rest of the splat
 * SoA returned by `decodeSFExtensions`).
 *
 * The accessor for SF_quat_smallest3 ROTATION is SCALAR / UNSIGNED_INT (5125)
 * per the producer, so the caller passes the raw u32 buffer (`Uint32Array`).
 */
export declare function decodeSmallest3QuatBuffer(packed: Uint32Array, componentBits: number, count?: number): Float32Array;
/** Top-level `SF_quat_smallest3` extension shape. */
export interface QuatSmallest3Ext {
    componentBits?: number;
    componentType?: number;
    layout?: string;
    tagBits?: number;
}
//# sourceMappingURL=smallest3.d.ts.map