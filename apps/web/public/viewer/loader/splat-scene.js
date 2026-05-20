/** Coefficient count for SH-rest given degree (degree 0 = no rest). */
export function shRestCoefCount(shDegree) {
    let c = 0;
    if (shDegree >= 1)
        c += 3;
    if (shDegree >= 2)
        c += 5;
    if (shDegree >= 3)
        c += 7;
    return c;
}
/** Tight bbox over positions. */
export function computeBbox(positions) {
    const N = positions.length / 3;
    if (N === 0)
        return { min: [0, 0, 0], max: [0, 0, 0] };
    let xMin = positions[0], yMin = positions[1], zMin = positions[2];
    let xMax = xMin, yMax = yMin, zMax = zMin;
    for (let i = 1; i < N; i++) {
        const x = positions[i * 3 + 0];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];
        if (x < xMin)
            xMin = x;
        else if (x > xMax)
            xMax = x;
        if (y < yMin)
            yMin = y;
        else if (y > yMax)
            yMax = y;
        if (z < zMin)
            zMin = z;
        else if (z > zMax)
            zMax = z;
    }
    return { min: [xMin, yMin, zMin], max: [xMax, yMax, zMax] };
}
/** Inria 3DGS sigmoid for opacity decode. */
export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
/** Inria SH-DC0 → linear color, used by PLY + GLB + SOG loaders. */
export const SH_C0 = 0.28209479177387814;
/** Clamp `x` to `[0, 1]`. */
export function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
/** Normalize one quaternion in-place into out[off..off+4]. Falls back to identity. */
export function normalizeQuatInto(out, off, x, y, z, w) {
    const n = Math.hypot(x, y, z, w);
    if (n === 0 || !Number.isFinite(n)) {
        out[off + 0] = 0;
        out[off + 1] = 0;
        out[off + 2] = 0;
        out[off + 3] = 1;
        return;
    }
    const inv = 1 / n;
    out[off + 0] = x * inv;
    out[off + 1] = y * inv;
    out[off + 2] = z * inv;
    out[off + 3] = w * inv;
}
//# sourceMappingURL=splat-scene.js.map