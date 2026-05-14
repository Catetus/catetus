/**
 * Tiny xorshift32 RNG. Deterministic given a seed; used only for jitter on
 * the `orbit` path so two runs of the same scene produce identical frames.
 */
export class SeededRng {
    state;
    constructor(seed) {
        // Avoid zero state which collapses xorshift.
        this.state = (seed | 0) || 0x1234_5678;
    }
    /** Next float in `[0, 1)`. */
    next() {
        let x = this.state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.state = x | 0;
        // Convert to unsigned then to unit interval.
        return ((x >>> 0) / 0x1_0000_0000);
    }
}
/** Return the center of an AABB. */
export function bboxCenter(b) {
    return [
        (b.min[0] + b.max[0]) * 0.5,
        (b.min[1] + b.max[1]) * 0.5,
        (b.min[2] + b.max[2]) * 0.5,
    ];
}
/** Return the half-diagonal length of an AABB. */
export function bboxRadius(b) {
    const dx = b.max[0] - b.min[0];
    const dy = b.max[1] - b.min[1];
    const dz = b.max[2] - b.min[2];
    return 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
}
/**
 * Produce `count` orbit yaw angles in `[0, 2π)`, evenly spaced.
 *
 * @example
 * ```ts
 * orbitFrames(8); // [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
 * ```
 */
export function orbitFrames(count) {
    if (count <= 0)
        return [];
    const out = new Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = (i / count) * Math.PI * 2;
    }
    return out;
}
/**
 * Build a {@link CameraPose} that frames `bbox` from yaw angle `yaw` at a
 * fixed elevation of 20 degrees.
 */
export function orbitPose(bbox, yaw, aspect) {
    const center = bboxCenter(bbox);
    const radius = Math.max(bboxRadius(bbox), 0.001) * 2.5;
    const elev = (20 * Math.PI) / 180;
    const cy = Math.cos(elev);
    const sy = Math.sin(elev);
    const x = center[0] + radius * cy * Math.sin(yaw);
    const y = center[1] + radius * sy;
    const z = center[2] + radius * cy * Math.cos(yaw);
    return {
        position: [x, y, z],
        target: center,
        up: [0, 1, 0],
        fovY: (60 * Math.PI) / 180,
        aspect,
        near: 0.01,
        far: radius * 10,
    };
}
