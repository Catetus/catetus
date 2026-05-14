/**
 * Deterministic orbit-camera utilities.
 *
 * The viewer uses these in headless mode (`cameraPath: 'orbit-8'`) to feed a
 * stable sequence of poses to the visual-regression harness in SPEC-0009.
 */
import type { Bbox, Vec3 } from './manifest.js';

/** Camera pose in world space. */
export interface CameraPose {
  /** Eye position. */
  position: Vec3;
  /** Look-at target. */
  target: Vec3;
  /** Up vector. Always `[0, 1, 0]` for the canned orbits. */
  up: Vec3;
  /** Vertical field-of-view in radians. */
  fovY: number;
  /** Aspect ratio (width / height). */
  aspect: number;
  /** Near clip. */
  near: number;
  /** Far clip. */
  far: number;
}

/**
 * Tiny xorshift32 RNG. Deterministic given a seed; used only for jitter on
 * the `orbit` path so two runs of the same scene produce identical frames.
 */
export class SeededRng {
  private state: number;
  constructor(seed: number) {
    // Avoid zero state which collapses xorshift.
    this.state = (seed | 0) || 0x1234_5678;
  }
  /** Next float in `[0, 1)`. */
  next(): number {
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
export function bboxCenter(b: Bbox): Vec3 {
  return [
    (b.min[0] + b.max[0]) * 0.5,
    (b.min[1] + b.max[1]) * 0.5,
    (b.min[2] + b.max[2]) * 0.5,
  ];
}

/** Return the half-diagonal length of an AABB. */
export function bboxRadius(b: Bbox): number {
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
export function orbitFrames(count: number): number[] {
  if (count <= 0) return [];
  const out: number[] = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = (i / count) * Math.PI * 2;
  }
  return out;
}

/**
 * Build a {@link CameraPose} that frames `bbox` from yaw angle `yaw` at a
 * fixed elevation of 20 degrees.
 */
export function orbitPose(bbox: Bbox, yaw: number, aspect: number): CameraPose {
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
