import { describe, expect, it } from 'vitest';
import {
  aabbFromObb12,
  aabbIntersectsFrustum,
  distanceFromCamera,
  extractFrustum,
} from '../frustum.js';
import { buildViewProj } from '../../renderer/math.js';
import type { CameraPose } from '../../camera.js';

function defaultCam(eye: [number, number, number], target: [number, number, number] = [0, 0, 0]): CameraPose {
  return {
    position: eye,
    target,
    up: [0, 1, 0],
    fovY: Math.PI / 3,
    aspect: 1,
    near: 0.1,
    far: 100,
  };
}

describe('aabbFromObb12', () => {
  it('round-trips an axis-aligned OBB', () => {
    const obb = [1, 2, 3,  4, 0, 0,  0, 5, 0,  0, 0, 6];
    const a = aabbFromObb12(obb);
    expect(a.min).toEqual([-3, -3, -3]);
    expect(a.max).toEqual([5, 7, 9]);
  });

  it('throws on bad length', () => {
    expect(() => aabbFromObb12([0, 0, 0])).toThrow(/aabb_invalid/);
  });

  it('handles a rotated OBB conservatively (still bounds it)', () => {
    // 45° rotation in XY plane: hx along (cos45, sin45, 0)
    const c = Math.SQRT1_2;
    const obb = [0, 0, 0,  c, c, 0,  -c, c, 0,  0, 0, 1];
    const a = aabbFromObb12(obb);
    // Conservative bound: hx + hy on x = 2c = sqrt(2)
    expect(a.max[0]).toBeCloseTo(Math.SQRT2, 5);
    expect(a.max[1]).toBeCloseTo(Math.SQRT2, 5);
    expect(a.min[2]).toBe(-1);
  });
});

describe('extractFrustum + aabbIntersectsFrustum', () => {
  it('includes a box at the origin when camera points at it', () => {
    const cam = defaultCam([0, 0, 5]);
    const { viewProj } = buildViewProj(cam, 1);
    const f = extractFrustum(viewProj);
    expect(
      aabbIntersectsFrustum({ min: [-1, -1, -1], max: [1, 1, 1] }, f),
    ).toBe(true);
  });

  it('rejects a box behind the camera', () => {
    const cam = defaultCam([0, 0, 5]);
    const { viewProj } = buildViewProj(cam, 1);
    const f = extractFrustum(viewProj);
    // Box at z = +10 is behind camera (cam looks at -z from z=5).
    expect(
      aabbIntersectsFrustum({ min: [-0.1, -0.1, 9.9], max: [0.1, 0.1, 10.1] }, f),
    ).toBe(false);
  });

  it('rejects a box far off to the side', () => {
    const cam = defaultCam([0, 0, 5]);
    const { viewProj } = buildViewProj(cam, 1);
    const f = extractFrustum(viewProj);
    expect(
      aabbIntersectsFrustum({ min: [100, -1, -1], max: [101, 1, 1] }, f),
    ).toBe(false);
  });
});

describe('distanceFromCamera', () => {
  it('returns euclidean distance to box center', () => {
    const d = distanceFromCamera({ min: [-1, -1, -1], max: [1, 1, 1] }, [3, 0, 0]);
    expect(d).toBeCloseTo(3, 5);
  });
  it('clamps a zero-distance to a small epsilon', () => {
    const d = distanceFromCamera({ min: [-1, -1, -1], max: [1, 1, 1] }, [0, 0, 0]);
    expect(d).toBeGreaterThan(0);
    expect(d).toBeLessThan(1e-3);
  });
});
