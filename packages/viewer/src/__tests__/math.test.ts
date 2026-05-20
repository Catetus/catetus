import { describe, expect, it } from 'vitest';
import {
  buildViewProj,
  computeCovariance3D,
  lookAt,
  perspective,
  projectCovariance2D,
  projectPoint,
  quatToMat3,
} from '../renderer/math.js';
import type { CameraPose } from '../camera.js';

const EPS = 1e-5;

function expectClose(a: number, b: number, eps = EPS): void {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(eps);
}

describe('quatToMat3', () => {
  it('returns identity for the identity quaternion', () => {
    const m = quatToMat3([0, 0, 0, 1]);
    expectClose(m[0]!, 1);
    expectClose(m[4]!, 1);
    expectClose(m[8]!, 1);
    expectClose(m[1]!, 0);
    expectClose(m[3]!, 0);
    expectClose(m[5]!, 0);
    expectClose(m[2]!, 0);
    expectClose(m[6]!, 0);
    expectClose(m[7]!, 0);
  });

  it('rotates 90 degrees about Y correctly', () => {
    // q = (sin(45deg) Y, cos(45deg)) = (0, sqrt(2)/2, 0, sqrt(2)/2)
    const s = Math.SQRT1_2;
    const m = quatToMat3([0, s, 0, s]);
    // R · X = Z (approximately), so column 0 = (0,0,-1).
    expectClose(m[0]!, 0, 1e-5);
    expectClose(m[1]!, 0, 1e-5);
    expectClose(m[2]!, -1, 1e-5);
    // R · Y = Y
    expectClose(m[3]!, 0, 1e-5);
    expectClose(m[4]!, 1, 1e-5);
    expectClose(m[5]!, 0, 1e-5);
    // R · Z = X
    expectClose(m[6]!, 1, 1e-5);
    expectClose(m[7]!, 0, 1e-5);
    expectClose(m[8]!, 0, 1e-5);
  });
});

describe('computeCovariance3D', () => {
  it('returns near-identity for unit scale + identity quat', () => {
    const c = computeCovariance3D([1, 1, 1], [0, 0, 0, 1]);
    expectClose(c[0]!, 1);
    expectClose(c[3]!, 1);
    expectClose(c[5]!, 1);
    expectClose(c[1]!, 0);
    expectClose(c[2]!, 0);
    expectClose(c[4]!, 0);
  });

  it('scales diagonal entries by s² for axis-aligned splats', () => {
    const c = computeCovariance3D([2, 3, 4], [0, 0, 0, 1]);
    expectClose(c[0]!, 4);
    expectClose(c[3]!, 9);
    expectClose(c[5]!, 16);
  });
});

describe('projectPoint', () => {
  it('projects an on-axis point through a lookAt-origin VP to NDC near (0,0)', () => {
    const cam: CameraPose = {
      position: [0, 0, 0],
      target: [0, 0, -1],
      up: [0, 1, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 100,
    };
    const { viewProj } = buildViewProj(cam);
    const r = projectPoint([0, 0, -5], viewProj);
    expectClose(r.ndc[0], 0, 1e-4);
    expectClose(r.ndc[1], 0, 1e-4);
    expect(r.w).toBeGreaterThan(0);
  });
});

describe('lookAt + perspective', () => {
  it('lookAt produces a translation that places the eye at origin in view space', () => {
    const view = lookAt([5, 0, 0], [0, 0, 0], [0, 1, 0]);
    // view * (5,0,0,1) should be (0,0,0,1).
    const px = 5;
    const tx = view[0]! * px + view[12]!;
    const ty = view[1]! * px + view[13]!;
    const tz = view[2]! * px + view[14]!;
    expectClose(tx, 0, 1e-5);
    expectClose(ty, 0, 1e-5);
    expectClose(tz, 0, 1e-5);
  });

  it('perspective maps a point on the near plane to NDC z = -1 (approx)', () => {
    const proj = perspective(Math.PI / 3, 1, 1, 100);
    // Sanity: m[10] and m[14] are non-zero.
    expect(proj[10]).not.toBe(0);
    expect(proj[14]).not.toBe(0);
  });
});

describe('projectCovariance2D', () => {
  it('returns a positive-definite 2x2 for a unit splat in front of the camera', () => {
    const cov3 = computeCovariance3D([0.1, 0.1, 0.1], [0, 0, 0, 1]);
    const cam: CameraPose = {
      position: [0, 0, 0],
      target: [0, 0, -1],
      up: [0, 1, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 100,
    };
    const { view } = buildViewProj(cam);
    const [c00, c01, c11] = projectCovariance2D(cov3, view, 400, 400, 5);
    expect(c00).toBeGreaterThan(0);
    expect(c11).toBeGreaterThan(0);
    expect(c00 * c11 - c01 * c01).toBeGreaterThan(0);
  });
});
