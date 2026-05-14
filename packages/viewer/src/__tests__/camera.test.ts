import { describe, expect, it } from 'vitest';
import {
  SeededRng,
  bboxCenter,
  bboxRadius,
  orbitFrames,
  orbitPose,
} from '../camera.js';
import type { Bbox } from '../manifest.js';

const UNIT: Bbox = { min: [-1, -1, -1], max: [1, 1, 1] };

describe('orbitFrames', () => {
  it('returns evenly spaced yaws for count=8', () => {
    const f = orbitFrames(8);
    expect(f).toHaveLength(8);
    expect(f[0]).toBeCloseTo(0);
    expect(f[2]).toBeCloseTo(Math.PI / 2);
    expect(f[4]).toBeCloseTo(Math.PI);
    expect(f[7]).toBeCloseTo((7 * Math.PI) / 4);
  });

  it('returns [] for non-positive counts', () => {
    expect(orbitFrames(0)).toEqual([]);
    expect(orbitFrames(-3)).toEqual([]);
  });

  it('is deterministic across calls', () => {
    expect(orbitFrames(8)).toEqual(orbitFrames(8));
  });
});

describe('SeededRng', () => {
  it('two instances with the same seed produce the same stream', () => {
    const a = new SeededRng(42);
    const b = new SeededRng(42);
    for (let i = 0; i < 16; i++) {
      expect(a.next()).toBe(b.next());
    }
  });

  it('different seeds produce different streams', () => {
    const a = new SeededRng(1);
    const b = new SeededRng(2);
    expect(a.next()).not.toBe(b.next());
  });
});

describe('orbitPose', () => {
  it('frames the bbox at the configured radius', () => {
    const p0 = orbitPose(UNIT, 0, 1);
    expect(p0.target).toEqual(bboxCenter(UNIT));
    const radius = bboxRadius(UNIT) * 2.5;
    const dx = p0.position[0] - p0.target[0];
    const dz = p0.position[2] - p0.target[2];
    const horiz = Math.sqrt(dx * dx + dz * dz);
    // Horizontal distance equals radius * cos(20deg).
    expect(horiz).toBeCloseTo(radius * Math.cos((20 * Math.PI) / 180), 4);
  });

  it('produces identical poses for the same yaw seed', () => {
    const a = orbitPose(UNIT, Math.PI / 3, 1.7);
    const b = orbitPose(UNIT, Math.PI / 3, 1.7);
    expect(a).toEqual(b);
  });
});
