import { describe, expect, it } from 'vitest';
import {
  parseLodgeManifest,
  selectLodLevel,
  chunksByCameraProximity,
  sceneBboxCenter,
  sceneBboxRadius,
  LODGE_MANIFEST_VERSION,
} from '../lodge/manifest.js';

const FIXTURE = {
  version: 1,
  source: 'bonsai.ply',
  original_splat_count: 1000,
  bbox: [
    [-10, -10, -10],
    [10, 10, 10],
  ],
  levels: [
    {
      level: 0,
      splat_count: 1000,
      reduction: 1.0,
      depth_threshold: 0.0,
      chunks: [
        {
          index: 0,
          path: 'level_0/chunk_0000.ply',
          splat_count: 500,
          bbox: [
            [-10, -10, -10],
            [0, 10, 10],
          ],
          centroid: [-5, 0, 0],
          radius: 7.0,
          blake3: 'aa',
        },
        {
          index: 1,
          path: 'level_0/chunk_0001.ply',
          splat_count: 500,
          bbox: [
            [0, -10, -10],
            [10, 10, 10],
          ],
          centroid: [5, 0, 0],
          radius: 7.0,
          blake3: 'bb',
        },
      ],
    },
    {
      level: 1,
      splat_count: 500,
      reduction: 0.5,
      depth_threshold: 15.0,
      chunks: [
        {
          index: 0,
          path: 'level_1/chunk_0000.ply',
          splat_count: 500,
          bbox: [
            [-10, -10, -10],
            [10, 10, 10],
          ],
          centroid: [0, 0, 0],
          radius: 10.0,
          blake3: 'cc',
        },
      ],
    },
    {
      level: 2,
      splat_count: 100,
      reduction: 0.1,
      depth_threshold: 30.0,
      chunks: [
        {
          index: 0,
          path: 'level_2/chunk_0000.ply',
          splat_count: 100,
          bbox: [
            [-10, -10, -10],
            [10, 10, 10],
          ],
          centroid: [0, 0, 0],
          radius: 10.0,
          blake3: 'dd',
        },
      ],
    },
  ],
};

describe('parseLodgeManifest', () => {
  it('parses a v1 manifest', () => {
    const m = parseLodgeManifest(JSON.stringify(FIXTURE));
    expect(m.version).toBe(LODGE_MANIFEST_VERSION);
    expect(m.originalSplatCount).toBe(1000);
    expect(m.levels).toHaveLength(3);
    expect(m.levels[0]!.chunks).toHaveLength(2);
    expect(m.levels[0]!.chunks[0]!.path).toBe('level_0/chunk_0000.ply');
    expect(m.levels[0]!.chunks[0]!.blake3).toBe('aa');
    expect(m.bbox[1][0]).toBe(10);
  });

  it('rejects non-JSON input', () => {
    expect(() => parseLodgeManifest('not json')).toThrow(/lodge_manifest_invalid/);
  });

  it('rejects missing version', () => {
    expect(() => parseLodgeManifest(JSON.stringify({ levels: [] }))).toThrow(
      /lodge_manifest_invalid/,
    );
  });

  it('rejects missing levels array', () => {
    expect(() => parseLodgeManifest(JSON.stringify({ version: 1, bbox: [[0,0,0],[1,1,1]] }))).toThrow(
      /lodge_manifest_invalid/,
    );
  });

  it('rejects malformed bbox', () => {
    expect(() =>
      parseLodgeManifest(
        JSON.stringify({ version: 1, levels: [], bbox: 'oops' }),
      ),
    ).toThrow(/lodge_manifest_invalid/);
  });
});

describe('selectLodLevel', () => {
  const m = parseLodgeManifest(JSON.stringify(FIXTURE));

  it('returns 0 when camera is at the scene centroid', () => {
    expect(selectLodLevel(m, [0, 0, 0])).toBe(0);
  });

  it('picks coarser level as the camera recedes', () => {
    // d = 20 → past level1 threshold (15), before level2 (30).
    expect(selectLodLevel(m, [20, 0, 0])).toBe(1);
    // d = 50 → past level2 threshold (30) → coarsest.
    expect(selectLodLevel(m, [50, 0, 0])).toBe(2);
  });

  it('falls back to 0 when manifest is empty', () => {
    const empty = parseLodgeManifest(
      JSON.stringify({ version: 1, source: '', original_splat_count: 0, bbox: [[0,0,0],[1,1,1]], levels: [] }),
    );
    expect(selectLodLevel(empty, [10, 10, 10])).toBe(0);
  });
});

describe('chunksByCameraProximity', () => {
  const m = parseLodgeManifest(JSON.stringify(FIXTURE));

  it('orders by ascending distance to camera', () => {
    const cam: [number, number, number] = [10, 0, 0];
    const ordered = chunksByCameraProximity(m, 0, cam);
    expect(ordered.map((c) => c.index)).toEqual([1, 0]);
  });

  it('returns [] for an invalid level', () => {
    expect(chunksByCameraProximity(m, 99, [0, 0, 0])).toEqual([]);
  });
});

describe('scene-bbox helpers', () => {
  const m = parseLodgeManifest(JSON.stringify(FIXTURE));

  it('center is the midpoint', () => {
    expect(sceneBboxCenter(m)).toEqual([0, 0, 0]);
  });

  it('radius is the half-diagonal', () => {
    const r = sceneBboxRadius(m);
    // bbox is [-10,-10,-10]..[10,10,10]; diag = sqrt(20² + 20² + 20²) = 20*sqrt(3).
    expect(r).toBeCloseTo(10 * Math.sqrt(3), 5);
  });
});
