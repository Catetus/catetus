import { describe, expect, it } from 'vitest';
import {
  boundaryBlendT,
  selectChunkActivation,
  pickNearFarChunks,
  LOD_SLOT_NEAR,
  LOD_SLOT_FAR,
  LOD_MAX_LEVELS,
  type ChunkRecord,
  type LodSelectInputs,
} from '../lodge/lod-math.js';
import { parseLodgeManifest } from '../lodge/manifest.js';

const FIXTURE_MANIFEST = {
  version: 1,
  source: 'test.ply',
  original_splat_count: 3000,
  bbox: [
    [-10, -10, -10],
    [10, 10, 10],
  ],
  levels: [
    {
      level: 0,
      splat_count: 2000,
      reduction: 1.0,
      depth_threshold: 0.0,
      chunks: [
        {
          index: 0,
          path: 'level_0/chunk_0000.ply',
          splat_count: 1000,
          bbox: [
            [-10, -10, -10],
            [0, 10, 10],
          ],
          centroid: [-5, 0, 0],
          radius: 5,
          blake3: 'aa',
        },
        {
          index: 1,
          path: 'level_0/chunk_0001.ply',
          splat_count: 1000,
          bbox: [
            [0, -10, -10],
            [10, 10, 10],
          ],
          centroid: [5, 0, 0],
          radius: 5,
          blake3: 'bb',
        },
      ],
    },
    {
      level: 1,
      splat_count: 1000,
      reduction: 0.5,
      depth_threshold: 20.0,
      chunks: [
        {
          index: 0,
          path: 'level_1/chunk_0000.ply',
          splat_count: 1000,
          bbox: [
            [-10, -10, -10],
            [10, 10, 10],
          ],
          centroid: [0, 0, 0],
          radius: 10,
          blake3: 'cc',
        },
      ],
    },
  ],
};

describe('boundaryBlendT — LODGE eq. 4', () => {
  it('returns 0 when camera is at near centroid', () => {
    const t = boundaryBlendT([0, 0, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBe(0);
  });

  it('returns 1 when camera is at far centroid', () => {
    const t = boundaryBlendT([10, 0, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBe(1);
  });

  it('returns 0.5 at midpoint', () => {
    const t = boundaryBlendT([5, 0, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBeCloseTo(0.5, 6);
  });

  it('clamps to 0 when camera projects past near', () => {
    const t = boundaryBlendT([-5, 0, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBe(0);
  });

  it('clamps to 1 when camera projects past far', () => {
    const t = boundaryBlendT([15, 0, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBe(1);
  });

  it('projects off-axis camera onto the chunk-axis', () => {
    // Camera at (5, 100, 0): the y-offset is orthogonal to the axis
    // (0,0,0) → (10,0,0), so the projected t should still be 0.5.
    const t = boundaryBlendT([5, 100, 0], [0, 0, 0], [10, 0, 0]);
    expect(t).toBeCloseTo(0.5, 6);
  });

  it('handles degenerate near==far by returning 0', () => {
    const t = boundaryBlendT([0, 0, 0], [3, 3, 3], [3, 3, 3]);
    expect(t).toBe(0);
  });
});

describe('selectChunkActivation', () => {
  const baseInputs: LodSelectInputs = {
    cameraPos: [0, 0, 0],
    sceneCenter: [0, 0, 0],
    depthThresholds: [0, 20, Infinity, Infinity, Infinity, Infinity, Infinity, Infinity],
    nearCentroid: [-5, 0, 0],
    farCentroid: [5, 0, 0],
    ssSizeThreshold: 0, // disable SS bump for the basic test
    focalY: 1000,
  };

  it('picks level 0 when camera inside the finest band', () => {
    const rec: ChunkRecord = {
      centroid: [0, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const act = selectChunkActivation(rec, { ...baseInputs, cameraPos: [0, 0, 0] });
    expect(act.level).toBe(0);
    expect(act.active).toBe(1);
  });

  it('picks level 1 when camera far enough out', () => {
    const rec0: ChunkRecord = {
      centroid: [0, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const rec1: ChunkRecord = { ...rec0, level: 1 };
    const farInputs = { ...baseInputs, cameraPos: [50, 0, 0] as [number, number, number] };
    expect(selectChunkActivation(rec0, farInputs).active).toBe(0);
    expect(selectChunkActivation(rec1, farInputs).active).toBe(1);
  });

  it('bumps to coarser level when projected size below threshold', () => {
    // chunk radius 0.001 → projected radius 0.5 px @ dist=2 with focal 1000.
    // ss_size_threshold = 4 → should bump from level 0 to level 1.
    const rec0: ChunkRecord = {
      centroid: [0, 0, 0],
      radius: 0.001,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const rec1: ChunkRecord = { ...rec0, level: 1 };
    const inputs = { ...baseInputs, ssSizeThreshold: 4, cameraPos: [2, 0, 0] as [number, number, number] };
    expect(selectChunkActivation(rec0, inputs).active).toBe(0);
    expect(selectChunkActivation(rec1, inputs).active).toBe(1);
  });

  it('assigns SLOT_NEAR / SLOT_FAR by chunk-to-centroid proximity', () => {
    const recA: ChunkRecord = {
      centroid: [-5, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const recB: ChunkRecord = { ...recA, centroid: [5, 0, 0] };
    const inputs = baseInputs; // near=-5,0,0  far=5,0,0
    expect(selectChunkActivation(recA, inputs).slot).toBe(LOD_SLOT_NEAR);
    expect(selectChunkActivation(recB, inputs).slot).toBe(LOD_SLOT_FAR);
  });

  it('applies eq. 4 t-ramp: near gets (1-t), far gets t', () => {
    const recNear: ChunkRecord = {
      centroid: [-5, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const recFar: ChunkRecord = { ...recNear, centroid: [5, 0, 0] };
    // Camera at (0, 0, 0): t = ((0 - (-5)) · 10) / 100 = 0.5.
    const aN = selectChunkActivation(recNear, baseInputs);
    const aF = selectChunkActivation(recFar, baseInputs);
    expect(aN.tBlend).toBeCloseTo(0.5, 6);
    expect(aF.tBlend).toBeCloseTo(0.5, 6);
  });

  it('camera at near centroid: near opacity stays full, far fades to 0', () => {
    const recNear: ChunkRecord = {
      centroid: [-5, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const recFar: ChunkRecord = { ...recNear, centroid: [5, 0, 0] };
    const inputs = { ...baseInputs, cameraPos: [-5, 0, 0] as [number, number, number] };
    expect(selectChunkActivation(recNear, inputs).tBlend).toBeCloseTo(1.0, 6);
    expect(selectChunkActivation(recFar, inputs).tBlend).toBeCloseTo(0.0, 6);
  });

  it('camera at far centroid: far opacity stays full, near fades to 0', () => {
    const recNear: ChunkRecord = {
      centroid: [-5, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const recFar: ChunkRecord = { ...recNear, centroid: [5, 0, 0] };
    const inputs = { ...baseInputs, cameraPos: [5, 0, 0] as [number, number, number] };
    expect(selectChunkActivation(recNear, inputs).tBlend).toBeCloseTo(0.0, 6);
    expect(selectChunkActivation(recFar, inputs).tBlend).toBeCloseTo(1.0, 6);
  });

  it('single-chunk pinch (near==far) yields t_blend = 1', () => {
    const rec: ChunkRecord = {
      centroid: [0, 0, 0],
      radius: 1,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const inputs = {
      ...baseInputs,
      nearCentroid: [0, 0, 0] as [number, number, number],
      farCentroid: [0, 0, 0] as [number, number, number],
    };
    expect(selectChunkActivation(rec, inputs).tBlend).toBeCloseTo(1.0, 6);
  });

  it('chunk radius slack pulls in the band edge', () => {
    // Camera at distance 21 from a chunk with radius 5 → dist_eff = 16.
    // Threshold 20 should NOT activate level 1; level 0 still picked.
    const rec: ChunkRecord = {
      centroid: [0, 0, 0],
      radius: 5,
      level: 0,
      chunkIndex: 0,
      splatCount: 100,
    };
    const inputs = {
      ...baseInputs,
      cameraPos: [21, 0, 0] as [number, number, number],
    };
    const act = selectChunkActivation(rec, inputs);
    expect(act.level).toBe(0);
    expect(act.active).toBe(1);
  });

  it('respects LOD_MAX_LEVELS cap', () => {
    expect(LOD_MAX_LEVELS).toBe(8);
  });
});

describe('pickNearFarChunks', () => {
  const manifest = parseLodgeManifest(JSON.stringify(FIXTURE_MANIFEST));

  it('returns the two closest chunks on a two-chunk level', () => {
    const { near, far } = pickNearFarChunks(manifest, 0, [-3, 0, 0]);
    expect(near.index).toBe(0); // centroid (-5, 0, 0) is closer to (-3, 0, 0)
    expect(far.index).toBe(1);
  });

  it('on single-chunk level returns near == far', () => {
    const { near, far } = pickNearFarChunks(manifest, 1, [0, 0, 0]);
    expect(near.index).toBe(far.index);
  });

  it('throws on missing level', () => {
    expect(() => pickNearFarChunks(manifest, 99, [0, 0, 0])).toThrow(/lodge_lod_no_chunks/);
  });
});
