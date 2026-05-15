import { describe, expect, it } from 'vitest';
import { screenSpaceError, selectVisibleTiles } from '../lod_selector.js';
import { extractFrustum } from '../frustum.js';
import { buildViewProj } from '../../renderer/math.js';
import { parseTileset } from '../tileset_loader.js';

function makeTileset(): ReturnType<typeof parseTileset> {
  // Two-level tileset: coarse root + one child.
  const json = JSON.stringify({
    asset: { version: '1.1' },
    geometricError: 10,
    root: {
      boundingVolume: { box: [0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1] },
      geometricError: 10,
      refine: 'REPLACE',
      content: { uri: 'root.glb' },
      children: [
        {
          boundingVolume: { box: [0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1] },
          geometricError: 0,
          content: { uri: 'child.glb' },
        },
      ],
    },
  });
  return parseTileset(json, 'http://example.com/tileset.json');
}

describe('screenSpaceError', () => {
  it('returns 0 when geometricError is 0', () => {
    expect(screenSpaceError(0, 10, Math.PI / 3, 512)).toBe(0);
  });
  it('grows linearly with geometricError', () => {
    const a = screenSpaceError(1, 10, Math.PI / 3, 512);
    const b = screenSpaceError(2, 10, Math.PI / 3, 512);
    expect(b / a).toBeCloseTo(2, 5);
  });
  it('shrinks linearly with distance', () => {
    const a = screenSpaceError(1, 10, Math.PI / 3, 512);
    const b = screenSpaceError(1, 20, Math.PI / 3, 512);
    expect(a / b).toBeCloseTo(2, 5);
  });
});

describe('selectVisibleTiles', () => {
  it('renders coarse root when SSE is below threshold', () => {
    const ts = makeTileset();
    // Far enough that even the root tile's geometric error projects to
    // well under 16 pixels on screen.
    const cam = { position: [0, 0, 2000], fovY: Math.PI / 3 } as const;
    const { viewProj } = buildViewProj(
      {
        position: [...cam.position] as [number, number, number],
        target: [0, 0, 0],
        up: [0, 1, 0],
        fovY: cam.fovY,
        aspect: 1,
        near: 0.1,
        far: 10000,
      },
      1,
    );
    const f = extractFrustum(viewProj);
    const sel = selectVisibleTiles(ts.root, {
      eye: cam.position as [number, number, number],
      fovY: cam.fovY,
      viewportHeight: 256,
      frustum: f,
      maximumScreenSpaceError: 16,
    });
    expect(sel.render.map((t) => t.id)).toEqual(['root']);
  });

  it('refines into the leaf when close', () => {
    const ts = makeTileset();
    const cam = { position: [0, 0, 2], fovY: Math.PI / 3 } as const;
    const { viewProj } = buildViewProj(
      {
        position: [...cam.position] as [number, number, number],
        target: [0, 0, 0],
        up: [0, 1, 0],
        fovY: cam.fovY,
        aspect: 1,
        near: 0.1,
        far: 1000,
      },
      1,
    );
    const f = extractFrustum(viewProj);
    const sel = selectVisibleTiles(ts.root, {
      eye: cam.position as [number, number, number],
      fovY: cam.fovY,
      viewportHeight: 1024,
      frustum: f,
      maximumScreenSpaceError: 16,
    });
    expect(sel.render.map((t) => t.id)).toEqual(['root/0']);
  });

  it('falls back to ancestor when child not resident', () => {
    const ts = makeTileset();
    const cam = { position: [0, 0, 2], fovY: Math.PI / 3 } as const;
    const { viewProj } = buildViewProj(
      {
        position: [...cam.position] as [number, number, number],
        target: [0, 0, 0],
        up: [0, 1, 0],
        fovY: cam.fovY,
        aspect: 1,
        near: 0.1,
        far: 1000,
      },
      1,
    );
    const f = extractFrustum(viewProj);
    // Only root is resident.
    const sel = selectVisibleTiles(ts.root, {
      eye: cam.position as [number, number, number],
      fovY: cam.fovY,
      viewportHeight: 1024,
      frustum: f,
      maximumScreenSpaceError: 16,
      resident: new Set(['root']),
    });
    expect(sel.render.map((t) => t.id)).toEqual(['root']);
    // And the leaf is queued for fetch.
    expect(sel.fetch.map((t) => t.id)).toContain('root/0');
  });
});
