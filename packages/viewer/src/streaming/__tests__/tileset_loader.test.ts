import { describe, expect, it } from 'vitest';
import { parseTileset, resolveAgainst } from '../tileset_loader.js';

const SAMPLE = JSON.stringify({
  asset: { version: '1.1' },
  geometricError: 5,
  extensionsRequired: ['KHR_gaussian_splatting'],
  root: {
    boundingVolume: { box: [0, 0, 0,  2, 0, 0,  0, 2, 0,  0, 0, 2] },
    geometricError: 5,
    refine: 'REPLACE',
    content: { uri: 'lod3.glb' },
    children: [
      {
        boundingVolume: { box: [0, 0, 0,  2, 0, 0,  0, 2, 0,  0, 0, 2] },
        geometricError: 2.5,
        refine: 'REPLACE',
        content: { uri: 'lod2.glb' },
        children: [
          {
            boundingVolume: { box: [0, 0, 0,  2, 0, 0,  0, 2, 0,  0, 0, 2] },
            geometricError: 0,
            content: { uri: 'lod1.glb' },
          },
        ],
      },
    ],
  },
});

describe('parseTileset', () => {
  it('builds a tile tree with stable ids', () => {
    const t = parseTileset(SAMPLE, 'http://x/tileset.json');
    expect(t.root.id).toBe('root');
    expect(t.root.children[0]!.id).toBe('root/0');
    expect(t.root.children[0]!.children[0]!.id).toBe('root/0/0');
    expect(t.tiles).toHaveLength(3);
  });

  it('resolves content URIs against the tileset URL', () => {
    const t = parseTileset(SAMPLE, 'http://x/path/tileset.json');
    expect(t.root.contentUrl).toBe('http://x/path/lod3.glb');
  });

  it('carries parent pointers + depth', () => {
    const t = parseTileset(SAMPLE, 'http://x/tileset.json');
    const leaf = t.root.children[0]!.children[0]!;
    expect(leaf.depth).toBe(2);
    expect(leaf.parent?.id).toBe('root/0');
  });

  it('rejects malformed JSON', () => {
    expect(() => parseTileset('not json', 'x')).toThrow(/tileset_invalid/);
  });

  it('rejects a missing root', () => {
    expect(() => parseTileset('{}', 'x')).toThrow(/tileset_invalid/);
  });
});

describe('resolveAgainst', () => {
  it('treats absolute base directly', () => {
    expect(resolveAgainst('http://x/a/b.json', 'c.glb')).toBe('http://x/a/c.glb');
  });
});
