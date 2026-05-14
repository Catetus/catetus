import { describe, expect, it } from 'vitest';
import { parseManifest } from '../manifest.js';

describe('parseManifest', () => {
  it('parses a glTF with SF_spatial_streaming_index', () => {
    const json = JSON.stringify({
      asset: { version: '2.0' },
      extensionsUsed: ['KHR_gaussian_splatting', 'SF_spatial_streaming_index'],
      extensions: {
        KHR_gaussian_splatting: {
          splatCount: 1000,
          shDegree: 2,
          bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
        },
        SF_spatial_streaming_index: {
          chunks: [
            {
              uri: 'tile_0.bin',
              byteOffset: 0,
              byteLength: 512,
              splatCount: 500,
              bbox: { min: [-1, -1, -1], max: [0, 1, 1] },
              lod: 0,
              checksum: '',
              loadPriority: 0,
            },
            {
              uri: 'tile_1.bin',
              byteOffset: 0,
              byteLength: 512,
              splatCount: 500,
              bbox: { min: [0, -1, -1], max: [1, 1, 1] },
              lod: 0,
              checksum: '',
              loadPriority: 1,
            },
          ],
        },
      },
    });

    const m = parseManifest(json);
    expect(m.splatCount).toBe(1000);
    expect(m.shDegree).toBe(2);
    expect(m.chunks).toHaveLength(2);
    expect(m.chunks[0]!.uri).toBe('tile_0.bin');
    expect(m.chunks[0]!.splatCount).toBe(500);
    expect(m.chunks[1]!.loadPriority).toBe(1);
    expect(m.bbox.min).toEqual([-1, -1, -1]);
  });

  it('falls back to a synthetic chunk when SF_spatial_streaming_index is absent', () => {
    const json = JSON.stringify({
      asset: { version: '2.0' },
      buffers: [{ uri: 'scene.bin', byteLength: 4096 }],
      extensionsUsed: ['KHR_gaussian_splatting'],
      extensions: {
        KHR_gaussian_splatting: { splatCount: 42, shDegree: 0 },
      },
    });
    const m = parseManifest(json);
    expect(m.chunks).toHaveLength(1);
    expect(m.chunks[0]!.uri).toBe('scene.bin');
    expect(m.chunks[0]!.byteLength).toBe(4096);
    expect(m.chunks[0]!.splatCount).toBe(42);
  });

  it('sorts chunks by loadPriority ascending', () => {
    const json = JSON.stringify({
      extensions: {
        KHR_gaussian_splatting: { splatCount: 0 },
        SF_spatial_streaming_index: {
          chunks: [
            { uri: 'c.bin', loadPriority: 5 },
            { uri: 'a.bin', loadPriority: 1 },
            { uri: 'b.bin', loadPriority: 3 },
          ],
        },
      },
    });
    const m = parseManifest(json);
    expect(m.chunks.map((c) => c.uri)).toEqual(['a.bin', 'b.bin', 'c.bin']);
  });

  it('throws manifest_invalid for non-JSON', () => {
    expect(() => parseManifest('not json')).toThrow(/manifest_invalid/);
  });

  it('throws manifest_invalid when KHR_gaussian_splatting is missing', () => {
    expect(() =>
      parseManifest(JSON.stringify({ asset: { version: '2.0' } })),
    ).toThrow(/manifest_invalid/);
  });
});
