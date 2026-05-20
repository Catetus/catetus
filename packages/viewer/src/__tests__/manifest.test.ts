import { describe, expect, it } from 'vitest';
import { parseManifest } from '../manifest.js';

describe('parseManifest', () => {
  it('parses a glTF with CT_spatial_streaming_index', () => {
    const json = JSON.stringify({
      asset: { version: '2.0' },
      extensionsUsed: ['KHR_gaussian_splatting', 'CT_spatial_streaming_index'],
      extensions: {
        KHR_gaussian_splatting: {
          splatCount: 1000,
          shDegree: 2,
          bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
        },
        CT_spatial_streaming_index: {
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

  it('falls back to a synthetic chunk when CT_spatial_streaming_index is absent', () => {
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
        CT_spatial_streaming_index: {
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

  /**
   * Dual-layout reader: the same set of accessor + bufferView + buffer
   * resources must produce the same `SoaAttributeLayout` whether the
   * primitive's attribute → accessor table is authored in the legacy
   * (pre-RC) shape — bare keys inside `prim.extensions.KHR_gaussian_splatting.attributes` —
   * or the RC shape — namespaced keys on `prim.attributes` next to `mode`.
   */
  describe('KHR_gaussian_splatting attribute layout auto-detection', () => {
    const sharedResources = {
      asset: { version: '2.0' },
      buffers: [{ uri: 'chunk_0.bin', byteLength: 56 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 12 },
        { buffer: 0, byteOffset: 12, byteLength: 16 },
        { buffer: 0, byteOffset: 28, byteLength: 12 },
        { buffer: 0, byteOffset: 40, byteLength: 4 },
        { buffer: 0, byteOffset: 44, byteLength: 12 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 1, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 1] },
        { bufferView: 1, componentType: 5126, count: 1, type: 'VEC4' },
        { bufferView: 2, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 3, componentType: 5126, count: 1, type: 'SCALAR' },
        { bufferView: 4, componentType: 5126, count: 1, type: 'VEC3' },
      ],
      extensionsUsed: ['KHR_gaussian_splatting'],
      extensions: {
        KHR_gaussian_splatting: {
          splatCount: 1,
          shDegree: 0,
          bbox: { min: [0, 0, 0], max: [1, 1, 1] },
        },
      },
    } as const;

    const legacy = {
      ...sharedResources,
      meshes: [
        {
          primitives: [
            {
              mode: 0,
              extensions: {
                KHR_gaussian_splatting: {
                  attributes: {
                    POSITION: 0,
                    _ROTATION: 1,
                    _SCALE: 2,
                    _OPACITY: 3,
                    _COLOR_DC: 4,
                  },
                },
              },
            },
          ],
        },
      ],
    };

    const rc = {
      ...sharedResources,
      meshes: [
        {
          primitives: [
            {
              mode: 0,
              attributes: {
                'KHR_gaussian_splatting:POSITION': 0,
                'KHR_gaussian_splatting:ROTATION': 1,
                'KHR_gaussian_splatting:SCALE': 2,
                'KHR_gaussian_splatting:OPACITY': 3,
                'KHR_gaussian_splatting:COLOR_DC': 4,
              },
              extensions: { KHR_gaussian_splatting: {} },
            },
          ],
        },
      ],
    };

    it('legacy layout decodes attribute layout', () => {
      const m = parseManifest(JSON.stringify(legacy));
      expect(m.chunks).toHaveLength(1);
      const layout = m.chunks[0]!.attributeLayout;
      expect(layout).toBeDefined();
      expect(layout!.positions.byteOffset).toBe(0);
      expect(layout!.rotations.byteOffset).toBe(12);
      expect(layout!.scales.byteOffset).toBe(28);
      expect(layout!.opacities.byteOffset).toBe(40);
      expect(layout!.colorDC.byteOffset).toBe(44);
    });

    it('RC layout decodes attribute layout', () => {
      const m = parseManifest(JSON.stringify(rc));
      expect(m.chunks).toHaveLength(1);
      const layout = m.chunks[0]!.attributeLayout;
      expect(layout).toBeDefined();
      expect(layout!.positions.byteOffset).toBe(0);
      expect(layout!.rotations.byteOffset).toBe(12);
      expect(layout!.scales.byteOffset).toBe(28);
      expect(layout!.opacities.byteOffset).toBe(40);
      expect(layout!.colorDC.byteOffset).toBe(44);
    });

    it('both layouts produce byte-equal SoaAttributeLayout', () => {
      const a = parseManifest(JSON.stringify(legacy));
      const b = parseManifest(JSON.stringify(rc));
      expect(b.splatCount).toBe(a.splatCount);
      expect(b.shDegree).toBe(a.shDegree);
      expect(b.bbox).toEqual(a.bbox);
      expect(b.chunks[0]!.attributeLayout).toEqual(a.chunks[0]!.attributeLayout);
    });

    it('RC takes precedence when both shapes are present (forward-compat)', () => {
      const mixed = {
        ...sharedResources,
        meshes: [
          {
            primitives: [
              {
                mode: 0,
                attributes: {
                  'KHR_gaussian_splatting:POSITION': 0,
                  'KHR_gaussian_splatting:ROTATION': 1,
                  'KHR_gaussian_splatting:SCALE': 2,
                  'KHR_gaussian_splatting:OPACITY': 3,
                  'KHR_gaussian_splatting:COLOR_DC': 4,
                },
                extensions: {
                  KHR_gaussian_splatting: {
                    // Intentional bogus accessor indices — the reader should
                    // prefer the RC primitive-level table over this.
                    attributes: {
                      POSITION: 99,
                      _ROTATION: 99,
                      _SCALE: 99,
                      _OPACITY: 99,
                      _COLOR_DC: 99,
                    },
                  },
                },
              },
            ],
          },
        ],
      };
      const m = parseManifest(JSON.stringify(mixed));
      const layout = m.chunks[0]!.attributeLayout;
      // If the legacy table had won, byteOffsets would be undefined (no
      // accessor 99 exists) and `attributeLayout` would be undefined.
      expect(layout).toBeDefined();
      expect(layout!.rotations.byteOffset).toBe(12);
    });
  });
});
