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

describe('parseManifest — multi-primitive multi-buffer', () => {
  // Regression for the SH-1 branch bug: the encoder emits one primitive per
  // chunk (each pointing at its own buffer with chunk-local accessor indices).
  // The parser previously extracted attributes from only the first primitive,
  // so chunks 1..N would fail the bufferIdx match in `normalizeChunk` and
  // fall through to the legacy 14-float interleaved AoS path, decoding SoA
  // bytes as garbage AoS.
  it('uses per-primitive attribute tables for chunks in different buffers', () => {
    // Helper: build accessor indices for one chunk worth of SoA attributes
    // (POSITION, ROTATION, SCALE, OPACITY, COLOR_DC) all pointing at the same
    // bufferIdx through five sequential bufferViews.
    const buildChunk = (bufferIdx: number, splatCount: number, startBvIdx: number) => {
      const bytesPerSplat = { pos: 12, rot: 16, scl: 12, op: 4, dc: 12 };
      const bvs = [
        { buffer: bufferIdx, byteOffset: 0, byteLength: bytesPerSplat.pos * splatCount },
        { buffer: bufferIdx, byteOffset: bytesPerSplat.pos * splatCount, byteLength: bytesPerSplat.rot * splatCount },
        { buffer: bufferIdx, byteOffset: (bytesPerSplat.pos + bytesPerSplat.rot) * splatCount, byteLength: bytesPerSplat.scl * splatCount },
        { buffer: bufferIdx, byteOffset: (bytesPerSplat.pos + bytesPerSplat.rot + bytesPerSplat.scl) * splatCount, byteLength: bytesPerSplat.op * splatCount },
        { buffer: bufferIdx, byteOffset: (bytesPerSplat.pos + bytesPerSplat.rot + bytesPerSplat.scl + bytesPerSplat.op) * splatCount, byteLength: bytesPerSplat.dc * splatCount },
      ];
      const accessors = [
        { bufferView: startBvIdx + 0, componentType: 5126, count: splatCount, type: 'VEC3' },
        { bufferView: startBvIdx + 1, componentType: 5126, count: splatCount, type: 'VEC4' },
        { bufferView: startBvIdx + 2, componentType: 5126, count: splatCount, type: 'VEC3' },
        { bufferView: startBvIdx + 3, componentType: 5126, count: splatCount, type: 'SCALAR' },
        { bufferView: startBvIdx + 4, componentType: 5126, count: splatCount, type: 'VEC3' },
      ];
      const primitive = {
        mode: 0,
        attributes: {
          // POSITION is bare (core glTF 2.0 attribute); the rest carry the
          // KHR_gaussian_splatting namespace. Mirrors the Rust encoder
          // (`crates/splatforge-gltf/src/lib.rs`, `attr_name`).
          POSITION: startBvIdx + 0,
          'KHR_gaussian_splatting:ROTATION': startBvIdx + 1,
          'KHR_gaussian_splatting:SCALE': startBvIdx + 2,
          'KHR_gaussian_splatting:OPACITY': startBvIdx + 3,
          'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': startBvIdx + 4,
        },
        extensions: {
          KHR_gaussian_splatting: { shDegree: 0 },
        },
      };
      return { bvs, accessors, primitive };
    };

    const chunkA = buildChunk(0, 100, 0);
    const chunkB = buildChunk(1, 50, 5);
    const chunkC = buildChunk(2, 25, 10);

    const json = JSON.stringify({
      asset: { version: '2.0' },
      buffers: [
        { byteLength: 5600000, uri: 'buffers/chunk_0000.bin' },
        { byteLength: 2800000, uri: 'buffers/chunk_0001.bin' },
        { byteLength: 1400000, uri: 'buffers/chunk_0002.bin' },
      ],
      bufferViews: [...chunkA.bvs, ...chunkB.bvs, ...chunkC.bvs],
      accessors: [...chunkA.accessors, ...chunkB.accessors, ...chunkC.accessors],
      meshes: [{ primitives: [chunkA.primitive, chunkB.primitive, chunkC.primitive] }],
      extensionsUsed: ['KHR_gaussian_splatting', 'SF_spatial_streaming_index'],
      extensions: {
        KHR_gaussian_splatting: {
          splatCount: 175,
          shDegree: 0,
          bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
        },
        SF_spatial_streaming_index: {
          chunks: [
            { uri: 'buffers/chunk_0000.bin', buffer: 0, byteOffset: 0, byteLength: 5600000, splatCount: 100, bbox: { min: [-1, -1, -1], max: [1, 1, 1] }, lod: 0, checksum: '', loadPriority: 0 },
            { uri: 'buffers/chunk_0001.bin', buffer: 1, byteOffset: 0, byteLength: 2800000, splatCount: 50, bbox: { min: [-1, -1, -1], max: [1, 1, 1] }, lod: 0, checksum: '', loadPriority: 1 },
            { uri: 'buffers/chunk_0002.bin', buffer: 2, byteOffset: 0, byteLength: 1400000, splatCount: 25, bbox: { min: [-1, -1, -1], max: [1, 1, 1] }, lod: 0, checksum: '', loadPriority: 2 },
          ],
        },
      },
    });

    const m = parseManifest(json);
    expect(m.chunks).toHaveLength(3);

    // Every chunk must get a SoA attributeLayout. The bug caused chunks 1 and 2
    // to come back with `attributeLayout === undefined`, falling through to the
    // legacy 14-float AoS decoder and producing viewport-filling garbage.
    for (let i = 0; i < 3; i++) {
      const layout = m.chunks[i]!.attributeLayout;
      expect(layout, `chunk[${i}] attributeLayout`).toBeDefined();
      const splatCount = m.chunks[i]!.splatCount;
      expect(layout!.positions.byteLength).toBe(12 * splatCount);
      expect(layout!.rotations.byteLength).toBe(16 * splatCount);
      expect(layout!.scales.byteLength).toBe(12 * splatCount);
      expect(layout!.opacities.byteLength).toBe(4 * splatCount);
      expect(layout!.colorDC.byteLength).toBe(12 * splatCount);
      // Chunk-relative offsets: POSITION sits at the start of every chunk.
      expect(layout!.positions.byteOffset).toBe(0);
    }
  });
});
