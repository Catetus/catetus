/**
 * KHR_gaussian_splatting attribute-layout parity test.
 *
 * Builds two glTF documents that point at the same shared binary buffer
 * (identical bytes, identical accessors, identical bufferViews) — the only
 * difference is how the attribute → accessor index table is authored:
 *
 *  - Legacy: bare keys (`POSITION`, `_ROTATION`, ...) inside
 *    `prim.extensions.KHR_gaussian_splatting.attributes`.
 *
 *  - Release Candidate (RC): namespaced keys
 *    (`KHR_gaussian_splatting:POSITION`, `KHR_gaussian_splatting:ROTATION`, ...)
 *    on `prim.attributes` next to `mode`.
 *
 * The fully decoded `DecodedSplat[]` arrays must be byte-identical. This
 * guards against the dual-layout reader silently picking up different
 * accessors for one authoring style.
 */
import { describe, expect, it } from 'vitest';
import { parseManifest } from '../manifest.js';
import { decodeChunkBytes } from '../renderer/base.js';

describe('attribute-layout parity (legacy vs RC)', () => {
  // 2 splats, 56 bytes per splat. All values are deterministic and chosen so
  // re-quantization is a no-op (FLOAT components, no normalization).
  const N = 2;
  const stride = 56;
  const bin = new Uint8Array(N * stride);
  const view = new DataView(bin.buffer);
  const posOff = 0;
  const rotOff = N * 12;
  const sclOff = rotOff + N * 16;
  const opOff = sclOff + N * 12;
  const dcOff = opOff + N * 4;
  for (let i = 0; i < N; i++) {
    view.setFloat32(posOff + i * 12 + 0, 0.5 + i, true);
    view.setFloat32(posOff + i * 12 + 4, 1.5 + i, true);
    view.setFloat32(posOff + i * 12 + 8, 2.5 + i, true);
    view.setFloat32(rotOff + i * 16 + 0, 0.1 * (i + 1), true);
    view.setFloat32(rotOff + i * 16 + 4, 0.2 * (i + 1), true);
    view.setFloat32(rotOff + i * 16 + 8, 0.3 * (i + 1), true);
    view.setFloat32(rotOff + i * 16 + 12, 0.4 * (i + 1), true);
    view.setFloat32(sclOff + i * 12 + 0, 0.01 + i * 0.01, true);
    view.setFloat32(sclOff + i * 12 + 4, 0.02 + i * 0.01, true);
    view.setFloat32(sclOff + i * 12 + 8, 0.03 + i * 0.01, true);
    view.setFloat32(opOff + i * 4, 0.7 + i * 0.05, true);
    view.setFloat32(dcOff + i * 12 + 0, 0.9 - i * 0.1, true);
    view.setFloat32(dcOff + i * 12 + 4, 0.8 - i * 0.1, true);
    view.setFloat32(dcOff + i * 12 + 8, 0.7 - i * 0.1, true);
  }

  const sharedResources = {
    asset: { version: '2.0' },
    buffers: [{ uri: 'chunk_0.bin', byteLength: bin.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: posOff, byteLength: N * 12 },
      { buffer: 0, byteOffset: rotOff, byteLength: N * 16 },
      { buffer: 0, byteOffset: sclOff, byteLength: N * 12 },
      { buffer: 0, byteOffset: opOff, byteLength: N * 4 },
      { buffer: 0, byteOffset: dcOff, byteLength: N * 12 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: N, type: 'VEC3', min: [0.5, 1.5, 2.5], max: [1.5, 2.5, 3.5] },
      { bufferView: 1, componentType: 5126, count: N, type: 'VEC4' },
      { bufferView: 2, componentType: 5126, count: N, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: N, type: 'SCALAR' },
      { bufferView: 4, componentType: 5126, count: N, type: 'VEC3' },
    ],
    extensionsUsed: ['KHR_gaussian_splatting'],
    extensions: {
      KHR_gaussian_splatting: {
        splatCount: N,
        shDegree: 0,
        bbox: { min: [0.5, 1.5, 2.5], max: [1.5, 2.5, 3.5] },
      },
    },
  } as const;

  const legacyJson = JSON.stringify({
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
  });

  const rcJson = JSON.stringify({
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
  });

  it('decoded splats are byte-identical across both authoring layouts', () => {
    const legacyManifest = parseManifest(legacyJson);
    const rcManifest = parseManifest(rcJson);

    const legacyDecoded = decodeChunkBytes(bin, legacyManifest.chunks[0]!);
    const rcDecoded = decodeChunkBytes(bin, rcManifest.chunks[0]!);

    expect(rcDecoded).toHaveLength(N);
    expect(legacyDecoded).toHaveLength(N);
    expect(rcDecoded).toEqual(legacyDecoded);
  });

  it('manifest header fields parse identically', () => {
    const a = parseManifest(legacyJson);
    const b = parseManifest(rcJson);
    expect(b.splatCount).toBe(a.splatCount);
    expect(b.shDegree).toBe(a.shDegree);
    expect(b.bbox).toEqual(a.bbox);
    expect(b.chunks).toHaveLength(a.chunks.length);
  });
});
