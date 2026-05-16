import { describe, expect, it } from 'vitest';
import {
  SH_C1,
  decodeChunkBytes,
  evaluateSh1,
  type DecodedSplat,
} from '../renderer/base.js';
import type { ChunkDescriptor, SoaAttributeLayout } from '../manifest.js';
import { parseManifest } from '../manifest.js';
import { WebGL2Renderer } from '../renderer/webgl2.js';
import type { CameraPose } from '../camera.js';

/**
 * Build a one-splat SoA binary chunk:
 *
 *   POSITION         vec3 float  → 12 bytes
 *   ROTATION         vec4 float  → 16 bytes
 *   SCALE            vec3 float  → 12 bytes
 *   OPACITY          scalar float → 4 bytes
 *   COLOR_DC         vec3 float  → 12 bytes
 *   SH_DEGREE_1_COEF_0 vec3 float → 12 bytes
 *   SH_DEGREE_1_COEF_1 vec3 float → 12 bytes
 *   SH_DEGREE_1_COEF_2 vec3 float → 12 bytes
 *                                  ──────────
 *                                  92 bytes total
 *
 * Returns the byte buffer plus the matching `SoaAttributeLayout`.
 */
function makeSh1Splat(opts: {
  position: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  opacity: number;
  dc: [number, number, number];
  sh1: number[]; // length 9 — [c0.r,c0.g,c0.b, c1.r,c1.g,c1.b, c2.r,c2.g,c2.b]
}): { bytes: Uint8Array; layout: SoaAttributeLayout } {
  const buf = new ArrayBuffer(92);
  const dv = new DataView(buf);
  let off = 0;
  for (const v of opts.position) {
    dv.setFloat32(off, v, true);
    off += 4;
  }
  const posLen = off; // 12
  for (const v of opts.rotation) {
    dv.setFloat32(off, v, true);
    off += 4;
  }
  const rotEnd = off; // 28
  for (const v of opts.scale) {
    dv.setFloat32(off, v, true);
    off += 4;
  }
  const sclEnd = off; // 40
  dv.setFloat32(off, opts.opacity, true);
  off += 4;
  const opEnd = off; // 44
  for (const v of opts.dc) {
    dv.setFloat32(off, v, true);
    off += 4;
  }
  const dcEnd = off; // 56
  // Three vec3 SH coefs.
  const sh1Starts: number[] = [];
  for (let c = 0; c < 3; c++) {
    sh1Starts.push(off);
    for (let k = 0; k < 3; k++) {
      dv.setFloat32(off, opts.sh1[c * 3 + k]!, true);
      off += 4;
    }
  }
  expect(off).toBe(92);

  const layout: SoaAttributeLayout = {
    positions: { byteOffset: 0, byteLength: posLen, componentType: 5126 },
    rotations: { byteOffset: posLen, byteLength: rotEnd - posLen, componentType: 5126 },
    scales: { byteOffset: rotEnd, byteLength: sclEnd - rotEnd, componentType: 5126 },
    opacities: { byteOffset: sclEnd, byteLength: opEnd - sclEnd, componentType: 5126 },
    colorDC: { byteOffset: opEnd, byteLength: dcEnd - opEnd, componentType: 5126 },
    sh1Coef0: { byteOffset: sh1Starts[0]!, byteLength: 12, componentType: 5126 },
    sh1Coef1: { byteOffset: sh1Starts[1]!, byteLength: 12, componentType: 5126 },
    sh1Coef2: { byteOffset: sh1Starts[2]!, byteLength: 12, componentType: 5126 },
  };
  return { bytes: new Uint8Array(buf), layout };
}

describe('SH degree-1 evaluation', () => {
  it('SH_C1 matches 3DGS reference (0.4886025...)', () => {
    expect(SH_C1).toBeCloseTo(0.4886025119029199, 12);
  });

  it('evaluateSh1 returns zero for an all-zero coef vector', () => {
    const z = new Float32Array(9);
    const [r, g, b] = evaluateSh1(z, 0.5, 0.5, Math.SQRT1_2);
    expect(r).toBe(0);
    expect(g).toBe(0);
    expect(b).toBe(0);
  });

  it('evaluateSh1 produces different RGB for opposite view directions', () => {
    // c1 (paired with dirZ) carries non-zero contributions on all three
    // channels so a Z-flip swaps the sign in every channel.
    const sh1 = new Float32Array([
      0, 0, 0,         // c0 (Y_1^-1, dirY)
      0.5, 0.6, 0.7,   // c1 (Y_1^0,  dirZ)
      0, 0, 0,         // c2 (Y_1^1,  dirX)
    ]);
    const front = evaluateSh1(sh1, 0, 0, 1);
    const back = evaluateSh1(sh1, 0, 0, -1);
    // Z-channel should swap sign when dirZ flips.
    expect(front[1]).toBeCloseTo(SH_C1 * 0.6, 6);
    expect(back[1]).toBeCloseTo(-SH_C1 * 0.6, 6);
    // All three channels differ on a Z-flip.
    expect(front[0]).not.toBe(back[0]);
    expect(front[1]).not.toBe(back[1]);
    expect(front[2]).not.toBe(back[2]);
  });
});

describe('SH degree-1 decode pipeline', () => {
  it('decodeChunkBytes attaches sh1 floats when layout carries SH-1 coefs', () => {
    const sh1 = [
      0.1, 0.2, 0.3,
      0.4, 0.5, 0.6,
      0.7, 0.8, 0.9,
    ];
    const { bytes, layout } = makeSh1Splat({
      position: [0, 0, 0],
      rotation: [0, 0, 0, 1],
      scale: [0.1, 0.1, 0.1],
      opacity: 0.9,
      dc: [0.4, 0.5, 0.6],
      sh1,
    });
    const descriptor: ChunkDescriptor = {
      uri: 't.bin',
      byteOffset: 0,
      byteLength: bytes.byteLength,
      splatCount: 1,
      bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
      lod: 0,
      checksum: '',
      loadPriority: 0,
      attributeLayout: layout,
    };
    const splats = decodeChunkBytes(bytes, descriptor);
    expect(splats.length).toBe(1);
    const s = splats[0]!;
    expect(s.sh1).toBeDefined();
    // Float32 round-trip — compare with tolerance, not bit-exact.
    const got = Array.from(s.sh1!);
    expect(got.length).toBe(9);
    for (let i = 0; i < 9; i++) {
      expect(got[i]!).toBeCloseTo(sh1[i]!, 6);
    }
    expect(s.colorDC[0]!).toBeCloseTo(0.4, 6);
    expect(s.colorDC[1]!).toBeCloseTo(0.5, 6);
    expect(s.colorDC[2]!).toBeCloseTo(0.6, 6);
  });

  it('parseManifest extracts SH_DEGREE_1_COEF_{0,1,2} accessor indices', () => {
    const gltf = {
      asset: { version: '2.0' },
      extensionsUsed: ['KHR_gaussian_splatting'],
      buffers: [{ uri: 'tile.bin', byteLength: 92 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 12 },
        { buffer: 0, byteOffset: 12, byteLength: 16 },
        { buffer: 0, byteOffset: 28, byteLength: 12 },
        { buffer: 0, byteOffset: 40, byteLength: 4 },
        { buffer: 0, byteOffset: 44, byteLength: 12 },
        { buffer: 0, byteOffset: 56, byteLength: 12 },
        { buffer: 0, byteOffset: 68, byteLength: 12 },
        { buffer: 0, byteOffset: 80, byteLength: 12 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 1, type: 'VEC4' },
        { bufferView: 2, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 3, componentType: 5126, count: 1, type: 'SCALAR' },
        { bufferView: 4, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 5, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 6, componentType: 5126, count: 1, type: 'VEC3' },
        { bufferView: 7, componentType: 5126, count: 1, type: 'VEC3' },
      ],
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
                'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 4,
                'KHR_gaussian_splatting:SH_DEGREE_1_COEF_0': 5,
                'KHR_gaussian_splatting:SH_DEGREE_1_COEF_1': 6,
                'KHR_gaussian_splatting:SH_DEGREE_1_COEF_2': 7,
              },
              extensions: { KHR_gaussian_splatting: {} },
            },
          ],
        },
      ],
      extensions: {
        KHR_gaussian_splatting: {
          splatCount: 1,
          shDegree: 1,
          bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
        },
      },
    };
    const m = parseManifest(JSON.stringify(gltf));
    expect(m.shDegree).toBe(1);
    expect(m.chunks.length).toBe(1);
    const layout = m.chunks[0]!.attributeLayout!;
    expect(layout.colorDC.byteOffset).toBe(44);
    expect(layout.sh1Coef0?.byteOffset).toBe(56);
    expect(layout.sh1Coef1?.byteOffset).toBe(68);
    expect(layout.sh1Coef2?.byteOffset).toBe(80);
  });
});

/**
 * Headless visual-difference check: feed the WebGL2 renderer a one-splat
 * SH-degree-1 chunk, render from two opposite camera positions, capture the
 * per-instance RGB the renderer wrote into its instance VBO, and assert the
 * two RGB triples differ — i.e. the view direction actually drives color.
 */
describe('SH degree-1 view dependence (WebGL2 path)', () => {
  it('per-instance RGB differs across two opposite view directions', async () => {
    const sh1 = [
      0, 0, 0,        // c0
      0.7, -0.7, 0.7, // c1 — paired with dirZ, picks up the Z flip
      0, 0, 0,        // c2
    ];
    const { bytes, layout } = makeSh1Splat({
      position: [0, 0, 0],
      rotation: [0, 0, 0, 1],
      scale: [0.5, 0.5, 0.5],
      opacity: 0.9,
      dc: [0.5, 0.5, 0.5],
      sh1,
    });
    const descriptor: ChunkDescriptor = {
      uri: 't.bin',
      byteOffset: 0,
      byteLength: bytes.byteLength,
      splatCount: 1,
      bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
      lod: 0,
      checksum: '',
      loadPriority: 0,
      attributeLayout: layout,
    };

    // Capture the Float32Array uploaded via `bufferSubData`. The mock GL just
    // intercepts the calls we care about; everything else is a no-op.
    const captures: Float32Array[] = [];
    const { gl } = makeCapturingGl(captures);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvas = { getContext: (): WebGL2RenderingContext => gl, width: 64, height: 64 } as any;
    const r = new WebGL2Renderer();
    await r.init({ canvas });
    r.uploadChunk(descriptor, bytes);

    const camFront: CameraPose = {
      position: [0, 0, 3],
      target: [0, 0, 0],
      up: [0, 1, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 100,
    };
    const camBack: CameraPose = { ...camFront, position: [0, 0, -3] };

    await r.renderFrame(camFront);
    await r.renderFrame(camBack);

    // Two frames → two upload captures.
    expect(captures.length).toBe(2);
    const FLOATS_PER_INSTANCE = 12;
    const rgbOffset = 8;
    const front = [
      captures[0]![rgbOffset]!,
      captures[0]![rgbOffset + 1]!,
      captures[0]![rgbOffset + 2]!,
    ];
    const back = [
      captures[1]![rgbOffset]!,
      captures[1]![rgbOffset + 1]!,
      captures[1]![rgbOffset + 2]!,
    ];
    expect(front[0]).not.toBeCloseTo(back[0]!, 5);
    expect(front[1]).not.toBeCloseTo(back[1]!, 5);
    expect(front[2]).not.toBeCloseTo(back[2]!, 5);
    // Sanity: pre-SH-eval base color is 0.5,0.5,0.5; SH-1 has c1 only with
    // dir = (0,0,+1) for camFront and (0,0,-1) for camBack, so the
    // contributions are equal magnitude with opposite signs.
    expect(front[0]! + back[0]!).toBeCloseTo(1.0, 4);
    expect(front[1]! + back[1]!).toBeCloseTo(1.0, 4);
    expect(front[2]! + back[2]!).toBeCloseTo(1.0, 4);
    void FLOATS_PER_INSTANCE;
  });
});

/**
 * Mock WebGL2RenderingContext that snapshots every Float32Array passed into
 * `bufferSubData`. Other entry points are no-ops returning sane defaults.
 */
function makeCapturingGl(captures: Float32Array[]): { gl: WebGL2RenderingContext } {
  const constants: Record<string, number> = {};
  let next = 1;
  const target: Record<string, unknown> = {
    drawingBufferWidth: 64,
    drawingBufferHeight: 64,
  };
  const handler: ProxyHandler<Record<string, unknown>> = {
    get(t, prop): unknown {
      const key = String(prop);
      if (key in t) return t[key];
      if (
        key === 'createProgram' ||
        key === 'createShader' ||
        key === 'createBuffer' ||
        key === 'createVertexArray' ||
        key === 'createTexture' ||
        key === 'getUniformLocation'
      ) {
        return (..._args: unknown[]): Record<string, never> => ({});
      }
      if (key === 'getShaderParameter' || key === 'getProgramParameter') {
        return (..._args: unknown[]): boolean => true;
      }
      if (key === 'bufferSubData') {
        return (_target: number, _offset: number, data: ArrayBufferView): void => {
          if (data instanceof Float32Array) {
            // Copy — the renderer reuses the staging buffer across frames.
            captures.push(new Float32Array(data));
          }
        };
      }
      if (typeof key === 'string' && /^[A-Z0-9_]+$/.test(key)) {
        if (!(key in constants)) constants[key] = next++;
        return constants[key];
      }
      return (..._args: unknown[]): number => 0;
    },
  };
  const gl = new Proxy(target, handler) as unknown as WebGL2RenderingContext;
  return { gl };
}

// Suppress an unused-import diagnostic if these types stay unused in some
// future refactor — they're referenced only via JSDoc otherwise.
type _Used = DecodedSplat;
