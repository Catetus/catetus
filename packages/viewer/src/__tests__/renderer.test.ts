import { describe, expect, it } from 'vitest';
import { WebGL2Renderer } from '../renderer/webgl2.js';
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor } from '../manifest.js';

interface MockGlState {
  calls: string[];
  drawArgs: number[][];
}

/**
 * Returns a Proxy that masquerades as a `WebGL2RenderingContext`. Every
 * accessed property is treated as a callable that records the call. Numeric
 * constants get a unique synthetic id so any code path that branches on
 * `gl.SOMETHING === gl.OTHER` keeps a sane shape.
 */
function makeMockGl(): { gl: WebGL2RenderingContext; state: MockGlState } {
  const state: MockGlState = { calls: [], drawArgs: [] };
  // Cache "constants" so equality comparisons remain consistent.
  const constants: Record<string, number> = {};
  let next = 1;
  const opaqueObject = (): Record<string, never> => ({});
  // Use `unknown` then assert; an actual context isn't needed for the smoke test.
  const target: Record<string, unknown> = {
    drawingBufferWidth: 256,
    drawingBufferHeight: 128,
  };
  const handler: ProxyHandler<Record<string, unknown>> = {
    get(t, prop): unknown {
      const key = String(prop);
      if (key in t) return t[key];
      // Functions that need to return objects.
      if (key === 'createProgram' || key === 'createShader' || key === 'createBuffer' || key === 'createVertexArray' || key === 'createTexture' || key === 'getUniformLocation') {
        return (..._args: unknown[]): Record<string, never> => {
          state.calls.push(key);
          return opaqueObject();
        };
      }
      // Functions that should report success.
      if (key === 'getShaderParameter' || key === 'getProgramParameter') {
        return (..._args: unknown[]): boolean => {
          state.calls.push(key);
          return true;
        };
      }
      if (key === 'drawArraysInstanced') {
        return (..._args: number[]): void => {
          state.calls.push(key);
          state.drawArgs.push([..._args]);
        };
      }
      if (typeof key === 'string' && /^[A-Z0-9_]+$/.test(key)) {
        if (!(key in constants)) constants[key] = next++;
        return constants[key];
      }
      return (..._args: unknown[]): number => {
        state.calls.push(key);
        return 0;
      };
    },
  };
  const gl = new Proxy(target, handler) as unknown as WebGL2RenderingContext;
  return { gl, state };
}

/** Build a Float32Array buffer with N splats laid out for `decodeSplats`. */
function makeSplatBytes(count: number): Uint8Array {
  const floats = new Float32Array(count * 14);
  for (let i = 0; i < count; i++) {
    const o = i * 14;
    floats[o + 0] = i * 0.1; // px
    floats[o + 1] = 0;
    floats[o + 2] = -2;
    floats[o + 3] = 0.05; // sx
    floats[o + 4] = 0.05;
    floats[o + 5] = 0.05;
    floats[o + 6] = 0; // qx
    floats[o + 7] = 0;
    floats[o + 8] = 0;
    floats[o + 9] = 1; // qw
    floats[o + 10] = 0.8; // opacity
    floats[o + 11] = 0.9; // r
    floats[o + 12] = 0.5; // g
    floats[o + 13] = 0.3; // b
  }
  return new Uint8Array(floats.buffer);
}

const CAMERA: CameraPose = {
  position: [0, 0, 5],
  target: [0, 0, 0],
  up: [0, 1, 0],
  fovY: Math.PI / 3,
  aspect: 2,
  near: 0.1,
  far: 100,
};

const CHUNK: ChunkDescriptor = {
  uri: 'tile_0.bin',
  byteOffset: 0,
  byteLength: 0,
  splatCount: 3,
  bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
  lod: 0,
  checksum: '',
  loadPriority: 0,
};

describe('WebGL2Renderer (mock smoke test)', () => {
  it('issues exactly one drawArraysInstanced per frame with the right instance count', async () => {
    const { gl, state } = makeMockGl();
    // Cast to `any` to satisfy the canvas API surface; the renderer only
    // touches `getContext`.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvas = { getContext: (): WebGL2RenderingContext => gl, width: 256, height: 128 } as any;
    const r = new WebGL2Renderer();
    await r.init({ canvas });
    r.uploadChunk(CHUNK, makeSplatBytes(3));
    await r.renderFrame(CAMERA);

    const draws = state.calls.filter((c) => c === 'drawArraysInstanced');
    expect(draws.length).toBe(1);
    const args = state.drawArgs[0]!;
    // drawArraysInstanced(mode, first, count, instanceCount)
    expect(args[2]).toBe(4);
    expect(args[3]).toBe(3);
    expect(r.drawCallCount).toBe(1);
  });

  it('records zero draws when no splats have been uploaded', async () => {
    const { gl, state } = makeMockGl();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvas = { getContext: (): WebGL2RenderingContext => gl, width: 256, height: 128 } as any;
    const r = new WebGL2Renderer();
    await r.init({ canvas });
    await r.renderFrame(CAMERA);
    expect(state.calls.filter((c) => c === 'drawArraysInstanced').length).toBe(0);
    expect(r.drawCallCount).toBe(0);
  });
});
