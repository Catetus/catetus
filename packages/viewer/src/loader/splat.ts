// SPDX-License-Identifier: Apache-2.0
/**
 * antimatter15 .splat loader.
 *
 * Format (https://github.com/antimatter15/splat) — 32 bytes per splat,
 * tightly packed:
 *
 *   [0..12)  position xyz, float32 LE
 *   [12..24) scale xyz   , float32 LE (LINEAR scale, not log-scale)
 *   [24..28) color rgba  , uint8 (0..255 sRGB-ish, alpha = opacity)
 *   [28..32) quaternion  , uint8 packed: q = (b - 128) / 128
 *
 * Quaternion order in .splat is (w, x, y, z) by convention. We carry XYZW
 * internally, so we reorder on load.
 *
 * Scales are LINEAR in .splat (already `exp`d). We convert back to log here
 * so the SplatScene contract (`scales` is log-space, consumer applies `exp`)
 * holds.
 *
 * `.splat` carries no SH-rest. To keep `dcRaw` always-populated (so the
 * WebGPU SoA path doesn't special-case), we recover raw `f_dc` from the
 * baked `colorDC` by inverting the SH_C0 bake:
 *   `f_dc = (color - 0.5) / SH_C0`. This round-trips cleanly through the
 * `0.5 + SH_C0 * f_dc` re-bake in the WebGPU decode pipeline.
 */
import {
  computeBbox,
  normalizeQuatInto,
  SH_C0,
  type SplatScene,
} from './splat-scene.js';

const RECORD = 32;

export function loadSplat(buf: Uint8Array, sourceName: string): SplatScene {
  if (buf.byteLength % RECORD !== 0) {
    throw new Error(`splat: payload length ${buf.byteLength} not a multiple of ${RECORD}`);
  }
  const N = buf.byteLength / RECORD;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  const positions = new Float32Array(N * 3);
  const rotations = new Float32Array(N * 4);
  const scales    = new Float32Array(N * 3);
  const opacity   = new Float32Array(N);
  const colorDC   = new Float32Array(N * 3);
  const dcRaw     = new Float32Array(N * 3);
  const invSHC0 = 1 / SH_C0;

  for (let i = 0; i < N; i++) {
    const o = i * RECORD;
    positions[i * 3 + 0] = dv.getFloat32(o + 0, true);
    positions[i * 3 + 1] = dv.getFloat32(o + 4, true);
    positions[i * 3 + 2] = dv.getFloat32(o + 8, true);

    // Linear scale → log scale (downstream applies `exp`).
    const sx = dv.getFloat32(o + 12, true);
    const sy = dv.getFloat32(o + 16, true);
    const sz = dv.getFloat32(o + 20, true);
    scales[i * 3 + 0] = Math.log(Math.max(sx, 1e-12));
    scales[i * 3 + 1] = Math.log(Math.max(sy, 1e-12));
    scales[i * 3 + 2] = Math.log(Math.max(sz, 1e-12));

    // Color is already in [0,1] in the bake — passthrough into `colorDC`.
    // Also invert the bake to populate `dcRaw` so SoA consumers (WebGPU
    // compute decode) read a uniformly-shaped `f_dc` regardless of format.
    const cr = dv.getUint8(o + 24) / 255;
    const cg = dv.getUint8(o + 25) / 255;
    const cb = dv.getUint8(o + 26) / 255;
    colorDC[i * 3 + 0] = cr;
    colorDC[i * 3 + 1] = cg;
    colorDC[i * 3 + 2] = cb;
    dcRaw[i * 3 + 0] = (cr - 0.5) * invSHC0;
    dcRaw[i * 3 + 1] = (cg - 0.5) * invSHC0;
    dcRaw[i * 3 + 2] = (cb - 0.5) * invSHC0;
    opacity[i] = dv.getUint8(o + 27) / 255;

    // Quaternion: (w, x, y, z) bytes → reorder to XYZW. Bytes are packed
    // such that v = (b - 128) / 128 in [-1, 127/128].
    const qw = (dv.getUint8(o + 28) - 128) / 128;
    const qx = (dv.getUint8(o + 29) - 128) / 128;
    const qy = (dv.getUint8(o + 30) - 128) / 128;
    const qz = (dv.getUint8(o + 31) - 128) / 128;
    normalizeQuatInto(rotations, i * 4, qx, qy, qz, qw);
  }

  return {
    count: N,
    positions, rotations, scales, opacity, colorDC, dcRaw,
    bbox: computeBbox(positions),
    meta: { source: sourceName, format: 'splat' },
  };
}
