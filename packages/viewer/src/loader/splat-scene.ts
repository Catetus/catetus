// SPDX-License-Identifier: Apache-2.0
/**
 * Canonical splat scene shape returned by the format loaders in this
 * directory. All loaders normalize to this shape regardless of input format
 * (PLY, .splat, SOG, SF .glb, .v5tail sidecar).
 *
 * Mirrors the shape used by `packages/viewer-app/src/splat-scene.ts` so a
 * future dedupe pass (Phase 3) can collapse the two into one. Keep field
 * names, semantics, and helper signatures parallel.
 *
 * Conventions:
 *   - positions: XYZ float, world-space.
 *   - rotations: quaternion XYZW, unit-normalized.
 *   - scales:    log-space per-axis (consumer applies `exp`).
 *   - opacity:   linear [0,1].
 *   - colorDC:   linear RGB [0,1] already through `0.5 + SH_C0 * f_dc`
 *                (display-ready bake; renderer does NOT re-apply SH_C0).
 *   - dcRaw:     raw `f_dc` coefficient (no SH_C0 bake). Always populated
 *                even when shRest is absent so the WebGPU compute pipeline,
 *                which consumes raw DC in its SoA chunk layout, doesn't have
 *                to special-case format-by-format. Loaders that legitimately
 *                only have baked DC (currently: antimatter15 .splat) recover
 *                raw DC by inverting the bake: `dcRaw = (colorDC - 0.5)/SH_C0`.
 */
export interface SplatScene {
  count: number;
  positions: Float32Array;   // length count*3
  rotations: Float32Array;   // length count*4 (XYZW)
  scales: Float32Array;      // length count*3 (log-scale)
  opacity: Float32Array;     // length count   ([0,1])
  colorDC: Float32Array;     // length count*3 (linear sRGB-ish [0,1])
  /**
   * Optional SH-rest coefficients (l=1..shDegree). Layout is splat-major,
   * then coef-major, then channel-major:
   *
   *     shRest[i * coefCount * 3 + k * 3 + c]
   *
   * where `coefCount = 3 (l=1) + 5 (l=2) + 7 (l=3)` for the chosen
   * `shDegree`. Values are the RAW f_rest coefficients (NOT pre-multiplied
   * by SH band constants); a SH evaluator (Phase 2b WebGPU work) applies
   * the standard 3DGS SH band coefficients during shading.
   *
   * `undefined` for formats that don't store SH-rest (e.g. .splat).
   */
  shRest?: Float32Array;
  /** SH degree carried in `shRest` (1, 2, or 3). `undefined` when no SH-rest. */
  shDegree?: number;
  /**
   * Raw `f_dc` coefficient (length count*3) — i.e., `colorDC` *before* the
   * `0.5 + SH_C0 * f_dc` bake. ALWAYS populated (see field comment on
   * `dcRaw` in the docstring) so the WebGPU SoA path doesn't need a
   * special case.
   */
  dcRaw: Float32Array;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  /** Free-form metadata surfaced in the HUD. */
  meta: {
    source: string;          // file name(s)
    format: string;          // e.g. "ply", "splat", "sog", "sf-glb"
    psnr?: number;           // optional, surfaced if known
    extra?: Record<string, string | number>;
  };
}

/** Coefficient count for SH-rest given degree (degree 0 = no rest). */
export function shRestCoefCount(shDegree: number): number {
  let c = 0;
  if (shDegree >= 1) c += 3;
  if (shDegree >= 2) c += 5;
  if (shDegree >= 3) c += 7;
  return c;
}

/** Tight bbox over positions. */
export function computeBbox(positions: Float32Array): SplatScene['bbox'] {
  const N = positions.length / 3;
  if (N === 0) return { min: [0, 0, 0], max: [0, 0, 0] };
  let xMin = positions[0]!, yMin = positions[1]!, zMin = positions[2]!;
  let xMax = xMin, yMax = yMin, zMax = zMin;
  for (let i = 1; i < N; i++) {
    const x = positions[i * 3 + 0]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    if (x < xMin) xMin = x; else if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; else if (y > yMax) yMax = y;
    if (z < zMin) zMin = z; else if (z > zMax) zMax = z;
  }
  return { min: [xMin, yMin, zMin], max: [xMax, yMax, zMax] };
}

/** Inria 3DGS sigmoid for opacity decode. */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Inria SH-DC0 → linear color, used by PLY + GLB + SOG loaders. */
export const SH_C0 = 0.28209479177387814;

/** Clamp `x` to `[0, 1]`. */
export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Normalize one quaternion in-place into out[off..off+4]. Falls back to identity. */
export function normalizeQuatInto(
  out: Float32Array,
  off: number,
  x: number,
  y: number,
  z: number,
  w: number,
): void {
  const n = Math.hypot(x, y, z, w);
  if (n === 0 || !Number.isFinite(n)) {
    out[off + 0] = 0; out[off + 1] = 0; out[off + 2] = 0; out[off + 3] = 1;
    return;
  }
  const inv = 1 / n;
  out[off + 0] = x * inv;
  out[off + 1] = y * inv;
  out[off + 2] = z * inv;
  out[off + 3] = w * inv;
}
