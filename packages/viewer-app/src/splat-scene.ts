/**
 * Canonical splat scene shape consumed by the renderer. All loaders normalize
 * to this shape regardless of input format (PLY, .splat, SOG, SF .glb).
 *
 * Conventions match what the Catetus bench harness produces:
 *   - positions: XYZ float, world-space (no scene rotation applied here).
 *   - rotations: quaternion XYZW, unit-normalized.
 *   - scales:    log-space per-axis (renderer takes exp()).
 *   - opacity:   linear [0,1] (renderer multiplies into alpha).
 *   - colorDC:   linear RGB [0,1] already through `0.5 + SH_C0 * f_dc`
 *                (renderer does NOT re-apply SH_C0).
 *
 * Carrying both `colorDC` (display-ready) and `f_dc` is intentional — viewer
 * code only needs colorDC; consumers that want raw SH can read f_dc when set.
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
   * by SH band constants); the renderer applies the standard 3DGS SH band
   * coefficients during fragment-shader evaluation and adds the result on
   * top of `colorDC`.
   *
   * Set to `undefined` for formats that don't store SH-rest (e.g. .splat).
   */
  shRest?: Float32Array;
  /** SH degree carried in `shRest` (1, 2, or 3). `undefined` when no SH-rest. */
  shDegree?: number;
  /**
   * Raw DC f_dc coefficient (length count*3) — i.e., `colorDC` *before* the
   * `0.5 + SH_C0 * f_dc` bake. Only populated when `shRest` is present;
   * the renderer reconstructs view-dependent color as
   * `clamp(0.5 + SH_C0 * dc + Σ SH_k(view) * f_rest_k, 0, 1)`. Loaders that
   * already bake DC color into `colorDC` set this in parallel.
   */
  dcRaw?: Float32Array;
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
  let xMin = positions[0], yMin = positions[1], zMin = positions[2];
  let xMax = xMin, yMax = yMin, zMax = zMin;
  for (let i = 1; i < N; i++) {
    const x = positions[i * 3 + 0];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
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

/** Inria SH-DC0 → linear color, used by PLY + GLB loaders. */
export const SH_C0 = 0.28209479177387814;

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
