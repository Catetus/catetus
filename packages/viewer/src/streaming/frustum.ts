/**
 * Camera-frustum extraction and tile-bbox intersection helpers.
 *
 * The producer side (`tileset.rs`) emits each tile's bounding volume as the
 * 12-float Cesium OBB form: `[cx, cy, cz, hx,0,0, 0,hy,0, 0,0,hz]`. Because
 * SplatForge tilesets are local-frame axis-aligned, the three half-axis
 * vectors collapse to `(hx,0,0)`, `(0,hy,0)`, `(0,0,hz)` — i.e. the OBB *is*
 * an AABB in the asset's coordinate frame. We exploit that to keep the
 * intersection test branch-free.
 *
 * The frustum is extracted from a column-major view-projection matrix using
 * the Gribb-Hartmann method (Plane Extraction from the Combined
 * Modelview-Projection Matrix, GDC 2001). Each plane's equation is
 * `a·x + b·y + c·z + d = 0`; we keep the planes pointed inward so the
 * AABB test reduces to "the box's positive vertex is on the inside half-
 * space for every plane".
 *
 * All math is deterministic for a fixed `viewProj` — no branches depend on
 * IEEE-rounding edge cases that the WebGPU and WebGL2 paths would disagree
 * on.
 */

/** A single plane `n.x*x + n.y*y + n.z*z + d = 0`, `n` unit-length, pointing inward. */
export interface Plane {
  nx: number;
  ny: number;
  nz: number;
  d: number;
}

/** Six planes of the view frustum: left, right, bottom, top, near, far. */
export interface Frustum {
  planes: [Plane, Plane, Plane, Plane, Plane, Plane];
}

/** Axis-aligned box. */
export interface Aabb {
  min: [number, number, number];
  max: [number, number, number];
}

/**
 * Extract the 6 frustum planes from a column-major 4x4 viewProj matrix.
 *
 * The combined matrix `M = P · V` maps world → clip. A point is inside the
 * frustum when `-w <= clip.x <= w`, `-w <= clip.y <= w`, and `0 <= clip.z <= w`
 * for a `[0, w]` reverse-z perspective. Each inequality yields one plane:
 *
 *   left   : (row3 + row0) · p >= 0
 *   right  : (row3 - row0) · p >= 0
 *   bottom : (row3 + row1) · p >= 0
 *   top    : (row3 - row1) · p >= 0
 *   near   : (row3 + row2) · p >= 0   (matches `near = -w + z`)
 *   far    : (row3 - row2) · p >= 0
 *
 * @param m column-major Float32Array of length 16 (`m[col*4 + row]`).
 */
export function extractFrustum(m: Float32Array): Frustum {
  // Rows of the column-major matrix. `row[r][c] = m[c*4 + r]`.
  const r0x = m[0]!,  r0y = m[4]!,  r0z = m[8]!,  r0w = m[12]!;
  const r1x = m[1]!,  r1y = m[5]!,  r1z = m[9]!,  r1w = m[13]!;
  const r2x = m[2]!,  r2y = m[6]!,  r2z = m[10]!, r2w = m[14]!;
  const r3x = m[3]!,  r3y = m[7]!,  r3z = m[11]!, r3w = m[15]!;

  const planes: Plane[] = [
    normalizePlane({ nx: r3x + r0x, ny: r3y + r0y, nz: r3z + r0z, d: r3w + r0w }), // left
    normalizePlane({ nx: r3x - r0x, ny: r3y - r0y, nz: r3z - r0z, d: r3w - r0w }), // right
    normalizePlane({ nx: r3x + r1x, ny: r3y + r1y, nz: r3z + r1z, d: r3w + r1w }), // bottom
    normalizePlane({ nx: r3x - r1x, ny: r3y - r1y, nz: r3z - r1z, d: r3w - r1w }), // top
    normalizePlane({ nx: r3x + r2x, ny: r3y + r2y, nz: r3z + r2z, d: r3w + r2w }), // near
    normalizePlane({ nx: r3x - r2x, ny: r3y - r2y, nz: r3z - r2z, d: r3w - r2w }), // far
  ];
  return { planes: planes as Frustum['planes'] };
}

function normalizePlane(p: Plane): Plane {
  const len = Math.hypot(p.nx, p.ny, p.nz) || 1;
  return { nx: p.nx / len, ny: p.ny / len, nz: p.nz / len, d: p.d / len };
}

/**
 * Convert the 12-float Cesium OBB representation to an AABB. We assume the
 * three half-axis triples are axis-aligned (the only form the SplatForge
 * tileset emitter produces). If a tile carries a rotated OBB we fall back to
 * its bounding AABB by absolute-value summing the half-axes — a safe over-
 * approximation that may load slightly extra tiles but never culls a visible
 * one.
 */
export function aabbFromObb12(obb: number[]): Aabb {
  if (obb.length !== 12) {
    throw new Error('aabb_invalid: expected 12-element OBB');
  }
  const cx = obb[0]!, cy = obb[1]!, cz = obb[2]!;
  // hx*ex axis, hy*ey axis, hz*ez axis. Sum absolute components per axis
  // for the conservative AABB.
  const hx = Math.abs(obb[3]!)  + Math.abs(obb[4]!)  + Math.abs(obb[5]!);
  const hy = Math.abs(obb[6]!)  + Math.abs(obb[7]!)  + Math.abs(obb[8]!);
  const hz = Math.abs(obb[9]!) + Math.abs(obb[10]!) + Math.abs(obb[11]!);
  return {
    min: [cx - hx, cy - hy, cz - hz],
    max: [cx + hx, cy + hy, cz + hz],
  };
}

/**
 * Returns `true` iff the AABB is at least partially inside the frustum.
 *
 * Uses the standard "positive vertex" trick: for each plane, pick the box
 * corner that's furthest along the plane normal; if even that corner is
 * outside the plane, the entire box is outside.
 *
 * Determinism: pure float math with no branches that diverge between
 * structurally-equivalent inputs.
 */
export function aabbIntersectsFrustum(box: Aabb, frustum: Frustum): boolean {
  for (const p of frustum.planes) {
    // Positive vertex along plane normal.
    const px = p.nx >= 0 ? box.max[0] : box.min[0];
    const py = p.ny >= 0 ? box.max[1] : box.min[1];
    const pz = p.nz >= 0 ? box.max[2] : box.min[2];
    if (p.nx * px + p.ny * py + p.nz * pz + p.d < 0) {
      return false;
    }
  }
  return true;
}

/**
 * Center-to-eye distance for SSE math. Returned in world units — the same
 * unit `geometricError` is in, so the SSE ratio is dimensionless.
 */
export function distanceFromCamera(box: Aabb, eye: [number, number, number]): number {
  const cx = 0.5 * (box.min[0] + box.max[0]);
  const cy = 0.5 * (box.min[1] + box.max[1]);
  const cz = 0.5 * (box.min[2] + box.max[2]);
  const dx = cx - eye[0];
  const dy = cy - eye[1];
  const dz = cz - eye[2];
  // Clamp to a tiny epsilon so SSE doesn't blow up when the camera is inside
  // the box (geometricError / 0 → ∞ → always refine, the desired behavior
  // anyway, but we avoid Infinity contaminating downstream arithmetic).
  return Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6);
}
