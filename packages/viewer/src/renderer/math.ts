/**
 * Math helpers shared by the WebGPU and WebGL2 backends.
 *
 * All 4x4 matrices are stored as column-major {@link Float32Array} of length 16
 * to match the layout expected by GLSL `uniformMatrix4fv` and WGSL `mat4x4`.
 * Quaternions are `[x, y, z, w]` with `w` last, matching the on-wire layout.
 *
 * The functions implement the standard Gaussian-splat math from the Inria
 * 3DGS / gsplat reference: 3D covariance from scale+rotation, projection of
 * that covariance into screen space via the Jacobian of the perspective
 * projection, and a column-major lookAt + perspective for the view-projection
 * matrix.
 */
import type { CameraPose } from '../camera.js';

/**
 * Convert a quaternion `[x, y, z, w]` into a column-major 3x3 rotation matrix
 * stored as a `Float32Array` of length 9.
 *
 * The identity quaternion `[0,0,0,1]` returns the identity matrix.
 */
export function quatToMat3(q: [number, number, number, number]): Float32Array {
  const [x, y, z, w] = q;
  // Normalize defensively; quaternions stored on disk may drift slightly.
  const n = Math.hypot(x, y, z, w) || 1;
  const xn = x / n;
  const yn = y / n;
  const zn = z / n;
  const wn = w / n;
  const xx = xn * xn;
  const yy = yn * yn;
  const zz = zn * zn;
  const xy = xn * yn;
  const xz = xn * zn;
  const yz = yn * zn;
  const wx = wn * xn;
  const wy = wn * yn;
  const wz = wn * zn;
  const m = new Float32Array(9);
  // Column 0
  m[0] = 1 - 2 * (yy + zz);
  m[1] = 2 * (xy + wz);
  m[2] = 2 * (xz - wy);
  // Column 1
  m[3] = 2 * (xy - wz);
  m[4] = 1 - 2 * (xx + zz);
  m[5] = 2 * (yz + wx);
  // Column 2
  m[6] = 2 * (xz + wy);
  m[7] = 2 * (yz - wx);
  m[8] = 1 - 2 * (xx + yy);
  return m;
}

/**
 * Compute the upper-triangular 3D covariance `Σ = R · diag(s²) · Rᵀ` for a
 * splat with anisotropic scale `s` and quaternion rotation `rot`.
 *
 * @returns the six unique entries as `[σxx, σxy, σxz, σyy, σyz, σzz]`.
 */
export function computeCovariance3D(
  scale: [number, number, number],
  rot: [number, number, number, number],
): Float32Array {
  const r = quatToMat3(rot);
  const sx = scale[0];
  const sy = scale[1];
  const sz = scale[2];
  // M = R · diag(s). Then Σ = M · Mᵀ.
  const m00 = r[0]! * sx;
  const m10 = r[1]! * sx;
  const m20 = r[2]! * sx;
  const m01 = r[3]! * sy;
  const m11 = r[4]! * sy;
  const m21 = r[5]! * sy;
  const m02 = r[6]! * sz;
  const m12 = r[7]! * sz;
  const m22 = r[8]! * sz;
  const out = new Float32Array(6);
  out[0] = m00 * m00 + m01 * m01 + m02 * m02; // σxx
  out[1] = m00 * m10 + m01 * m11 + m02 * m12; // σxy
  out[2] = m00 * m20 + m01 * m21 + m02 * m22; // σxz
  out[3] = m10 * m10 + m11 * m11 + m12 * m12; // σyy
  out[4] = m10 * m20 + m11 * m21 + m12 * m22; // σyz
  out[5] = m20 * m20 + m21 * m21 + m22 * m22; // σzz
  return out;
}

/**
 * Project a packed symmetric 3D covariance into a 2D screen-space covariance
 * via the Jacobian of the perspective projection, evaluated at view-space
 * depth `depth` (positive in front of the camera). `viewMatrix` is the
 * column-major world-to-view 4x4 used for the rotation part of the transform.
 *
 * Returns the three unique entries of the 2x2 covariance as `[c00, c01, c11]`,
 * already in screen pixels (since `focalX`/`focalY` are pixel focal lengths).
 */
export function projectCovariance2D(
  cov3d: Float32Array,
  viewMatrix: Float32Array,
  focalX: number,
  focalY: number,
  depth: number,
): [number, number, number] {
  // Reconstruct the symmetric 3D covariance matrix.
  const V = [
    cov3d[0]!, cov3d[1]!, cov3d[2]!,
    cov3d[1]!, cov3d[3]!, cov3d[4]!,
    cov3d[2]!, cov3d[4]!, cov3d[5]!,
  ];
  // Rotation part of view matrix (upper-left 3x3, column-major).
  const r00 = viewMatrix[0]!;
  const r10 = viewMatrix[1]!;
  const r20 = viewMatrix[2]!;
  const r01 = viewMatrix[4]!;
  const r11 = viewMatrix[5]!;
  const r21 = viewMatrix[6]!;
  const r02 = viewMatrix[8]!;
  const r12 = viewMatrix[9]!;
  const r22 = viewMatrix[10]!;
  // T = W · V, then Σ_view = T · Wᵀ. Build the symmetric Σ_view directly.
  // Treat W as rows of the rotation: row0 = (r00, r01, r02), etc.
  // (Note: column-major storage means viewMatrix[i*4+j] is row j, column i; we
  // map back to row-vectors here.)
  const w0x = r00, w0y = r01, w0z = r02;
  const w1x = r10, w1y = r11, w1z = r12;
  void r20; void r21; void r22;
  // Σ_view = W V Wᵀ — compute first row W V, then dot with rows of W.
  const a00 = w0x * V[0]! + w0y * V[3]! + w0z * V[6]!;
  const a01 = w0x * V[1]! + w0y * V[4]! + w0z * V[7]!;
  const a02 = w0x * V[2]! + w0y * V[5]! + w0z * V[8]!;
  const a10 = w1x * V[0]! + w1y * V[3]! + w1z * V[6]!;
  const a11 = w1x * V[1]! + w1y * V[4]! + w1z * V[7]!;
  const a12 = w1x * V[2]! + w1y * V[5]! + w1z * V[8]!;
  const vxx = a00 * w0x + a01 * w0y + a02 * w0z;
  const vxy = a00 * w1x + a01 * w1y + a02 * w1z;
  const vyy = a10 * w1x + a11 * w1y + a12 * w1z;
  // Jacobian of the perspective projection at (0,0,depth):
  //   J = [ fx/z,    0, 0 ]
  //       [    0, fy/z, 0 ]
  // The full 2D covariance is J · Σ_view · Jᵀ but with only the upper-left
  // 2x2 block of Σ_view participating because J zeros out the z column.
  const z = Math.max(Math.abs(depth), 1e-4);
  const jx = focalX / z;
  const jy = focalY / z;
  const c00 = jx * jx * vxx;
  const c01 = jx * jy * vxy;
  const c11 = jy * jy * vyy;
  // Add a small low-pass term so single-pixel splats stay drawable, matching
  // the canonical 3DGS rasterizer.
  const reg = 0.3;
  return [c00 + reg, c01, c11 + reg];
}

/**
 * Project a world-space point `p` by the column-major view-projection matrix
 * `viewProj`. Returns the NDC coordinates, the (positive-forward) view-space
 * depth, and the clip-space `w` for downstream divide-by-w decisions.
 */
export function projectPoint(
  p: [number, number, number],
  viewProj: Float32Array,
): { ndc: [number, number, number]; depth: number; w: number } {
  const x = p[0], y = p[1], z = p[2];
  const cx = viewProj[0]! * x + viewProj[4]! * y + viewProj[8]! * z + viewProj[12]!;
  const cy = viewProj[1]! * x + viewProj[5]! * y + viewProj[9]! * z + viewProj[13]!;
  const cz = viewProj[2]! * x + viewProj[6]! * y + viewProj[10]! * z + viewProj[14]!;
  const cw = viewProj[3]! * x + viewProj[7]! * y + viewProj[11]! * z + viewProj[15]!;
  const invW = cw !== 0 ? 1 / cw : 1;
  return {
    ndc: [cx * invW, cy * invW, cz * invW],
    depth: cw,
    w: cw,
  };
}

/**
 * Build column-major `view`, `proj`, and `viewProj = proj * view` matrices for
 * a {@link CameraPose}.
 *
 * - `view` is a right-handed `lookAt` looking down `-Z` in view space.
 * - `proj` is a reverse-`z`-positive perspective with `fovY` vertical FOV.
 *
 * The optional `aspect` and `fov` parameters override the values on the pose
 * (useful when the swap-chain size diverges from what the pose was created
 * for, e.g. across a `resize`).
 */
export function buildViewProj(
  camera: CameraPose,
  aspect?: number,
  fov?: number,
): { view: Float32Array; proj: Float32Array; viewProj: Float32Array } {
  const view = lookAt(camera.position, camera.target, camera.up);
  const a = aspect ?? camera.aspect;
  const f = fov ?? camera.fovY;
  const proj = perspective(f, a, camera.near, camera.far);
  const viewProj = mulMat4(proj, view);
  return { view, proj, viewProj };
}

/**
 * Right-handed `lookAt`. Returns a column-major 4x4 that transforms world
 * coordinates into a view space where `+X` is right, `+Y` is up, and `-Z`
 * points toward the target.
 */
export function lookAt(
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number],
): Float32Array {
  const fx = target[0] - eye[0];
  const fy = target[1] - eye[1];
  const fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  const f0 = fx / fl, f1 = fy / fl, f2 = fz / fl;
  // s = normalize(cross(f, up))
  let sx = f1 * up[2] - f2 * up[1];
  let sy = f2 * up[0] - f0 * up[2];
  let sz = f0 * up[1] - f1 * up[0];
  const sl = Math.hypot(sx, sy, sz) || 1;
  sx /= sl; sy /= sl; sz /= sl;
  // u = cross(s, f)
  const ux = sy * f2 - sz * f1;
  const uy = sz * f0 - sx * f2;
  const uz = sx * f1 - sy * f0;
  const m = new Float32Array(16);
  m[0] = sx;  m[4] = sy;  m[8]  = sz;   m[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  m[1] = ux;  m[5] = uy;  m[9]  = uz;   m[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  m[2] = -f0; m[6] = -f1; m[10] = -f2;  m[14] = (f0 * eye[0] + f1 * eye[1] + f2 * eye[2]);
  m[3] = 0;   m[7] = 0;   m[11] = 0;    m[15] = 1;
  return m;
}

/**
 * Build a right-handed, depth-`[0,1]` perspective projection (clip-space `z`
 * in `[0, w]`). Column-major.
 */
export function perspective(
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1 / Math.tan(fovY * 0.5);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

/** Column-major 4x4 multiply `out = a · b`. */
export function mulMat4(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let v = 0;
      for (let k = 0; k < 4; k++) {
        v += a[k * 4 + r]! * b[c * 4 + k]!;
      }
      o[c * 4 + r] = v;
    }
  }
  return o;
}
