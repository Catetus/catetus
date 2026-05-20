/**
 * Orbit camera state + a tiny row-major-friendly matrix kit.
 *
 * The camera orbits around `target` at `distance`. `yaw` rotates around world Y,
 * `pitch` rotates above the XZ plane. `roll` keyboard-driven, defaults 0.
 *
 * Matrices are stored as column-major Float32Array(16) per WebGL convention.
 */

export interface CameraState {
  target: [number, number, number];
  distance: number;
  yaw: number;     // radians, around Y, 0 = +Z looking down
  pitch: number;  // radians, around X, 0 = horizon, +PI/2 = top-down
  roll: number;   // radians
  fovYRad: number;
  near: number;
  far: number;
}

export function defaultCamera(): CameraState {
  return {
    target: [0, 0, 0],
    distance: 4,
    yaw: 0,
    pitch: 0,
    roll: 0,
    fovYRad: 50 * Math.PI / 180,
    near: 0.01,
    far: 1000,
  };
}

export function frameCameraToBbox(
  cam: CameraState,
  bbox: { min: [number, number, number]; max: [number, number, number] },
): void {
  const cx = (bbox.min[0] + bbox.max[0]) * 0.5;
  const cy = (bbox.min[1] + bbox.max[1]) * 0.5;
  const cz = (bbox.min[2] + bbox.max[2]) * 0.5;
  cam.target = [cx, cy, cz];
  const dx = bbox.max[0] - bbox.min[0];
  const dy = bbox.max[1] - bbox.min[1];
  const dz = bbox.max[2] - bbox.min[2];
  const radius = 0.5 * Math.hypot(dx, dy, dz);
  // Fit so the bbox's bounding-sphere fills ~80% of vertical FOV.
  const fitDist = radius / Math.tan(cam.fovYRad * 0.5) * 1.3;
  cam.distance = Math.max(fitDist, 0.001);
  cam.near = Math.max(cam.distance * 0.001, 0.001);
  cam.far = Math.max(cam.distance * 1000, 100);
}

/**
 * Compute a "tight" bbox by clipping each axis to a percentile range — defeats
 * the bbox-blow-up caused by 3DGS training floaters (outlier splats at near-
 * infinity that drag min/max wildly outside the actual scene).
 *
 * `clip` = 0.02 means we drop the bottom 1% and top 1% of values per axis.
 * Returns the inflated bbox over the inlier splats, matching the type
 * `frameCameraToBbox` expects.
 */
export function tightBbox(
  positions: Float32Array,
  clip = 0.02,
): { min: [number, number, number]; max: [number, number, number] } {
  const N = positions.length / 3;
  if (N === 0) return { min: [-1, -1, -1], max: [1, 1, 1] };
  const lo = clip * 0.5;
  const hi = 1 - lo;
  const out = { min: [0, 0, 0] as [number, number, number], max: [0, 0, 0] as [number, number, number] };
  // Sampled percentile over a stride to keep this O(M log M) with small M
  // (~50 k samples is plenty for percentile accuracy on a typical 1-2 M scene).
  const stride = Math.max(1, Math.floor(N / 50000));
  const samples = new Float32Array(Math.ceil(N / stride));
  for (let axis = 0; axis < 3; axis++) {
    let n = 0;
    for (let i = 0; i < N; i += stride) samples[n++] = positions[i * 3 + axis];
    const slice = samples.subarray(0, n).slice().sort();
    out.min[axis] = slice[Math.floor(slice.length * lo)];
    out.max[axis] = slice[Math.floor(slice.length * hi)];
  }
  return out;
}

export interface ViewProj {
  view: Float32Array;
  proj: Float32Array;
  viewProj: Float32Array;
  eye: [number, number, number];
}

export function viewProjMatrix(cam: CameraState, aspect: number): ViewProj {
  // Spherical eye position from yaw/pitch/distance. Pitch is unbounded.
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const ex = cam.target[0] + cam.distance * cp * sy;
  const ey = cam.target[1] + cam.distance * sp;
  const ez = cam.target[2] + cam.distance * cp * cy;

  // Flip world-up when the camera is upside-down so lookAt stays coherent and
  // can continuously roll over the poles. At the exact ±π/2 singularity the
  // cross product would be zero; nudge with a tiny epsilon so we never land
  // exactly there.
  const upSign = cp >= 0 ? 1 : -1;
  const epsX = Math.abs(cp) < 1e-4 ? 1e-4 * Math.sign(sy || 1) : 0;
  const view = lookAt([ex + epsX, ey, ez], cam.target, [0, upSign, 0], cam.roll);
  const proj = perspective(cam.fovYRad, aspect, cam.near, cam.far);
  const viewProj = mul(proj, view);
  return { view, proj, viewProj, eye: [ex, ey, ez] };
}

function lookAt(
  eye: [number, number, number],
  center: [number, number, number],
  up: [number, number, number],
  roll: number,
): Float32Array {
  let fx = center[0] - eye[0];
  let fy = center[1] - eye[1];
  let fz = center[2] - eye[2];
  const fl = 1 / Math.hypot(fx, fy, fz);
  fx *= fl; fy *= fl; fz *= fl;

  let sx = fy * up[2] - fz * up[1];
  let sy = fz * up[0] - fx * up[2];
  let sz = fx * up[1] - fy * up[0];
  const sl = 1 / Math.hypot(sx, sy, sz);
  sx *= sl; sy *= sl; sz *= sl;

  let ux = sy * fz - sz * fy;
  let uy = sz * fx - sx * fz;
  let uz = sx * fy - sy * fx;

  // Apply roll: rotate (s, u) about forward axis.
  if (roll !== 0) {
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const sx2 =  cr * sx + sr * ux;
    const sy2 =  cr * sy + sr * uy;
    const sz2 =  cr * sz + sr * uz;
    const ux2 = -sr * sx + cr * ux;
    const uy2 = -sr * sy + cr * uy;
    const uz2 = -sr * sz + cr * uz;
    sx = sx2; sy = sy2; sz = sz2;
    ux = ux2; uy = uy2; uz = uz2;
  }

  // Column-major view matrix:
  const v = new Float32Array(16);
  v[0]  = sx;   v[4]  = sy;   v[8]  = sz;   v[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  v[1]  = ux;   v[5]  = uy;   v[9]  = uz;   v[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  v[2]  = -fx;  v[6]  = -fy;  v[10] = -fz;  v[14] =  (fx * eye[0] + fy * eye[1] + fz * eye[2]);
  v[3]  = 0;    v[7]  = 0;    v[11] = 0;    v[15] = 1;
  return v;
}

function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY * 0.5);
  const nf = 1 / (near - far);
  const p = new Float32Array(16);
  p[0] = f / aspect;
  p[5] = f;
  p[10] = (far + near) * nf;
  p[11] = -1;
  p[14] = 2 * far * near * nf;
  return p;
}

function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      o[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0]
                   + a[1 * 4 + r] * b[c * 4 + 1]
                   + a[2 * 4 + r] * b[c * 4 + 2]
                   + a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}
