//! Pure-function matrix math, line-for-line port of
//! `packages/viewer/src/renderer/math.ts`.
//!
//! Matrices are 16-float column-major (`[col0_row0, col0_row1, ...]`) so they
//! can be uploaded straight to Metal/GL without a transpose.

/// Right-handed `lookAt`.
pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let (fx, fy, fz) = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
    let fl = (fx * fx + fy * fy + fz * fz).sqrt().max(1e-8);
    let (f0, f1, f2) = (fx / fl, fy / fl, fz / fl);
    let mut sx = f1 * up[2] - f2 * up[1];
    let mut sy = f2 * up[0] - f0 * up[2];
    let mut sz = f0 * up[1] - f1 * up[0];
    let sl = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-8);
    sx /= sl;
    sy /= sl;
    sz /= sl;
    let ux = sy * f2 - sz * f1;
    let uy = sz * f0 - sx * f2;
    let uz = sx * f1 - sy * f0;
    let mut m = [0.0_f32; 16];
    m[0] = sx;
    m[4] = sy;
    m[8] = sz;
    m[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
    m[1] = ux;
    m[5] = uy;
    m[9] = uz;
    m[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    m[2] = -f0;
    m[6] = -f1;
    m[10] = -f2;
    m[14] = f0 * eye[0] + f1 * eye[1] + f2 * eye[2];
    m[15] = 1.0;
    m
}

/// Right-handed perspective, clip-`z` in `[0, w]`.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    let mut m = [0.0_f32; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1.0;
    m[14] = 2.0 * far * near * nf;
    m
}

/// Column-major `out = a · b`.
pub fn mul_mat4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut o = [0.0_f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            let mut v = 0.0;
            for k in 0..4 {
                v += a[k * 4 + r] * b[c * 4 + k];
            }
            o[c * 4 + r] = v;
        }
    }
    o
}

/// Project a world-space point by a column-major view-projection matrix.
/// Returns NDC `[x, y, z]` and clip-space `w` (so callers can compute view
/// depth for the depth sort).
pub fn project_point(p: [f32; 3], vp: &[f32; 16]) -> ([f32; 3], f32) {
    let (x, y, z) = (p[0], p[1], p[2]);
    let cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    let cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    let cz = vp[2] * x + vp[6] * y + vp[10] * z + vp[14];
    let cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    let inv_w = if cw != 0.0 { 1.0 / cw } else { 1.0 };
    ([cx * inv_w, cy * inv_w, cz * inv_w], cw)
}
