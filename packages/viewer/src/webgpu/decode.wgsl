// SPDX-License-Identifier: Apache-2.0
//
// WebGPU compute decode shader for Catetus SoA splat chunks.
//
// Input  (storage, read):  packed bytes of a single chunk laid out as
//                          POSITION u16x3 normalized  | min, max from uniform
//                          ROTATION f32x4
//                          SCALE    u8x3  normalized  | min, max from uniform
//                          OPACITY  u8    normalized  | min=0, max=1
//                          COLOR_DC u8x3  normalized  | min=0, max=1
//
// Output (storage, write): array<DecodedSplat> — one record per splat in the
//                          canonical decoded form fed to the project pass.
//
// One workgroup-thread = one splat. Workgroup size 256 chosen to comfortably
// fit on every WebGPU implementation (Apple Silicon, Adreno, Mali) without
// hitting the 16384-byte workgroup shared-memory cap; we don't use shared
// memory in this pass.
//
// The math mirrors `decodeAttribute` in renderer/base.ts so the GPU and CPU
// paths produce bit-identical results modulo float-precision noise.

struct DecodedSplat {
  // vec4 packing keeps storage-buffer alignment at 16B. The shader emits
  //   pos.xyz   = world position
  //   pos.w     = opacity (so we save 12 bytes vs a separate vec4).
  //   scale.xyz = anisotropic scale
  //   scale.w   = reserved (0)
  //   rot.xyzw  = quaternion (x,y,z,w)
  //   color.rgb = DC SH coefficient as linear RGB
  //   color.a   = reserved (1)
  pos:   vec4<f32>,
  scale: vec4<f32>,
  rot:   vec4<f32>,
  color: vec4<f32>,
};

struct AttributeSlice {
  byte_offset: u32,
  // Each element is f32 even for ints — we burn 16B per slice but get
  // straight-line WGSL with no scalar-load conversions.
  comp_type:   u32, // 5121 u8, 5123 u16, 5126 f32
  normalized:  u32, // 0/1
  _pad:        u32,
  // Per-component dequant bounds. comp 0..3.
  vmin: vec4<f32>,
  vmax: vec4<f32>,
};

struct DecodeUniforms {
  splat_count:  u32,
  // chunk_offset: first splat index covered by this dispatch. Multi-dispatch
  // wrappers update only this slot between chunks (see multi-dispatch.ts).
  chunk_offset: u32,
  _pad1:        u32,
  _pad2:        u32,
  positions:    AttributeSlice,
  rotations:    AttributeSlice,
  scales:       AttributeSlice,
  opacities:    AttributeSlice,
  color_dc:     AttributeSlice,
};

@group(0) @binding(0) var<storage, read>       src_bytes : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_splats : array<DecodedSplat>;
@group(0) @binding(2) var<uniform>             u : DecodeUniforms;

// Load an unsigned byte from a u32 storage buffer at absolute byte offset `b`.
fn load_u8(b: u32) -> u32 {
  let word = src_bytes[b >> 2u];
  let sh   = (b & 3u) * 8u;
  return (word >> sh) & 0xffu;
}

// Load an unsigned 16-bit little-endian short.
fn load_u16(b: u32) -> u32 {
  // Two byte fetches always work, including across u32 boundaries. The two
  // are folded by the driver when aligned.
  let lo = load_u8(b);
  let hi = load_u8(b + 1u);
  return lo | (hi << 8u);
}

// Load a little-endian f32 by reading 4 contiguous bytes. WebGPU storage
// buffers are byte-addressable but not byte-typed; we reassemble.
fn load_f32(b: u32) -> f32 {
  let b0 = load_u8(b);
  let b1 = load_u8(b + 1u);
  let b2 = load_u8(b + 2u);
  let b3 = load_u8(b + 3u);
  return bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}

// Decode one scalar attribute component at byte offset `b` according to
// the slice's comp_type / normalized / min / max.
fn decode_component(slice: AttributeSlice, b: u32, k: u32) -> f32 {
  if (slice.comp_type == 5126u) {
    return load_f32(b);
  }
  if (slice.comp_type == 5123u) {
    let raw = f32(load_u16(b));
    if (slice.normalized == 1u) {
      let lo = slice.vmin[k];
      let hi = slice.vmax[k];
      return lo + (raw / 65535.0) * (hi - lo);
    }
    return raw;
  }
  // default: 5121 u8
  let raw = f32(load_u8(b));
  if (slice.normalized == 1u) {
    let lo = slice.vmin[k];
    let hi = slice.vmax[k];
    return lo + (raw / 255.0) * (hi - lo);
  }
  return raw;
}

fn comp_stride(slice: AttributeSlice) -> u32 {
  if (slice.comp_type == 5126u) { return 4u; }
  if (slice.comp_type == 5123u) { return 2u; }
  return 1u;
}

@compute @workgroup_size(256)
fn cs_decode(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + u.chunk_offset;
  if (i >= u.splat_count) { return; }

  // POSITION (vec3)
  let p_stride = comp_stride(u.positions);
  let p_base = u.positions.byte_offset + i * 3u * p_stride;
  let px = decode_component(u.positions, p_base + 0u * p_stride, 0u);
  let py = decode_component(u.positions, p_base + 1u * p_stride, 1u);
  let pz = decode_component(u.positions, p_base + 2u * p_stride, 2u);

  // ROTATION (vec4) — always f32 on the wire today.
  let r_stride = comp_stride(u.rotations);
  let r_base = u.rotations.byte_offset + i * 4u * r_stride;
  let rx = decode_component(u.rotations, r_base + 0u * r_stride, 0u);
  let ry = decode_component(u.rotations, r_base + 1u * r_stride, 1u);
  let rz = decode_component(u.rotations, r_base + 2u * r_stride, 2u);
  let rw = decode_component(u.rotations, r_base + 3u * r_stride, 3u);

  // SCALE (vec3)
  let s_stride = comp_stride(u.scales);
  let s_base = u.scales.byte_offset + i * 3u * s_stride;
  let sx = decode_component(u.scales, s_base + 0u * s_stride, 0u);
  let sy = decode_component(u.scales, s_base + 1u * s_stride, 1u);
  let sz = decode_component(u.scales, s_base + 2u * s_stride, 2u);

  // OPACITY (scalar)
  let o_stride = comp_stride(u.opacities);
  let opacity = decode_component(u.opacities, u.opacities.byte_offset + i * o_stride, 0u);

  // COLOR_DC (vec3)
  let c_stride = comp_stride(u.color_dc);
  let c_base = u.color_dc.byte_offset + i * 3u * c_stride;
  let cr = decode_component(u.color_dc, c_base + 0u * c_stride, 0u);
  let cg = decode_component(u.color_dc, c_base + 1u * c_stride, 1u);
  let cb = decode_component(u.color_dc, c_base + 2u * c_stride, 2u);

  // DC color wire format is RAW f_dc (per to-soa.ts contract). Bake the
  // SH degree-0 polynomial here: `color = 0.5 + SH_C0 * f_dc`, clamped
  // to [0,1]. SH_C0 = 1/(2·sqrt(π)) ≈ 0.28209479. Without this, the
  // shader treated raw SH coefficients (centered ~0) as already-baked
  // RGB and the PLY/.splat paths rendered as solid black.
  let SH_C0 = 0.28209479177387814;
  let baked = clamp(vec3<f32>(0.5) + SH_C0 * vec3<f32>(cr, cg, cb),
                    vec3<f32>(0.0), vec3<f32>(1.0));

  var out: DecodedSplat;
  out.pos   = vec4<f32>(px, py, pz, opacity);
  out.scale = vec4<f32>(sx, sy, sz, 0.0);
  out.rot   = vec4<f32>(rx, ry, rz, rw);
  out.color = vec4<f32>(baked, 1.0);
  dst_splats[i] = out;
}

// ---------------------------------------------------------------------------
// Project pass: canonical splats + camera → per-instance render attributes
// (clipPos, 2D covariance, premultiplied color) + sortable depth keys.
// ---------------------------------------------------------------------------

struct ProjectUniforms {
  view: mat4x4<f32>,
  view_proj: mat4x4<f32>,
  viewport: vec2<f32>,
  focal:    vec2<f32>,
  splat_count:  u32,
  chunk_offset: u32,
  _pad2:        vec2<u32>,
  // Phase 2b: SH-rest evaluator state. Mirrors the layout used in
  // cs_project_gather.wgsl so the same projectUniforms buffer can be bound
  // to either pipeline. See that file for field semantics.
  cam_pos:      vec4<f32>,
  sh_params:    vec4<u32>,
};

// Per-instance vertex-buffer record (must match FLOATS_PER_INSTANCE=12).
struct Instance {
  clip_pos: vec4<f32>,
  cov:      vec4<f32>,
  color:    vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       p_splats   : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> p_inst     : array<Instance>;
@group(0) @binding(2) var<storage, read_write> p_keys     : array<u32>;
@group(0) @binding(3) var<storage, read_write> p_indices  : array<u32>;
@group(0) @binding(4) var<uniform>             pu : ProjectUniforms;
// Phase 2b: per-splat SH-rest blob. See cs_project_gather.wgsl for layout.
@group(0) @binding(5) var<storage, read>       p_sh_rest : array<f32>;

// SH-rest evaluator (mirrors cs_project_gather.wgsl). Named with a `_p`
// suffix so the two files can coexist after the embed-wgsl pass merges them
// (they don't today, but the suffix makes intent + naming clear).
const SH_C0_P : f32 = 0.28209479177387814;
const SH_C1_P : f32 = 0.4886025119029199;
const SH_C2_0_P : f32 =  1.0925484305920792;
const SH_C2_1_P : f32 = -1.0925484305920792;
const SH_C2_2_P : f32 =  0.31539156525252005;
const SH_C2_3_P : f32 = -1.0925484305920792;
const SH_C2_4_P : f32 =  0.5462742152960396;
const SH_C3_0_P : f32 = -0.5900435899266435;
const SH_C3_1_P : f32 =  2.890611442640554;
const SH_C3_2_P : f32 = -0.4570457994644658;
const SH_C3_3_P : f32 =  0.3731763325901154;
const SH_C3_4_P : f32 = -0.4570457994644658;
const SH_C3_5_P : f32 =  1.445305721320277;
const SH_C3_6_P : f32 = -0.5900435899266435;

fn fetch_sh_rest_k_p(splat_idx: u32, k: u32, coef_count: u32) -> vec3<f32> {
  let base = splat_idx * coef_count * 3u + k * 3u;
  return vec3<f32>(p_sh_rest[base], p_sh_rest[base + 1u], p_sh_rest[base + 2u]);
}

fn eval_sh_rest_p(splat_idx: u32, dir: vec3<f32>, degree: u32, coef_count: u32) -> vec3<f32> {
  if (degree == 0u) { return vec3<f32>(0.0); }
  let x = dir.x; let y = dir.y; let z = dir.z;
  var col = vec3<f32>(0.0);
  col = col + SH_C1_P * (-y) * fetch_sh_rest_k_p(splat_idx, 0u, coef_count);
  col = col + SH_C1_P * ( z) * fetch_sh_rest_k_p(splat_idx, 1u, coef_count);
  col = col + SH_C1_P * (-x) * fetch_sh_rest_k_p(splat_idx, 2u, coef_count);
  if (degree < 2u) { return col; }
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let yz = y*z; let xz = x*z;
  col = col + SH_C2_0_P * (xy)               * fetch_sh_rest_k_p(splat_idx, 3u, coef_count);
  col = col + SH_C2_1_P * (yz)               * fetch_sh_rest_k_p(splat_idx, 4u, coef_count);
  col = col + SH_C2_2_P * (2.0*zz - xx - yy) * fetch_sh_rest_k_p(splat_idx, 5u, coef_count);
  col = col + SH_C2_3_P * (xz)               * fetch_sh_rest_k_p(splat_idx, 6u, coef_count);
  col = col + SH_C2_4_P * (xx - yy)          * fetch_sh_rest_k_p(splat_idx, 7u, coef_count);
  if (degree < 3u) { return col; }
  col = col + SH_C3_0_P * y * (3.0*xx - yy)             * fetch_sh_rest_k_p(splat_idx, 8u, coef_count);
  col = col + SH_C3_1_P * xy * z                        * fetch_sh_rest_k_p(splat_idx, 9u, coef_count);
  col = col + SH_C3_2_P * y * (4.0*zz - xx - yy)        * fetch_sh_rest_k_p(splat_idx, 10u, coef_count);
  col = col + SH_C3_3_P * z * (2.0*zz - 3.0*xx - 3.0*yy) * fetch_sh_rest_k_p(splat_idx, 11u, coef_count);
  col = col + SH_C3_4_P * x * (4.0*zz - xx - yy)        * fetch_sh_rest_k_p(splat_idx, 12u, coef_count);
  col = col + SH_C3_5_P * z * (xx - yy)                 * fetch_sh_rest_k_p(splat_idx, 13u, coef_count);
  col = col + SH_C3_6_P * x * (xx - 3.0*yy)             * fetch_sh_rest_k_p(splat_idx, 14u, coef_count);
  return col;
}

// Build the upper-triangular 3D covariance Σ = R · diag(s²) · Rᵀ.
// Returns the six unique entries as (σxx, σxy, σxz, σyy, σyz, σzz).
fn cov3d(scale: vec3<f32>, q: vec4<f32>) -> array<f32, 6> {
  // Normalize the quaternion defensively (mirrors quatToMat3).
  let n = max(length(q), 1e-8);
  let qn = q / n;
  let x = qn.x; let y = qn.y; let z = qn.z; let w = qn.w;
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let xz = x*z; let yz = y*z;
  let wx = w*x; let wy = w*y; let wz = w*z;
  // Column-major rotation matrix.
  let r00 = 1.0 - 2.0*(yy + zz);
  let r10 = 2.0*(xy + wz);
  let r20 = 2.0*(xz - wy);
  let r01 = 2.0*(xy - wz);
  let r11 = 1.0 - 2.0*(xx + zz);
  let r21 = 2.0*(yz + wx);
  let r02 = 2.0*(xz + wy);
  let r12 = 2.0*(yz - wx);
  let r22 = 1.0 - 2.0*(xx + yy);
  let sx = scale.x; let sy = scale.y; let sz = scale.z;
  let m00 = r00*sx; let m10 = r10*sx; let m20 = r20*sx;
  let m01 = r01*sy; let m11 = r11*sy; let m21 = r21*sy;
  let m02 = r02*sz; let m12 = r12*sz; let m22 = r22*sz;
  return array<f32, 6>(
    m00*m00 + m01*m01 + m02*m02,
    m00*m10 + m01*m11 + m02*m12,
    m00*m20 + m01*m21 + m02*m22,
    m10*m10 + m11*m11 + m12*m12,
    m10*m20 + m11*m21 + m12*m22,
    m20*m20 + m21*m21 + m22*m22,
  );
}

@compute @workgroup_size(256)
fn cs_project(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + pu.chunk_offset;
  if (i >= pu.splat_count) { return; }

  let s = p_splats[i];
  let pos = s.pos.xyz;
  let opacity = s.pos.w;

  // Clip-space.
  let clip = pu.view_proj * vec4<f32>(pos, 1.0);
  let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
  let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

  // View-space splat center. row k of view (column-major mat4x4 indexed
  // m[col][row]) dotted with (pos,1) gives the k-th view-space component.
  let vx = pu.view[0][0] * pos.x + pu.view[1][0] * pos.y + pu.view[2][0] * pos.z + pu.view[3][0];
  let vy = pu.view[0][1] * pos.x + pu.view[1][1] * pos.y + pu.view[2][1] * pos.z + pu.view[3][1];
  let vz = pu.view[0][2] * pos.x + pu.view[1][2] * pos.y + pu.view[2][2] * pos.z + pu.view[3][2];
  let depth = -vz;
  // Frustum + near-plane cull (mirrors viewer-app's `if (vz >= -1e-4)` and
  // 1.2× clip-margin tests). Without this, splats outside the view frustum
  // — especially close-but-off-screen ones — project through 1/z to hundreds
  // of pixels of radius and smear the visible region with low-alpha blobs.
  // CRITICAL with the full Jacobian below: j02 = fx*vx/vz² and j12 = fy*vy/vz²
  // grow unboundedly for near-camera off-axis splats.
  // GL-convention clip-space cull (clip.z in [-w, w]) — matches the
  // projection matrix in camera-state.ts/perspective() and viewer-app's
  // identical clip-space test. The previous `clip.z < 0` was the WebGPU
  // 0..w convention and was OVER-CULLING every close-to-camera splat
  // whose GL NDC z fell in [-1, 0], leaving background splats visible
  // through missing foreground geometry (the "smear / X-ray" regression).
  let clip_margin = 1.2 * clip.w;
  let off_frustum = clip.w <= 0.0 ||
    clip.x < -clip_margin || clip.x > clip_margin ||
    clip.y < -clip_margin || clip.y > clip_margin ||
    clip.z < -clip.w || clip.z > clip.w;
  let behind = depth <= 1e-4 || off_frustum;

  // Reconstruct full 3D covariance Σ_w (6 unique entries).
  let V = cov3d(s.scale.xyz, s.rot);
  //   V[0]=σxx, V[1]=σxy, V[2]=σxz, V[3]=σyy, V[4]=σyz, V[5]=σzz.
  let r0 = vec3<f32>(pu.view[0][0], pu.view[1][0], pu.view[2][0]);
  let r1 = vec3<f32>(pu.view[0][1], pu.view[1][1], pu.view[2][1]);
  let r2 = vec3<f32>(pu.view[0][2], pu.view[1][2], pu.view[2][2]);
  // T = R · Σ_w  (3×3, dense). Mirrors viewer-app/src/renderer.ts.
  let t00 = r0.x*V[0] + r0.y*V[1] + r0.z*V[2];
  let t01 = r0.x*V[1] + r0.y*V[3] + r0.z*V[4];
  let t02 = r0.x*V[2] + r0.y*V[4] + r0.z*V[5];
  let t10 = r1.x*V[0] + r1.y*V[1] + r1.z*V[2];
  let t11 = r1.x*V[1] + r1.y*V[3] + r1.z*V[4];
  let t12 = r1.x*V[2] + r1.y*V[4] + r1.z*V[5];
  let t20 = r2.x*V[0] + r2.y*V[1] + r2.z*V[2];
  let t21 = r2.x*V[1] + r2.y*V[3] + r2.z*V[4];
  let t22 = r2.x*V[2] + r2.y*V[4] + r2.z*V[5];
  // Σ_v = T · Rᵀ (symmetric, upper triangle).
  let v00 = t00*r0.x + t01*r0.y + t02*r0.z;
  let v01 = t00*r1.x + t01*r1.y + t02*r1.z;
  let v02 = t00*r2.x + t01*r2.y + t02*r2.z;
  let v11 = t10*r1.x + t11*r1.y + t12*r1.z;
  let v12 = t10*r2.x + t11*r2.y + t12*r2.z;
  let v22 = t20*r2.x + t21*r2.y + t22*r2.z;
  // Full 2×3 image-plane Jacobian. The j02/j12 perspective-skew columns
  // are CRITICAL — dropping them produces wrong-shape elliptical Gaussians
  // that smear off-axis. (Was the primary bug behind the WebGPU-vs-viewer-app
  // visual regression — see experiments/webgpu-quality-regression/ROOT_CAUSE.md.)
  let inv_vz  = 1.0 / vz;
  let inv_vz2 = inv_vz * inv_vz;
  let j00 = -pu.focal.x * inv_vz;
  let j02 =  pu.focal.x * vx * inv_vz2;
  let j11 = -pu.focal.y * inv_vz;
  let j12 =  pu.focal.y * vy * inv_vz2;
  // Σ_2D = J · Σ_v · Jᵀ.
  let m00 = j00*v00 + j02*v02;
  let m01 = j00*v01 + j02*v12;
  let m02 = j00*v02 + j02*v22;
  let m11 = j11*v11 + j12*v12;
  let m12 = j11*v12 + j12*v22;
  let reg = 0.3; // SF_EWA_DILATION
  let c00 = m00*j00 + m02*j02 + reg;
  let c01 = m01*j11 + m02*j12;
  let c11 = m11*j11 + m12*j12 + reg;

  // 2.5σ captures 98.8% of mass; 3σ was 99.7% but 30% more fillrate.
  // Matches viewer-app Stage 7b (fc0fb09) and cs_project_gather.wgsl.
  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 0.0);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  var radius = 2.5 * sqrt(max(lambda_max, 0.0));
  // Cap radius to half the viewport so a single rogue splat can't fill the
  // whole frame (mirrors viewer-app's `rMax = max(viewport)/2`).
  let r_max = max(pu.viewport.x, pu.viewport.y) * 0.5;
  radius = min(radius, r_max);
  if (behind) { radius = 0.0; }

  var inst: Instance;
  let zc = clamp(ndc.z, 0.0, 1.0);
  if (behind) {
    inst.clip_pos = vec4<f32>(2.0, 2.0, 1.0, 1.0); // off-screen; radius=0 also kills it
    inst.cov      = vec4<f32>(1.0, 0.0, 1.0, 0.0);
  } else {
    inst.clip_pos = vec4<f32>(ndc.xy, zc, clip.w);
    inst.cov      = vec4<f32>(c00, c01, c11, radius);
  }
  inst.color = vec4<f32>(s.color.rgb, opacity);
  // Phase 2b: view-dependent SH-rest evaluation. See cs_project_gather.wgsl
  // for the symmetric block that this mirrors. The two post-blocks must stay
  // logically identical so the fused vs separate output remains byte-equal
  // when SH-rest is enabled.
  if (pu.cam_pos.w > 0.5) {
    let degree = pu.sh_params.x;
    let coef_count = pu.sh_params.y;
    if (degree > 0u) {
      let dir = normalize(pos - pu.cam_pos.xyz);
      let sh = eval_sh_rest_p(i, dir, degree, coef_count);
      // See cs_project_gather.wgsl for the rationale: s.color.rgb is already
      // baked via `0.5 + SH_C0 * f_dc` in cs_decode.
      let rgb = clamp(s.color.rgb + sh, vec3<f32>(0.0), vec3<f32>(1.0));
      inst.color = vec4<f32>(rgb, opacity);
    }
  }
  p_inst[i] = inst;

  // Sort key: bigger view-space depth = drawn first (back-to-front). We bit-
  // pack a u32 from the float. To get correct ascending order over both
  // signs of f32 IEEE-754 reinterpret + a fix-up flip is required, but here
  // we sort positive depths only — behind-camera splats get depth=0 and are
  // suppressed by radius=0 anyway.
  let dpos = max(depth, 0.0);
  // We want back-to-front => largest depth FIRST. Radix sort below sorts
  // ascending, so we invert: key = 0xFFFFFFFF - bitcast(d) gives descending
  // semantics. NaN-safe because dpos is clamped.
  let kd = bitcast<u32>(dpos);
  p_keys[i] = 0xffffffffu - kd;
  p_indices[i] = i;
}
