// SPDX-License-Identifier: Apache-2.0
//
// Fused project + gather compute shaders.
//
// The non-fused pipeline (decode.wgsl) writes per-instance vertex records
// into an *unsorted* scratch buffer (`instUnsorted`, 64 B/splat = 640 MB at
// 10 M splats), then a separate `cs_gather` kernel reorders those records
// into the final vertex buffer using the radix-sort output indices. That
// scratch read+write is pure DRAM bandwidth — it dominates at high splat
// counts (~30 % of the frame at 10 M, per bench/4090-clean-single-tenant).
//
// The fused path replaces that with:
//
//   1. cs_keygen          — 1 thread per splat. Reads canonical splat
//                           position only, computes the back-to-front depth
//                           sort key, writes (keys[i], indices[i]).
//                           No projection work, no covariance, no instance
//                           record. Cheap.
//   2. radix sort (existing) — sorts indices by depth.
//   3. cs_project_gather  — 1 thread per *output* slot. Thread `i` reads
//                           `splat_idx = indices[i]`, projects the splat
//                           at that index, and writes the resulting
//                           Instance record directly to `instanceBuffer[i]`.
//
// Net effect at 10 M splats:
//   - `instUnsorted` (640 MB) is never allocated, written, or read.
//   - Projection math runs ONCE (in cs_project_gather), not twice.
//   - Saves one full 640 MB write (cs_project → instUnsorted) and one full
//     640 MB read (cs_gather ← instUnsorted) per frame.
//
// Math invariant: cs_project_gather produces bit-identical Instance records
// to the non-fused (cs_project + cs_gather) path. We achieve this by
// keeping the projection math in this file *byte-for-byte identical* to the
// math in decode.wgsl::cs_project (same operation ordering, same constants).
// The fused-vs-separate parity test in __tests__/webgpu/fused_project_gather
// pins this invariant.

struct DecodedSplat {
  pos:   vec4<f32>,
  scale: vec4<f32>,
  rot:   vec4<f32>,
  color: vec4<f32>,
};

struct Instance {
  clip_pos: vec4<f32>,
  cov:      vec4<f32>,
  color:    vec4<f32>,
};

struct ProjectUniforms {
  view: mat4x4<f32>,
  view_proj: mat4x4<f32>,
  viewport: vec2<f32>,
  focal:    vec2<f32>,
  splat_count:  u32,
  chunk_offset: u32,
  _pad2:        vec2<u32>,
  // Phase 2b: SH-rest evaluator state. Kept in this uniform so the
  // bind-group layout for cs_project_gather only grows by one storage
  // binding (the SH-rest blob) — camera position and the runtime flags
  // ride along here without a second uniform binding.
  // .xyz = camera world-space position; .w = 1 when sh_enabled, else 0.
  cam_pos:      vec4<f32>,
  // .x = sh_degree (0/1/2/3); .y = coef_count (0/3/8/15); .z/.w pad.
  sh_params:    vec4<u32>,
};

// ---------------------------------------------------------------------------
// cs_keygen — depth-only pre-pass.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read>       k_splats  : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> k_keys    : array<u32>;
@group(0) @binding(2) var<storage, read_write> k_indices : array<u32>;
@group(0) @binding(3) var<uniform>             ku        : ProjectUniforms;

@compute @workgroup_size(256)
fn cs_keygen(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + ku.chunk_offset;
  if (i >= ku.splat_count) { return; }
  let s = k_splats[i];
  let pos = s.pos.xyz;
  // View-space depth: column-major mat4 indexed as m[col][row]. Row 2 of view
  // dotted with (pos,1) is -z of the view-space point. MUST match cs_project.
  let vz = ku.view[0][2] * pos.x + ku.view[1][2] * pos.y + ku.view[2][2] * pos.z + ku.view[3][2];
  let depth = -vz;
  let dpos = max(depth, 0.0);
  let kd = bitcast<u32>(dpos);
  k_keys[i] = 0xffffffffu - kd;
  k_indices[i] = i;
}

// ---------------------------------------------------------------------------
// cs_project_gather — full projection in sorted order, direct to vertex buf.
//
// IMPORTANT: the body of this function is the *exact* same math as cs_project
// in decode.wgsl. Any change here must be mirrored there or the parity test
// will fail.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read>       g_splats   : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read>       g_indices  : array<u32>;
@group(0) @binding(2) var<storage, read_write> g_inst_out : array<Instance>;
@group(0) @binding(3) var<uniform>             gu         : ProjectUniforms;
// Phase 2b: per-splat SH-rest coefficients (degrees 1..gu.sh_params.x).
// Layout is splat-major / k-major / channel-minor: index
// `splat_idx * coef_count * 3 + k * 3 + c`. When SH is disabled this binding
// is still bound (to a dummy 16-byte buffer) so the WGSL stays the same.
@group(0) @binding(4) var<storage, read>       g_sh_rest  : array<f32>;

// ---------------------------------------------------------------------------
// SH-rest evaluator (degrees 1..3).
//
// Polynomial form is the Inria 3DGS reference, matching the WebGL2 viewer-app
// fragment shader (packages/viewer-app/src/renderer.ts:116 — evalShRest). The
// constants are the standard Condon-Shortley-phase real spherical harmonics
// basis coefficients. Final color is `clamp(0.5 + SH_C0 * f_dc + sh, 0, 1)`.
//
// `dir` is the world-space view direction, normalized: `normalize(pos - cam)`.
// ---------------------------------------------------------------------------
const SH_C0_CONST : f32 = 0.28209479177387814;
const SH_C1       : f32 = 0.4886025119029199;
const SH_C2_0     : f32 =  1.0925484305920792;
const SH_C2_1     : f32 = -1.0925484305920792;
const SH_C2_2     : f32 =  0.31539156525252005;
const SH_C2_3     : f32 = -1.0925484305920792;
const SH_C2_4     : f32 =  0.5462742152960396;
const SH_C3_0     : f32 = -0.5900435899266435;
const SH_C3_1     : f32 =  2.890611442640554;
const SH_C3_2     : f32 = -0.4570457994644658;
const SH_C3_3     : f32 =  0.3731763325901154;
const SH_C3_4     : f32 = -0.4570457994644658;
const SH_C3_5     : f32 =  1.445305721320277;
const SH_C3_6     : f32 = -0.5900435899266435;

fn fetch_sh_rest_k_g(splat_idx: u32, k: u32, coef_count: u32) -> vec3<f32> {
  let base = splat_idx * coef_count * 3u + k * 3u;
  return vec3<f32>(g_sh_rest[base], g_sh_rest[base + 1u], g_sh_rest[base + 2u]);
}

fn eval_sh_rest_g(splat_idx: u32, dir: vec3<f32>, degree: u32, coef_count: u32) -> vec3<f32> {
  if (degree == 0u) { return vec3<f32>(0.0); }
  let x = dir.x; let y = dir.y; let z = dir.z;
  var col = vec3<f32>(0.0);
  // l=1 (k = 0..2)
  col = col + SH_C1 * (-y) * fetch_sh_rest_k_g(splat_idx, 0u, coef_count);
  col = col + SH_C1 * ( z) * fetch_sh_rest_k_g(splat_idx, 1u, coef_count);
  col = col + SH_C1 * (-x) * fetch_sh_rest_k_g(splat_idx, 2u, coef_count);
  if (degree < 2u) { return col; }
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let yz = y*z; let xz = x*z;
  // l=2 (k = 3..7)
  col = col + SH_C2_0 * (xy)               * fetch_sh_rest_k_g(splat_idx, 3u, coef_count);
  col = col + SH_C2_1 * (yz)               * fetch_sh_rest_k_g(splat_idx, 4u, coef_count);
  col = col + SH_C2_2 * (2.0*zz - xx - yy) * fetch_sh_rest_k_g(splat_idx, 5u, coef_count);
  col = col + SH_C2_3 * (xz)               * fetch_sh_rest_k_g(splat_idx, 6u, coef_count);
  col = col + SH_C2_4 * (xx - yy)          * fetch_sh_rest_k_g(splat_idx, 7u, coef_count);
  if (degree < 3u) { return col; }
  // l=3 (k = 8..14)
  col = col + SH_C3_0 * y * (3.0*xx - yy)             * fetch_sh_rest_k_g(splat_idx, 8u, coef_count);
  col = col + SH_C3_1 * xy * z                        * fetch_sh_rest_k_g(splat_idx, 9u, coef_count);
  col = col + SH_C3_2 * y * (4.0*zz - xx - yy)        * fetch_sh_rest_k_g(splat_idx, 10u, coef_count);
  col = col + SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy) * fetch_sh_rest_k_g(splat_idx, 11u, coef_count);
  col = col + SH_C3_4 * x * (4.0*zz - xx - yy)        * fetch_sh_rest_k_g(splat_idx, 12u, coef_count);
  col = col + SH_C3_5 * z * (xx - yy)                 * fetch_sh_rest_k_g(splat_idx, 13u, coef_count);
  col = col + SH_C3_6 * x * (xx - 3.0*yy)             * fetch_sh_rest_k_g(splat_idx, 14u, coef_count);
  return col;
}

// Build the upper-triangular 3D covariance Σ = R · diag(s²) · Rᵀ.
// Returns (σxx, σxy, σxz, σyy, σyz, σzz). Mirrors cov3d() in decode.wgsl.
fn cov3d_fg(scale: vec3<f32>, q: vec4<f32>) -> array<f32, 6> {
  let n = max(length(q), 1e-8);
  let qn = q / n;
  let x = qn.x; let y = qn.y; let z = qn.z; let w = qn.w;
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let xz = x*z; let yz = y*z;
  let wx = w*x; let wy = w*y; let wz = w*z;
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
fn cs_project_gather(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gu.chunk_offset;
  if (i >= gu.splat_count) { return; }

  // Indirection: thread `i` writes the i-th sorted slot; we read the splat
  // index from the radix-sort output, then project that splat.
  let splat_idx = g_indices[i];
  let s = g_splats[splat_idx];
  let pos = s.pos.xyz;
  let opacity = s.pos.w;

  // Clip-space. (Identical to cs_project in decode.wgsl.)
  let clip = gu.view_proj * vec4<f32>(pos, 1.0);
  let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
  let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

  // View-space splat center. row 2 of view (column-major mat4x4 indexed
  // m[col][row]) dotted with (pos,1) gives -z of the view-space point.
  let vx = gu.view[0][0] * pos.x + gu.view[1][0] * pos.y + gu.view[2][0] * pos.z + gu.view[3][0];
  let vy = gu.view[0][1] * pos.x + gu.view[1][1] * pos.y + gu.view[2][1] * pos.z + gu.view[3][1];
  let vz = gu.view[0][2] * pos.x + gu.view[1][2] * pos.y + gu.view[2][2] * pos.z + gu.view[3][2];
  let depth = -vz;
  // Frustum + near-plane cull (matches viewer-app). See decode.wgsl for
  // the rationale: keeps splats outside the clip volume from projecting
  // through 1/z to enormous radii — critical now that the full Jacobian
  // below carries j02 = fx*vx/vz² and j12 = fy*vy/vz² perspective-skew
  // terms (the j**² · vzz contribution overflows for near-camera off-axis
  // splats without this cull).
  // GL-convention clip-space cull (clip.z in [-w, w]) — matches the
  // projection matrix in camera-state.ts/perspective() and viewer-app's
  // identical clip-space test. The previous `clip.z < 0` was the WebGPU
  // 0..w convention and was OVER-CULLING any splat whose GL NDC z fell
  // in [-1, 0] — i.e. EVERY close-to-camera foreground splat. That left
  // background splats visible through the missing foreground, producing
  // the visible "smear / X-ray" effect on bonsai + kitchen.
  let clip_margin = 1.2 * clip.w;
  let off_frustum = clip.w <= 0.0 ||
    clip.x < -clip_margin || clip.x > clip_margin ||
    clip.y < -clip_margin || clip.y > clip_margin ||
    clip.z < -clip.w || clip.z > clip.w;
  let behind = depth <= 1e-4 || off_frustum;

  // Reconstruct full 3D covariance Σ_w (6 unique entries).
  let V = cov3d_fg(s.scale.xyz, s.rot);
  // World→view rotation rows. row0 = (m00, m01, m02) maps world-x.
  let r0 = vec3<f32>(gu.view[0][0], gu.view[1][0], gu.view[2][0]);
  let r1 = vec3<f32>(gu.view[0][1], gu.view[1][1], gu.view[2][1]);
  let r2 = vec3<f32>(gu.view[0][2], gu.view[1][2], gu.view[2][2]);
  // T = R · Σ_w  (3×3). Σ_w is symmetric so we expand from the 6 entries
  // (V[0]=σxx, V[1]=σxy, V[2]=σxz, V[3]=σyy, V[4]=σyz, V[5]=σzz).
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
  // Full 2×3 image-plane Jacobian (matches viewer-app/src/renderer.ts:228).
  // The j02/j12 perspective-skew columns are CRITICAL — dropping them
  // produces wrong-shape elliptical Gaussians that smear off-axis.
  // The frustum + near-plane cull above bounds 1/vz² so these terms don't
  // explode for near-camera splats.
  let inv_vz  = 1.0 / vz;
  let inv_vz2 = inv_vz * inv_vz;
  let j00 = -gu.focal.x * inv_vz;
  let j02 =  gu.focal.x * vx * inv_vz2;
  let j11 = -gu.focal.y * inv_vz;
  let j12 =  gu.focal.y * vy * inv_vz2;
  // Σ_2D = J · Σ_v · Jᵀ. The 2×3 J has zeros in (0,1) and (1,0), so this
  // reduces to the symmetric block below.
  let m00 = j00*v00 + j02*v02;
  let m01 = j00*v01 + j02*v12;
  let m02 = j00*v02 + j02*v22;
  let m11 = j11*v11 + j12*v12;
  let m12 = j11*v12 + j12*v22;
  let reg = 0.3; // SF_EWA_DILATION
  let c00 = m00*j00 + m02*j02 + reg;
  let c01 = m01*j11 + m02*j12;
  let c11 = m11*j11 + m12*j12 + reg;

  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 0.0);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  // 2.5σ captures 98.8% of mass; 3σ was 99.7% but 30% more fillrate. Matches viewer-app Stage 7b (fc0fb09).
  var radius = 2.5 * sqrt(max(lambda_max, 0.0));
  let r_max = max(gu.viewport.x, gu.viewport.y) * 0.5;
  radius = min(radius, r_max);
  if (behind) { radius = 0.0; }

  var inst: Instance;
  let zc = clamp(ndc.z, 0.0, 1.0);
  if (behind) {
    inst.clip_pos = vec4<f32>(2.0, 2.0, 1.0, 1.0);
    inst.cov      = vec4<f32>(1.0, 0.0, 1.0, 0.0);
  } else {
    inst.clip_pos = vec4<f32>(ndc.xy, zc, clip.w);
    inst.cov      = vec4<f32>(c00, c01, c11, radius);
  }
  inst.color = vec4<f32>(s.color.rgb, opacity);
  // Phase 2b: view-dependent SH-rest evaluation. When sh_enabled (cam_pos.w == 1)
  // and degree > 0, replace the passthrough DC color with the WebGL2-equivalent
  // bake `clamp(0.5 + SH_C0 * f_dc + sh_eval(dir), 0, 1)`. When disabled, the
  // pre-existing DC-only color stays exactly as written above — byte-identical
  // to the OFF baseline.
  if (gu.cam_pos.w > 0.5) {
    let degree = gu.sh_params.x;
    let coef_count = gu.sh_params.y;
    if (degree > 0u) {
      let dir = normalize(pos - gu.cam_pos.xyz);
      let sh = eval_sh_rest_g(splat_idx, dir, degree, coef_count);
      // s.color.rgb is the *baked* DC color (cs_decode applied `0.5 + SH_C0*f_dc`
      // and clamped). Add the view-dependent SH-rest contribution and re-clamp.
      // This matches WebGL2 viewer-app's `clamp(0.5 + SH_C0*f_dc + shCol, 0, 1)`
      // because the bake is associative.
      let rgb = clamp(s.color.rgb + sh, vec3<f32>(0.0), vec3<f32>(1.0));
      inst.color = vec4<f32>(rgb, opacity);
    }
  }
  g_inst_out[i] = inst;
}
