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
  splat_count: u32,
  _pad: u32,
  _pad2: vec2<u32>,
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
  let i = gid.x;
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
  let i = gid.x;
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

  let vz = gu.view[0][2] * pos.x + gu.view[1][2] * pos.y + gu.view[2][2] * pos.z + gu.view[3][2];
  let depth = -vz;
  let behind = depth <= 0.0;

  let V = cov3d_fg(s.scale.xyz, s.rot);
  let w0 = vec3<f32>(gu.view[0][0], gu.view[1][0], gu.view[2][0]);
  let w1 = vec3<f32>(gu.view[0][1], gu.view[1][1], gu.view[2][1]);
  let a0 = vec3<f32>(
    w0.x * V[0] + w0.y * V[1] + w0.z * V[2],
    w0.x * V[1] + w0.y * V[3] + w0.z * V[4],
    w0.x * V[2] + w0.y * V[4] + w0.z * V[5],
  );
  let a1 = vec3<f32>(
    w1.x * V[0] + w1.y * V[1] + w1.z * V[2],
    w1.x * V[1] + w1.y * V[3] + w1.z * V[4],
    w1.x * V[2] + w1.y * V[4] + w1.z * V[5],
  );
  let vxx = dot(a0, w0);
  let vxy = dot(a0, w1);
  let vyy = dot(a1, w1);

  let z = max(abs(depth), 1e-4);
  let jx = gu.focal.x / z;
  let jy = gu.focal.y / z;
  let reg = 0.3; // SF_EWA_DILATION
  let c00 = jx * jx * vxx + reg;
  let c01 = jx * jy * vxy;
  let c11 = jy * jy * vyy + reg;

  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 0.0);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  var radius = 3.0 * sqrt(max(lambda_max, 0.0));
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
  g_inst_out[i] = inst;
}
