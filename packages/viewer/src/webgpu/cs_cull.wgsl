// SPDX-License-Identifier: Apache-2.0
//
// Opacity-radius pre-sort cull.
//
// Drops splats that cannot contribute a visible pixel before they enter the
// project / sort / gather pipeline. Implemented as three compute kernels:
//
//   1. cs_cull         - 1 thread / splat. Computes the 2D screen covariance
//                        and writes alive_flag[i] = 1 if the splat survives,
//                        0 otherwise. The predicate is:
//                          alpha >= tau    (opacity above the floor)        AND
//                          depth > 0       (in front of the camera)         AND
//                          r_max >= 0.5 px (the iso-opacity threshold
//                                           ellipse exceeds one pixel)
//                        with r_max = sigma * sqrt(2 * ln(alpha / tau)).
//                        The flag is written to TWO buffers:
//                          flag_buffer[i]   = the original 0/1 (read by
//                                             cs_compact to gate the write).
//                          prefix_buffer[i] = same 0/1, consumed in-place by
//                                             the scan_multiblock kernels
//                                             which turn it into an exclusive
//                                             prefix sum.
//
//   2. cs_compact      - 1 thread / splat. Reads flag_buffer[i] (alive?) and
//                        prefix_buffer[i] (scatter target). If alive, writes
//                        compact_indices[prefix_buffer[i]] = i.
//
//   3. cs_project_cmpct - 1 thread / survivor. Reads compact_indices[i] ->
//                         splat index si, re-runs the projection on
//                         splats[si], writes inst[i] / keys[i] / values[i].
//                         keys[i]   = depth-bits packed for descending sort.
//                         values[i] = i (the compact-position index - so the
//                         gather stage finds the per-survivor Instance via
//                         dst[k] = src[sorted_values[k]]).
//
// Determinism: no storage-buffer atomics. The prefix sum used by cs_compact
// is provided by the scan_multiblock kernels which are atomic-free.
//
// Math invariant: cs_project_cmpct produces bit-identical Instance records
// to cs_project for the surviving subset; the projection math is copied
// verbatim and the only difference is the index indirection.

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

struct CullUniforms {
  view:        mat4x4<f32>,
  view_proj:   mat4x4<f32>,
  viewport:    vec2<f32>,
  focal:       vec2<f32>,
  splat_count: u32,
  tau:         f32,
  _pad:        vec2<u32>,
};

// =============================================================================
// cs_cull bindings:
//   0 (read)        : splats[]
//   1 (read_write)  : flag_buffer[]    (final 0/1 flags, read by cs_compact)
//   2 (read_write)  : prefix_buffer[]  (also 0/1; turned into exclusive prefix
//                                       in place by the scan_multiblock pass)
//   3 (uniform)     : CullUniforms
// =============================================================================
@group(0) @binding(0) var<storage, read>       cu_splats : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> cu_flag   : array<u32>;
@group(0) @binding(2) var<storage, read_write> cu_prefix : array<u32>;
@group(0) @binding(3) var<uniform>             cu_u      : CullUniforms;

fn cov3d_cull(scale: vec3<f32>, q: vec4<f32>) -> array<f32, 6> {
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
fn cs_cull(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= cu_u.splat_count) { return; }

  let s = cu_splats[i];
  let pos = s.pos.xyz;
  let alpha = s.pos.w;

  var alive: u32 = 0u;

  // Cheap reject 1: opacity below the perceptual floor.
  if (alpha >= cu_u.tau) {
    // View-space depth.
    let vz = cu_u.view[0][2] * pos.x + cu_u.view[1][2] * pos.y + cu_u.view[2][2] * pos.z + cu_u.view[3][2];
    let depth = -vz;
    if (depth > 0.0) {
      // Project to 2D covariance.
      let V = cov3d_cull(s.scale.xyz, s.rot);
      let w0 = vec3<f32>(cu_u.view[0][0], cu_u.view[1][0], cu_u.view[2][0]);
      let w1 = vec3<f32>(cu_u.view[0][1], cu_u.view[1][1], cu_u.view[2][1]);
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
      let jx = cu_u.focal.x / z;
      let jy = cu_u.focal.y / z;
      let reg = 0.3;
      let c00 = jx * jx * vxx + reg;
      let c01 = jx * jy * vxy;
      let c11 = jy * jy * vyy + reg;

      let half_trace = 0.5 * (c00 + c11);
      let det = max(c00 * c11 - c01 * c01, 0.0);
      let term = sqrt(max(half_trace * half_trace - det, 0.0));
      let lambda_max = max(half_trace + term, 0.0);
      let sigma = sqrt(lambda_max);

      // r_max - radius at which alpha * exp(-r^2 / 2 sigma^2) = tau.
      //   r_max^2 = 2 * sigma^2 * ln(alpha/tau).
      let ln_ratio = max(log(alpha / cu_u.tau), 0.0);
      let r_max = sigma * sqrt(2.0 * ln_ratio);

      // Sub-pixel cull.
      if (r_max >= 0.5) {
        alive = 1u;
      }
    }
  }

  cu_flag[i]   = alive;
  cu_prefix[i] = alive;
}

// =============================================================================
// cs_compact bindings:
//   0 (read)        : flag_buffer[]    (0/1 alive flag from cs_cull)
//   1 (read)        : prefix_buffer[]  (exclusive prefix sum, post-scan)
//   2 (read_write)  : compact_indices[]
//   3 (uniform)     : { splat_count, _pad, _pad, _pad }
// =============================================================================
struct CompactUniforms {
  splat_count: u32,
  _pad: vec3<u32>,
};
@group(0) @binding(0) var<storage, read>       co_flag    : array<u32>;
@group(0) @binding(1) var<storage, read>       co_prefix  : array<u32>;
@group(0) @binding(2) var<storage, read_write> co_compact : array<u32>;
@group(0) @binding(3) var<uniform>             co_u       : CompactUniforms;

@compute @workgroup_size(256)
fn cs_compact(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= co_u.splat_count) { return; }
  if (co_flag[i] == 1u) {
    co_compact[co_prefix[i]] = i;
  }
}

// =============================================================================
// cs_project_cmpct bindings (mirrors cs_project from decode.wgsl except for
// the compact_indices indirection at binding 5):
//   0 (read)        : splats[]
//   1 (read_write)  : inst[]
//   2 (read_write)  : keys[]
//   3 (read_write)  : values[]
//   4 (uniform)     : ProjectUniforms  (.splat_count = survivors)
//   5 (read)        : compact_indices[]
// Dispatch shape: ceil(survivors / 256).
// =============================================================================
struct ProjectUniforms {
  view: mat4x4<f32>,
  view_proj: mat4x4<f32>,
  viewport: vec2<f32>,
  focal:    vec2<f32>,
  splat_count: u32,
  _pad: u32,
  _pad2: vec2<u32>,
};
@group(0) @binding(0) var<storage, read>       pc_splats   : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> pc_inst     : array<Instance>;
@group(0) @binding(2) var<storage, read_write> pc_keys     : array<u32>;
@group(0) @binding(3) var<storage, read_write> pc_indices  : array<u32>;
@group(0) @binding(4) var<uniform>             pc_u        : ProjectUniforms;
@group(0) @binding(5) var<storage, read>       pc_compact  : array<u32>;

fn cov3d_pc(scale: vec3<f32>, q: vec4<f32>) -> array<f32, 6> {
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
fn cs_project_cmpct(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= pc_u.splat_count) { return; }   // splat_count here = survivors

  let si = pc_compact[i];
  let s = pc_splats[si];
  let pos = s.pos.xyz;
  let opacity = s.pos.w;

  let clip = pc_u.view_proj * vec4<f32>(pos, 1.0);
  let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
  let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

  let vz = pc_u.view[0][2] * pos.x + pc_u.view[1][2] * pos.y + pc_u.view[2][2] * pos.z + pc_u.view[3][2];
  let depth = -vz;
  let behind = depth <= 0.0;

  let V = cov3d_pc(s.scale.xyz, s.rot);
  let w0 = vec3<f32>(pc_u.view[0][0], pc_u.view[1][0], pc_u.view[2][0]);
  let w1 = vec3<f32>(pc_u.view[0][1], pc_u.view[1][1], pc_u.view[2][1]);
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
  let jx = pc_u.focal.x / z;
  let jy = pc_u.focal.y / z;
  let reg = 0.3;
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
  pc_inst[i] = inst;

  let dpos = max(depth, 0.0);
  let kd = bitcast<u32>(dpos);
  pc_keys[i] = 0xffffffffu - kd;
  pc_indices[i] = i;
}
