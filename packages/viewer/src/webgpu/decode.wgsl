// SPDX-License-Identifier: Apache-2.0
//
// WebGPU compute decode shader for SplatForge SoA splat chunks.
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

  var out: DecodedSplat;
  out.pos   = vec4<f32>(px, py, pz, opacity);
  out.scale = vec4<f32>(sx, sy, sz, 0.0);
  out.rot   = vec4<f32>(rx, ry, rz, rw);
  out.color = vec4<f32>(cr, cg, cb, 1.0);
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

  // View-space depth: column-major mat4x4 indexed as m[col][row]; row 2 of the
  // view matrix dotted with (pos, 1) is `-z` of the view-space point.
  let vz = pu.view[0][2] * pos.x + pu.view[1][2] * pos.y + pu.view[2][2] * pos.z + pu.view[3][2];
  let depth = -vz;
  let behind = depth <= 0.0;

  // Reconstruct 3D covariance, project into 2D screen covariance.
  let V = cov3d(s.scale.xyz, s.rot);
  // Symmetric Σ entries:
  //   V[0]=σxx, V[1]=σxy, V[2]=σxz, V[3]=σyy, V[4]=σyz, V[5]=σzz.
  // World→view rotation rows.
  let w0 = vec3<f32>(pu.view[0][0], pu.view[1][0], pu.view[2][0]);
  let w1 = vec3<f32>(pu.view[0][1], pu.view[1][1], pu.view[2][1]);
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
  let jx = pu.focal.x / z;
  let jy = pu.focal.y / z;
  let reg = 0.3; // SF_EWA_DILATION
  let c00 = jx * jx * vxx + reg;
  let c01 = jx * jy * vxy;
  let c11 = jy * jy * vyy + reg;

  // 3σ radius (px) from largest eigenvalue of the 2x2.
  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 0.0);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  var radius = 3.0 * sqrt(max(lambda_max, 0.0));
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
