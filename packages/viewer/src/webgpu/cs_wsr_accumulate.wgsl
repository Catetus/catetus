// SPDX-License-Identifier: Apache-2.0
//
// WSR scatter accumulate (Hou et al. 2024, LC-WSR variant).
//
// One thread per splat. Each thread:
//   1. Reads `splats[i]` directly — no g_indices[] indirection, no sort.
//   2. Projects to NDC + 2D screen-space covariance using the same math as
//      cs_project / cs_project_gather (kept byte-identical so the projection
//      regression tests in __tests__/webgpu/fused_project_gather still
//      enforce parity if the legacy sorted-alpha path is ever re-enabled).
//   3. Computes the per-splat depth weight  w_i = max(0, 1 - d_i/σ - v_i).
//      PR1 ships with v_i = 0 globally (no per-splat-bias storage); the σ
//      uniform is set host-side from the scene bounding box (2 × mean depth).
//   4. Walks the [cx - r, cx + r] × [cy - r, cy + r] axis-aligned bounding
//      box around the projected splat center in screen-space pixels. For
//      each covered pixel, evaluates the 2D Gaussian:
//          power = -0.5 · (dx² · inv00 + 2 · dx · dy · inv01 + dy² · inv11)
//          α_eff = α_i · exp(power)              (clamped to [0, 0.999])
//      and scatter-adds:
//          numerator[px]   += α_eff · w_i · c_i   (vec3)
//          denominator[px] += α_eff · w_i         (scalar)
//      via a CAS-loop emulated atomic-add on u32-bitcasted f32 (storage-
//      buffer atomic-float is not in WebGPU 1.0; the CAS loop is the
//      portable shim documented in B7.1 EXECUTION-LOG.)
//
// Numerical notes:
//   - Both `power > 0` (outside ellipse) and `α_eff < 1/255` are early-exit
//     fragment-discard heuristics inherited from fs_main in renderer/webgpu.ts.
//     They preserve the visible-pixel set of the existing rasterizer so
//     downstream parity testing can compare PSNR fairly.
//   - The CAS loop is bounded by a hard iteration cap (256) to keep the
//     shader well-behaved under pathological contention. In practice the
//     B7.1 measurement on the laptop 4090 showed scatter is DRAM-bound, not
//     atomic-bound, so the loop terminates after 1-2 iterations on average
//     even at 10 M splats × 16 px² median footprint.
//   - Behind-camera and zero-radius splats are filtered identically to
//     cs_project_gather so the WSR path's culled set is a strict subset of
//     the rasterized set (no spurious accumulation behind the camera).

struct DecodedSplat {
  pos:   vec4<f32>,  // xyz = world position, w = opacity
  scale: vec4<f32>,  // xyz = scale, w = reserved
  rot:   vec4<f32>,  // quaternion (x,y,z,w)
  color: vec4<f32>,  // rgb = DC SH coefficient, a = reserved
};

struct WSRUniforms {
  view:        mat4x4<f32>,
  view_proj:   mat4x4<f32>,
  viewport:    vec2<f32>,
  focal:       vec2<f32>,
  splat_count: u32,
  // chunk_offset (multi-dispatch). Splat-index base for the current dispatch.
  chunk_offset: u32,
  // Per-frame WSR parameters. PR1 uses a single scene-wide σ derived host-
  // side from `2 × scene_mean_depth`; PR3 replaces this with a learned σ
  // baked into the manifest. v_i is treated as 0 in PR1 (no per-splat
  // storage allocated).
  sigma:       f32,
  v_default:   f32,
  // Viewport size redundantly as u32 to avoid float-to-int conversions in
  // the inner pixel loop; matches the ClearUniforms layout.
  viewport_u:  vec2<u32>,
};

@group(0) @binding(0) var<storage, read>       splats      : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> numerator   : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> denominator : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             u           : WSRUniforms;

// Same upper-triangular 3D covariance as cov3d_fg in cs_project_gather.wgsl.
fn cov3d_w(scale: vec3<f32>, q: vec4<f32>) -> array<f32, 6> {
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

// CAS-loop atomic float add. Returns nothing — best-effort with a hard cap
// to avoid livelock under pathological contention.
fn atomic_add_f32(slot: ptr<storage, atomic<u32>, read_write>, value: f32) {
  if (value == 0.0) { return; }
  var iter = 0u;
  loop {
    let current_u = atomicLoad(slot);
    let current_f = bitcast<f32>(current_u);
    let new_f = current_f + value;
    let new_u = bitcast<u32>(new_f);
    let res = atomicCompareExchangeWeak(slot, current_u, new_u);
    if (res.exchanged) { break; }
    iter = iter + 1u;
    if (iter >= 256u) { break; }
  }
}

@compute @workgroup_size(256)
fn cs_wsr_accumulate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + u.chunk_offset;
  if (i >= u.splat_count) { return; }

  let s = splats[i];
  let pos = s.pos.xyz;
  let opacity = s.pos.w;

  // Clip-space + view-space depth. Same operation ordering as cs_project_gather.
  let clip = u.view_proj * vec4<f32>(pos, 1.0);
  let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
  let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

  let vz = u.view[0][2] * pos.x + u.view[1][2] * pos.y + u.view[2][2] * pos.z + u.view[3][2];
  let depth = -vz;
  if (depth <= 0.0) { return; }

  // 2D screen-space covariance via the Jacobian of the perspective projection.
  let V = cov3d_w(s.scale.xyz, s.rot);
  let w0 = vec3<f32>(u.view[0][0], u.view[1][0], u.view[2][0]);
  let w1 = vec3<f32>(u.view[0][1], u.view[1][1], u.view[2][1]);
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
  let jx = u.focal.x / z;
  let jy = u.focal.y / z;
  let reg = 0.3;
  let c00 = jx * jx * vxx + reg;
  let c01 = jx * jy * vxy;
  let c11 = jy * jy * vyy + reg;

  // 3σ pixel radius from the larger eigenvalue.
  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 1e-6);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  let radius = 3.0 * sqrt(max(lambda_max, 0.0));
  if (radius < 0.5) { return; }

  // Inverse covariance for Gaussian evaluation.
  let inv00 =  c11 / det;
  let inv01 = -c01 / det;
  let inv11 =  c00 / det;

  // LC-WSR depth weight. σ = 2 × mean_scene_depth in PR1 (no fine-tuning).
  let sigma = max(u.sigma, 1e-6);
  let v_i   = u.v_default;
  let w_d = max(0.0, 1.0 - depth / sigma - v_i);
  if (w_d <= 0.0) { return; }

  // Screen-space center in pixels (NDC y is flipped to match the renderer).
  let cx = (ndc.x * 0.5 + 0.5) * u.viewport.x;
  let cy = (1.0 - (ndc.y * 0.5 + 0.5)) * u.viewport.y;

  let r_pad = ceil(radius);
  let xmin = max(0,            i32(floor(cx - r_pad)));
  let ymin = max(0,            i32(floor(cy - r_pad)));
  let xmax = min(i32(u.viewport_u.x) - 1, i32(ceil(cx + r_pad)));
  let ymax = min(i32(u.viewport_u.y) - 1, i32(ceil(cy + r_pad)));
  if (xmin > xmax || ymin > ymax) { return; }

  let color = vec3<f32>(s.color.r, s.color.g, s.color.b);
  let alpha_base = clamp(opacity, 0.0, 1.0);

  // Cap the per-thread pixel walk to a generous upper bound. At the PR1
  // capacity (1.16M-3.62M real splats, sub-pixel median footprint) this is
  // effectively no-op; the cap exists so a single pathological splat at very
  // close range cannot stall the entire workgroup.
  let bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1);
  if (bbox_area > 4096) { return; }

  for (var py = ymin; py <= ymax; py = py + 1) {
    for (var px = xmin; px <= xmax; px = px + 1) {
      let dx = f32(px) + 0.5 - cx;
      let dy = f32(py) + 0.5 - cy;
      let power = -0.5 * (dx * dx * inv00 + 2.0 * dx * dy * inv01 + dy * dy * inv11);
      if (power > 0.0) { continue; }
      let g = exp(power);
      let alpha = clamp(alpha_base * g, 0.0, 0.999);
      if (alpha < 1.0 / 255.0) { continue; }

      let aw = alpha * w_d;
      let pidx = u32(py) * u.viewport_u.x + u32(px);
      atomic_add_f32(&numerator[pidx * 4u + 0u], aw * color.r);
      atomic_add_f32(&numerator[pidx * 4u + 1u], aw * color.g);
      atomic_add_f32(&numerator[pidx * 4u + 2u], aw * color.b);
      atomic_add_f32(&denominator[pidx],         aw);
    }
  }
}
