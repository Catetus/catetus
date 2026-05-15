// SPDX-License-Identifier: Apache-2.0
//
// WSR tile binning kernel (Phase 1 of the tile-prefix-sum scatter path).
//
// Salvages the WSR rendering path after B8 PR2 KILL: the original
// cs_wsr_accumulate.wgsl uses a CAS-loop atomic float-add scatter, which
// hit the 256-iter cap on pixels covered by many splats (bonsai: 0.29 fps
// / 17 dB PSNR). The tile-prefix-sum path replaces per-pixel atomic
// contention with two cheap stages:
//
//   1. (this kernel) bin each splat into the per-tile splat list, using
//      ONE atomic counter per (splat, overlapping-tile) pair. The atomic
//      domain is now ~256 tiles (1080p / 16²) rather than ~2M pixels —
//      ≈4 orders of magnitude lower contention.
//   2. (cs_wsr_tile_accumulate) one workgroup per tile, reads the tile's
//      splat list into workgroup-shared memory, accumulates per-pixel
//      contributions into thread-private registers (NO atomics inside the
//      workgroup), final coalesced write to numerator/denominator.
//
// Per-tile list cap (`MAX_PER_TILE`) is enforced via an atomic-add fence:
// if the post-add count exceeds the cap, the contribution is silently
// dropped (the count saturates). Real-world tile occupancy on the
// validation scenes is well below the cap; pathological scenes will
// degrade gracefully (some splats will not render in over-saturated tiles)
// rather than livelock.
//
// Compatibility: the splat unpack + projection + footprint math is kept
// byte-identical to cs_wsr_accumulate.wgsl so the two paths produce the
// same visible-pixel set (modulo the per-tile overflow behaviour above).

struct DecodedSplat {
  pos:   vec4<f32>,  // xyz = world position, w = opacity
  scale: vec4<f32>,  // xyz = scale, w = reserved
  rot:   vec4<f32>,  // quaternion (x,y,z,w)
  color: vec4<f32>,  // rgb = DC SH coefficient, a = reserved
};

struct TileBinUniforms {
  view:        mat4x4<f32>,
  view_proj:   mat4x4<f32>,
  viewport:    vec2<f32>,
  focal:       vec2<f32>,
  splat_count: u32,
  tile_size:   u32,          // tile edge in pixels (16)
  tiles_x:     u32,          // ceil(W / tile_size)
  tiles_y:     u32,          // ceil(H / tile_size)
  max_per_tile: u32,         // hard cap on per-tile list length
  _pad0:        u32,
  sigma:        f32,
  v_default:    f32,
};

@group(0) @binding(0) var<storage, read>       splats      : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> tile_count  : array<atomic<u32>>; // [tiles_x * tiles_y]
@group(0) @binding(2) var<storage, read_write> tile_lists  : array<u32>;         // [tiles_x * tiles_y * max_per_tile]
@group(0) @binding(3) var<uniform>             u           : TileBinUniforms;

// Same upper-triangular 3D covariance helper as cs_wsr_accumulate.wgsl.
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

@compute @workgroup_size(256)
fn cs_tile_bin(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u.splat_count) { return; }

  let s = splats[i];
  let pos = s.pos.xyz;

  let clip = u.view_proj * vec4<f32>(pos, 1.0);
  let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
  let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

  let vz = u.view[0][2] * pos.x + u.view[1][2] * pos.y + u.view[2][2] * pos.z + u.view[3][2];
  let depth = -vz;
  if (depth <= 0.0) { return; }

  // Per-splat LC-WSR depth weight gate: if w_d ≤ 0 the splat contributes
  // nothing to the WSR sum, so it doesn't need a tile-list slot.
  let sigma = max(u.sigma, 1e-6);
  let w_d = max(0.0, 1.0 - depth / sigma - u.v_default);
  if (w_d <= 0.0) { return; }

  // 2D screen-space covariance (same math as cs_wsr_accumulate.wgsl).
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

  let half_trace = 0.5 * (c00 + c11);
  let det = max(c00 * c11 - c01 * c01, 1e-6);
  let term = sqrt(max(half_trace * half_trace - det, 0.0));
  let lambda_max = half_trace + term;
  let radius = 3.0 * sqrt(max(lambda_max, 0.0));
  if (radius < 0.5) { return; }

  // Screen-space center in pixels (NDC y flipped to match renderer).
  let cx = (ndc.x * 0.5 + 0.5) * u.viewport.x;
  let cy = (1.0 - (ndc.y * 0.5 + 0.5)) * u.viewport.y;

  let r_pad = ceil(radius);
  let xmin = max(0.0, floor(cx - r_pad));
  let ymin = max(0.0, floor(cy - r_pad));
  let xmax = min(u.viewport.x - 1.0, ceil(cx + r_pad));
  let ymax = min(u.viewport.y - 1.0, ceil(cy + r_pad));
  if (xmin > xmax || ymin > ymax) { return; }

  // Pathological-footprint guard: same 4096-pixel cap as cs_wsr_accumulate.
  // A splat covering > 4096 px is almost certainly a numerical degenerate
  // (very close to camera). We skip it in both binning and accumulate to
  // keep the two paths' visible-pixel set identical.
  let bbox_area = (xmax - xmin + 1.0) * (ymax - ymin + 1.0);
  if (bbox_area > 4096.0) { return; }

  let ts = f32(u.tile_size);
  let tx_min = u32(floor(xmin / ts));
  let ty_min = u32(floor(ymin / ts));
  // Clamp to tile-grid extent (ceil-divide upper bound).
  let tx_max = min(u.tiles_x - 1u, u32(floor(xmax / ts)));
  let ty_max = min(u.tiles_y - 1u, u32(floor(ymax / ts)));

  for (var ty = ty_min; ty <= ty_max; ty = ty + 1u) {
    for (var tx = tx_min; tx <= tx_max; tx = tx + 1u) {
      let tile_idx = ty * u.tiles_x + tx;
      let slot = atomicAdd(&tile_count[tile_idx], 1u);
      if (slot < u.max_per_tile) {
        tile_lists[tile_idx * u.max_per_tile + slot] = i;
      }
      // Overflow: counter keeps incrementing; the accumulator pass reads
      // min(count, max_per_tile) splats. Drop is silent + bounded.
    }
  }
}

@compute @workgroup_size(64)
fn cs_tile_bin_clear(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let total = u.tiles_x * u.tiles_y;
  if (i >= total) { return; }
  atomicStore(&tile_count[i], 0u);
}
