// SPDX-License-Identifier: Apache-2.0
//
// WSR per-tile accumulate kernel (Phase 2 of the tile-prefix-sum scatter path).
//
// One workgroup per tile, 16×16 = 256 threads, one thread per pixel.
//
// The B8 PR2 KILL (CAS-loop atomic float-add scatter, 0.29 fps / 17 dB on
// bonsai) was caused by per-pixel atomic contention: many splats landing
// on the same hot pixel hit the 256-iter CAS retry cap. This kernel
// eliminates per-pixel atomic contention by inverting the loop:
//
//   for each tile (one workgroup):
//     load tile's splat-index list (cooperative read into shared mem)
//     for each thread (one pixel):
//       num_r = num_g = num_b = den = 0   (thread-private registers)
//       for each splat in the tile's list:
//         (read splat, compute α·w(d)·c at this pixel, add to registers)
//       write {num, den} once to global storage (no atomic, no contention)
//
// The only contention point in the whole tile-prefix-sum pipeline is the
// cs_tile_bin atomic-add into per-tile counters (~256 tiles, ~10^4× less
// contention than per-pixel atomics).
//
// Per-tile splat list is read directly from global memory by each thread.
// We do NOT materialize the splat records in shared memory because the
// per-tile list can be up to MAX_PER_TILE (e.g. 16384) and the per-splat
// record is 64 B → that would need 1 MB of shared memory, far above the
// 16 KB workgroup-shared cap.
//
// Instead, we use a small `WORKGROUP_BATCH` chunk of indices: the
// workgroup cooperatively loads `WORKGROUP_BATCH` splat indices into
// shared memory, every thread evaluates that batch against its pixel,
// then we advance to the next batch. This amortises the global-memory
// index read across all 256 threads.

struct DecodedSplat {
  pos:   vec4<f32>,
  scale: vec4<f32>,
  rot:   vec4<f32>,
  color: vec4<f32>,
};

struct TileAccumUniforms {
  view:        mat4x4<f32>,
  view_proj:   mat4x4<f32>,
  viewport:    vec2<f32>,
  focal:       vec2<f32>,
  splat_count: u32,
  tile_size:   u32,
  tiles_x:     u32,
  tiles_y:     u32,
  max_per_tile: u32,
  viewport_x:  u32,
  viewport_y:  u32,
  _pad0:       u32,
  sigma:       f32,
  v_default:   f32,
  _pad1:       u32,
  _pad2:       u32,
};

@group(0) @binding(0) var<storage, read>       splats      : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read>       tile_count  : array<u32>;
@group(0) @binding(2) var<storage, read>       tile_lists  : array<u32>;
@group(0) @binding(3) var<storage, read_write> numerator   : array<u32>;   // bitcast f32, 4 slots/px
@group(0) @binding(4) var<storage, read_write> denominator : array<u32>;   // bitcast f32, 1 slot/px
@group(0) @binding(5) var<uniform>             u           : TileAccumUniforms;

// Workgroup-shared batch of splat indices. Sized so the whole batch fits in
// ~1 KB of shared memory (256 u32). Each accumulate iteration:
//   - 256 threads cooperatively load 256 indices from the tile list.
//   - All threads evaluate every loaded splat against their pixel.
const BATCH : u32 = 256u;
var<workgroup> shared_idx : array<u32, 256>;

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

@compute @workgroup_size(16, 16)
fn cs_wsr_tile_accumulate(
  @builtin(workgroup_id) wg_id : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(local_invocation_index) lin : u32,
) {
  let tile_idx = wg_id.y * u.tiles_x + wg_id.x;
  if (wg_id.x >= u.tiles_x || wg_id.y >= u.tiles_y) { return; }

  // This thread's pixel in screen-space.
  let px_x = wg_id.x * u.tile_size + lid.x;
  let px_y = wg_id.y * u.tile_size + lid.y;
  let in_bounds = (px_x < u.viewport_x) && (px_y < u.viewport_y);

  // Per-thread accumulators (registers — no atomics).
  var num_r = 0.0;
  var num_g = 0.0;
  var num_b = 0.0;
  var den   = 0.0;

  let raw_count = tile_count[tile_idx];
  let count = min(raw_count, u.max_per_tile);

  let fx = f32(px_x) + 0.5;
  let fy = f32(px_y) + 0.5;
  let sigma = max(u.sigma, 1e-6);
  let v_i = u.v_default;

  // Batched loop: cooperatively prefetch BATCH indices then have every
  // thread evaluate those splats against its pixel.
  var base : u32 = 0u;
  loop {
    if (base >= count) { break; }
    let remaining = count - base;
    let take = min(BATCH, remaining);

    // Each of the 256 threads loads (at most) one index into shared mem.
    if (lin < take) {
      shared_idx[lin] = tile_lists[tile_idx * u.max_per_tile + base + lin];
    }
    workgroupBarrier();

    if (in_bounds) {
      // Evaluate every loaded splat against this thread's pixel.
      for (var k : u32 = 0u; k < take; k = k + 1u) {
        let si = shared_idx[k];
        let s = splats[si];
        let pos = s.pos.xyz;
        let opacity = s.pos.w;

        let clip = u.view_proj * vec4<f32>(pos, 1.0);
        let invW = select(1.0, 1.0 / clip.w, abs(clip.w) > 1e-12);
        let ndc = vec3<f32>(clip.x * invW, clip.y * invW, clip.z * invW);

        let vz = u.view[0][2] * pos.x + u.view[1][2] * pos.y + u.view[2][2] * pos.z + u.view[3][2];
        let depth = -vz;
        if (depth <= 0.0) { continue; }

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

        let det = max(c00 * c11 - c01 * c01, 1e-6);
        let inv00 =  c11 / det;
        let inv01 = -c01 / det;
        let inv11 =  c00 / det;

        let w_d = max(0.0, 1.0 - depth / sigma - v_i);
        if (w_d <= 0.0) { continue; }

        let cx = (ndc.x * 0.5 + 0.5) * u.viewport.x;
        let cy = (1.0 - (ndc.y * 0.5 + 0.5)) * u.viewport.y;
        let dx = fx - cx;
        let dy = fy - cy;
        let power = -0.5 * (dx * dx * inv00 + 2.0 * dx * dy * inv01 + dy * dy * inv11);
        if (power > 0.0) { continue; }
        let g = exp(power);
        let alpha_base = clamp(opacity, 0.0, 1.0);
        let alpha = clamp(alpha_base * g, 0.0, 0.999);
        if (alpha < 1.0 / 255.0) { continue; }

        let aw = alpha * w_d;
        num_r = num_r + aw * s.color.r;
        num_g = num_g + aw * s.color.g;
        num_b = num_b + aw * s.color.b;
        den   = den   + aw;
      }
    }

    workgroupBarrier();
    base = base + take;
  }

  // Final write — exactly one write per pixel slot, no atomics needed.
  if (in_bounds) {
    let pidx = px_y * u.viewport_x + px_x;
    numerator[pidx * 4u + 0u] = bitcast<u32>(num_r);
    numerator[pidx * 4u + 1u] = bitcast<u32>(num_g);
    numerator[pidx * 4u + 2u] = bitcast<u32>(num_b);
    numerator[pidx * 4u + 3u] = 0u;
    denominator[pidx]         = bitcast<u32>(den);
  }
}
