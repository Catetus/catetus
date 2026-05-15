// SPDX-License-Identifier: Apache-2.0
//
// WSR resolve pass.
//
// Reads the per-pixel accumulator buffers (numerator RGB sum, denominator
// scalar sum) and computes the final RGB:
//
//     C_px = (w_B · c_B + N_px) / (w_B + D_px)
//
// with constants `w_B = 1e-4` and `c_B = (0, 0, 0)` (black background — PR1
// scope; scene-aware background is a PR5 follow-up). The 1e-4 floor on the
// denominator prevents division by zero where no splat covered the pixel.
//
// Writes the final RGBA (alpha = 1) to the output storage texture. The
// caller decides whether to present that texture to the canvas directly
// (renderer integration) or to read it back for offline comparison (the
// `wsr.test.ts` unit test path).
//
// Buffer layout (matches cs_wsr_clear.wgsl and cs_wsr_accumulate.wgsl):
//   numerator[px*4 + 0..2] = bit-pattern of f32 channel sums
//   numerator[px*4 + 3]    = 0 (unused, kept for vec4 alignment)
//   denominator[px]        = bit-pattern of f32 scalar sum

struct ResolveUniforms {
  viewport: vec2<u32>,
  _pad:     vec2<u32>,
  bg_color: vec4<f32>,  // (R, G, B, w_B). Default (0, 0, 0, 1e-4).
};

@group(0) @binding(0) var<storage, read>       numerator   : array<u32>;
@group(0) @binding(1) var<storage, read>       denominator : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_rgba : array<u32>;
@group(0) @binding(3) var<uniform>             u           : ResolveUniforms;

// Pack a vec4<f32> RGBA in [0,1] into a single u32 with R in the low byte.
// Matches the canonical `rgba8unorm` byte order so the buffer can be used as
// a `COPY_SRC` for `copyBufferToTexture(rgba8unorm, ...)` downstream.
fn pack_rgba8(c: vec4<f32>) -> u32 {
  let v = clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + vec4<f32>(0.5);
  let r = u32(v.x);
  let g = u32(v.y);
  let b = u32(v.z);
  let a = u32(v.w);
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(16, 16)
fn cs_wsr_resolve(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= u.viewport.x || y >= u.viewport.y) { return; }
  let px = y * u.viewport.x + x;

  let n_r = bitcast<f32>(numerator[px * 4u + 0u]);
  let n_g = bitcast<f32>(numerator[px * 4u + 1u]);
  let n_b = bitcast<f32>(numerator[px * 4u + 2u]);
  let d   = bitcast<f32>(denominator[px]);

  let w_b   = u.bg_color.a;
  let c_b   = u.bg_color.rgb;
  let denom = w_b + d;
  let c     = (w_b * c_b + vec3<f32>(n_r, n_g, n_b)) / denom;

  output_rgba[px] = pack_rgba8(vec4<f32>(c.r, c.g, c.b, 1.0));
}
