// SPDX-License-Identifier: Apache-2.0
//
// WSR per-frame accumulator clear.
//
// Zeros the numerator (RGB + unused, packed as 4 × u32-bitcast-of-f32) and
// denominator (1 × u32-bitcast-of-f32) atomic accumulator buffers at the
// start of each frame. WebGPU 1.0 has no portable storage-buffer clear
// command for atomic-typed bindings, so we do it in a compute kernel.
//
// 16 × 16 workgroup; one invocation per pixel. Each invocation writes 5 u32s.
//
// Buffer layout (must match cs_wsr_accumulate.wgsl and cs_wsr_resolve.wgsl):
//   numerator[px*4 + 0] = bit-pattern of f32 Σ (α·w·c.r)
//   numerator[px*4 + 1] = bit-pattern of f32 Σ (α·w·c.g)
//   numerator[px*4 + 2] = bit-pattern of f32 Σ (α·w·c.b)
//   numerator[px*4 + 3] = 0 (unused; kept for 16-byte vec4 alignment)
//   denominator[px]     = bit-pattern of f32 Σ (α·w)
//
// Zero is the bit-pattern 0x00000000 for both signed-zero f32 and u32, so a
// plain `atomicStore(..., 0u)` is correct and sets the accumulator to +0.0.

struct ClearUniforms {
  viewport: vec2<u32>,
  _pad:     vec2<u32>,
};

@group(0) @binding(0) var<storage, read_write> numerator   : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> denominator : array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             cu          : ClearUniforms;

@compute @workgroup_size(16, 16)
fn cs_wsr_clear(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= cu.viewport.x || y >= cu.viewport.y) { return; }
  let px = y * cu.viewport.x + x;
  atomicStore(&numerator[px * 4u + 0u], 0u);
  atomicStore(&numerator[px * 4u + 1u], 0u);
  atomicStore(&numerator[px * 4u + 2u], 0u);
  atomicStore(&numerator[px * 4u + 3u], 0u);
  atomicStore(&denominator[px], 0u);
}
