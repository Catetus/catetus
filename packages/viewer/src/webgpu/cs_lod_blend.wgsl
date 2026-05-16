// SPDX-License-Identifier: Apache-2.0
//
// LODGE Phase A.3 — per-splat boundary-blend opacity modulation
// (LODGE eq. 4).
//
// Input  : decoded splats[] (the canonical 64-byte DecodedSplat layout)
//          + per-splat `chunk_id` side buffer
//          + per-chunk activation records (output of cs_lod_select)
// Output : modulated splats[] (opacity field rewritten with t_blend)
//
// Pipeline:
//
//   1. cs_lod_select decided per-chunk `active` + per-chunk `t_blend`.
//   2. This kernel touches every decoded splat, looks up its chunk's
//      activation record, and writes:
//        - If `active == 0`: opacity := 0 (the splat is dropped by the
//          downstream cull predicate which gates on opacity ≥ tau).
//        - If `active == 1`: opacity := opacity * t_blend.
//      In both cases the splat record stays in the buffer; the cull /
//      sort / project flow handles the rest. We never delete splats —
//      just zero their alpha so the existing fast-path predicate prunes
//      them with no extra plumbing.
//
// Determinism: 1 thread per splat, pure functional update to splat.pos.w
// (which is where the canonical decoded splat stores opacity — see
// DecodedSplat struct in decode.wgsl).
//
// Math: LODGE paper eq. 4. The per-chunk multiplier is pre-computed in
// cs_lod_select (boundary_blend_t + slot resolution) so this kernel is
// branch-free per splat. The "near" chunk gets `t_blend = 1 - t` and the
// "far" chunk gets `t_blend = t`; intersection / single-chunk pinch gets
// `t_blend = 1`.

struct DecodedSplat {
  pos:   vec4<f32>,   // .xyz position, .w opacity
  scale: vec4<f32>,
  rot:   vec4<f32>,
  color: vec4<f32>,
};

struct ChunkActivation {
  level:   u32,
  active:  u32,
  slot:    u32,
  t_blend: f32,
};

struct LodBlendUniforms {
  splat_count: u32,
  // We can't reach the activation buffer via an SoA chunk_id directly
  // (chunk_ids are stored alongside splats but indexed per-splat). The
  // host packs the active set as a flattened per-splat u32 — the
  // `chunk_id` side buffer — so the GPU just does a single indirect
  // lookup. The activation buffer is indexed by chunk_id.
  chunk_count: u32,
  // When `force_passthrough == 1`, this kernel becomes a no-op (used by
  // the test harness to bypass the blend without re-creating buffers).
  force_passthrough: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> lb_splats     : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read>       lb_chunk_id   : array<u32>;
@group(0) @binding(2) var<storage, read>       lb_activation : array<ChunkActivation>;
@group(0) @binding(3) var<uniform>             lb_u          : LodBlendUniforms;

@compute @workgroup_size(256)
fn cs_lod_blend(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= lb_u.splat_count) { return; }
  if (lb_u.force_passthrough == 1u) { return; }

  let cid = lb_chunk_id[i];
  // Bounds-check: out-of-range chunk_id implies a streaming bug; drop
  // the splat by zeroing its opacity instead of letting the OOB read
  // poison the activation.
  if (cid >= lb_u.chunk_count) {
    lb_splats[i].pos.w = 0.0;
    return;
  }

  let act = lb_activation[cid];

  // Active chunks: modulate opacity by the LODGE eq. 4 ramp.
  // Inactive chunks (wrong level for this camera, OR culled by the
  // CPU-side budgetier): zero the alpha so the cull/project pipeline
  // skips them.
  var alpha = lb_splats[i].pos.w;
  if (act.active == 0u) {
    alpha = 0.0;
  } else {
    alpha = alpha * act.t_blend;
  }
  lb_splats[i].pos.w = alpha;
}

// =============================================================================
// cs_lod_blend_reset — restore splat opacities from a backup buffer.
//
// The blend kernel mutates `splats[].pos.w` in place. To support the
// "blend → project → restore → re-blend next frame" cycle without
// re-decoding the chunks, we keep a side `original_alpha[]` buffer
// populated at chunk-upload time, and this kernel writes it back into
// the splat records before each new blend. This is cheaper than
// re-decoding (which would re-run cs_decode over the same SoA bytes).
//
// Bind group:
//   0 (read_write) : splats[]
//   1 (read)       : original_alpha[]   (f32, 1 per splat)
//   2 (uniform)    : { splat_count, _pad×3 }
// =============================================================================
struct ResetUniforms {
  splat_count: u32,
  _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read_write> lr_splats : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read>       lr_alpha  : array<f32>;
@group(0) @binding(2) var<uniform>             lr_u      : ResetUniforms;

@compute @workgroup_size(256)
fn cs_lod_alpha_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= lr_u.splat_count) { return; }
  lr_splats[i].pos.w = lr_alpha[i];
}
