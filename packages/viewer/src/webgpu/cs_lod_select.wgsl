// SPDX-License-Identifier: Apache-2.0
//
// LODGE Phase A.3 — per-chunk LOD selection compute pass.
//
// One thread per chunk (workgroup 64). Reads the chunk's centroid + radius
// + per-level depth-threshold table, then writes a per-chunk activation
// record:
//
//   - `level`   : chosen LOD level for this chunk (0 = finest)
//   - `active`  : 1 = chunk is in the union of "near" + "far" slots this
//                 frame, 0 = drop. The runtime allocates `active=1` chunks
//                 to either the near slot or the far slot per LODGE eq. 4.
//   - `slot`    : 0 = near (camera-side), 1 = far (transition partner).
//                 Used by cs_lod_blend to read the correct centroid pair
//                 when computing the boundary-blend t-ramp.
//   - `t_blend` : LODGE eq. 4 boundary-blend parameter, normalized to
//                 [0, 1]. The runtime applies this as `alpha' = alpha *
//                 t_blend` for the symmetric-difference splats inside
//                 cs_lod_blend. Splats in the union intersection get
//                 `t_blend = 1` (fully opaque).
//
// The kernel is the GPU-side embodiment of LODGE eq. 2 (per-band active
// set) + eq. 4 (boundary-blend ramp). The CPU-side selector in
// `lodge/manifest.ts::selectLodLevel` is a coarser one-level-per-frame
// fallback that we keep for browsers without compute-shader access.
//
// Determinism: pure functions of (camera, chunk centroids, depth
// thresholds, per-level radius). No atomics, no inter-thread comms.
// Two runs with the same uniforms produce bit-identical outputs.
//
// Bind group:
//   0 (read)    : chunks[]              — ChunkDesc per chunk (all levels
//                                          flattened; level/index encoded in
//                                          the record).
//   1 (read)    : levels[]              — per-level uniform record
//                                          (depth_threshold, level index).
//                                          MAX_LOD_LEVELS = 8 slots; unused
//                                          entries carry depth_threshold =
//                                          +inf so the level is never
//                                          chosen.
//   2 (read_write) : chunk_activation[] — per-chunk output record.
//   3 (uniform) : LodSelectUniforms.

struct ChunkDesc {
  centroid: vec4<f32>,   // .xyz = centroid, .w = radius
  // The "owning" level for this chunk record. Phase A.3 emits one
  // ChunkDesc per (level, chunk_index); the kernel picks the level whose
  // depth-threshold band contains the camera distance.
  level: u32,
  chunk_index: u32,
  splat_count: u32,
  _pad: u32,
};

struct LevelDesc {
  depth_threshold: f32,  // LODGE eq. 2 band edge for this level
  level: u32,            // == array index, but stored for safety
  _pad0: u32,
  _pad1: u32,
};

struct ChunkActivation {
  level: u32,            // chosen level (0 if !active)
  active: u32,           // 1 = include this chunk this frame
  slot: u32,             // 0 = near, 1 = far
  t_blend: f32,          // LODGE eq. 4 ramp in [0, 1]
};

struct LodSelectUniforms {
  camera_pos:        vec4<f32>,   // .xyz = world-space camera position
  scene_center:      vec4<f32>,   // .xyz = scene-bbox centroid (anchor for
                                  //         level pick)
  // Per-level depth thresholds (LODGE eq. 2). We mirror the per-level
  // record in `LevelDesc` here for cache locality — one uniform read per
  // thread instead of an indirect into the level buffer.
  // depth_threshold[i] = nominal band edge for level i. depth_threshold[0]
  // is always 0. Unused slots are set to +inf.
  depth_thresholds:  vec4<f32>,   // levels 0..3
  depth_thresholds2: vec4<f32>,   // levels 4..7
  // Packed (chunk_count, num_levels, pad, pad) — explicit vec4-of-u32 to
  // keep WGSL alignment in lock-step with the JS-side byte encoder.
  // Without this packing, WGSL would silently pad after `num_levels` to
  // re-align `near_centroid: vec4<f32>`, and the JS encoder would have
  // to mirror that pad — easy to get wrong.
  counts:            vec4<u32>,
  // Near-chunk centroid for the boundary blend (LODGE eq. 4). The runtime
  // picks the camera-closest chunk on the CPU side and writes its centroid
  // here so the GPU has the m_o anchor for the ramp.
  near_centroid:     vec4<f32>,   // .xyz = m_o (closer-chunk centroid)
  far_centroid:      vec4<f32>,   // .xyz = m_f (transition partner)
  // (ss_size_threshold, focal_y, _pad, _pad) — same packing rationale as
  // `counts` above. Treat as a vec4<f32> so the WGSL struct is exactly 8
  // vec4 = 128 bytes total, identical to the JS encoder layout.
  ss_focal:          vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       ls_chunks      : array<ChunkDesc>;
@group(0) @binding(1) var<storage, read>       ls_levels      : array<LevelDesc>;
@group(0) @binding(2) var<storage, read_write> ls_activation  : array<ChunkActivation>;
@group(0) @binding(3) var<uniform>             ls_u           : LodSelectUniforms;

/// Read level i's depth threshold from the packed vec4 uniforms.
fn depth_threshold_at(i: u32) -> f32 {
  switch (i) {
    case 0u: { return ls_u.depth_thresholds.x; }
    case 1u: { return ls_u.depth_thresholds.y; }
    case 2u: { return ls_u.depth_thresholds.z; }
    case 3u: { return ls_u.depth_thresholds.w; }
    case 4u: { return ls_u.depth_thresholds2.x; }
    case 5u: { return ls_u.depth_thresholds2.y; }
    case 6u: { return ls_u.depth_thresholds2.z; }
    default: { return ls_u.depth_thresholds2.w; }
  }
}

/// LODGE eq. 4 boundary-blend ramp. Computes the projection of
/// `(c - m_o)` onto `(m_f - m_o)`, normalized by `||m_o - m_f||²`, clamped
/// to [0, 1]. The result is 0 when the camera is at m_o, 1 when at m_f.
/// For symmetric-difference splats: opacity multiplier is t for splats
/// owned by m_f, (1 - t) for splats owned by m_o. Intersection splats keep
/// t = 1.
fn boundary_blend_t(camera: vec3<f32>, m_o: vec3<f32>, m_f: vec3<f32>) -> f32 {
  let v = m_f - m_o;
  let denom = max(dot(v, v), 1e-8);
  let raw = dot(camera - m_o, v) / denom;
  return clamp(raw, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn cs_lod_select(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ls_u.counts.x) { return; }

  let chunk = ls_chunks[i];
  let centroid = chunk.centroid.xyz;
  let radius = chunk.centroid.w;
  let camera = ls_u.camera_pos.xyz;

  // Distance from camera to chunk centroid (eq. 2 anchor).
  let dvec = camera - centroid;
  let dist = sqrt(dot(dvec, dvec));
  // LODGE allows a "near" slack of `radius` so chunks at the camera edge
  // don't pop. Subtract the chunk radius from the band-edge comparison.
  let dist_eff = max(dist - radius, 0.0);

  // ----- Camera-distance band: pick the coarsest level whose
  //       depth_threshold ≤ dist_eff. -----
  var chosen_level: u32 = 0u;
  for (var l: u32 = 0u; l < ls_u.counts.y; l = l + 1u) {
    let t = depth_threshold_at(l);
    if (dist_eff >= t) {
      chosen_level = l;
    }
  }

  // ----- Screen-space size heuristic: if the chunk's projected screen
  //       radius (focal * radius / dist) drops below the SS threshold,
  //       bump to the next coarser level. -----
  let safe_dist = max(dist, 1e-3);
  let ss_radius = ls_u.ss_focal.y * radius / safe_dist;
  if (ss_radius < ls_u.ss_focal.x && chosen_level + 1u < ls_u.counts.y) {
    chosen_level = chosen_level + 1u;
  }

  // ----- Activation: a chunk is active iff its own record's level matches
  //       the chosen level. The runtime emits one ChunkDesc per
  //       (level, chunk_index); only the records on the chosen level get
  //       `active = 1`. This is the GPU-side embodiment of "select the
  //       right LOD level for this spatial region".
  var act: ChunkActivation;
  act.level = chosen_level;
  act.active = 0u;
  act.slot = 0u;
  act.t_blend = 1.0;

  if (chunk.level == chosen_level) {
    act.active = 1u;

    // ----- Slot + boundary blend. Compute distance from this chunk's
    //       centroid to the near and far centroids; this chunk lives in
    //       the slot whose centroid is closer.
    let d_near = length(centroid - ls_u.near_centroid.xyz);
    let d_far  = length(centroid - ls_u.far_centroid.xyz);
    let is_far = d_far < d_near;
    act.slot = select(0u, 1u, is_far);

    // ----- LODGE eq. 4 t-ramp. The blend parameter is shared by every
    //       chunk in the active union; we evaluate it once per chunk so
    //       the downstream blend kernel can apply it as a per-splat
    //       opacity multiplier without re-fetching the uniforms. -----
    let t = boundary_blend_t(camera, ls_u.near_centroid.xyz, ls_u.far_centroid.xyz);
    // Near chunk: opacity *= (1 - t). Far chunk: opacity *= t. We store
    // the per-chunk multiplier directly so the blend kernel doesn't have
    // to branch on slot.
    act.t_blend = select(1.0 - t, t, is_far);

    // For chunks in BOTH near and far set ranges (i.e. intersection),
    // the runtime collapses the duplicate by setting t_blend = 1 on the
    // canonical entry. Detection: if near == far centroids, this is a
    // single-chunk pinch; force t = 1.
    let near_eq_far = distance(ls_u.near_centroid.xyz, ls_u.far_centroid.xyz) < 1e-4;
    if (near_eq_far) {
      act.t_blend = 1.0;
    }
  }

  ls_activation[i] = act;
}
