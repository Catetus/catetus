// SPDX-License-Identifier: Apache-2.0
/**
 * LODGE Phase A.3 — TypeScript-side reference math for the per-chunk
 * LOD-selection compute pass + LODGE eq. 4 boundary-blend ramp.
 *
 * These functions are the JS-side equivalents of `cs_lod_select.wgsl` and
 * `cs_lod_blend.wgsl`. They exist for three reasons:
 *
 *   1. Unit-testable reference for the WGSL — see
 *      `__tests__/lodge-lod-math.test.ts`. The headless test environment
 *      (vitest) cannot run real WebGPU, so we exercise the algorithm
 *      with these JS equivalents and compare against expected outputs
 *      derived by hand from the LODGE paper.
 *   2. CPU-side fallback path for browsers / harnesses without compute-
 *      shader access. The {@link LodgeLODPipeline} can drive a render
 *      with the JS math producing the same per-chunk active-set decision
 *      that the GPU kernel would have.
 *   3. The same math is needed on the build side (CLI / bench) to
 *      precompute per-frame survivor counts when measuring fps proxies
 *      against `.lodge` directories without spinning up a browser.
 */

import type { Vec3 } from '../manifest.js';
import type { LodgeChunkDescriptor, LodgeManifest } from './manifest.js';

/** Slot id matching the WGSL `ChunkActivation.slot` field. */
export const LOD_SLOT_NEAR = 0;
/** Slot id matching the WGSL `ChunkActivation.slot` field. */
export const LOD_SLOT_FAR = 1;

/** Maximum LOD levels supported by `cs_lod_select.wgsl` (uniform fits 2×vec4 of f32). */
export const LOD_MAX_LEVELS = 8;

/**
 * Per-chunk activation decision produced by the LOD selector. Mirrors the
 * WGSL `ChunkActivation` struct field-for-field; the JS-side reference
 * test compares this directly against a CPU-emulated kernel output.
 */
export interface ChunkActivation {
  /** Chosen LOD level (0 = finest). 0 when inactive. */
  level: number;
  /** 1 = include in the active union this frame; 0 = drop. */
  active: number;
  /** {@link LOD_SLOT_NEAR} | {@link LOD_SLOT_FAR}. Meaningless when inactive. */
  slot: number;
  /** LODGE eq. 4 ramp value in [0, 1]. 1.0 means "no boundary fade". */
  tBlend: number;
}

/** One ChunkDesc record as fed to the WGSL `ls_chunks[]` buffer. */
export interface ChunkRecord {
  /** Centroid in world space. */
  centroid: Vec3;
  /** Bounding-sphere radius (max splat distance from centroid). */
  radius: number;
  /** Level this chunk record belongs to. */
  level: number;
  /** Chunk index inside its level. */
  chunkIndex: number;
  /** Splat count for budget tracking. */
  splatCount: number;
}

/** Per-frame inputs that the GPU uniform buffer carries. */
export interface LodSelectInputs {
  /** Camera world-space position. */
  cameraPos: Vec3;
  /** Scene-bbox centroid (anchor used for the JS-side fallback heuristic). */
  sceneCenter: Vec3;
  /** Per-level depth thresholds; LODGE eq. 2 band edges. Index = level. */
  depthThresholds: number[];
  /** Pre-computed near-chunk centroid (`m_o` in eq. 4). */
  nearCentroid: Vec3;
  /** Pre-computed far-chunk centroid (`m_f` in eq. 4). */
  farCentroid: Vec3;
  /** Min projected screen radius (pixels) before bumping to a coarser level. */
  ssSizeThreshold: number;
  /** Focal length in pixels (image_height / (2 tan(fov_y/2))). */
  focalY: number;
}

/**
 * LODGE eq. 4 boundary-blend ramp. Returns t ∈ [0, 1] as the projection of
 * `(camera - m_o)` onto `(m_f - m_o)`, normalized by `||m_o - m_f||²` and
 * clamped.
 *
 *   t = clamp( ((c - m_o) · (m_f - m_o)) / ||m_o - m_f||² , 0, 1 )
 *
 * At `camera == m_o` the result is 0; at `camera == m_f` the result is 1;
 * along the line between them it ramps linearly. Off-axis cameras
 * project onto the line — so the ramp stays smooth even when the camera
 * crosses chunk boundaries obliquely (paper §3.4).
 *
 * Degenerate case: if `m_o == m_f`, returns 0 (no fade).
 */
export function boundaryBlendT(camera: Vec3, mo: Vec3, mf: Vec3): number {
  const vx = mf[0] - mo[0];
  const vy = mf[1] - mo[1];
  const vz = mf[2] - mo[2];
  const denom = vx * vx + vy * vy + vz * vz;
  if (denom < 1e-12) return 0.0;
  const rx = camera[0] - mo[0];
  const ry = camera[1] - mo[1];
  const rz = camera[2] - mo[2];
  const raw = (rx * vx + ry * vy + rz * vz) / denom;
  return Math.max(0, Math.min(1, raw));
}

/**
 * Pick the per-chunk activation record for one chunk. JS reference for
 * `cs_lod_select.wgsl::cs_lod_select`. Used by the boundary-blend test +
 * by the headless fps proxy bench.
 *
 * Decision tree (mirror of the WGSL):
 *
 *   1. Distance band: walk depth_thresholds fine→coarse, pick the
 *      coarsest level whose threshold ≤ (||camera - centroid|| - radius).
 *   2. Screen-space size: if `focal_y * radius / dist < ss_threshold`,
 *      bump to the next coarser level (clamped at L-1).
 *   3. Activation: a record at `record.level == chosen_level` gets
 *      active=1. All other records on the chunk are inactive.
 *   4. Slot / t_blend: slot = whichever of (near_centroid, far_centroid)
 *      is closer to this chunk's centroid. t_blend = LODGE eq. 4 ramp,
 *      with the near chunk using (1 - t) and the far chunk using t.
 *      When near and far centroids coincide (single-chunk pinch),
 *      t_blend = 1.
 */
export function selectChunkActivation(
  record: ChunkRecord,
  inputs: LodSelectInputs,
): ChunkActivation {
  const cx = record.centroid[0] - inputs.cameraPos[0];
  const cy = record.centroid[1] - inputs.cameraPos[1];
  const cz = record.centroid[2] - inputs.cameraPos[2];
  const dist = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const distEff = Math.max(dist - record.radius, 0);

  // 1. Camera-distance band — coarsest level whose threshold ≤ distEff.
  let chosen = 0;
  const L = Math.min(inputs.depthThresholds.length, LOD_MAX_LEVELS);
  for (let l = 0; l < L; l++) {
    if (distEff >= (inputs.depthThresholds[l] ?? 0)) chosen = l;
  }

  // 2. Screen-space size heuristic.
  const safeDist = Math.max(dist, 1e-3);
  const ssRadius = (inputs.focalY * record.radius) / safeDist;
  if (ssRadius < inputs.ssSizeThreshold && chosen + 1 < L) {
    chosen += 1;
  }

  const act: ChunkActivation = {
    level: chosen,
    active: 0,
    slot: LOD_SLOT_NEAR,
    tBlend: 1.0,
  };

  if (record.level !== chosen) return act;
  act.active = 1;

  // 3. Slot + boundary blend.
  const dNear = euclid(record.centroid, inputs.nearCentroid);
  const dFar = euclid(record.centroid, inputs.farCentroid);
  const isFar = dFar < dNear;
  act.slot = isFar ? LOD_SLOT_FAR : LOD_SLOT_NEAR;

  const t = boundaryBlendT(inputs.cameraPos, inputs.nearCentroid, inputs.farCentroid);
  act.tBlend = isFar ? t : 1 - t;

  // 4. Single-chunk pinch: identical centroids ⇒ no fade.
  if (euclid(inputs.nearCentroid, inputs.farCentroid) < 1e-4) {
    act.tBlend = 1.0;
  }

  return act;
}

/** Run the per-chunk selector across an entire manifest, returning one
 *  activation per (level, chunk) record in manifest order. The order
 *  matters: the WGSL kernel and the streaming uploader both index into
 *  this flattened array, so callers can swap the JS path for the GPU
 *  path without re-deriving indices. */
export function selectAllActivations(
  manifest: LodgeManifest,
  inputs: LodSelectInputs,
): { records: ChunkRecord[]; activations: ChunkActivation[] } {
  const records: ChunkRecord[] = [];
  const activations: ChunkActivation[] = [];
  for (const lvl of manifest.levels) {
    for (const c of lvl.chunks) {
      const rec: ChunkRecord = {
        centroid: c.centroid,
        radius: c.radius,
        level: lvl.level,
        chunkIndex: c.index,
        splatCount: c.splatCount,
      };
      records.push(rec);
      activations.push(selectChunkActivation(rec, inputs));
    }
  }
  return { records, activations };
}

/**
 * Find the two closest chunks to the camera within a single level (LODGE
 * eq. 4 inputs `m_o` and `m_f`). The "near" chunk is the unconditional
 * closest. The "far" chunk is the next-closest whose centroid is on the
 * other side of the camera-near axis — i.e. the most likely "transition
 * partner" as the camera moves. We use a simple Euclidean tiebreak:
 *
 *   near = argmin_c   ||c.centroid - camera||
 *   far  = argmin_c { ||c.centroid - camera|| : c ≠ near }
 *
 * Returns `{ near, far }` references into the manifest's chunk list. When
 * the level only has one chunk, near == far and the runtime degenerates
 * to no-fade rendering. The bench harness uses this to pre-compute the
 * eq. 4 uniforms each frame.
 */
export function pickNearFarChunks(
  manifest: LodgeManifest,
  levelIndex: number,
  cameraPos: Vec3,
): { near: LodgeChunkDescriptor; far: LodgeChunkDescriptor } {
  const lvl = manifest.levels[levelIndex];
  if (!lvl || lvl.chunks.length === 0) {
    throw new Error(`lodge_lod_no_chunks: level ${levelIndex}`);
  }
  let nearIdx = 0;
  let nearD2 = Infinity;
  for (let i = 0; i < lvl.chunks.length; i++) {
    const c = lvl.chunks[i]!;
    const d2 = dist2(cameraPos, c.centroid);
    if (d2 < nearD2) {
      nearD2 = d2;
      nearIdx = i;
    }
  }
  if (lvl.chunks.length === 1) {
    const only = lvl.chunks[0]!;
    return { near: only, far: only };
  }
  let farIdx = nearIdx === 0 ? 1 : 0;
  let farD2 = Infinity;
  for (let i = 0; i < lvl.chunks.length; i++) {
    if (i === nearIdx) continue;
    const c = lvl.chunks[i]!;
    const d2 = dist2(cameraPos, c.centroid);
    if (d2 < farD2) {
      farD2 = d2;
      farIdx = i;
    }
  }
  return { near: lvl.chunks[nearIdx]!, far: lvl.chunks[farIdx]! };
}

function euclid(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function dist2(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}
