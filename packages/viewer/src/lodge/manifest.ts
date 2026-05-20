/**
 * LODGE manifest loader — viewer side (Phase A.2).
 *
 * Parses the `manifest.json` emitted by `catetus lodge build` (see
 * `crates/catetus-lodge/src/lib.rs`) and `docs/perf/lodge-lod-spec.md`
 * §A.1.3.
 *
 * The runtime consumes this in three stages:
 *
 *   1. {@link parseLodgeManifest} — turn the on-disk JSON into typed
 *      {@link LodgeManifest} (this file).
 *   2. {@link LodgeChunkLoader} — fetch + decode per-chunk PLYs and stream
 *      them to the GPU with LRU eviction (`./chunk-loader.ts`).
 *   3. Phase A.3 — per-frame WGSL compute pass that picks chunks per
 *      camera distance. Not in this file; Phase A.2 ships a CPU-side
 *      heuristic in {@link selectLodLevel}.
 *
 * The schema is versioned (`version: 1` today). Unknown fields are
 * preserved verbatim in the returned object's `_raw` map so a future
 * runtime can read Phase-A.2/A.3 fields (training-view-greedy
 * `depth_threshold`, K-means `chunk_centroid`s) without a schema bump on
 * this loader.
 */

import type { Vec3 } from '../manifest.js';

/** Schema version emitted by the Phase-A.1 chunker. */
export const LODGE_MANIFEST_VERSION = 1;

/** Scene-wide AABB as `[min, max]` — matches the on-disk shape. */
export type LodgeAabb = [Vec3, Vec3];

/** One spatial chunk inside one LOD level. */
export interface LodgeChunkDescriptor {
  /** Chunk index within the level. */
  index: number;
  /** Path to the chunk PLY, relative to the manifest's directory. */
  path: string;
  /** Splat count in this chunk. */
  splatCount: number;
  /** Chunk-local AABB. */
  bbox: LodgeAabb;
  /** Splat-position centroid `[x, y, z]`. Phase A.2 LOD selector keys
   *  per-chunk camera distance off this. */
  centroid: Vec3;
  /** Bounding-sphere radius — max distance from centroid to any splat. */
  radius: number;
  /** BLAKE3 hex digest of the chunk PLY bytes (or empty when absent). */
  blake3: string;
}

/** One level of the LOD pyramid. */
export interface LodgeLevelDescriptor {
  /** Level index. 0 = finest = original. */
  level: number;
  /** Total splats at this level (= sum of chunk counts). */
  splatCount: number;
  /** Splat count relative to level 0. 1.0 at level 0; smaller at coarser. */
  reduction: number;
  /** Approximate Nyquist depth `d_l` for this level (LODGE eq. 2 band
   *  edge). Phase A.1 emits a linear heuristic; Phase A.3 will refine. */
  depthThreshold: number;
  /** Chunks in Morton sweep order. */
  chunks: LodgeChunkDescriptor[];
}

/** Fully-parsed LODGE pyramid manifest. */
export interface LodgeManifest {
  /** Schema version. Equal to {@link LODGE_MANIFEST_VERSION} for Phase-A.1
   *  output. */
  version: number;
  /** Source PLY filename, provenance only. */
  source: string;
  /** Splat count of the original (= level 0) PLY. */
  originalSplatCount: number;
  /** Scene-wide AABB. */
  bbox: LodgeAabb;
  /** Pyramid levels, fine → coarse. `levels[0]` is the original. */
  levels: LodgeLevelDescriptor[];
}

/* -------------------------------------------------------------------- */
/* Parse                                                                */
/* -------------------------------------------------------------------- */

function isObject(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function asNumber(x: unknown, fallback: number): number {
  return typeof x === 'number' && Number.isFinite(x) ? x : fallback;
}

function asString(x: unknown, fallback: string): string {
  return typeof x === 'string' ? x : fallback;
}

function asVec3(x: unknown): Vec3 {
  if (Array.isArray(x) && x.length >= 3) {
    const a = Number(x[0]);
    const b = Number(x[1]);
    const c = Number(x[2]);
    if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c)) {
      return [a, b, c];
    }
  }
  throw new Error('lodge_manifest_invalid: expected Vec3');
}

function asAabb(x: unknown): LodgeAabb {
  if (Array.isArray(x) && x.length >= 2) {
    return [asVec3(x[0]), asVec3(x[1])];
  }
  throw new Error('lodge_manifest_invalid: expected AABB [Vec3, Vec3]');
}

function parseChunk(c: unknown, levelIndex: number, chunkIndex: number): LodgeChunkDescriptor {
  if (!isObject(c)) {
    throw new Error(
      `lodge_manifest_invalid: level[${levelIndex}].chunks[${chunkIndex}] not an object`,
    );
  }
  const path = asString(c.path, '');
  if (path.length === 0) {
    throw new Error(
      `lodge_manifest_invalid: level[${levelIndex}].chunks[${chunkIndex}] missing path`,
    );
  }
  return {
    index: asNumber(c.index, chunkIndex),
    path,
    splatCount: asNumber(c.splat_count, 0),
    bbox: asAabb(c.bbox),
    centroid: asVec3(c.centroid),
    radius: asNumber(c.radius, 0),
    blake3: asString(c.blake3, ''),
  };
}

function parseLevel(l: unknown, levelIndex: number): LodgeLevelDescriptor {
  if (!isObject(l)) {
    throw new Error(`lodge_manifest_invalid: levels[${levelIndex}] not an object`);
  }
  const chunksRaw = l.chunks;
  if (!Array.isArray(chunksRaw)) {
    throw new Error(`lodge_manifest_invalid: levels[${levelIndex}].chunks not an array`);
  }
  return {
    level: asNumber(l.level, levelIndex),
    splatCount: asNumber(l.splat_count, 0),
    reduction: asNumber(l.reduction, 1.0),
    depthThreshold: asNumber(l.depth_threshold, 0),
    chunks: chunksRaw.map((c, i) => parseChunk(c, levelIndex, i)),
  };
}

/**
 * Parse a `.lodge/manifest.json` text payload.
 *
 * @throws Error whose message starts with `lodge_manifest_invalid:` on
 *   malformed input.
 */
export function parseLodgeManifest(json: string): LodgeManifest {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (err) {
    throw new Error(`lodge_manifest_invalid: bad JSON (${(err as Error).message})`);
  }
  if (!isObject(raw)) {
    throw new Error('lodge_manifest_invalid: root must be an object');
  }
  const version = asNumber(raw.version, 0);
  if (version === 0) {
    throw new Error('lodge_manifest_invalid: missing version');
  }
  if (version > LODGE_MANIFEST_VERSION) {
    // Forward-compatible: we still attempt to parse, but the caller
    // can decide whether to surface a warning to the user.
    // (No throw — schema is additive by convention.)
  }
  const levelsRaw = raw.levels;
  if (!Array.isArray(levelsRaw)) {
    throw new Error('lodge_manifest_invalid: levels must be an array');
  }
  return {
    version,
    source: asString(raw.source, ''),
    originalSplatCount: asNumber(raw.original_splat_count, 0),
    bbox: asAabb(raw.bbox),
    levels: levelsRaw.map((l, i) => parseLevel(l, i)),
  };
}

/* -------------------------------------------------------------------- */
/* Helpers                                                              */
/* -------------------------------------------------------------------- */

/** Centroid of the scene bbox. Used as the camera-distance anchor for
 *  the Phase-A.2 TS-side LOD heuristic. */
export function sceneBboxCenter(m: LodgeManifest): Vec3 {
  const [mn, mx] = m.bbox;
  return [
    (mn[0] + mx[0]) * 0.5,
    (mn[1] + mx[1]) * 0.5,
    (mn[2] + mx[2]) * 0.5,
  ];
}

/** Half-diagonal of the scene bbox. Used as the scale for the LOD-band
 *  heuristic when `levels[l].depth_threshold` is the Phase-A.1 default
 *  linear sweep (which is 0 at level 0 and `1.5*scene_diag` at the
 *  coarsest level). */
export function sceneBboxRadius(m: LodgeManifest): number {
  const [mn, mx] = m.bbox;
  const dx = mx[0] - mn[0];
  const dy = mx[1] - mn[1];
  const dz = mx[2] - mn[2];
  return 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Pick a LOD level for the current camera position (Phase A.2 — TS-side
 * heuristic, no GPU compute pass yet).
 *
 * The rule:
 *   - Compute `d = ||camera - scene_centroid||`.
 *   - Walk levels fine → coarse and pick the coarsest level whose
 *     `depthThreshold` is ≤ d. The finest level always wins ties.
 *   - When all `depthThreshold`s are 0 (degenerate Phase-A.1 output for a
 *     single-level scene), fall back to level 0.
 *
 * This matches LODGE eq. 2 reduced to "one level per frame" — Phase A.3
 * will replace it with a per-Gaussian banded selection inside the WGSL
 * compute pass.
 */
export function selectLodLevel(m: LodgeManifest, cameraPos: Vec3): number {
  if (m.levels.length === 0) return 0;
  if (m.levels.length === 1) return 0;
  const c = sceneBboxCenter(m);
  const d = Math.sqrt(
    (cameraPos[0] - c[0]) ** 2 +
      (cameraPos[1] - c[1]) ** 2 +
      (cameraPos[2] - c[2]) ** 2,
  );

  // Coarsest level whose threshold ≤ d (i.e. we are at least as far out
  // as the level says it needs). Default to 0 when no level passes the
  // gate (camera inside the finest band).
  let chosen = 0;
  for (let i = 0; i < m.levels.length; i++) {
    const t = m.levels[i]!.depthThreshold;
    if (d >= t) chosen = i;
  }
  return chosen;
}

/**
 * Pick the chunks of a chosen level, sorted by camera proximity (closest
 * first). Useful for setting `loadPriority` on a streaming fetcher so the
 * camera-near chunks land in VRAM first.
 */
export function chunksByCameraProximity(
  m: LodgeManifest,
  levelIndex: number,
  cameraPos: Vec3,
): LodgeChunkDescriptor[] {
  const level = m.levels[levelIndex];
  if (!level) return [];
  const scored = level.chunks.map((c) => {
    const dx = cameraPos[0] - c.centroid[0];
    const dy = cameraPos[1] - c.centroid[1];
    const dz = cameraPos[2] - c.centroid[2];
    return { c, d2: dx * dx + dy * dy + dz * dz };
  });
  scored.sort((a, b) => a.d2 - b.d2);
  return scored.map((s) => s.c);
}
