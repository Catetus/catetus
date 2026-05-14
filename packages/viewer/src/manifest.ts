/**
 * glTF + extension parser for SplatForge optimized assets.
 *
 * Two extensions are read:
 *   - `KHR_gaussian_splatting`            — point primitive header.
 *   - `SF_spatial_streaming_index`        — per-chunk LOD / priority table.
 */

/** Vec3 tuple convenience alias. */
export type Vec3 = [number, number, number];

/** Axis-aligned bounding box. */
export interface Bbox {
  min: Vec3;
  max: Vec3;
}

/**
 * Streaming chunk descriptor. One entry per binary tile of splats.
 *
 * `loadPriority` is an integer where lower values load first.
 */
export interface ChunkDescriptor {
  uri: string;
  byteOffset: number;
  byteLength: number;
  splatCount: number;
  bbox: Bbox;
  lod: number;
  /** Hex BLAKE3 digest, or empty string if absent. */
  checksum: string;
  loadPriority: number;
}

/** Fully-parsed manifest produced by {@link parseManifest}. */
export interface Manifest {
  splatCount: number;
  bbox: Bbox;
  chunks: ChunkDescriptor[];
  /** Spherical-harmonics degree from KHR_gaussian_splatting. */
  shDegree: number;
}

/* --------------------------------------------------------------------- */
/* Raw glTF JSON shape — narrowed to just what we read. Everything is    */
/* `unknown` until validated by guards below.                            */
/* --------------------------------------------------------------------- */

interface RawGltf {
  asset?: { version?: string };
  buffers?: Array<{ uri?: string; byteLength?: number }>;
  extensionsUsed?: string[];
  extensions?: Record<string, unknown>;
  meshes?: Array<{
    primitives?: Array<{
      extensions?: Record<string, unknown>;
    }>;
  }>;
}

interface RawGaussianSplatting {
  splatCount?: number;
  shDegree?: number;
  bbox?: { min?: number[]; max?: number[] };
}

interface RawStreamingIndex {
  chunks?: Array<{
    uri?: string;
    byteOffset?: number;
    byteLength?: number;
    splatCount?: number;
    bbox?: { min?: number[]; max?: number[] };
    lod?: number;
    checksum?: string;
    loadPriority?: number;
  }>;
}

const GS_EXT = 'KHR_gaussian_splatting';
const SF_EXT = 'SF_spatial_streaming_index';

function isObject(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function asVec3(x: unknown, fallback: Vec3): Vec3 {
  if (Array.isArray(x) && x.length >= 3) {
    const [a, b, c] = x;
    if (typeof a === 'number' && typeof b === 'number' && typeof c === 'number') {
      return [a, b, c];
    }
  }
  return fallback;
}

/**
 * Locate the `KHR_gaussian_splatting` extension block, searching first the
 * top-level `extensions` object, then mesh-primitive extensions.
 */
function findGaussianExt(g: RawGltf): RawGaussianSplatting | undefined {
  const top = g.extensions?.[GS_EXT];
  if (isObject(top)) return top as RawGaussianSplatting;
  for (const mesh of g.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const e = prim.extensions?.[GS_EXT];
      if (isObject(e)) return e as RawGaussianSplatting;
    }
  }
  return undefined;
}

function findStreamingIndex(g: RawGltf): RawStreamingIndex | undefined {
  const top = g.extensions?.[SF_EXT];
  if (isObject(top)) return top as RawStreamingIndex;
  return undefined;
}

/**
 * Parse a glTF JSON text and return a normalized {@link Manifest}.
 *
 * Falls back to a single synthetic chunk pointing at the primary buffer when
 * `SF_spatial_streaming_index` is absent.
 *
 * @throws Error with message starting `manifest_invalid:` for malformed input.
 */
export function parseManifest(json: string): Manifest {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (err) {
    throw new Error(`manifest_invalid: not valid JSON (${(err as Error).message})`);
  }
  if (!isObject(raw)) {
    throw new Error('manifest_invalid: root must be an object');
  }
  const g = raw as RawGltf;

  const gs = findGaussianExt(g);
  if (!gs) {
    throw new Error(`manifest_invalid: missing ${GS_EXT} extension`);
  }

  const splatCount = typeof gs.splatCount === 'number' ? gs.splatCount : 0;
  const shDegree = typeof gs.shDegree === 'number' ? gs.shDegree : 0;
  const bbox: Bbox = {
    min: asVec3(gs.bbox?.min, [-1, -1, -1]),
    max: asVec3(gs.bbox?.max, [1, 1, 1]),
  };

  const index = findStreamingIndex(g);
  let chunks: ChunkDescriptor[];

  if (index && Array.isArray(index.chunks) && index.chunks.length > 0) {
    chunks = index.chunks.map((c, i) => normalizeChunk(c, i, bbox, splatCount));
  } else {
    // Synthetic single chunk pointing at the primary buffer.
    const buf = g.buffers?.[0];
    if (!buf?.uri) {
      throw new Error('manifest_invalid: no streaming index and no primary buffer uri');
    }
    chunks = [
      {
        uri: buf.uri,
        byteOffset: 0,
        byteLength: typeof buf.byteLength === 'number' ? buf.byteLength : 0,
        splatCount,
        bbox,
        lod: 0,
        checksum: '',
        loadPriority: 0,
      },
    ];
  }

  // Stable sort by loadPriority ascending, then by lod ascending.
  chunks.sort((a, b) =>
    a.loadPriority !== b.loadPriority ? a.loadPriority - b.loadPriority : a.lod - b.lod,
  );

  return { splatCount, bbox, chunks, shDegree };
}

function normalizeChunk(
  c: NonNullable<RawStreamingIndex['chunks']>[number],
  index: number,
  sceneBbox: Bbox,
  totalSplats: number,
): ChunkDescriptor {
  if (typeof c.uri !== 'string' || c.uri.length === 0) {
    throw new Error(`manifest_invalid: chunk[${index}] missing uri`);
  }
  return {
    uri: c.uri,
    byteOffset: typeof c.byteOffset === 'number' ? c.byteOffset : 0,
    byteLength: typeof c.byteLength === 'number' ? c.byteLength : 0,
    splatCount: typeof c.splatCount === 'number' ? c.splatCount : totalSplats,
    bbox: {
      min: asVec3(c.bbox?.min, sceneBbox.min),
      max: asVec3(c.bbox?.max, sceneBbox.max),
    },
    lod: typeof c.lod === 'number' ? c.lod : 0,
    checksum: typeof c.checksum === 'string' ? c.checksum : '',
    loadPriority: typeof c.loadPriority === 'number' ? c.loadPriority : index,
  };
}
