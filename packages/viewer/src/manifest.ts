/**
 * glTF + extension parser for Catetus optimized assets.
 *
 * Two extensions are read:
 *   - `KHR_gaussian_splatting`            — point primitive header.
 *   - `CT_spatial_streaming_index`        — per-chunk LOD / priority table.
 */

/** Vec3 tuple convenience alias. */
export type Vec3 = [number, number, number];

/** Axis-aligned bounding box. */
export interface Bbox {
  min: Vec3;
  max: Vec3;
}

/**
 * glTF accessor component type. Matches the underlying GL enum values so we
 * can compare directly against the JSON.
 */
export const FLOAT_COMPONENT = 5126;
export const UBYTE_COMPONENT = 5121;
export const USHORT_COMPONENT = 5123;

/**
 * Per-attribute slice into a structure-of-arrays binary chunk. Offsets are
 * relative to the start of the chunk's `byteOffset`. Component widths follow
 * `KHR_gaussian_splatting`: POSITION/SCALE/COLOR_DC are vec3, ROTATION is vec4,
 * OPACITY is scalar.
 *
 * When `componentType` is one of the integer types (5121, 5123) and
 * `normalized` is true, the SPEC-0013 `KHR_mesh_quantization` extension is in
 * use and the bytes must be dequantized against `min` / `max` (one entry per
 * component) before being passed to the renderer.
 */
export interface SoaAttributeSlice {
  byteOffset: number;
  byteLength: number;
  componentType?: number;
  normalized?: boolean;
  min?: number[];
  max?: number[];
}

/**
 * SoA buffer layout for a single chunk. Present when the source glTF stores
 * splats attribute-by-attribute (one bufferView per attribute), which is what
 * `catetus-gltf` emits. Absent for legacy interleaved AoS test fixtures.
 *
 * `shRest` (optional) carries SH degrees 1..N as a per-splat blob of
 * `shRestCoefCount(shDegree) * 3` float32s, laid out as
 * `shRest[i * coefCount * 3 + k * 3 + c]` (splat-major, k-major over the
 * `coefCount` coefficients per channel, channel-minor). Phase 2b
 * (WebGPU SH-rest evaluator) consumes this directly as a storage buffer.
 */
export interface SoaAttributeLayout {
  positions: SoaAttributeSlice;
  rotations: SoaAttributeSlice;
  scales: SoaAttributeSlice;
  opacities: SoaAttributeSlice;
  colorDC: SoaAttributeSlice;
  /** Optional per-splat SH-rest blob (degrees 1..shDegree). */
  shRest?: SoaAttributeSlice;
  /** SH-rest degree (1, 2, or 3). Required iff `shRest` is set. */
  shDegree?: number;
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
  /** Optional SoA layout — see {@link SoaAttributeLayout}. */
  attributeLayout?: SoaAttributeLayout;
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
  bufferViews?: Array<{ buffer?: number; byteOffset?: number; byteLength?: number }>;
  accessors?: Array<{
    bufferView?: number;
    componentType?: number;
    normalized?: boolean;
    count?: number;
    type?: string;
    min?: number[];
    max?: number[];
  }>;
  extensionsUsed?: string[];
  extensions?: Record<string, unknown>;
  meshes?: Array<{
    primitives?: Array<{
      mode?: number;
      attributes?: Record<string, number>;
      extensions?: Record<string, unknown>;
    }>;
  }>;
}

/**
 * Canonical (layout-neutral) attribute index table. Both the legacy
 * (pre-RC) layout — where attribute indices live inside the
 * `KHR_gaussian_splatting` extension under bare keys (`_ROTATION`, etc.) —
 * and the RC layout — where attributes live on the primitive itself under
 * namespaced keys (`KHR_gaussian_splatting:ROTATION`, etc.) — are
 * normalized into this shape by {@link extractGaussianAttributes}.
 */
interface RawGaussianAttributes {
  POSITION?: number;
  _ROTATION?: number;
  _SCALE?: number;
  _OPACITY?: number;
  _COLOR_DC?: number;
  _COLOR_SH?: number;
}

/** Source layout the manifest was authored in. */
export type GaussianAttributeLayout = 'rc' | 'legacy';

interface RawGaussianSplatting {
  splatCount?: number;
  shDegree?: number;
  bbox?: { min?: number[]; max?: number[] };
  attributes?: RawGaussianAttributes;
}

interface RawStreamingIndex {
  chunks?: Array<{
    uri?: string;
    buffer?: number;
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
const SF_EXT = 'CT_spatial_streaming_index';

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
 * Locate the top-level `KHR_gaussian_splatting` extension (scene-wide
 * splatCount + bbox), if any. The Rust writer emits this since v0.1.1.
 */
function findSceneGaussianExt(g: RawGltf): RawGaussianSplatting | undefined {
  const top = g.extensions?.[GS_EXT];
  return isObject(top) ? (top as RawGaussianSplatting) : undefined;
}

/**
 * Locate the per-primitive `KHR_gaussian_splatting` extension which carries
 * (in the legacy layout) the attribute → accessor table needed to decode
 * binary chunks. Returns the extension object plus a flag indicating whether
 * it actually carried `.attributes` (legacy) or not (RC, where attributes
 * live on the primitive itself).
 */
function findPrimitiveGaussianExt(g: RawGltf): RawGaussianSplatting | undefined {
  for (const mesh of g.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const e = prim.extensions?.[GS_EXT];
      if (isObject(e)) return e as RawGaussianSplatting;
    }
  }
  return undefined;
}

/**
 * Namespaced-attribute keys per the KHR_gaussian_splatting Release Candidate
 * (RC). These live on the primitive's `attributes` map (not inside the
 * extension object) alongside `mode`. We sniff for these first; absence falls
 * back to the legacy in-extension `.attributes` table.
 */
const RC_ATTR_KEYS = {
  // Bare `POSITION` is what the Catetus Rust writer (SpecVersion::RcMay2026)
  // emits — it places POSITION on the primitive's core glTF attribute slot
  // rather than under the namespaced key. Earlier RC drafts (and some
  // third-party authors) use the namespaced `KHR_gaussian_splatting:POSITION`
  // form; we accept either so a viewer fed a manifest from either source
  // decodes correctly.
  POSITION: 'POSITION',
  POSITION_RC: `${GS_EXT}:POSITION`,
  ROTATION: `${GS_EXT}:ROTATION`,
  SCALE: `${GS_EXT}:SCALE`,
  OPACITY: `${GS_EXT}:OPACITY`,
  // KHR RC (May 2026): the DC color is named `SH_DEGREE_0_COEF_0` because the
  // DC term is the degree-0 spherical-harmonic coefficient. Earlier drafts
  // used `COLOR_DC`; we accept either so manifests written by the Rust writer
  // (which switched to `SH_DEGREE_0_COEF_0` in `SpecVersion::RcMay2026`)
  // load correctly. Without this alias the viewer silently misses the color
  // accessor, falls through to layout=undefined, and the renderer decodes the
  // SoA buffer with the legacy 56-byte AoS reader — producing a black canvas.
  COLOR_DC: `${GS_EXT}:COLOR_DC`,
  COLOR_DC_RC: `${GS_EXT}:SH_DEGREE_0_COEF_0`,
  COLOR_SH: `${GS_EXT}:COLOR_SH`,
} as const;

/**
 * Extract the attribute → accessor index table from the first splat primitive,
 * supporting both the RC (namespaced keys on `primitive.attributes`) and
 * legacy (bare keys inside `prim.extensions.KHR_gaussian_splatting.attributes`)
 * layouts. Schema sniff: presence of any `KHR_gaussian_splatting:*` key on a
 * primitive's `attributes` map is taken as authoritative for RC; otherwise we
 * fall back to legacy.
 */
function extractGaussianAttributes(
  g: RawGltf,
): { attrs: RawGaussianAttributes; layout: GaussianAttributeLayout } {
  for (const mesh of g.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const primAttrs = prim.attributes;
      if (isObject(primAttrs)) {
        const pa = primAttrs as Record<string, number>;
        const hasRc = Object.keys(pa).some((k) => k.startsWith(`${GS_EXT}:`));
        if (hasRc) {
          const position =
            typeof pa[RC_ATTR_KEYS.POSITION] === 'number'
              ? pa[RC_ATTR_KEYS.POSITION]
              : typeof pa[RC_ATTR_KEYS.POSITION_RC] === 'number'
                ? pa[RC_ATTR_KEYS.POSITION_RC]
                : undefined;
          return {
            attrs: {
              POSITION: position,
              _ROTATION: typeof pa[RC_ATTR_KEYS.ROTATION] === 'number' ? pa[RC_ATTR_KEYS.ROTATION] : undefined,
              _SCALE: typeof pa[RC_ATTR_KEYS.SCALE] === 'number' ? pa[RC_ATTR_KEYS.SCALE] : undefined,
              _OPACITY: typeof pa[RC_ATTR_KEYS.OPACITY] === 'number' ? pa[RC_ATTR_KEYS.OPACITY] : undefined,
              _COLOR_DC: typeof pa[RC_ATTR_KEYS.COLOR_DC_RC] === 'number'
                ? pa[RC_ATTR_KEYS.COLOR_DC_RC]
                : (typeof pa[RC_ATTR_KEYS.COLOR_DC] === 'number' ? pa[RC_ATTR_KEYS.COLOR_DC] : undefined),
              _COLOR_SH: typeof pa[RC_ATTR_KEYS.COLOR_SH] === 'number' ? pa[RC_ATTR_KEYS.COLOR_SH] : undefined,
            },
            layout: 'rc',
          };
        }
      }
      const e = prim.extensions?.[GS_EXT];
      if (isObject(e)) {
        const legacy = (e as RawGaussianSplatting).attributes;
        if (isObject(legacy)) {
          return { attrs: legacy as RawGaussianAttributes, layout: 'legacy' };
        }
      }
    }
  }
  return { attrs: {}, layout: 'legacy' };
}

function findStreamingIndex(g: RawGltf): RawStreamingIndex | undefined {
  const top = g.extensions?.[SF_EXT];
  if (isObject(top)) return top as RawStreamingIndex;
  return undefined;
}

/**
 * Resolve a single attribute accessor index into a slice into the chunk's
 * binary payload. Offsets here are relative to the buffer (chunk-relative
 * conversion happens at chunk-construction time).
 */
function accessorSlice(
  g: RawGltf,
  accIdx: number | undefined,
):
  | {
      bufferIdx: number;
      byteOffset: number;
      byteLength: number;
      componentType?: number;
      normalized?: boolean;
      min?: number[];
      max?: number[];
    }
  | undefined {
  if (typeof accIdx !== 'number') return undefined;
  const acc = g.accessors?.[accIdx];
  if (!acc || typeof acc.bufferView !== 'number') return undefined;
  const bv = g.bufferViews?.[acc.bufferView];
  if (!bv) return undefined;
  return {
    bufferIdx: typeof bv.buffer === 'number' ? bv.buffer : 0,
    byteOffset: typeof bv.byteOffset === 'number' ? bv.byteOffset : 0,
    byteLength: typeof bv.byteLength === 'number' ? bv.byteLength : 0,
    componentType: typeof acc.componentType === 'number' ? acc.componentType : undefined,
    normalized: typeof acc.normalized === 'boolean' ? acc.normalized : undefined,
    min: Array.isArray(acc.min) ? acc.min : undefined,
    max: Array.isArray(acc.max) ? acc.max : undefined,
  };
}

/**
 * Parse a glTF JSON text and return a normalized {@link Manifest}.
 *
 * Falls back to a single synthetic chunk pointing at the primary buffer when
 * `CT_spatial_streaming_index` is absent.
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

  const primExt = findPrimitiveGaussianExt(g);
  const sceneExt = findSceneGaussianExt(g);
  if (!primExt && !sceneExt) {
    throw new Error(`manifest_invalid: missing ${GS_EXT} extension`);
  }

  // Resolve attribute → SoA byte-slice (relative to its buffer's start).
  // Auto-detect RC (namespaced primitive-level attributes) vs legacy
  // (bare attributes inside the per-primitive extension object).
  const { attrs } = extractGaussianAttributes(g);
  const posSlice = accessorSlice(g, attrs.POSITION);
  const rotSlice = accessorSlice(g, attrs._ROTATION);
  const sclSlice = accessorSlice(g, attrs._SCALE);
  const opSlice = accessorSlice(g, attrs._OPACITY);
  const dcSlice = accessorSlice(g, attrs._COLOR_DC);

  // splatCount: prefer scene-level extension; fall back to POSITION accessor
  // count; fall back to streaming-index records (handled below).
  let splatCount = typeof sceneExt?.splatCount === 'number' ? sceneExt.splatCount : 0;
  if (splatCount === 0 && typeof attrs.POSITION === 'number') {
    const posAcc = g.accessors?.[attrs.POSITION];
    if (posAcc && typeof posAcc.count === 'number') splatCount = posAcc.count;
  }

  // shDegree: scene-level wins, else primitive-level.
  const shDegree =
    typeof sceneExt?.shDegree === 'number'
      ? sceneExt.shDegree
      : typeof primExt?.shDegree === 'number'
        ? primExt.shDegree
        : 0;

  // bbox: prefer scene-level; else POSITION accessor's min/max; else unit cube.
  let bbox: Bbox = {
    min: asVec3(sceneExt?.bbox?.min, [-1, -1, -1]),
    max: asVec3(sceneExt?.bbox?.max, [1, 1, 1]),
  };
  if (!sceneExt?.bbox && typeof attrs.POSITION === 'number') {
    const posAcc = g.accessors?.[attrs.POSITION];
    if (posAcc?.min && posAcc.max) {
      bbox = {
        min: asVec3(posAcc.min, bbox.min),
        max: asVec3(posAcc.max, bbox.max),
      };
    }
  }

  const index = findStreamingIndex(g);
  let chunks: ChunkDescriptor[];

  if (index && Array.isArray(index.chunks) && index.chunks.length > 0) {
    chunks = index.chunks.map((c, i) =>
      normalizeChunk(c, i, bbox, splatCount, g, attrs),
    );
  } else {
    // Synthetic single chunk pointing at the primary buffer. The SoA layout is
    // derived from the per-primitive attribute accessors so the renderer can
    // re-interleave bytes at decode time.
    const buf = g.buffers?.[0];
    if (!buf?.uri) {
      throw new Error('manifest_invalid: no streaming index and no primary buffer uri');
    }
    const layout = buildAttributeLayout(posSlice, rotSlice, sclSlice, opSlice, dcSlice, 0);
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
        attributeLayout: layout,
      },
    ];
  }

  // Stable sort by loadPriority ascending, then by lod ascending.
  chunks.sort((a, b) =>
    a.loadPriority !== b.loadPriority ? a.loadPriority - b.loadPriority : a.lod - b.lod,
  );

  return { splatCount, bbox, chunks, shDegree };
}

/**
 * Combine five per-attribute accessor slices into a chunk-relative layout.
 * Returns `undefined` if any required attribute is missing (caller falls back
 * to the legacy interleaved AoS path).
 *
 * `chunkByteOffset` is the buffer-relative offset of the chunk; subtracting it
 * yields offsets the renderer can use directly against the fetched bytes.
 */
type RawSlice = {
  byteOffset: number;
  byteLength: number;
  componentType?: number;
  normalized?: boolean;
  min?: number[];
  max?: number[];
};

function buildAttributeLayout(
  positions: RawSlice | undefined,
  rotations: RawSlice | undefined,
  scales: RawSlice | undefined,
  opacities: RawSlice | undefined,
  colorDC: RawSlice | undefined,
  chunkByteOffset: number,
): SoaAttributeLayout | undefined {
  if (!positions || !rotations || !scales || !opacities || !colorDC) {
    return undefined;
  }
  const rebase = (s: RawSlice): SoaAttributeSlice => ({
    byteOffset: s.byteOffset - chunkByteOffset,
    byteLength: s.byteLength,
    componentType: s.componentType,
    normalized: s.normalized,
    min: s.min,
    max: s.max,
  });
  return {
    positions: rebase(positions),
    rotations: rebase(rotations),
    scales: rebase(scales),
    opacities: rebase(opacities),
    colorDC: rebase(colorDC),
  };
}

function normalizeChunk(
  c: NonNullable<RawStreamingIndex['chunks']>[number],
  index: number,
  sceneBbox: Bbox,
  totalSplats: number,
  g: RawGltf,
  attrs: RawGaussianAttributes,
): ChunkDescriptor {
  // The chunk may carry `uri` (preferred) or `buffer` (index into root.buffers).
  // The Rust writer emits both since v0.1.1 but we accept either to remain
  // forward-compatible with future packers.
  let uri: string | undefined =
    typeof c.uri === 'string' && c.uri.length > 0 ? c.uri : undefined;
  if (!uri && typeof c.buffer === 'number') {
    uri = g.buffers?.[c.buffer]?.uri;
  }
  if (!uri || uri.length === 0) {
    throw new Error(`manifest_invalid: chunk[${index}] missing uri`);
  }

  const chunkByteOffset = typeof c.byteOffset === 'number' ? c.byteOffset : 0;
  // Per-chunk SoA layout: only emit when this chunk owns the buffer used by
  // the primitive attributes. For the common case where Rust emits one buffer
  // per chunk this always holds; treating it conservatively keeps the legacy
  // path active when assumptions don't.
  const bufIdx = typeof c.buffer === 'number' ? c.buffer : undefined;
  const slice = (accIdx: number | undefined):
    | { byteOffset: number; byteLength: number }
    | undefined => {
    const s = accessorSlice(g, accIdx);
    if (!s) return undefined;
    if (bufIdx !== undefined && s.bufferIdx !== bufIdx) return undefined;
    return { byteOffset: s.byteOffset, byteLength: s.byteLength };
  };
  const layout =
    bufIdx !== undefined
      ? buildAttributeLayout(
          slice(attrs.POSITION),
          slice(attrs._ROTATION),
          slice(attrs._SCALE),
          slice(attrs._OPACITY),
          slice(attrs._COLOR_DC),
          chunkByteOffset,
        )
      : undefined;

  return {
    uri,
    byteOffset: chunkByteOffset,
    byteLength: typeof c.byteLength === 'number' ? c.byteLength : 0,
    splatCount: typeof c.splatCount === 'number' ? c.splatCount : totalSplats,
    bbox: {
      min: asVec3(c.bbox?.min, sceneBbox.min),
      max: asVec3(c.bbox?.max, sceneBbox.max),
    },
    lod: typeof c.lod === 'number' ? c.lod : 0,
    checksum: typeof c.checksum === 'string' ? c.checksum : '',
    loadPriority: typeof c.loadPriority === 'number' ? c.loadPriority : index,
    attributeLayout: layout,
  };
}
