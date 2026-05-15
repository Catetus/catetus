/**
 * Minimal binary-glTF (GLB v2) reader for the streaming-tile loader.
 *
 * A GLB is a tiny container: 12-byte header, a JSON chunk, and a BIN chunk.
 * The producer side (`splatforge-gltf::write_glb` + `splatforge-optimize::tileset`)
 * always emits both chunks back-to-back. We split them out so the existing
 * `parseManifest` (which is JSON-only) can be reused for per-tile parsing,
 * and the BIN bytes can be range-sliced by chunk byteOffset on the renderer
 * side without re-fetching.
 *
 * The reader is intentionally strict: it rejects anything that isn't
 * `magic = glTF`, `version = 2`. We deliberately do not depend on the heavier
 * three.js / @gltf-transform readers — every byte they add to the bundle eats
 * into the v2 mobile size budget.
 */

import type { ChunkDescriptor, Manifest, SoaAttributeLayout } from '../manifest.js';

/** Result of {@link decodeGlb}. */
export interface DecodedGlb {
  /** Parsed JSON text (the glTF asset's structure). */
  json: string;
  /** Raw bytes of the BIN chunk (or zero-length when absent). */
  bin: Uint8Array;
}

const MAGIC_GLTF = 0x46546c67; // 'glTF' LE
const CHUNK_JSON = 0x4e4f534a; // 'JSON' LE
const CHUNK_BIN = 0x004e4942;  // 'BIN\0' LE

/**
 * Decode a GLB blob's JSON + BIN chunks. Throws an `Error` whose message
 * starts with `glb_invalid:` for malformed input — the streaming layer
 * surfaces this as a `tileset_invalid` warning rather than crashing the
 * viewer.
 *
 * Determinism: the function reads bytes in a fixed order and performs no
 * allocation beyond the two output slices, so two identical inputs produce
 * byte-identical outputs.
 */
export function decodeGlb(bytes: Uint8Array): DecodedGlb {
  if (bytes.byteLength < 12) {
    throw new Error('glb_invalid: header too short');
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = dv.getUint32(0, true);
  const version = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);
  if (magic !== MAGIC_GLTF) {
    throw new Error('glb_invalid: bad magic (not "glTF")');
  }
  if (version !== 2) {
    throw new Error(`glb_invalid: unsupported version ${version}`);
  }
  if (length > bytes.byteLength) {
    throw new Error(`glb_invalid: declared length ${length} > buffer ${bytes.byteLength}`);
  }

  let cursor = 12;
  let json = '';
  let bin: Uint8Array = new Uint8Array(new ArrayBuffer(0));
  const decoder = new TextDecoder('utf-8');

  while (cursor + 8 <= length) {
    const chunkLen = dv.getUint32(cursor, true);
    const chunkType = dv.getUint32(cursor + 4, true);
    cursor += 8;
    if (cursor + chunkLen > length) {
      throw new Error('glb_invalid: chunk extends past EOF');
    }
    const slice = new Uint8Array(bytes.buffer, bytes.byteOffset + cursor, chunkLen);
    if (chunkType === CHUNK_JSON) {
      json = decoder.decode(stripTrailingPad(slice, 0x20));
    } else if (chunkType === CHUNK_BIN) {
      bin = slice;
    }
    // Unknown chunks are silently skipped per the spec.
    cursor += chunkLen;
  }

  if (json.length === 0) {
    throw new Error('glb_invalid: missing JSON chunk');
  }
  return { json, bin };
}

/**
 * Strip a trailing run of `pad` bytes. GLB chunks are 4-byte padded with
 * `0x20` (space) for JSON and `0x00` for BIN; the JSON parser handles
 * trailing space fine but stripping it keeps test fixtures comparable.
 */
function stripTrailingPad(b: Uint8Array, pad: number): Uint8Array {
  let end = b.byteLength;
  while (end > 0 && b[end - 1] === pad) end--;
  return b.subarray(0, end);
}

/* ------------------------------------------------------------------------ */
/* GLB → SplatForge Manifest                                                */
/*                                                                          */
/* The viewer's `parseManifest` expects an external buffer URI; a binary    */
/* glTF (GLB) embeds the buffer inline. We synthesize a one-chunk           */
/* {@link Manifest} directly from the GLB's JSON + BIN, mirroring the SoA   */
/* layout the existing CPU + compute decode paths already understand.       */
/* ------------------------------------------------------------------------ */

interface RawGltf {
  buffers?: Array<{ uri?: string | null; byteLength?: number }>;
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
  meshes?: Array<{
    primitives?: Array<{
      mode?: number;
      attributes?: Record<string, number>;
      extensions?: Record<string, unknown>;
    }>;
  }>;
  extensions?: Record<string, unknown>;
}

interface GsAttributes {
  POSITION?: number;
  _ROTATION?: number;
  _SCALE?: number;
  _OPACITY?: number;
  _COLOR_DC?: number;
}

interface GsExt {
  splatCount?: number;
  shDegree?: number;
  bbox?: { min?: number[]; max?: number[] };
  attributes?: GsAttributes;
}

const GS_EXT_NAME = 'KHR_gaussian_splatting';
const RC_KEYS = {
  POSITION: `${GS_EXT_NAME}:POSITION`,
  ROTATION: `${GS_EXT_NAME}:ROTATION`,
  SCALE: `${GS_EXT_NAME}:SCALE`,
  OPACITY: `${GS_EXT_NAME}:OPACITY`,
  COLOR_DC: `${GS_EXT_NAME}:COLOR_DC`,
};

/**
 * Pull a normalized attribute index table out of the first splat primitive
 * supporting both KHR_gaussian_splatting RC (namespaced primitive-level
 * attributes) and the legacy in-extension layout. Returns `undefined` when
 * neither shape is present.
 */
function readPrimitiveAttributes(g: RawGltf): GsAttributes | undefined {
  for (const mesh of g.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const pa = prim.attributes;
      if (pa && typeof pa === 'object') {
        const rec = pa as Record<string, number>;
        if (Object.keys(rec).some((k) => k.startsWith(`${GS_EXT_NAME}:`))) {
          return {
            POSITION: typeof rec[RC_KEYS.POSITION] === 'number' ? rec[RC_KEYS.POSITION] : undefined,
            _ROTATION: typeof rec[RC_KEYS.ROTATION] === 'number' ? rec[RC_KEYS.ROTATION] : undefined,
            _SCALE: typeof rec[RC_KEYS.SCALE] === 'number' ? rec[RC_KEYS.SCALE] : undefined,
            _OPACITY: typeof rec[RC_KEYS.OPACITY] === 'number' ? rec[RC_KEYS.OPACITY] : undefined,
            _COLOR_DC: typeof rec[RC_KEYS.COLOR_DC] === 'number' ? rec[RC_KEYS.COLOR_DC] : undefined,
          };
        }
      }
      const e = prim.extensions?.[GS_EXT_NAME];
      if (e && typeof e === 'object' && !Array.isArray(e)) {
        const legacy = (e as GsExt).attributes;
        if (legacy && typeof legacy === 'object') return legacy;
      }
    }
  }
  return undefined;
}

function accessorSlice(g: RawGltf, idx: number | undefined):
  | { byteOffset: number; byteLength: number; componentType?: number; normalized?: boolean; min?: number[]; max?: number[] }
  | undefined {
  if (typeof idx !== 'number') return undefined;
  const acc = g.accessors?.[idx];
  if (!acc || typeof acc.bufferView !== 'number') return undefined;
  const bv = g.bufferViews?.[acc.bufferView];
  if (!bv) return undefined;
  return {
    byteOffset: bv.byteOffset ?? 0,
    byteLength: bv.byteLength ?? 0,
    componentType: acc.componentType,
    normalized: acc.normalized,
    min: acc.min,
    max: acc.max,
  };
}

/**
 * Build a one-chunk {@link Manifest} for a GLB by treating the BIN chunk as
 * the chunk payload. The chunk's `byteOffset` is 0 (offsets in the layout
 * are relative to BIN, matching what `decodeChunkBytes` expects).
 *
 * Throws `glb_invalid:` when the GLB doesn't carry `KHR_gaussian_splatting`
 * attributes — these tiles are useless to the renderer regardless.
 */
export function manifestFromGlb(glb: DecodedGlb): { manifest: Manifest; bin: Uint8Array } {
  let raw: unknown;
  try {
    raw = JSON.parse(glb.json);
  } catch (err) {
    throw new Error(`glb_invalid: bad JSON (${(err as Error).message})`);
  }
  const g = raw as RawGltf;
  const sceneExt = (g.extensions?.['KHR_gaussian_splatting'] ?? {}) as GsExt;
  // Auto-detect RC (namespaced primitive-level attributes) vs legacy
  // (attributes nested in the per-primitive extension object).
  const attrs = readPrimitiveAttributes(g);
  if (!attrs) {
    throw new Error('glb_invalid: missing KHR_gaussian_splatting primitive attributes');
  }
  const pos = accessorSlice(g, attrs.POSITION);
  const rot = accessorSlice(g, attrs._ROTATION);
  const scl = accessorSlice(g, attrs._SCALE);
  const op = accessorSlice(g, attrs._OPACITY);
  const dc = accessorSlice(g, attrs._COLOR_DC);
  if (!pos || !rot || !scl || !op || !dc) {
    throw new Error('glb_invalid: incomplete splat attribute set');
  }
  const splatCount =
    typeof sceneExt.splatCount === 'number'
      ? sceneExt.splatCount
      : typeof attrs.POSITION === 'number'
        ? g.accessors?.[attrs.POSITION]?.count ?? 0
        : 0;
  const bbox = {
    min: (sceneExt.bbox?.min as [number, number, number] | undefined) ?? [-1, -1, -1] as [number, number, number],
    max: (sceneExt.bbox?.max as [number, number, number] | undefined) ?? [1, 1, 1] as [number, number, number],
  };
  const layout: SoaAttributeLayout = {
    positions: pos,
    rotations: rot,
    scales: scl,
    opacities: op,
    colorDC: dc,
  };
  const chunk: ChunkDescriptor = {
    uri: 'glb:embedded',
    byteOffset: 0,
    byteLength: glb.bin.byteLength,
    splatCount,
    bbox,
    lod: 0,
    checksum: '',
    loadPriority: 0,
    attributeLayout: layout,
  };
  const manifest: Manifest = {
    splatCount,
    bbox,
    chunks: [chunk],
    shDegree: typeof sceneExt.shDegree === 'number' ? sceneExt.shDegree : 0,
  };
  return { manifest, bin: glb.bin };
}
