/**
 * @catetus/glb-polyfill
 *
 * Decode Catetus custom glTF extensions (CT_zstd_split_buffer,
 * CT_gaussian_splatting_palette, CT_quat_smallest3) so that any
 * Three.js / Babylon / model-viewer pipeline can ingest Catetus GLBs
 * without bundling the production viewer.
 *
 * Zero hard dependencies beyond `fzstd` (browser zstd). Pure functions, no
 * I/O, no DOM access — caller provides the GLB bytes and any sidecar bytes.
 */

import { decompress as fzstdDecompress } from 'fzstd';

import { decompressZstdSplitBuffer, type ZstdSplitBufferExt, type ZstdDecompress } from './zstd-split.js';
import { decodeShPaletteSidecar, paletteShRestForSplat, type ShPaletteExt, type ShPalette, VQ_DIM } from './palette.js';
import { decodeSmallest3QuatBuffer, type QuatSmallest3Ext } from './smallest3.js';

export { decompressZstdSplitBuffer } from './zstd-split.js';
export type { ZstdSplitBufferExt, ZstdSplitView, ZstdDecompress } from './zstd-split.js';
export { decodeShPaletteSidecar, paletteShRestForSplat, VQ_DIM } from './palette.js';
export type { ShPalette, ShPaletteExt } from './palette.js';
export { decodeSmallest3Quat, decodeSmallest3QuatBuffer } from './smallest3.js';
export type { QuatSmallest3Ext } from './smallest3.js';
export { decodeV5TailBytes, applyV5TailToScene } from './v5tail.js';
export type { V5TailHeader, DecodedV5Tail, ApplyTargetScene } from './v5tail.js';

/** Normalized decoded splat scene returned by {@link decodeSFExtensions}. */
export interface DecodedSplats {
  /** Splat count = positions.length / 3. */
  count: number;
  /** Float32 positions, flat XYZ (length `count*3`). */
  positions: Float32Array;
  /** Float32 quaternion rotations, flat XYZW (length `count*4`). Normalized. */
  rotations: Float32Array;
  /**
   * Float32 per-axis LINEAR scales, flat XYZ (length `count*3`). Always linear
   * — when the source GLB stores `CT_log_quant_attrs` the polyfill eagerly
   * applies `exp()` so callers never have to inspect a flag. This matches the
   * Rust decoder (`crates/catetus-gltf/src/lib.rs::apply_log_quant_attrs`).
   */
  scales: Float32Array;
  /**
   * Float32 LINEAR opacity in `[0, 1]` (length `count`). Always linear — when
   * the source GLB stores `CT_log_quant_attrs` the polyfill eagerly applies
   * `sigmoid()`. See `scales` above for the rationale.
   */
  opacities: Float32Array;
  /**
   * Float32 raw DC SH coefficients per channel, flat RGB (length `count*3`).
   *
   * Canonical public name is `dcRaw` (matches `ApplyTargetScene.dcRaw` and the
   * viewer-app `SplatScene.dcRaw`). `dc_color` is a deprecated alias kept for
   * backwards compatibility — it will be removed before the first npm publish.
   *
   * Internal/parser layer: the JSON-level key is
   * `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` (spec); we map that to
   * `dcRaw` at the public API boundary in `decodeDCColor`.
   */
  dcRaw: Float32Array;
  /**
   * @deprecated Use `dcRaw`. Same buffer, snake-case name kept for
   * back-compat through one release. Will be removed before npm publish.
   */
  dc_color: Float32Array;
  /**
   * Float32 SH-rest coefficients (degrees 1..shDegree), interleaved per splat
   * as `[k=0..coefCount-1][rgb]`. Length `count * coefCount * 3`. `null` when
   * the GLB declares no SH-rest data and no palette is attached.
   */
  sh_rest: Float32Array | null;
  /** SH-rest degree actually reconstructed (0..3). */
  shDegree: number;
  /** Scene bbox if present (KHR_gaussian_splatting.bbox or POSITION accessor min/max). */
  bbox: { min: [number, number, number]; max: [number, number, number] } | null;
  /** Which SF extensions were applied during decode. */
  extensionsApplied: {
    zstdSplitBuffer: boolean;
    palette: boolean;
    smallest3: boolean;
    /**
     * `true` iff the source GLB declared `CT_log_quant_attrs` and the
     * polyfill eagerly de-logged scales / de-logited opacities. This is
     * informational only — the output `scales` / `opacities` are already
     * linear regardless. Provided so consumers can record provenance.
     */
    logQuantAttrs: boolean;
  };
}

/* ------------------------------------------------------------------ */
/* glTF subset types                                                  */
/* ------------------------------------------------------------------ */

interface RawGltfBufferView {
  buffer?: number;
  byteOffset?: number;
  byteLength?: number;
}

interface RawGltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType?: number;
  normalized?: boolean;
  count?: number;
  type?: string;
  min?: number[];
  max?: number[];
}

interface RawGltfPrim {
  attributes?: Record<string, number>;
  extensions?: Record<string, unknown>;
}

interface RawGltf {
  buffers?: Array<{ uri?: string | null; byteLength?: number }>;
  bufferViews?: RawGltfBufferView[];
  accessors?: RawGltfAccessor[];
  meshes?: Array<{ primitives?: RawGltfPrim[] }>;
  extensions?: Record<string, unknown>;
  extensionsUsed?: string[];
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

const SH_COEFS_PER_DEGREE = [1, 3, 5, 7] as const;

/**
 * Phase 5 back-compat: GLB files encoded before the 2026-05-19 Catetus →
 * Catetus rename use `SF_*` extension keys; the current encoder writes `CT_*`.
 * For one minor version cycle the decoder accepts both — when we see an
 * `SF_*` key we rewrite it to `CT_*` (idempotent, no-op if `CT_*` is already
 * present) and emit a one-time console warning. Drop this shim and the
 * associated tests when the encoder has been at CT_* for ≥ 1 minor version
 * (target: 4-6 weeks from rename).
 */
const LEGACY_EXTENSION_REMAP: Record<string, string> = {
  CT_zstd_split_buffer: 'CT_zstd_split_buffer',
  CT_gaussian_splatting_palette: 'CT_gaussian_splatting_palette',
  CT_log_quant_attrs: 'CT_log_quant_attrs',
  CT_quat_smallest3: 'CT_quat_smallest3',
  CT_v5_tail_residual: 'CT_v5_tail_residual',
  SF_brotli_buffer: 'CT_brotli_buffer',
  CT_spatial_streaming_index: 'CT_spatial_streaming_index',
};
const warnedLegacyKeys = new Set<string>();

function normalizeLegacyExtensions(ext: Record<string, unknown> | undefined): void {
  if (!ext) return;
  for (const [legacy, modern] of Object.entries(LEGACY_EXTENSION_REMAP)) {
    if (ext[legacy] != null) {
      if (!warnedLegacyKeys.has(legacy)) {
        // eslint-disable-next-line no-console
        console.warn(
          `[catetus] Reading deprecated ${legacy} extension. Re-encode with the current encoder to use ${modern}.`,
        );
        warnedLegacyKeys.add(legacy);
      }
      if (ext[modern] == null) ext[modern] = ext[legacy];
      delete ext[legacy];
    }
  }
}

function normalizeLegacyExtensionsUsed(used: string[] | undefined): void {
  if (!used) return;
  for (let i = 0; i < used.length; i++) {
    const modern = LEGACY_EXTENSION_REMAP[used[i]];
    if (modern) used[i] = modern;
  }
}

/**
 * Decode the SF_* extensions on an in-memory GLB asset and return a normalized
 * splat scene (positions, rotations, scales, opacities, DC color, SH-rest).
 *
 * The function expects the GLB's JSON chunk already parsed and the BIN chunk
 * as a `Uint8Array` — consumers like Three.js `GLTFLoader` already split those
 * for you, so this is a small "in the loader" hook rather than a fresh GLB
 * reader.
 *
 * @param gltfJson Parsed glTF JSON document.
 * @param binBuffer Raw BIN chunk bytes (still compressed if CT_zstd_split_buffer
 *   is present — this function decompresses it for you).
 * @param sidecars Optional `{ [uri]: ArrayBuffer }` map of `.shpal` sidecars
 *   referenced by `CT_gaussian_splatting_palette.uri`.
 * @param zstdDecompress Optional zstd decoder. Defaults to `fzstd.decompress`.
 */
export function decodeSFExtensions(
  gltfJson: unknown,
  binBuffer: Uint8Array,
  sidecars?: Record<string, ArrayBuffer | Uint8Array>,
  zstdDecompress?: ZstdDecompress,
): DecodedSplats {
  const g = gltfJson as RawGltf;
  const decoder: ZstdDecompress = zstdDecompress ?? ((b) => fzstdDecompress(b));

  // Phase 5 back-compat: rewrite any legacy SF_* extension keys to CT_*
  // before any downstream lookup. Mutates `g` in place (the caller's GLB
  // JSON is typically discarded after decode).
  normalizeLegacyExtensions(g.extensions);
  normalizeLegacyExtensionsUsed(g.extensionsUsed);
  if (g.meshes) {
    for (const mesh of g.meshes) {
      if (!mesh.primitives) continue;
      for (const prim of mesh.primitives) {
        normalizeLegacyExtensions(prim.extensions);
      }
    }
  }

  // 1. Unwrap CT_zstd_split_buffer (if present).
  const zstdExt = g.extensions?.['CT_zstd_split_buffer'] as ZstdSplitBufferExt | undefined;
  const bin = zstdExt ? decompressZstdSplitBuffer(binBuffer, zstdExt, decoder) : binBuffer;

  // 1b. CT_log_quant_attrs is a marker extension carrying
  //     `{ "scale": "ln", "opacity": "logit" }` (see Rust writer in
  //     crates/catetus-gltf/src/lib.rs around `CT_LOG_QUANT_ATTRS`).
  //     When present, SCALE accessor values are in log-space and OPACITY
  //     accessor values are in logit-space. We **eagerly** apply the
  //     inverse (`exp` / `sigmoid`) below so the public `scales` /
  //     `opacities` arrays are ALWAYS linear regardless of source format.
  //     This mirrors the Rust decoder in `crates/catetus-gltf/src/lib.rs`
  //     (`apply_log_quant_attrs` / `DecodeExtensions::log_quant_attrs`) and
  //     closes the foot-gun where callers could forget the
  //     `logQuantAttrsApplied` flag and either double-apply (garbage) or
  //     skip it (renderer sees logit-space opacities and produces giant
  //     half-transparent splats — the bonsai blob bug, task #113).
  const lqaExt = g.extensions?.['CT_log_quant_attrs'] as
    | { scale?: string; opacity?: string }
    | undefined;
  const dequantScale = !!lqaExt && (lqaExt.scale ?? 'ln') === 'ln';
  const dequantOpacity = !!lqaExt && (lqaExt.opacity ?? 'logit') === 'logit';

  // 2. Load palette sidecar (if present).
  const palExt = g.extensions?.['CT_gaussian_splatting_palette'] as ShPaletteExt | undefined;
  let palette: ShPalette | null = null;
  if (palExt) {
    if (!sidecars || !(palExt.uri in sidecars)) {
      throw new Error(`CT_gaussian_splatting_palette: missing sidecar for uri "${palExt.uri}"`);
    }
    const sc = sidecars[palExt.uri];
    const bytes = sc instanceof Uint8Array ? sc : new Uint8Array(sc);
    palette = decodeShPaletteSidecar(bytes, palExt, decoder);
  }

  // 3. Locate splat primitive + KHR_gaussian_splatting attributes.
  const prim = g.meshes?.[0]?.primitives?.[0];
  if (!prim || !prim.attributes) {
    throw new Error('decodeSFExtensions: GLB has no splat primitive (mesh[0].primitives[0])');
  }
  const attrs = prim.attributes;
  const posIdx = attrs['POSITION'];
  if (typeof posIdx !== 'number') {
    throw new Error('decodeSFExtensions: missing POSITION accessor');
  }
  const posView = accessorView(g, bin, posIdx);
  const count = g.accessors![posIdx].count ?? 0;

  // 4. Decode POSITION.
  const positions = decodePositions(posView, count, g.accessors![posIdx]);

  // 5. Decode ROTATION (handles CT_quat_smallest3).
  const rotIdx = attrs['KHR_gaussian_splatting:ROTATION'];
  if (typeof rotIdx !== 'number') {
    throw new Error('decodeSFExtensions: missing KHR_gaussian_splatting:ROTATION accessor');
  }
  const rotAcc = g.accessors![rotIdx];
  const rotView = accessorView(g, bin, rotIdx);
  const s3Ext = g.extensions?.['CT_quat_smallest3'] as QuatSmallest3Ext | undefined;
  let rotations: Float32Array;
  if (s3Ext && rotAcc.componentType === 5125) {
    // SCALAR UNSIGNED_INT packed quaternions.
    const u32 = new Uint32Array(rotView.buffer, rotView.byteOffset, count);
    rotations = decodeSmallest3QuatBuffer(u32, s3Ext.componentBits ?? 10, count);
  } else {
    rotations = decodeQuatRaw(rotView, count, rotAcc);
  }

  // 6. SCALE — VEC3, FLOAT or normalized UBYTE.
  //    With CT_log_quant_attrs the accessor carries `ln(scale)`; we eagerly
  //    `exp()` so the public `scales` is always linear.
  const sclIdx = attrs['KHR_gaussian_splatting:SCALE'];
  if (typeof sclIdx !== 'number') {
    throw new Error('decodeSFExtensions: missing KHR_gaussian_splatting:SCALE accessor');
  }
  const scales = decodeVec3(accessorView(g, bin, sclIdx), count, g.accessors![sclIdx]);
  if (dequantScale) {
    for (let i = 0; i < scales.length; i++) scales[i] = Math.exp(scales[i]);
  }

  // 7. OPACITY — SCALAR, FLOAT or normalized UBYTE.
  //
  // For UBYTE we MUST honor the accessor min/max for the affine dequant — when
  // `CT_log_quant_attrs` is set the writer stored `logit(opacity)` in a
  // logit-space range (typically ≈[-12, +12]); ignoring min/max and using
  // `arr/255` collapses that to `[0, 1]` and the viewer renders giant
  // half-transparent splats. (Same family of bug as the EPSILON clamp in the
  // Rust PLY writer fixed by task #86.)
  const opIdx = attrs['KHR_gaussian_splatting:OPACITY'];
  const opacities = new Float32Array(count);
  if (typeof opIdx === 'number') {
    const oA = g.accessors![opIdx];
    const oView = accessorView(g, bin, opIdx);
    if (oA.componentType === 5126) {
      const dv = new DataView(oView.buffer, oView.byteOffset, count * 4);
      for (let i = 0; i < count; i++) opacities[i] = dv.getFloat32(i * 4, true);
    } else if (oA.componentType === 5121) {
      const arr = new Uint8Array(oView.buffer, oView.byteOffset, count);
      const lo = oA.min?.[0] ?? 0;
      const hi = oA.max?.[0] ?? 1;
      const range = hi - lo;
      for (let i = 0; i < count; i++) opacities[i] = lo + (arr[i] / 255) * range;
    } else if (oA.componentType === 5123) {
      // USHORT-normalized (rare but allowed by glTF).
      const dv = new DataView(oView.buffer, oView.byteOffset, count * 2);
      const lo = oA.min?.[0] ?? 0;
      const hi = oA.max?.[0] ?? 1;
      const range = hi - lo;
      for (let i = 0; i < count; i++) opacities[i] = lo + (dv.getUint16(i * 2, true) / 65535) * range;
    } else {
      opacities.fill(1);
    }
  } else {
    opacities.fill(1);
  }
  // Eagerly de-logit when CT_log_quant_attrs declares logit-space opacity.
  // After this step `opacities` is always linear in [0, 1].
  if (dequantOpacity) {
    for (let i = 0; i < count; i++) {
      opacities[i] = 1 / (1 + Math.exp(-opacities[i]));
    }
  }

  // 8. DC color — KHR_gaussian_splatting:SH_DEGREE_0_COEF_0 (raw DC coefficients).
  //    The polyfill returns RAW DC (no SH_C0 bake, no +0.5 bias) to match how
  //    the Catetus bench harness consumes the value; renderers that want
  //    sRGB-ish color do `color = clamp(SH_C0 * dc + 0.5, 0, 1)` downstream.
  //
  //    Naming boundary: the glTF JSON-level attribute is
  //    `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` (snake-ish spec name); the
  //    public API field is `dcRaw` (camelCase, matches viewer-app SplatScene
  //    and ApplyTargetScene). `dc_color` is a back-compat alias on the same
  //    buffer.
  const dcRaw = decodeDCColor(g, bin, attrs, count);

  // 9. SH-rest from palette (if present). The polyfill does not currently
  //    decode the raw KHR_gaussian_splatting:SH_DEGREE_l_COEF_n accessors —
  //    those don't use any SF_* extension and any standard loader can pull
  //    them straight out of the (decompressed) BIN.
  let sh_rest: Float32Array | null = null;
  let shDegree = 0;
  if (palette && palette.shDegree > 0) {
    shDegree = palette.shDegree;
    let coefCount = 0;
    for (let l = 1; l <= shDegree; l++) coefCount += SH_COEFS_PER_DEGREE[l];
    sh_rest = new Float32Array(count * coefCount * 3);
    for (let i = 0; i < count; i++) {
      const v = paletteShRestForSplat(palette, i, shDegree);
      if (v) sh_rest.set(v, i * coefCount * 3);
    }
  }

  // 10. bbox.
  const sceneExt = (g.extensions?.['KHR_gaussian_splatting'] as { bbox?: { min?: number[]; max?: number[] } } | undefined) ?? {};
  let bbox: DecodedSplats['bbox'] = null;
  if (sceneExt.bbox?.min && sceneExt.bbox?.max && sceneExt.bbox.min.length >= 3 && sceneExt.bbox.max.length >= 3) {
    bbox = {
      min: [sceneExt.bbox.min[0], sceneExt.bbox.min[1], sceneExt.bbox.min[2]],
      max: [sceneExt.bbox.max[0], sceneExt.bbox.max[1], sceneExt.bbox.max[2]],
    };
  } else {
    const posAcc = g.accessors![posIdx];
    if (posAcc.min && posAcc.max && posAcc.min.length >= 3 && posAcc.max.length >= 3) {
      bbox = {
        min: [posAcc.min[0], posAcc.min[1], posAcc.min[2]],
        max: [posAcc.max[0], posAcc.max[1], posAcc.max[2]],
      };
    }
  }

  return {
    count,
    positions,
    rotations,
    scales,
    opacities,
    dcRaw,
    // Back-compat alias — same buffer, deprecated. Will be dropped before
    // npm publish; new code should read `dcRaw`.
    dc_color: dcRaw,
    sh_rest,
    shDegree,
    bbox,
    extensionsApplied: {
      zstdSplitBuffer: !!zstdExt,
      palette: !!palette,
      smallest3: !!s3Ext,
      logQuantAttrs: !!lqaExt,
    },
  };
}

/* ------------------------------------------------------------------ */
/* Ergonomic one-shot wrapper                                          */
/* ------------------------------------------------------------------ */

/**
 * Convenience wrapper around {@link decodeSFExtensions} for the common case:
 * "I have raw GLB bytes and no sidecars, just give me the splats."
 *
 * Splits the GLB into JSON + BIN chunks internally and calls
 * `decodeSFExtensions(json, bin)` with no sidecar map. If the GLB declares
 * `CT_gaussian_splatting_palette` (which requires a `.shpal` sidecar) this
 * will throw — use `decodeSFExtensions(json, bin, { uri: bytes })` for the
 * sidecar path.
 *
 * Synchronous. No I/O. No DOM.
 */
export function decodeGlb(
  bytes: Uint8Array,
  zstdDecompress?: ZstdDecompress,
): DecodedSplats {
  const { json, bin } = splitGlbChunks(bytes);
  return decodeSFExtensions(json, bin, undefined, zstdDecompress);
}

/** Split a GLB into its JSON + BIN chunks. Internal helper for `decodeGlb`. */
function splitGlbChunks(bytes: Uint8Array): { json: unknown; bin: Uint8Array } {
  if (bytes.byteLength < 12) {
    throw new Error('decodeGlb: GLB shorter than 12 B header');
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546c67) {
    throw new Error(`decodeGlb: bad GLB magic 0x${magic.toString(16)}`);
  }
  const version = dv.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`decodeGlb: unsupported GLB version ${version}`);
  }
  const total = dv.getUint32(8, true);
  let cursor = 12;
  let json: unknown = null;
  let bin: Uint8Array | null = null;
  while (cursor + 8 <= total && cursor + 8 <= bytes.byteLength) {
    const chunkLen = dv.getUint32(cursor + 0, true);
    const chunkType = dv.getUint32(cursor + 4, true);
    const body = bytes.subarray(cursor + 8, cursor + 8 + chunkLen);
    if (chunkType === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(body));
    } else if (chunkType === 0x004e4942) {
      bin = body;
    }
    cursor += 8 + chunkLen;
  }
  if (!json) throw new Error('decodeGlb: GLB missing JSON chunk');
  return { json, bin: bin ?? new Uint8Array(0) };
}

/* ------------------------------------------------------------------ */
/* Accessor helpers                                                   */
/* ------------------------------------------------------------------ */

interface ViewSlice {
  buffer: ArrayBufferLike;
  byteOffset: number;
  byteLength: number;
}

function accessorView(g: RawGltf, bin: Uint8Array, idx: number): ViewSlice {
  const acc = g.accessors?.[idx];
  if (!acc || typeof acc.bufferView !== 'number') {
    throw new Error(`decodeSFExtensions: accessor ${idx} missing or has no bufferView`);
  }
  const bv = g.bufferViews?.[acc.bufferView];
  if (!bv) throw new Error(`decodeSFExtensions: bufferView ${acc.bufferView} missing`);
  const bvOff = bv.byteOffset ?? 0;
  const accOff = acc.byteOffset ?? 0;
  const off = bin.byteOffset + bvOff + accOff;
  const len = (bv.byteLength ?? 0) - accOff;
  return { buffer: bin.buffer, byteOffset: off, byteLength: len };
}

function decodePositions(view: ViewSlice, N: number, acc: RawGltfAccessor): Float32Array {
  const out = new Float32Array(N * 3);
  if (acc.componentType === 5126) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 12);
    for (let i = 0; i < N; i++) {
      out[i * 3 + 0] = dv.getFloat32(i * 12 + 0, true);
      out[i * 3 + 1] = dv.getFloat32(i * 12 + 4, true);
      out[i * 3 + 2] = dv.getFloat32(i * 12 + 8, true);
    }
  } else if (acc.componentType === 5123 && acc.normalized && acc.min && acc.max) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 6);
    const xMin = acc.min[0], yMin = acc.min[1], zMin = acc.min[2];
    const xR = acc.max[0] - xMin, yR = acc.max[1] - yMin, zR = acc.max[2] - zMin;
    for (let i = 0; i < N; i++) {
      out[i * 3 + 0] = xMin + (dv.getUint16(i * 6 + 0, true) / 65535) * xR;
      out[i * 3 + 1] = yMin + (dv.getUint16(i * 6 + 2, true) / 65535) * yR;
      out[i * 3 + 2] = zMin + (dv.getUint16(i * 6 + 4, true) / 65535) * zR;
    }
  } else {
    throw new Error(`POSITION: unsupported componentType=${acc.componentType} normalized=${acc.normalized}`);
  }
  return out;
}

function decodeVec3(view: ViewSlice, N: number, acc: RawGltfAccessor): Float32Array {
  const out = new Float32Array(N * 3);
  if (acc.componentType === 5126) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 12);
    for (let i = 0; i < N; i++) {
      out[i * 3 + 0] = dv.getFloat32(i * 12 + 0, true);
      out[i * 3 + 1] = dv.getFloat32(i * 12 + 4, true);
      out[i * 3 + 2] = dv.getFloat32(i * 12 + 8, true);
    }
  } else if (acc.componentType === 5121 && acc.normalized) {
    const arr = new Uint8Array(view.buffer, view.byteOffset, N * 3);
    const lo = acc.min ?? [0, 0, 0];
    const hi = acc.max ?? [1, 1, 1];
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        const t = arr[i * 3 + c] / 255;
        out[i * 3 + c] = lo[c] + t * (hi[c] - lo[c]);
      }
    }
  } else {
    throw new Error(`VEC3: unsupported componentType=${acc.componentType}`);
  }
  return out;
}

function decodeQuatRaw(view: ViewSlice, N: number, acc: RawGltfAccessor): Float32Array {
  const out = new Float32Array(N * 4);
  if (acc.componentType === 5126) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 16);
    for (let i = 0; i < N; i++) {
      out[i * 4 + 0] = dv.getFloat32(i * 16 + 0, true);
      out[i * 4 + 1] = dv.getFloat32(i * 16 + 4, true);
      out[i * 4 + 2] = dv.getFloat32(i * 16 + 8, true);
      out[i * 4 + 3] = dv.getFloat32(i * 16 + 12, true);
    }
  } else if (acc.componentType === 5122 && acc.normalized) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 8);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 4; c++) {
        const q = dv.getInt16(i * 8 + c * 2, true);
        out[i * 4 + c] = Math.max(q / 32767, -1);
      }
    }
  } else if (acc.componentType === 5121) {
    const arr = new Uint8Array(view.buffer, view.byteOffset, N * 4);
    const lo = acc.min ?? [-1, -1, -1, -1];
    const hi = acc.max ?? [1, 1, 1, 1];
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 4; c++) {
        const t = arr[i * 4 + c] / 255;
        out[i * 4 + c] = lo[c] + t * (hi[c] - lo[c]);
      }
    }
  } else {
    throw new Error(`ROTATION: unsupported componentType=${acc.componentType} normalized=${acc.normalized}`);
  }
  return out;
}

function decodeDCColor(
  g: RawGltf,
  bin: Uint8Array,
  attrs: Record<string, number>,
  N: number,
): Float32Array {
  const sh0Idx = attrs['KHR_gaussian_splatting:SH_DEGREE_0_COEF_0'];
  const dc = new Float32Array(N * 3);
  if (typeof sh0Idx !== 'number') {
    // Fallback: KHR_gaussian_splatting:COLOR (UBYTE RGB).
    const cIdx = attrs['KHR_gaussian_splatting:COLOR'];
    if (typeof cIdx === 'number') {
      const cAcc = g.accessors![cIdx];
      const view = accessorView(g, bin, cIdx);
      if (cAcc.componentType === 5121) {
        const arr = new Uint8Array(view.buffer, view.byteOffset, N * 3);
        for (let i = 0; i < N * 3; i++) dc[i] = arr[i] / 255;
      }
    }
    return dc;
  }
  const sAcc = g.accessors![sh0Idx];
  const view = accessorView(g, bin, sh0Idx);
  const lo = Array.isArray(sAcc.min) && sAcc.min.length >= 3 ? sAcc.min : null;
  const hi = Array.isArray(sAcc.max) && sAcc.max.length >= 3 ? sAcc.max : null;
  if (sAcc.componentType === 5126) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 12);
    for (let i = 0; i < N; i++) {
      dc[i * 3 + 0] = dv.getFloat32(i * 12 + 0, true);
      dc[i * 3 + 1] = dv.getFloat32(i * 12 + 4, true);
      dc[i * 3 + 2] = dv.getFloat32(i * 12 + 8, true);
    }
  } else if (sAcc.componentType === 5121) {
    const arr = new Uint8Array(view.buffer, view.byteOffset, N * 3);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        const t = arr[i * 3 + c] / 255;
        const l = lo ? lo[c] : 0;
        const h = hi ? hi[c] : 1;
        dc[i * 3 + c] = l + t * (h - l);
      }
    }
  } else if (sAcc.componentType === 5123) {
    const dv = new DataView(view.buffer, view.byteOffset, N * 6);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        const t = dv.getUint16((i * 3 + c) * 2, true) / 65535;
        const l = lo ? lo[c] : 0;
        const h = hi ? hi[c] : 1;
        dc[i * 3 + c] = l + t * (h - l);
      }
    }
  } else {
    throw new Error(`SH_DEGREE_0_COEF_0: unsupported componentType=${sAcc.componentType}`);
  }
  // Silence unused-import warning for VQ_DIM in environments that drop the
  // type-only re-export — this is a no-op at runtime.
  void VQ_DIM;
  return dc;
}
