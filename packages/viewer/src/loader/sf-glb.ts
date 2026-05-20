// SPDX-License-Identifier: Apache-2.0
/**
 * Catetus .glb loader. Handles both plain `KHR_gaussian_splatting` GLBs
 * and SF-extended GLBs (`CT_zstd_split_buffer`,
 * `CT_gaussian_splatting_palette`, `CT_quat_smallest3`) via
 * `@catetus/glb-polyfill`.
 *
 * Optional sidecars:
 *   1. `.glb.shpal` — SH-rest palette. Carried by GLB's
 *      `CT_gaussian_splatting_palette` extension `uri`; passed straight
 *      through if dropped together, else auto-fetched from `baseUrl`.
 *   2. `.glb.v5tail` — V5.2 joint-tail residual sidecar. Applied on top of
 *      the polyfill's reconstruction (mirrors the SOG loader's v5tail path).
 *
 * Parity: keep field names + signatures parallel to
 * `packages/viewer-app/src/loaders/sf-glb.ts` so a future dedupe collapses
 * the two.
 */
import {
  applyV5TailToScene,
  decodeSFExtensions,
  decodeV5TailBytes,
  type ApplyTargetScene,
} from '@catetus/glb-polyfill';
import {
  clamp01,
  computeBbox,
  normalizeQuatInto,
  SH_C0,
  shRestCoefCount,
  type SplatScene,
} from './splat-scene.js';

interface GlbParts { json: unknown; bin: Uint8Array; }

/** Split a GLB blob into its JSON + BIN chunks. */
function parseGlb(buf: Uint8Array): GlbParts {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546C67) throw new Error('glb: bad magic'); // 'glTF'
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error(`glb: unsupported version ${version}`);

  let cursor = 12;
  let json: unknown = null;
  let bin: Uint8Array | null = null;
  while (cursor + 8 <= buf.byteLength) {
    const chunkLen = dv.getUint32(cursor + 0, true);
    const chunkType = dv.getUint32(cursor + 4, true);
    const body = buf.subarray(cursor + 8, cursor + 8 + chunkLen);
    if (chunkType === 0x4E4F534A) {                       // 'JSON'
      json = JSON.parse(new TextDecoder().decode(body));
    } else if (chunkType === 0x004E4942) {                // 'BIN\0'
      bin = body;
    }
    cursor += 8 + chunkLen;
  }
  if (!json) throw new Error('glb: missing JSON chunk');
  return { json, bin: bin ?? new Uint8Array(0) };
}

export interface LoadGlbOpts {
  /** Map of sidecar uri → bytes. Filled from drag-drop or auto-fetch. */
  sidecars?: Record<string, Uint8Array | ArrayBuffer>;
  /** Absolute or relative URL prefix used to fetch missing sidecars. */
  baseUrl?: string;
}

export async function loadSfGlb(
  buf: Uint8Array,
  sourceName: string,
  opts: LoadGlbOpts = {},
): Promise<SplatScene> {
  const { json, bin } = parseGlb(buf);

  // If the GLB declares a palette extension and the caller didn't supply the
  // sidecar bytes, fetch it from baseUrl (siblings-on-server pattern).
  const ext = (json as { extensions?: Record<string, unknown> }).extensions ?? {};
  const palExt = ext['CT_gaussian_splatting_palette'] as { uri?: string } | undefined;
  const sidecars: Record<string, Uint8Array | ArrayBuffer> = { ...(opts.sidecars ?? {}) };
  if (palExt?.uri && !(palExt.uri in sidecars)) {
    if (!opts.baseUrl) {
      throw new Error(`sf-glb: GLB declares CT_gaussian_splatting_palette uri="${palExt.uri}" ` +
        `but no sidecar bytes were provided. Drop the .glb AND the .glb.shpal together, ` +
        `or load via URL so the sidecar can be auto-fetched.`);
    }
    const url = new URL(palExt.uri, opts.baseUrl).toString();
    const res = await fetch(url);
    if (!res.ok) throw new Error(`sf-glb: failed to fetch sidecar ${url}: HTTP ${res.status}`);
    sidecars[palExt.uri] = await res.arrayBuffer();
  }

  const decoded = decodeSFExtensions(json, bin, sidecars);
  const N = decoded.count;

  // Polyfill returns:
  //   dcRaw     : raw SH DC (no SH_C0 bake) — what we need for SoA.
  //   scales    : LINEAR.
  //   opacities : LINEAR [0, 1].
  // We bake DC → colorDC, defensively re-normalize quats, and re-`ln` scales.
  const positions = decoded.positions;
  const colorDC = new Float32Array(N * 3);
  const dcRaw = new Float32Array(decoded.dcRaw);
  for (let i = 0; i < N * 3; i++) {
    colorDC[i] = clamp01(0.5 + SH_C0 * decoded.dcRaw[i]!);
  }
  const rotations = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    normalizeQuatInto(rotations, i * 4,
      decoded.rotations[i * 4 + 0]!,
      decoded.rotations[i * 4 + 1]!,
      decoded.rotations[i * 4 + 2]!,
      decoded.rotations[i * 4 + 3]!);
  }

  const opacity = new Float32Array(N);
  opacity.set(decoded.opacities);

  // SplatScene.scales is log-space; polyfill returns LINEAR. Floor at
  // f32::MIN_POSITIVE so an underflow can't push us below ln(MIN_POSITIVE).
  const scales = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) {
    scales[i] = Math.log(Math.max(decoded.scales[i]!, 1.175494e-38));
  }

  // SH-rest passes through unchanged (canonical layout already).
  let shRest: Float32Array | undefined = decoded.sh_rest ?? undefined;
  let shDegree: number | undefined = decoded.shDegree > 0 ? decoded.shDegree : undefined;

  const extras: Record<string, string | number> = {};
  if (decoded.extensionsApplied.palette) extras.palette = 'SF';
  if (decoded.extensionsApplied.zstdSplitBuffer) extras.zstd = 'split';
  if (decoded.extensionsApplied.smallest3) extras.quat = 'smallest3';

  // ---- V5.2 joint-tail sidecar (optional, mirrors SOG path) -----------
  const v5tailKey = pickV5TailKey(opts.sidecars, sourceName);
  let v5tailBytes: Uint8Array | null = v5tailKey
    ? toUint8(opts.sidecars![v5tailKey]!)
    : null;
  if (!v5tailBytes && opts.baseUrl) {
    const sidecarUrl = new URL(`${sourceName}.v5tail`, opts.baseUrl).toString();
    try {
      const res = await fetch(sidecarUrl);
      if (res.ok) v5tailBytes = new Uint8Array(await res.arrayBuffer());
    } catch {
      // Silent fallback — the GLB renders fine without the sidecar.
    }
  }
  if (v5tailBytes) {
    try {
      const dec = decodeV5TailBytes(v5tailBytes);
      let applyShRest: Float32Array | null = shRest ?? null;
      let applyShRestCoefs = shRest ? shRestCoefCount(shDegree ?? 0) : 0;
      if (!applyShRest && dec.header.shRestCoefs > 0) {
        applyShRest = new Float32Array(N * dec.header.shRestCoefs * 3);
        applyShRestCoefs = dec.header.shRestCoefs;
      }
      const target: ApplyTargetScene = {
        positions, rotations, scales, opacities: opacity,
        dcRaw, shRest: applyShRest, shRestCoefs: applyShRestCoefs,
      };
      const modified = applyV5TailToScene(target, dec);
      if (applyShRest) {
        shRest = applyShRest;
        if (!shDegree || shDegree < 1) shDegree = 3;
      }
      // Re-bake colorDC after dcRaw mutations.
      for (let i = 0; i < N * 3; i++) {
        colorDC[i] = clamp01(0.5 + SH_C0 * dcRaw[i]!);
      }
      extras.v5tail = 'applied';
      extras.v5tailK = modified;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[sf-glb] failed to apply v5tail sidecar — falling back to vanilla GLB: ${String(err)}`);
    }
  }

  const bbox = decoded.bbox ?? computeBbox(positions);
  return {
    count: N,
    positions,
    rotations,
    scales,
    opacity,
    colorDC,
    shRest,
    shDegree,
    dcRaw,
    bbox: {
      min: [bbox.min[0]!, bbox.min[1]!, bbox.min[2]!],
      max: [bbox.max[0]!, bbox.max[1]!, bbox.max[2]!],
    },
    meta: { source: sourceName, format: 'sf-glb', extra: extras },
  };
}

/** Resolve the v5tail sidecar key in the bag. */
function pickV5TailKey(
  sidecars: Record<string, Uint8Array | ArrayBuffer> | undefined,
  sourceName: string,
): string | null {
  if (!sidecars) return null;
  const want = `${sourceName}.v5tail`;
  if (want in sidecars) return want;
  const base = `${sourceName.split('/').pop()}.v5tail`;
  if (base in sidecars) return base;
  for (const k of Object.keys(sidecars)) {
    if (/\.v5tail$/i.test(k)) return k;
  }
  return null;
}

function toUint8(b: Uint8Array | ArrayBuffer): Uint8Array {
  return b instanceof Uint8Array ? b : new Uint8Array(b);
}
