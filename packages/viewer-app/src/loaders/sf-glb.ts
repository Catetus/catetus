/**
 * Catetus .glb loader. Handles both plain `KHR_gaussian_splatting` GLBs and
 * SF-extended GLBs (`CT_zstd_split_buffer`, `CT_gaussian_splatting_palette`,
 * `CT_quat_smallest3`) via `@catetus/glb-polyfill`.
 *
 * Sidecar handling:
 *   1. The GLB's `CT_gaussian_splatting_palette` extension carries a `.shpal`
 *      `uri`. If the caller drops both files together, we pass the bytes
 *      straight through.
 *   2. Otherwise we auto-fetch the sibling sidecar from the same directory the
 *      GLB was loaded from. For local file picks/drops there is no directory;
 *      we surface a clear error in that case.
 *
 * Future hook for V5.2 joint-tail residual (task #109):
 *   Once the `CT_v5_tail_residual` decoder ships, plumb a second sidecar
 *   (`.v5tail`) through this same `sidecars` map and call the residual
 *   apply-on-top pass right after `decodeSFExtensions` returns. The viewer
 *   needs no other changes — the residual writes through the same
 *   {DC, rest, opacity, scale} channels.
 */
import { decodeSFExtensions } from '@catetus/glb-polyfill';
import {
  computeBbox,
  normalizeQuatInto,
  SH_C0,
  type SplatScene,
} from '../splat-scene.js';

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

  // Polyfill now eagerly dequants everything:
  //   - `dcRaw`     : raw SH DC (no SH_C0 bake)
  //   - `scales`    : LINEAR (polyfill applied exp if CT_log_quant_attrs)
  //   - `opacities` : LINEAR [0, 1] (polyfill applied sigmoid if CT_log_quant_attrs)
  // We only need to (1) bake DC → linear color, (2) re-normalize quats
  // defensively, and (3) re-`ln` scales for the SplatScene contract (the
  // shader applies `exp` so the scene-level field is log-space). The old
  // conditional-on-flag path is gone — see task #113 / polyfill cleanup pass.
  const positions = decoded.positions;
  const colorDC = new Float32Array(N * 3);
  const dcRaw = new Float32Array(decoded.dcRaw);
  for (let i = 0; i < N * 3; i++) {
    colorDC[i] = clamp01(0.5 + SH_C0 * decoded.dcRaw[i]);
  }
  const rotations = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    normalizeQuatInto(rotations, i * 4,
      decoded.rotations[i * 4 + 0],
      decoded.rotations[i * 4 + 1],
      decoded.rotations[i * 4 + 2],
      decoded.rotations[i * 4 + 3]);
  }

  const opacity = new Float32Array(N);
  opacity.set(decoded.opacities);

  const bbox = decoded.bbox ?? computeBbox(positions);

  const extras: Record<string, string | number> = {};
  if (decoded.extensionsApplied.palette) extras.palette = 'SF';
  if (decoded.extensionsApplied.zstdSplitBuffer) extras.zstd = 'split';
  if (decoded.extensionsApplied.smallest3) extras.quat = 'smallest3';

  // SH-rest passes through unchanged: polyfill returns Float32Array of length
  // count*coefCount*3 ordered as [splat][k][rgb], which is exactly our canonical
  // SplatScene.shRest layout.
  const shRest = decoded.sh_rest ?? undefined;
  const shDegree = decoded.shDegree > 0 ? decoded.shDegree : undefined;

  // SplatScene.scales is log-space (shader applies exp). The polyfill now
  // always returns LINEAR scales, so we `ln` them unconditionally. Floor at
  // f32::MIN_POSITIVE so an underflow can't push us below ln(MIN_POSITIVE)
  // ≈ -87 (task #86 family).
  const scales = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) {
    scales[i] = Math.log(Math.max(decoded.scales[i], 1.175494e-38));
  }

  return {
    count: N,
    positions,
    rotations,
    scales,
    opacity,
    colorDC,
    shRest,
    shDegree,
    dcRaw: shRest ? dcRaw : undefined,
    bbox: {
      min: [bbox.min[0], bbox.min[1], bbox.min[2]],
      max: [bbox.max[0], bbox.max[1], bbox.max[2]],
    },
    meta: { source: sourceName, format: 'sf-glb', extra: extras },
  };
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
