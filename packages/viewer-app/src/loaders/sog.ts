/**
 * PlayCanvas .sog loader (V2 — codebook-quantized).
 *
 * SOG is a ZIP archive that bundles `meta.json` + per-attribute WebP textures.
 * V2 layout (the only one we encode against today):
 *
 *   meta.json
 *   means_l.webp, means_u.webp        (positions as 16-bit log-transform lerps)
 *   quats.webp                        (smallest-three quaternion)
 *   scales.webp                       (256-entry codebook indices)
 *   sh0.webp                          (rgb codebook indices + sigmoid'd opacity)
 *   shN_centroids.webp, shN_labels.webp (optional SH palette; decoded into
 *                                        SplatScene.shRest when present)
 *
 * The on-disk layout mirrors `@playcanvas/splat-transform`'s `read-sog.ts`.
 *
 * Implementation notes:
 *   - We use `fflate` for the ZIP read (it handles STORE + DEFLATE).
 *   - WebP decode is delegated to the browser via `createImageBitmap`, then we
 *     pull RGBA back through an `OffscreenCanvas` (no extra wasm).
 */
import { unzipSync } from 'fflate';
import {
  applyV5TailToScene,
  decodeV5TailBytes,
  type ApplyTargetScene,
} from '@catetus/glb-polyfill';
import {
  computeBbox,
  SH_C0,
  shRestCoefCount,
  type SplatScene,
} from '../splat-scene.js';

/** Options for [`loadSog`] — companion files + URL-load auto-fetch. */
export interface LoadSogOpts {
  /** Map of sidecar filename → bytes. Filled from drag-drop or auto-fetch.
   *  Currently the only key the SOG loader honors is `<sogname>.v5tail`. */
  sidecars?: Record<string, Uint8Array | ArrayBuffer>;
  /** Absolute or relative URL prefix used to fetch missing sidecars (e.g.
   *  the URL the SOG itself was loaded from, sans filename). */
  baseUrl?: string;
}

interface SogMetaV2 {
  version: 2;
  count: number;
  means: { mins: number[]; maxs: number[]; files: string[] };
  scales: { codebook: number[]; files: string[] };
  quats: { files: string[] };
  sh0: { codebook: number[]; files: string[] };
  shN?: { count: number; bands: number; codebook: number[]; files: string[] };
}

/** Inverse of logTransform: sign(x) * (exp(|x|) - 1). */
function invLogTransform(v: number): number {
  const a = Math.abs(v);
  const e = Math.exp(a) - 1;
  return v < 0 ? -e : e;
}

/** Inverse sigmoid: y in [0,1] → logit. Used here to recover a logit, but the
 *  viewer wants linear opacity in [0,1], so we just use y directly. */
function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

/** Unpack the smallest-three quat from (px,py,pz,tag). Returns (w, x, y, z). */
function unpackQuat(px: number, py: number, pz: number, tag: number): [number, number, number, number] {
  const maxComp = tag - 252;
  const a = px / 255 * 2 - 1;
  const b = py / 255 * 2 - 1;
  const c = pz / 255 * 2 - 1;
  const sqrt2 = Math.SQRT2;
  const comps: [number, number, number, number] = [0, 0, 0, 0];
  const idx = [
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
  ][maxComp];
  comps[idx[0]] = a / sqrt2;
  comps[idx[1]] = b / sqrt2;
  comps[idx[2]] = c / sqrt2;
  const t = 1 - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
  comps[maxComp] = Math.sqrt(Math.max(0, t));
  return comps;
}

async function decodeWebpToRGBA(bytes: Uint8Array): Promise<{ rgba: Uint8ClampedArray; w: number; h: number }> {
  const blob = new Blob([bytes.slice().buffer], { type: 'image/webp' });
  const bitmap = await createImageBitmap(blob);
  const w = bitmap.width;
  const h = bitmap.height;
  // OffscreenCanvas is available in all modern browsers (Safari 16.4+).
  const canvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const ctx = (canvas as OffscreenCanvas).getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('sog: 2D context unavailable for WebP decode');
  ctx.drawImage(bitmap as unknown as CanvasImageSource, 0, 0);
  const img = ctx.getImageData(0, 0, w, h);
  bitmap.close();
  return { rgba: img.data, w, h };
}

export async function loadSog(
  buf: Uint8Array,
  sourceName: string,
  opts: LoadSogOpts = {},
): Promise<SplatScene> {
  const entries = unzipSync(buf);
  const metaRaw = entries['meta.json'];
  if (!metaRaw) throw new Error('sog: missing meta.json in archive');
  const meta = JSON.parse(new TextDecoder().decode(metaRaw)) as SogMetaV2;
  if (meta.version !== 2) {
    throw new Error(`sog: unsupported meta version "${meta.version}" (only V2)`);
  }
  const N = meta.count;

  const grab = (name: string): Uint8Array => {
    const e = entries[name];
    if (!e) throw new Error(`sog: archive missing "${name}"`);
    return e;
  };

  const meansLo = await decodeWebpToRGBA(grab(meta.means.files[0]));
  const meansHi = await decodeWebpToRGBA(grab(meta.means.files[1]));
  const quats = await decodeWebpToRGBA(grab(meta.quats.files[0]));
  const sclTex = await decodeWebpToRGBA(grab(meta.scales.files[0]));
  const sh0Tex = await decodeWebpToRGBA(grab(meta.sh0.files[0]));
  // SH-rest (optional) — palette of centroid pixels × per-splat label index.
  let shNCentroids: { rgba: Uint8ClampedArray; w: number; h: number } | null = null;
  let shNLabels: { rgba: Uint8ClampedArray; w: number; h: number } | null = null;
  if (meta.shN && meta.shN.files.length >= 2) {
    shNCentroids = await decodeWebpToRGBA(grab(meta.shN.files[0]));
    shNLabels = await decodeWebpToRGBA(grab(meta.shN.files[1]));
  }

  if (meansLo.w * meansLo.h < N) throw new Error('sog: means texture smaller than count');

  const positions = new Float32Array(N * 3);
  const rotations = new Float32Array(N * 4);
  const scales    = new Float32Array(N * 3);
  const opacity   = new Float32Array(N);
  const colorDC   = new Float32Array(N * 3);

  // Positions.
  const { mins, maxs } = meta.means;
  const xMin = mins[0], xR = (maxs[0] - mins[0]) || 1;
  const yMin = mins[1], yR = (maxs[1] - mins[1]) || 1;
  const zMin = mins[2], zR = (maxs[2] - mins[2]) || 1;
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    const xs = meansLo.rgba[o + 0] | (meansHi.rgba[o + 0] << 8);
    const ys = meansLo.rgba[o + 1] | (meansHi.rgba[o + 1] << 8);
    const zs = meansLo.rgba[o + 2] | (meansHi.rgba[o + 2] << 8);
    positions[i * 3 + 0] = invLogTransform(xMin + xR * (xs / 65535));
    positions[i * 3 + 1] = invLogTransform(yMin + yR * (ys / 65535));
    positions[i * 3 + 2] = invLogTransform(zMin + zR * (zs / 65535));
  }

  // Quaternions (smallest-3 packed). PlayCanvas stores (w, x, y, z); we carry XYZW.
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    const tag = quats.rgba[o + 3];
    if (tag < 252 || tag > 255) {
      rotations[i * 4 + 0] = 0; rotations[i * 4 + 1] = 0;
      rotations[i * 4 + 2] = 0; rotations[i * 4 + 3] = 1;
      continue;
    }
    const [w, x, y, z] = unpackQuat(quats.rgba[o + 0], quats.rgba[o + 1], quats.rgba[o + 2], tag);
    // Normalize again (rounding/clamps); reorder w→last.
    const nrm = Math.hypot(x, y, z, w) || 1;
    rotations[i * 4 + 0] = x / nrm;
    rotations[i * 4 + 1] = y / nrm;
    rotations[i * 4 + 2] = z / nrm;
    rotations[i * 4 + 3] = w / nrm;
  }

  // Scales (codebook lookup, log-space).
  const sCode = meta.scales.codebook;
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    scales[i * 3 + 0] = sCode[sclTex.rgba[o + 0]] ?? 0;
    scales[i * 3 + 1] = sCode[sclTex.rgba[o + 1]] ?? 0;
    scales[i * 3 + 2] = sCode[sclTex.rgba[o + 2]] ?? 0;
  }

  // SH-0 (color codebook) + opacity (sigmoid value byte).
  //
  // PlayCanvas's `read-sog.ts` runs `sigmoidInv` on opacity to recover a logit
  // (because their splat math expects a logit). We instead want LINEAR opacity
  // in [0,1] for the renderer, so we pass the byte value through directly.
  //
  // The SH-0 codebook stores RAW f_dc coefficients (the Inria PLY's f_dc_0..2
  // columns) in pre-activation space — typical range [-2, +8]. To get
  // display-ready DC color we apply the standard 3DGS activation
  //   colorDC = clamp(0.5 + SH_C0 * f_dc, 0, 1)
  // Earlier code (incorrectly) clamped the raw codebook values to [0,1],
  // which clipped every splat with negative f_dc to 0 → black scene.
  const cCode = meta.sh0.codebook;
  // Pre-build a `dcRaw` array now; we always have it for SOG (regardless of
  // whether shRest is decoded below). The shader uses dcRaw + SH-rest for
  // view-dependent shading; colorDC is the bake-only fallback for DC-only.
  const dcRawSog = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    const r = cCode[sh0Tex.rgba[o + 0]] ?? 0;
    const g = cCode[sh0Tex.rgba[o + 1]] ?? 0;
    const b = cCode[sh0Tex.rgba[o + 2]] ?? 0;
    dcRawSog[i * 3 + 0] = r;
    dcRawSog[i * 3 + 1] = g;
    dcRawSog[i * 3 + 2] = b;
    colorDC[i * 3 + 0] = clamp01(0.5 + SH_C0 * r);
    colorDC[i * 3 + 1] = clamp01(0.5 + SH_C0 * g);
    colorDC[i * 3 + 2] = clamp01(0.5 + SH_C0 * b);
    opacity[i] = sh0Tex.rgba[o + 3] / 255;
  }

  // SH-rest decode. The centroids texture stores `paletteSize` centroid rows
  // of `C` pixels each (C = SH coefficient count). Each pixel encodes
  // (R_codebookIdx, G_codebookIdx, B_codebookIdx, 0xff). The labels texture
  // stores per-splat 2-byte palette index packed as (lo, hi, 0, 0xff). We
  // dereference both into a Float32 [splat][k][rgb] array.
  let shRest: Float32Array | undefined = undefined;
  let shDegree: number | undefined = undefined;
  if (meta.shN && shNCentroids && shNLabels) {
    const bands = meta.shN.bands;
    const C = shRestCoefCount(bands);              // 3, 8, or 15
    if (C > 0) {
      const cb = meta.shN.codebook;                // 256-entry float scalars
      const cent = shNCentroids.rgba;
      const centW = shNCentroids.w;                // entry pixel = 4 bytes
      const lab = shNLabels.rgba;
      const rest = new Float32Array(N * C * 3);
      for (let i = 0; i < N; i++) {
        const lo = lab[i * 4 + 0];
        const hi = lab[i * 4 + 1];
        const palIdx = lo | (hi << 8);
        // Centroid row stride is C*4 bytes per entry; rows hold 64 entries.
        // Pixel coord of centroid k in centroid `palIdx`:
        //   pxCol = (palIdx % 64) * C + k
        //   pxRow = floor(palIdx / 64)
        const colBase = (palIdx & 63) * C;
        const row = palIdx >> 6;
        const rowBase = row * centW * 4;
        for (let k = 0; k < C; k++) {
          const pxOff = rowBase + (colBase + k) * 4;
          const ri = cent[pxOff + 0];
          const gi = cent[pxOff + 1];
          const bi = cent[pxOff + 2];
          const dst = (i * C + k) * 3;
          rest[dst + 0] = cb[ri] ?? 0;
          rest[dst + 1] = cb[gi] ?? 0;
          rest[dst + 2] = cb[bi] ?? 0;
        }
      }
      shRest = rest;
      shDegree = bands;
    }
  }

  // dcRaw already populated above (the SOG codebook IS the raw f_dc).
  let dcRaw = shRest ? dcRawSog : undefined;

  // ---- V5.2 joint-tail sidecar ----------------------------------------
  // If the caller dropped a `<sourceName>.v5tail` (or any *.v5tail in the
  // sidecars bag), decode it and add the residuals on top of the SOG-
  // reconstructed splats. Mirrors the GLB-path v5tail apply.
  //
  // After applying the residuals we recompute the bbox + the SH_C0-baked
  // colorDC (only the K selected splats had their DC modified, so the
  // bbox can drift; we re-derive both rather than incrementally fix-up).
  let extra: Record<string, string | number> | undefined;
  const v5tailKey = pickV5TailKey(opts.sidecars, sourceName);
  let v5tailBytes: Uint8Array | null = v5tailKey
    ? toUint8(opts.sidecars![v5tailKey])
    : null;
  if (!v5tailBytes && opts.baseUrl) {
    // URL-load mode: auto-fetch sibling `<source>.v5tail` if it 200s.
    const sidecarUrl = new URL(`${sourceName}.v5tail`, opts.baseUrl).toString();
    try {
      const res = await fetch(sidecarUrl);
      if (res.ok) {
        v5tailBytes = new Uint8Array(await res.arrayBuffer());
      }
    } catch {
      // Silent fallback — the SOG renders fine without the sidecar.
    }
  }
  if (v5tailBytes) {
    try {
      const decoded = decodeV5TailBytes(v5tailBytes);
      // V5.2 residuals always carry a non-zero SH-rest channel because the
      // codec selects splats by joint-J score; without an SH-rest scene
      // buffer the apply path can still write back pos / rot / opa / sca /
      // dc, but to handle shr we need a backing array. Allocate one of the
      // sidecar's coef shape when the SOG itself didn't carry shRest.
      let applyShRest: Float32Array | null = shRest ?? null;
      let applyShRestCoefs = shRest ? shRestCoefCount(meta.shN?.bands ?? 0) : 0;
      const sideShrCoefs = decoded.header.shRestCoefs;
      if (!applyShRest && sideShrCoefs > 0) {
        applyShRest = new Float32Array(N * sideShrCoefs * 3);
        applyShRestCoefs = sideShrCoefs;
      }
      if (!dcRaw) {
        // The polyfill expects raw DC; promote dcRawSog (already set above).
        dcRaw = dcRawSog;
      }
      const target: ApplyTargetScene = {
        positions,
        rotations,
        scales,
        opacities: opacity,
        dcRaw,
        shRest: applyShRest,
        shRestCoefs: applyShRestCoefs,
      };
      const modified = applyV5TailToScene(target, decoded);
      shRest = applyShRest ?? shRest;
      if (applyShRest && (!shDegree || shDegree < 1)) {
        // The sidecar grants us degree-3 SH-rest data even on a DC-only SOG.
        shDegree = 3;
      }
      // Recompute the SH_C0-baked colorDC after dcRaw mutations.
      for (let i = 0; i < N * 3; i++) {
        colorDC[i] = clamp01(0.5 + SH_C0 * dcRaw[i]);
      }
      extra = { v5tail: 'applied', v5tailK: modified };
      // eslint-disable-next-line no-console
      console.log(`[sog] applied V5.2 sidecar (${modified} splats modified)`);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[sog] failed to apply v5tail sidecar — falling back to vanilla SOG: ${err}`);
    }
  }

  // Bbox after residual apply (positions may have shifted on the top-K).
  const bbox = computeBbox(positions);
  return {
    count: N,
    positions, rotations, scales, opacity, colorDC,
    shRest, shDegree, dcRaw,
    bbox,
    meta: { source: sourceName, format: 'sog', extra },
  };
}

/** Resolve the v5tail sidecar key in the bag. Accepts `<source>.v5tail`,
 *  the basename of that, or any single `*.v5tail` entry. */
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
