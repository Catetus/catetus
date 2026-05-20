// SPDX-License-Identifier: Apache-2.0
/**
 * Browser-side PlayCanvas .sog container loader.
 *
 * `.sog` is a ZIP archive that bundles `meta.json` + per-attribute WebP
 * textures. This loader is the runtime cousin of `./sog.ts` (which is the
 * splat-transform reference reader used by the bench harness) — both ports
 * decode the same V2 layout; this one assumes a real browser (uses
 * `unzipSync` + `createImageBitmap` + `OffscreenCanvas`) and emits the
 * canonical {@link SplatScene}.
 *
 * V2 layout (the only one we encode against today):
 *
 *   meta.json
 *   means_l.webp, means_u.webp           — positions, 16-bit log-transform lerp
 *   quats.webp                           — smallest-three quaternion bytes
 *   scales.webp                          — 256-entry codebook indices
 *   sh0.webp                             — RGB codebook indices + sigmoid'd opacity
 *   shN_centroids.webp, shN_labels.webp  — optional SH palette
 *
 * Optional sidecar:
 *   `.v5tail` — V5.2 joint-tail residual; applied on top of the decoded SOG
 *   via the polyfill's `applyV5TailToScene`. Mirrors the GLB v5tail path.
 *
 * Parity: keep field names + signatures parallel to
 * `packages/viewer-app/src/loaders/sog.ts`. Phase 3 will dedupe.
 */
import { unzipSync } from 'fflate';
import { applyV5TailToScene, decodeV5TailBytes, } from '@catetus/glb-polyfill';
import { clamp01, computeBbox, SH_C0, shRestCoefCount, } from './splat-scene.js';
/** Inverse of logTransform: sign(x) * (exp(|x|) - 1). */
function invLogTransform(v) {
    const a = Math.abs(v);
    const e = Math.exp(a) - 1;
    return v < 0 ? -e : e;
}
/** Unpack the smallest-three quat from (px,py,pz,tag). Returns (w, x, y, z). */
function unpackQuat(px, py, pz, tag) {
    const maxComp = tag - 252;
    const a = px / 255 * 2 - 1;
    const b = py / 255 * 2 - 1;
    const c = pz / 255 * 2 - 1;
    const sqrt2 = Math.SQRT2;
    const comps = [0, 0, 0, 0];
    const idxTable = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ];
    const idx = idxTable[maxComp];
    comps[idx[0]] = a / sqrt2;
    comps[idx[1]] = b / sqrt2;
    comps[idx[2]] = c / sqrt2;
    const t = 1 - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
    comps[maxComp] = Math.sqrt(Math.max(0, t));
    return comps;
}
async function decodeWebpToRGBA(bytes) {
    const blob = new Blob([bytes.slice().buffer], { type: 'image/webp' });
    const bitmap = await createImageBitmap(blob);
    const w = bitmap.width;
    const h = bitmap.height;
    const canvas = typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(w, h)
        : Object.assign(document.createElement('canvas'), { width: w, height: h });
    const ctx = canvas.getContext('2d');
    if (!ctx)
        throw new Error('sog: 2D context unavailable for WebP decode');
    ctx.drawImage(bitmap, 0, 0);
    const img = ctx.getImageData(0, 0, w, h);
    bitmap.close();
    return { rgba: img.data, w, h };
}
export async function loadSogContainer(buf, sourceName, opts = {}) {
    const entries = unzipSync(buf);
    const metaRaw = entries['meta.json'];
    if (!metaRaw)
        throw new Error('sog: missing meta.json in archive');
    const meta = JSON.parse(new TextDecoder().decode(metaRaw));
    if (meta.version !== 2) {
        throw new Error(`sog: unsupported meta version "${String(meta.version)}" (only V2)`);
    }
    const N = meta.count;
    const grab = (name) => {
        const e = entries[name];
        if (!e)
            throw new Error(`sog: archive missing "${name}"`);
        return e;
    };
    const meansLo = await decodeWebpToRGBA(grab(meta.means.files[0]));
    const meansHi = await decodeWebpToRGBA(grab(meta.means.files[1]));
    const quats = await decodeWebpToRGBA(grab(meta.quats.files[0]));
    const sclTex = await decodeWebpToRGBA(grab(meta.scales.files[0]));
    const sh0Tex = await decodeWebpToRGBA(grab(meta.sh0.files[0]));
    let shNCentroids = null;
    let shNLabels = null;
    if (meta.shN && meta.shN.files.length >= 2) {
        shNCentroids = await decodeWebpToRGBA(grab(meta.shN.files[0]));
        shNLabels = await decodeWebpToRGBA(grab(meta.shN.files[1]));
    }
    if (meansLo.w * meansLo.h < N)
        throw new Error('sog: means texture smaller than count');
    const positions = new Float32Array(N * 3);
    const rotations = new Float32Array(N * 4);
    const scales = new Float32Array(N * 3);
    const opacity = new Float32Array(N);
    const colorDC = new Float32Array(N * 3);
    const dcRaw = new Float32Array(N * 3);
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
            rotations[i * 4 + 0] = 0;
            rotations[i * 4 + 1] = 0;
            rotations[i * 4 + 2] = 0;
            rotations[i * 4 + 3] = 1;
            continue;
        }
        const [w, x, y, z] = unpackQuat(quats.rgba[o + 0], quats.rgba[o + 1], quats.rgba[o + 2], tag);
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
    // SH-0 codebook (raw f_dc) + opacity byte → linear.
    const cCode = meta.sh0.codebook;
    for (let i = 0; i < N; i++) {
        const o = i * 4;
        const r = cCode[sh0Tex.rgba[o + 0]] ?? 0;
        const g = cCode[sh0Tex.rgba[o + 1]] ?? 0;
        const b = cCode[sh0Tex.rgba[o + 2]] ?? 0;
        dcRaw[i * 3 + 0] = r;
        dcRaw[i * 3 + 1] = g;
        dcRaw[i * 3 + 2] = b;
        colorDC[i * 3 + 0] = clamp01(0.5 + SH_C0 * r);
        colorDC[i * 3 + 1] = clamp01(0.5 + SH_C0 * g);
        colorDC[i * 3 + 2] = clamp01(0.5 + SH_C0 * b);
        opacity[i] = sh0Tex.rgba[o + 3] / 255;
    }
    // SH-rest decode (optional). See viewer-app/loaders/sog.ts for the layout
    // walkthrough — centroid rows of `C` pixels each, 64 entries per row.
    let shRest = undefined;
    let shDegree = undefined;
    if (meta.shN && shNCentroids && shNLabels) {
        const bands = meta.shN.bands;
        const C = shRestCoefCount(bands);
        if (C > 0) {
            const cb = meta.shN.codebook;
            const cent = shNCentroids.rgba;
            const centW = shNCentroids.w;
            const lab = shNLabels.rgba;
            const rest = new Float32Array(N * C * 3);
            for (let i = 0; i < N; i++) {
                const lo = lab[i * 4 + 0];
                const hi = lab[i * 4 + 1];
                const palIdx = lo | (hi << 8);
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
    // ---- V5.2 joint-tail sidecar -----------------------------------------
    const extras = {};
    const v5tailKey = pickV5TailKey(opts.sidecars, sourceName);
    let v5tailBytes = v5tailKey
        ? toUint8(opts.sidecars[v5tailKey])
        : null;
    if (!v5tailBytes && opts.baseUrl) {
        const sidecarUrl = new URL(`${sourceName}.v5tail`, opts.baseUrl).toString();
        try {
            const res = await fetch(sidecarUrl);
            if (res.ok)
                v5tailBytes = new Uint8Array(await res.arrayBuffer());
        }
        catch {
            // Silent fallback.
        }
    }
    if (v5tailBytes) {
        try {
            const dec = decodeV5TailBytes(v5tailBytes);
            let applyShRest = shRest ?? null;
            let applyShRestCoefs = shRest ? shRestCoefCount(meta.shN?.bands ?? 0) : 0;
            if (!applyShRest && dec.header.shRestCoefs > 0) {
                applyShRest = new Float32Array(N * dec.header.shRestCoefs * 3);
                applyShRestCoefs = dec.header.shRestCoefs;
            }
            const target = {
                positions, rotations, scales, opacities: opacity,
                dcRaw, shRest: applyShRest, shRestCoefs: applyShRestCoefs,
            };
            const modified = applyV5TailToScene(target, dec);
            if (applyShRest) {
                shRest = applyShRest;
                if (!shDegree || shDegree < 1)
                    shDegree = 3;
            }
            // Re-bake colorDC after dcRaw mutations.
            for (let i = 0; i < N * 3; i++) {
                colorDC[i] = clamp01(0.5 + SH_C0 * dcRaw[i]);
            }
            extras.v5tail = 'applied';
            extras.v5tailK = modified;
        }
        catch (err) {
            // eslint-disable-next-line no-console
            console.warn(`[sog] failed to apply v5tail sidecar — falling back to vanilla SOG: ${String(err)}`);
        }
    }
    const bbox = computeBbox(positions);
    return {
        count: N,
        positions, rotations, scales, opacity, colorDC,
        shRest, shDegree, dcRaw,
        bbox,
        meta: {
            source: sourceName,
            format: 'sog',
            ...(Object.keys(extras).length > 0 ? { extra: extras } : {}),
        },
    };
}
function pickV5TailKey(sidecars, sourceName) {
    if (!sidecars)
        return null;
    const want = `${sourceName}.v5tail`;
    if (want in sidecars)
        return want;
    const base = `${sourceName.split('/').pop()}.v5tail`;
    if (base in sidecars)
        return base;
    for (const k of Object.keys(sidecars)) {
        if (/\.v5tail$/i.test(k))
            return k;
    }
    return null;
}
function toUint8(b) {
    return b instanceof Uint8Array ? b : new Uint8Array(b);
}
//# sourceMappingURL=sog-container.js.map