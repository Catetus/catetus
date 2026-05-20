// SPDX-License-Identifier: Apache-2.0
/**
 * Format dispatch: takes a bag of files (or URL + bytes) and returns a
 * normalized {@link SplatScene}.
 *
 * Detection order:
 *   1. Extension on the primary file (`.ply` / `.splat` / `.sog` / `.glb`).
 *   2. Magic-byte sniff if the extension is unknown.
 *
 * Sidecar handling:
 *   - `.glb`  — `.glb.shpal` (SH-rest palette) + optional `.glb.v5tail`.
 *   - `.sog`  — optional `.sog.v5tail`.
 *   The primary file is the first non-sidecar; remaining files are passed
 *   through as `sidecars[name] → bytes` (also keyed by basename when the
 *   name carries a directory prefix).
 *
 * Returns {@link SplatScene}, the canonical structure consumed by
 * {@link splatSceneToSoaChunk} (for the WebGPU renderer) or any future
 * direct-render path.
 *
 * Parity: mirrors the dispatch API in
 * `packages/viewer-app/src/loaders/dispatch.ts`. Both packages keep
 * independent (but identical-shape) copies until Phase 3 dedupes.
 */
import { isLikely3DGSPly, loadPly } from './ply.js';
import { loadSplat } from './splat.js';
import { loadSogContainer } from './sog-container.js';
import { loadSfGlb } from './sf-glb.js';
/** Load from a bag of dropped/picked files. Picks the primary splat file
 *  and bundles any sibling sidecars under both their full name and basename. */
export async function loadFromFiles(files) {
    if (files.length === 0)
        throw new Error('no files supplied');
    const primary = files.find((f) => !isSidecarName(f.name)) ?? files[0];
    const sidecars = {};
    for (const f of files) {
        if (f === primary)
            continue;
        sidecars[f.name] = f.bytes;
    }
    return loadByName(primary, sidecars);
}
/** Load from a single URL-resolved name + bytes pair. `baseUrl` enables
 *  sidecar auto-fetch for SF GLBs and v5tail-equipped SOGs / GLBs. */
export async function loadFromUrl(url, bytes) {
    const fname = url.split('/').pop().split('?')[0];
    return loadByName({ name: fname, bytes }, {}, url);
}
async function loadByName(primary, sidecarsByName, baseUrl) {
    const ext = detectFormat(primary);
    switch (ext) {
        case 'ply':
            return loadPly(primary.bytes, primary.name);
        case 'splat':
            return loadSplat(primary.bytes, primary.name);
        case 'sog': {
            const sidecars = withBasenameAliases(sidecarsByName);
            return loadSogContainer(primary.bytes, primary.name, { sidecars, baseUrl });
        }
        case 'glb': {
            const sidecars = withBasenameAliases(sidecarsByName);
            return loadSfGlb(primary.bytes, primary.name, { sidecars, baseUrl });
        }
        default:
            throw new Error(`dispatch: unknown format for "${primary.name}"`);
    }
}
/** Mirror every sidecar entry under its basename too, so a GLB's
 *  `uri: "scene.glb.shpal"` matches a drop that came in with directory prefix. */
function withBasenameAliases(input) {
    const out = {};
    for (const [name, bytes] of Object.entries(input)) {
        out[name] = bytes;
        const base = name.split('/').pop();
        if (base !== name)
            out[base] = bytes;
    }
    return out;
}
function isSidecarName(name) {
    return /\.(shpal|v5tail|bin)$/i.test(name);
}
function detectFormat(f) {
    const lower = f.name.toLowerCase();
    if (lower.endsWith('.ply'))
        return 'ply';
    if (lower.endsWith('.splat'))
        return 'splat';
    if (lower.endsWith('.sog'))
        return 'sog';
    if (lower.endsWith('.glb'))
        return 'glb';
    // Magic-byte fallback.
    const b = f.bytes;
    // GLB: "glTF" (0x46546C67 LE).
    if (b.length >= 4 && b[0] === 0x67 && b[1] === 0x6C && b[2] === 0x54 && b[3] === 0x46)
        return 'glb';
    // ZIP (SOG): "PK\x03\x04".
    if (b.length >= 4 && b[0] === 0x50 && b[1] === 0x4B && b[2] === 0x03 && b[3] === 0x04)
        return 'sog';
    // PLY: "ply\n".
    if (b.length >= 4 && b[0] === 0x70 && b[1] === 0x6C && b[2] === 0x79) {
        return isLikely3DGSPly(b) ? 'ply' : 'ply';
    }
    // .splat heuristic: 32-byte multiples + plausible float positions.
    if (b.length >= 32 && b.length % 32 === 0) {
        const dv = new DataView(b.buffer, b.byteOffset, 32);
        const x = dv.getFloat32(0, true);
        if (Number.isFinite(x) && Math.abs(x) < 1e6)
            return 'splat';
    }
    return null;
}
//# sourceMappingURL=dispatch.js.map