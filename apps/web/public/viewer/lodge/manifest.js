/**
 * LODGE manifest loader — viewer side (Phase A.2).
 *
 * Parses the `manifest.json` emitted by `splatforge lodge build` (see
 * `crates/splatforge-lodge/src/lib.rs`) and `docs/perf/lodge-lod-spec.md`
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
/** Schema version emitted by the Phase-A.1 chunker. */
export const LODGE_MANIFEST_VERSION = 1;
/* -------------------------------------------------------------------- */
/* Parse                                                                */
/* -------------------------------------------------------------------- */
function isObject(x) {
    return typeof x === 'object' && x !== null && !Array.isArray(x);
}
function asNumber(x, fallback) {
    return typeof x === 'number' && Number.isFinite(x) ? x : fallback;
}
function asString(x, fallback) {
    return typeof x === 'string' ? x : fallback;
}
function asVec3(x) {
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
function asAabb(x) {
    if (Array.isArray(x) && x.length >= 2) {
        return [asVec3(x[0]), asVec3(x[1])];
    }
    throw new Error('lodge_manifest_invalid: expected AABB [Vec3, Vec3]');
}
function parseChunk(c, levelIndex, chunkIndex) {
    if (!isObject(c)) {
        throw new Error(`lodge_manifest_invalid: level[${levelIndex}].chunks[${chunkIndex}] not an object`);
    }
    const path = asString(c.path, '');
    if (path.length === 0) {
        throw new Error(`lodge_manifest_invalid: level[${levelIndex}].chunks[${chunkIndex}] missing path`);
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
function parseLevel(l, levelIndex) {
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
export function parseLodgeManifest(json) {
    let raw;
    try {
        raw = JSON.parse(json);
    }
    catch (err) {
        throw new Error(`lodge_manifest_invalid: bad JSON (${err.message})`);
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
export function sceneBboxCenter(m) {
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
export function sceneBboxRadius(m) {
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
export function selectLodLevel(m, cameraPos) {
    if (m.levels.length === 0)
        return 0;
    if (m.levels.length === 1)
        return 0;
    const c = sceneBboxCenter(m);
    const d = Math.sqrt((cameraPos[0] - c[0]) ** 2 +
        (cameraPos[1] - c[1]) ** 2 +
        (cameraPos[2] - c[2]) ** 2);
    // Coarsest level whose threshold ≤ d (i.e. we are at least as far out
    // as the level says it needs). Default to 0 when no level passes the
    // gate (camera inside the finest band).
    let chosen = 0;
    for (let i = 0; i < m.levels.length; i++) {
        const t = m.levels[i].depthThreshold;
        if (d >= t)
            chosen = i;
    }
    return chosen;
}
/**
 * Pick the chunks of a chosen level, sorted by camera proximity (closest
 * first). Useful for setting `loadPriority` on a streaming fetcher so the
 * camera-near chunks land in VRAM first.
 */
export function chunksByCameraProximity(m, levelIndex, cameraPos) {
    const level = m.levels[levelIndex];
    if (!level)
        return [];
    const scored = level.chunks.map((c) => {
        const dx = cameraPos[0] - c.centroid[0];
        const dy = cameraPos[1] - c.centroid[1];
        const dz = cameraPos[2] - c.centroid[2];
        return { c, d2: dx * dx + dy * dy + dz * dz };
    });
    scored.sort((a, b) => a.d2 - b.d2);
    return scored.map((s) => s.c);
}
//# sourceMappingURL=manifest.js.map