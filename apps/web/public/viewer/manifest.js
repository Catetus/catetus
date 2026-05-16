/**
 * glTF + extension parser for SplatForge optimized assets.
 *
 * Two extensions are read:
 *   - `KHR_gaussian_splatting`            — point primitive header.
 *   - `SF_spatial_streaming_index`        — per-chunk LOD / priority table.
 */
/**
 * glTF accessor component type. Matches the underlying GL enum values so we
 * can compare directly against the JSON.
 */
export const FLOAT_COMPONENT = 5126;
export const UBYTE_COMPONENT = 5121;
export const USHORT_COMPONENT = 5123;
const GS_EXT = 'KHR_gaussian_splatting';
const SF_EXT = 'SF_spatial_streaming_index';
function isObject(x) {
    return typeof x === 'object' && x !== null && !Array.isArray(x);
}
function asVec3(x, fallback) {
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
function findSceneGaussianExt(g) {
    const top = g.extensions?.[GS_EXT];
    return isObject(top) ? top : undefined;
}
/**
 * Locate the per-primitive `KHR_gaussian_splatting` extension which carries
 * (in the legacy layout) the attribute → accessor table needed to decode
 * binary chunks. Returns the extension object plus a flag indicating whether
 * it actually carried `.attributes` (legacy) or not (RC, where attributes
 * live on the primitive itself).
 */
function findPrimitiveGaussianExt(g) {
    for (const mesh of g.meshes ?? []) {
        for (const prim of mesh.primitives ?? []) {
            const e = prim.extensions?.[GS_EXT];
            if (isObject(e))
                return e;
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
    POSITION: `${GS_EXT}:POSITION`,
    ROTATION: `${GS_EXT}:ROTATION`,
    SCALE: `${GS_EXT}:SCALE`,
    OPACITY: `${GS_EXT}:OPACITY`,
    COLOR_DC: `${GS_EXT}:COLOR_DC`,
    COLOR_SH: `${GS_EXT}:COLOR_SH`,
    // KHR_gaussian_splatting RC pre-2026: DC color lives in the SH degree-0
    // slot (`SH_DEGREE_0_COEF_0`). The Rust writer emits this name; treat it
    // as a synonym for COLOR_DC so we can read both layouts.
    COLOR_DC_RC: `${GS_EXT}:SH_DEGREE_0_COEF_0`,
    SH1_COEF_0: `${GS_EXT}:SH_DEGREE_1_COEF_0`,
    SH1_COEF_1: `${GS_EXT}:SH_DEGREE_1_COEF_1`,
    SH1_COEF_2: `${GS_EXT}:SH_DEGREE_1_COEF_2`,
};
/**
 * Extract the attribute → accessor index table from the first splat primitive,
 * supporting both the RC (namespaced keys on `primitive.attributes`) and
 * legacy (bare keys inside `prim.extensions.KHR_gaussian_splatting.attributes`)
 * layouts. Schema sniff: presence of any `KHR_gaussian_splatting:*` key on a
 * primitive's `attributes` map is taken as authoritative for RC; otherwise we
 * fall back to legacy.
 */
function extractGaussianAttributes(g) {
    for (const mesh of g.meshes ?? []) {
        for (const prim of mesh.primitives ?? []) {
            const primAttrs = prim.attributes;
            if (isObject(primAttrs)) {
                const pa = primAttrs;
                const hasRc = Object.keys(pa).some((k) => k.startsWith(`${GS_EXT}:`));
                if (hasRc) {
                    // RC DC may live under either `COLOR_DC` (early RC drafts and the
                    // SOG synthesizer) or the canonical SH-degree-0 slot
                    // (`SH_DEGREE_0_COEF_0`, what the Rust writer emits). Prefer the
                    // explicit COLOR_DC key, fall back to SH_DEGREE_0_COEF_0.
                    const dcIdx = typeof pa[RC_ATTR_KEYS.COLOR_DC] === 'number'
                        ? pa[RC_ATTR_KEYS.COLOR_DC]
                        : typeof pa[RC_ATTR_KEYS.COLOR_DC_RC] === 'number'
                            ? pa[RC_ATTR_KEYS.COLOR_DC_RC]
                            : undefined;
                    return {
                        attrs: {
                            POSITION: typeof pa[RC_ATTR_KEYS.POSITION] === 'number' ? pa[RC_ATTR_KEYS.POSITION] : undefined,
                            _ROTATION: typeof pa[RC_ATTR_KEYS.ROTATION] === 'number' ? pa[RC_ATTR_KEYS.ROTATION] : undefined,
                            _SCALE: typeof pa[RC_ATTR_KEYS.SCALE] === 'number' ? pa[RC_ATTR_KEYS.SCALE] : undefined,
                            _OPACITY: typeof pa[RC_ATTR_KEYS.OPACITY] === 'number' ? pa[RC_ATTR_KEYS.OPACITY] : undefined,
                            _COLOR_DC: dcIdx,
                            _COLOR_SH: typeof pa[RC_ATTR_KEYS.COLOR_SH] === 'number' ? pa[RC_ATTR_KEYS.COLOR_SH] : undefined,
                            _SH1_COEF_0: typeof pa[RC_ATTR_KEYS.SH1_COEF_0] === 'number' ? pa[RC_ATTR_KEYS.SH1_COEF_0] : undefined,
                            _SH1_COEF_1: typeof pa[RC_ATTR_KEYS.SH1_COEF_1] === 'number' ? pa[RC_ATTR_KEYS.SH1_COEF_1] : undefined,
                            _SH1_COEF_2: typeof pa[RC_ATTR_KEYS.SH1_COEF_2] === 'number' ? pa[RC_ATTR_KEYS.SH1_COEF_2] : undefined,
                        },
                        layout: 'rc',
                    };
                }
            }
            const e = prim.extensions?.[GS_EXT];
            if (isObject(e)) {
                const legacy = e.attributes;
                if (isObject(legacy)) {
                    return { attrs: legacy, layout: 'legacy' };
                }
            }
        }
    }
    return { attrs: {}, layout: 'legacy' };
}
function findStreamingIndex(g) {
    const top = g.extensions?.[SF_EXT];
    if (isObject(top))
        return top;
    return undefined;
}
/**
 * Resolve a single attribute accessor index into a slice into the chunk's
 * binary payload. Offsets here are relative to the buffer (chunk-relative
 * conversion happens at chunk-construction time).
 */
function accessorSlice(g, accIdx) {
    if (typeof accIdx !== 'number')
        return undefined;
    const acc = g.accessors?.[accIdx];
    if (!acc || typeof acc.bufferView !== 'number')
        return undefined;
    const bv = g.bufferViews?.[acc.bufferView];
    if (!bv)
        return undefined;
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
 * `SF_spatial_streaming_index` is absent.
 *
 * @throws Error with message starting `manifest_invalid:` for malformed input.
 */
export function parseManifest(json) {
    let raw;
    try {
        raw = JSON.parse(json);
    }
    catch (err) {
        throw new Error(`manifest_invalid: not valid JSON (${err.message})`);
    }
    if (!isObject(raw)) {
        throw new Error('manifest_invalid: root must be an object');
    }
    const g = raw;
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
    const sh1c0Slice = accessorSlice(g, attrs._SH1_COEF_0);
    const sh1c1Slice = accessorSlice(g, attrs._SH1_COEF_1);
    const sh1c2Slice = accessorSlice(g, attrs._SH1_COEF_2);
    // splatCount: prefer scene-level extension; fall back to POSITION accessor
    // count; fall back to streaming-index records (handled below).
    let splatCount = typeof sceneExt?.splatCount === 'number' ? sceneExt.splatCount : 0;
    if (splatCount === 0 && typeof attrs.POSITION === 'number') {
        const posAcc = g.accessors?.[attrs.POSITION];
        if (posAcc && typeof posAcc.count === 'number')
            splatCount = posAcc.count;
    }
    // shDegree: scene-level wins, else primitive-level.
    const shDegree = typeof sceneExt?.shDegree === 'number'
        ? sceneExt.shDegree
        : typeof primExt?.shDegree === 'number'
            ? primExt.shDegree
            : 0;
    // bbox: prefer scene-level; else POSITION accessor's min/max; else unit cube.
    let bbox = {
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
    let chunks;
    if (index && Array.isArray(index.chunks) && index.chunks.length > 0) {
        chunks = index.chunks.map((c, i) => normalizeChunk(c, i, bbox, splatCount, g, attrs));
    }
    else {
        // Synthetic single chunk pointing at the primary buffer. The SoA layout is
        // derived from the per-primitive attribute accessors so the renderer can
        // re-interleave bytes at decode time.
        const buf = g.buffers?.[0];
        if (!buf?.uri) {
            throw new Error('manifest_invalid: no streaming index and no primary buffer uri');
        }
        const layout = buildAttributeLayout(posSlice, rotSlice, sclSlice, opSlice, dcSlice, 0, {
            c0: sh1c0Slice,
            c1: sh1c1Slice,
            c2: sh1c2Slice,
        });
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
    chunks.sort((a, b) => a.loadPriority !== b.loadPriority ? a.loadPriority - b.loadPriority : a.lod - b.lod);
    return { splatCount, bbox, chunks, shDegree };
}
function buildAttributeLayout(positions, rotations, scales, opacities, colorDC, chunkByteOffset, sh1 = {}) {
    if (!positions || !rotations || !scales || !opacities || !colorDC) {
        return undefined;
    }
    const rebase = (s) => ({
        byteOffset: s.byteOffset - chunkByteOffset,
        byteLength: s.byteLength,
        componentType: s.componentType,
        normalized: s.normalized,
        min: s.min,
        max: s.max,
    });
    const layout = {
        positions: rebase(positions),
        rotations: rebase(rotations),
        scales: rebase(scales),
        opacities: rebase(opacities),
        colorDC: rebase(colorDC),
    };
    if (sh1.c0 && sh1.c1 && sh1.c2) {
        layout.sh1Coef0 = rebase(sh1.c0);
        layout.sh1Coef1 = rebase(sh1.c1);
        layout.sh1Coef2 = rebase(sh1.c2);
    }
    return layout;
}
function normalizeChunk(c, index, sceneBbox, totalSplats, g, attrs) {
    // The chunk may carry `uri` (preferred) or `buffer` (index into root.buffers).
    // The Rust writer emits both since v0.1.1 but we accept either to remain
    // forward-compatible with future packers.
    let uri = typeof c.uri === 'string' && c.uri.length > 0 ? c.uri : undefined;
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
    const fullSlice = (accIdx) => {
        const s = accessorSlice(g, accIdx);
        if (!s)
            return undefined;
        if (bufIdx !== undefined && s.bufferIdx !== bufIdx)
            return undefined;
        return s;
    };
    const layout = bufIdx !== undefined
        ? buildAttributeLayout(fullSlice(attrs.POSITION), fullSlice(attrs._ROTATION), fullSlice(attrs._SCALE), fullSlice(attrs._OPACITY), fullSlice(attrs._COLOR_DC), chunkByteOffset, {
            c0: fullSlice(attrs._SH1_COEF_0),
            c1: fullSlice(attrs._SH1_COEF_1),
            c2: fullSlice(attrs._SH1_COEF_2),
        })
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
//# sourceMappingURL=manifest.js.map