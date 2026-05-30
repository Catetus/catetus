/**
 * Minimal binary-glTF (GLB v2) reader for the streaming-tile loader.
 *
 * A GLB is a tiny container: 12-byte header, a JSON chunk, and a BIN chunk.
 * The producer side (`catetus-gltf::write_glb` + `catetus-optimize::tileset`)
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
const MAGIC_GLTF = 0x46546c67; // 'glTF' LE
const CHUNK_JSON = 0x4e4f534a; // 'JSON' LE
const CHUNK_BIN = 0x004e4942; // 'BIN\0' LE
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
export function decodeGlb(bytes) {
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
    let bin = new Uint8Array(new ArrayBuffer(0));
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
        }
        else if (chunkType === CHUNK_BIN) {
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
function stripTrailingPad(b, pad) {
    let end = b.byteLength;
    while (end > 0 && b[end - 1] === pad)
        end--;
    return b.subarray(0, end);
}
const GS_EXT_NAME = 'KHR_gaussian_splatting';
const RC_KEYS = {
    POSITION: `${GS_EXT_NAME}:POSITION`,
    ROTATION: `${GS_EXT_NAME}:ROTATION`,
    SCALE: `${GS_EXT_NAME}:SCALE`,
    OPACITY: `${GS_EXT_NAME}:OPACITY`,
    COLOR_DC: `${GS_EXT_NAME}:COLOR_DC`,
    // SH-elided shared-palette tiles store the DC color as the SH degree-0
    // coefficient accessor rather than a dedicated COLOR_DC attribute. The two
    // are the same 3-float-per-splat DC term; accept either name.
    SH_DC: `${GS_EXT_NAME}:SH_DEGREE_0_COEF_0`,
};
/**
 * Pull a normalized attribute index table out of the first splat primitive
 * supporting both KHR_gaussian_splatting RC (namespaced primitive-level
 * attributes) and the legacy in-extension layout. Returns `undefined` when
 * neither shape is present.
 */
function readPrimitiveAttributes(g) {
    for (const mesh of g.meshes ?? []) {
        for (const prim of mesh.primitives ?? []) {
            const pa = prim.attributes;
            if (pa && typeof pa === 'object') {
                const rec = pa;
                if (Object.keys(rec).some((k) => k.startsWith(`${GS_EXT_NAME}:`))) {
                    // POSITION is a STANDARD glTF attribute, so the producer writes it
                    // as the bare `POSITION` key (not namespaced). Accept the plain key
                    // first, falling back to a namespaced form if a producer ever uses it.
                    const pos = typeof rec['POSITION'] === 'number'
                        ? rec['POSITION']
                        : typeof rec[RC_KEYS.POSITION] === 'number'
                            ? rec[RC_KEYS.POSITION]
                            : undefined;
                    // SH-elided shared-palette tiles store the DC color as the SH
                    // degree-0 coefficient accessor rather than a dedicated COLOR_DC
                    // attribute; accept either name.
                    const dc = typeof rec[RC_KEYS.COLOR_DC] === 'number'
                        ? rec[RC_KEYS.COLOR_DC]
                        : typeof rec[RC_KEYS.SH_DC] === 'number'
                            ? rec[RC_KEYS.SH_DC]
                            : undefined;
                    return {
                        POSITION: pos,
                        _ROTATION: typeof rec[RC_KEYS.ROTATION] === 'number' ? rec[RC_KEYS.ROTATION] : undefined,
                        _SCALE: typeof rec[RC_KEYS.SCALE] === 'number' ? rec[RC_KEYS.SCALE] : undefined,
                        _OPACITY: typeof rec[RC_KEYS.OPACITY] === 'number' ? rec[RC_KEYS.OPACITY] : undefined,
                        _COLOR_DC: dc,
                    };
                }
            }
            const e = prim.extensions?.[GS_EXT_NAME];
            if (e && typeof e === 'object' && !Array.isArray(e)) {
                const legacy = e.attributes;
                if (legacy && typeof legacy === 'object')
                    return legacy;
            }
        }
    }
    return undefined;
}
function accessorSlice(g, idx) {
    if (typeof idx !== 'number')
        return undefined;
    const acc = g.accessors?.[idx];
    if (!acc || typeof acc.bufferView !== 'number')
        return undefined;
    const bv = g.bufferViews?.[acc.bufferView];
    if (!bv)
        return undefined;
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
export function manifestFromGlb(glb) {
    let raw;
    try {
        raw = JSON.parse(glb.json);
    }
    catch (err) {
        throw new Error(`glb_invalid: bad JSON (${err.message})`);
    }
    const g = raw;
    const sceneExt = (g.extensions?.['KHR_gaussian_splatting'] ?? {});
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
    const splatCount = typeof sceneExt.splatCount === 'number'
        ? sceneExt.splatCount
        : typeof attrs.POSITION === 'number'
            ? g.accessors?.[attrs.POSITION]?.count ?? 0
            : 0;
    const bbox = {
        min: sceneExt.bbox?.min ?? [-1, -1, -1],
        max: sceneExt.bbox?.max ?? [1, 1, 1],
    };
    const layout = {
        positions: pos,
        rotations: rot,
        scales: scl,
        opacities: op,
        colorDC: dc,
    };
    const chunk = {
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
    const manifest = {
        splatCount,
        bbox,
        chunks: [chunk],
        shDegree: typeof sceneExt.shDegree === 'number' ? sceneExt.shDegree : 0,
    };
    return { manifest, bin: glb.bin };
}
/**
 * Reverse `SF_zstd_split_buffer`: decompress each per-bufferView zstd frame and
 * un-transpose the byte planes back into the original interleaved layout. The
 * returned buffer is a drop-in replacement for the GLB's BIN chunk — every
 * accessor's `byteOffset` resolves to the same bytes it would on an
 * uncompressed asset, so the rest of the decode pipeline is unchanged.
 *
 * @param compressed The compressed BIN chunk (i.e. `glb.bin` after `decodeGlb`).
 * @param ext The `extensions.SF_zstd_split_buffer` object from the GLB JSON.
 * @param zstdDecompress Pure zstd frame decoder (e.g. `fzstd.decompress`).
 */
export function decompressZstdSplitBuffer(compressed, ext, zstdDecompress) {
    const out = new Uint8Array(ext.uncompressedByteLength | 0);
    for (const v of ext.views) {
        const origOffset = v.origOffset | 0;
        const origLength = v.origLength | 0;
        const stride = (v.stride ?? 1) | 0;
        const splitApplied = !!v.splitApplied;
        const compOffset = v.compOffset | 0;
        const compLength = v.compLength | 0;
        if (origLength === 0 || compLength === 0)
            continue;
        const frame = compressed.subarray(compOffset, compOffset + compLength);
        const decoded = zstdDecompress(frame);
        if (decoded.length !== origLength) {
            throw new Error(`SF_zstd_split_buffer: view length mismatch ${decoded.length} != ${origLength}`);
        }
        if (splitApplied && stride > 1) {
            // Reverse byte-plane transpose: src[b*count + i] -> dst[i*stride + b].
            const count = origLength / stride;
            for (let b = 0; b < stride; b++) {
                const srcBase = b * count;
                for (let i = 0; i < count; i++) {
                    out[origOffset + i * stride + b] = decoded[srcBase + i];
                }
            }
        }
        else {
            out.set(decoded, origOffset);
        }
    }
    return out;
}
/**
 * Decode a `.shpal` sidecar emitted by `catetus optimize`'s
 * `VQPaletteShRest` pass. The sidecar is a zstd-compressed binary with a
 * 16-byte header, 4-byte alignment pad, 45 floats of per-coefficient ranges,
 * `K*45` quantized codebook entries, and `N` u16 indices.
 *
 * Wire format mirrors `catetus-optimize::vq_palette::ShRestPaletteSidetable`.
 *
 * @param compressed Raw bytes of the `.shpal` file.
 * @param ext Optional extension metadata for sanity-checking (paletteSize,
 *   splatCount, codebookBits). Pass `null` to skip checks.
 * @param zstdDecompress Pure zstd frame decoder (e.g. `fzstd.decompress`).
 */
export function decodeShPaletteSidecar(compressed, ext, zstdDecompress) {
    const raw = zstdDecompress(compressed);
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const magic = dv.getUint32(0, true);
    // The Rust writer encodes `0x53485041u32` LE — bytes [0x41,0x50,0x48,0x53]
    // = "APHS" on disk, which is "SHPA" read big-endian.
    if (magic !== 0x53485041) {
        throw new Error(`.shpal magic mismatch: 0x${magic.toString(16)}`);
    }
    const version = dv.getUint32(4, true);
    if (version !== 1)
        throw new Error(`unsupported .shpal version: ${version}`);
    const K = dv.getUint32(8, true);
    const N = dv.getUint32(12, true);
    const codebookBits = dv.getUint8(16);
    if (ext) {
        if (ext.paletteSize !== undefined && ext.paletteSize !== K) {
            throw new Error(`.shpal paletteSize mismatch: ${ext.paletteSize} vs ${K}`);
        }
        if (ext.splatCount !== undefined && ext.splatCount !== N) {
            throw new Error(`.shpal splatCount mismatch: ${ext.splatCount} vs ${N}`);
        }
        if (ext.codebookBits !== undefined && ext.codebookBits !== codebookBits) {
            throw new Error(`.shpal codebookBits mismatch: ${ext.codebookBits} vs ${codebookBits}`);
        }
    }
    // 16 header bytes + 4-byte alignment pad before the float ranges.
    let off = 20;
    const VQ_DIM = 45;
    const ranges = new Float32Array(VQ_DIM);
    for (let d = 0; d < VQ_DIM; d++) {
        ranges[d] = dv.getFloat32(off, true);
        off += 4;
    }
    const codebook = new Float32Array(K * VQ_DIM);
    if (codebookBits <= 8) {
        const levels = 127.0;
        for (let c = 0; c < K; c++) {
            for (let d = 0; d < VQ_DIM; d++) {
                const q = dv.getInt8(off);
                off += 1;
                codebook[c * VQ_DIM + d] = (q / levels) * ranges[d];
            }
        }
    }
    else {
        const levels = 32767.0;
        for (let c = 0; c < K; c++) {
            for (let d = 0; d < VQ_DIM; d++) {
                const q = dv.getInt16(off, true);
                off += 2;
                codebook[c * VQ_DIM + d] = (q / levels) * ranges[d];
            }
        }
    }
    const indices = new Uint16Array(N);
    for (let i = 0; i < N; i++) {
        indices[i] = dv.getUint16(off, true);
        off += 2;
    }
    return {
        K,
        N,
        codebookBits,
        ranges,
        codebook,
        indices,
        shDegree: ext?.shDegree ?? 0,
    };
}
/**
 * Decode a per-tile `.glb.shpalx` index sidecar emitted by the shared-palette
 * tileset codec (`catetus-tileset::shared_palette`). Wire format (little-endian,
 * after zstd decompression of the whole file), SPLX-v1:
 *   magic "SPLX" u32 (0x53504c58 LE) | version u32 (==1) | n u32 | k u32 |
 *   indices u16[n]
 * Each `indices[j]` is the shared-codebook centroid index for tile splat `j`
 * (tile-local order, matching the tile GLB's splat order). The codebook itself
 * lives ONCE at the tileset root (`palette.shpal`, decoded via
 * {@link decodeShPaletteSidecar}); per-tile SH-rest = `codebook[indices[j]]`.
 *
 * @param compressed Raw bytes of the `.glb.shpalx` file.
 * @param zstdDecompress Pure zstd frame decoder (e.g. from `makeZstd()`).
 */
export function decodeTileIndices(compressed, zstdDecompress) {
    const raw = zstdDecompress(compressed);
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    if (raw.byteLength < 16) {
        throw new Error(`.shpalx too small: ${raw.byteLength} bytes`);
    }
    const magic = dv.getUint32(0, true);
    if (magic !== 0x53504c58) {
        throw new Error(`.shpalx magic mismatch: 0x${magic.toString(16)}`);
    }
    const version = dv.getUint32(4, true);
    if (version !== 1)
        throw new Error(`unsupported .shpalx version: ${version}`);
    const n = dv.getUint32(8, true);
    const k = dv.getUint32(12, true);
    if (raw.byteLength < 16 + n * 2) {
        throw new Error('.shpalx truncated in indices');
    }
    const indices = new Uint16Array(n);
    let off = 16;
    for (let i = 0; i < n; i++) {
        indices[i] = dv.getUint16(off, true);
        off += 2;
    }
    return { n, k, indices };
}
/**
 * Reconstruct an SH-rest blob (degree-3, 45 floats/splat) for a shared-palette
 * tile from its per-splat codebook indices and the shared root codebook.
 *
 * Output layout matches what `splatSceneToSoaChunk` / the WebGPU SoA path
 * expect: splat-major, then coef-major, then channel-minor (interleaved RGB):
 *     out[splat * 45 + k * 3 + c]   (coefCount = 15 for degree 3)
 *
 * The codebook (from `decodeShPaletteSidecar`) is stored CHANNEL-major to match
 * Inria PLY's `f_rest_X` convention: `codebook[idx*45 + c*15 + k]`. This is the
 * exact transpose `glb-polyfill/palette.js::paletteShRestForSplat` performs for
 * the single-file path — kept bit-identical here so streamed tiles render the
 * same view-dependent color as a dropped-in full GLB.
 *
 * @param indices Per-tile-splat codebook indices (from {@link decodeTileIndices}).
 * @param codebook Decoded shared codebook (`decodeShPaletteSidecar(...).codebook`,
 *   a Float32Array of length K*45).
 * @param splatCount Tile splat count (== indices.length).
 * @returns Float32Array of length splatCount*45 (degree-3 SH-rest, interleaved).
 */
export function reconstructShRestBlob(indices, codebook, splatCount) {
    const VQ_DIM = 45;
    const COEF = 15; // degree-3 coefficient count (per channel)
    const STRIDE = 15; // intra-centroid channel stride (fixed at VQ_DIM/3)
    const out = new Float32Array(splatCount * VQ_DIM);
    for (let s = 0; s < splatCount; s++) {
        const cbBase = indices[s] * VQ_DIM;
        const dst = s * VQ_DIM;
        for (let kk = 0; kk < COEF; kk++) {
            out[dst + kk * 3 + 0] = codebook[cbBase + 0 * STRIDE + kk];
            out[dst + kk * 3 + 1] = codebook[cbBase + 1 * STRIDE + kk];
            out[dst + kk * 3 + 2] = codebook[cbBase + 2 * STRIDE + kk];
        }
    }
    return out;
}
//# sourceMappingURL=glb.js.map