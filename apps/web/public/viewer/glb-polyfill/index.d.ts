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
import { type ZstdDecompress } from './zstd-split.js';
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
    bbox: {
        min: [number, number, number];
        max: [number, number, number];
    } | null;
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
export declare function decodeSFExtensions(gltfJson: unknown, binBuffer: Uint8Array, sidecars?: Record<string, ArrayBuffer | Uint8Array>, zstdDecompress?: ZstdDecompress): DecodedSplats;
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
export declare function decodeGlb(bytes: Uint8Array, zstdDecompress?: ZstdDecompress): DecodedSplats;
//# sourceMappingURL=index.d.ts.map