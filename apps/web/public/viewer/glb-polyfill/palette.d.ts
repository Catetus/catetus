/**
 * Decode `SF_gaussian_splatting_palette` `.shpal` sidecars.
 *
 * Wire format (little-endian, after zstd decompression of the whole file):
 *   magic "SHPA" u32 (0x53485041 LE)
 *   version u32 (==1)
 *   palette_size K u32
 *   splat_count N u32
 *   codebook_bits u8 (then 3 bytes alignment pad)
 *   ranges f32[45]
 *   codebook int8[K*45] when codebook_bits <= 8 else int16[K*45] (LE)
 *   indices u16[N] (LE)
 *
 * This is a direct port of the reference decoder in
 * `packages/viewer/src/streaming/glb.ts::decodeShPaletteSidecar`.
 */
import type { ZstdDecompress } from './zstd-split.js';
/** Top-level `SF_gaussian_splatting_palette` extension shape. */
export interface ShPaletteExt {
    uri: string;
    shDegree?: number;
    paletteSize?: number;
    splatCount?: number;
    codebookBits?: number;
}
/** Decoded `.shpal` sidecar (45-D VQ codebook + per-splat indices). */
export interface ShPalette {
    /** Codebook entry count (palette size). */
    K: number;
    /** Splat count (matches the GLB's POSITION accessor.count). */
    N: number;
    /** Quantization bit-width (<=8: int8, else int16). */
    codebookBits: number;
    /** Per-coefficient range for dequantization (length 45). */
    ranges: Float32Array;
    /** Codebook entries, row-major: `codebook[c * 45 + d]` (length K*45). */
    codebook: Float32Array;
    /** Per-splat palette indices (length N). */
    indices: Uint16Array;
    /** SH degree the palette covers (1, 2, or 3). */
    shDegree: number;
}
/** VQ vector dimensionality used by the v1 sidecar (covers SH-rest degrees 1..3). */
export declare const VQ_DIM = 45;
/**
 * Decode a `.shpal` sidecar.
 *
 * @param compressed Raw bytes of the `.shpal` file.
 * @param ext Optional extension metadata for sanity-checking. Pass `null` to skip.
 * @param zstdDecompress Pure zstd frame decoder (e.g. `fzstd.decompress`).
 */
export declare function decodeShPaletteSidecar(compressed: Uint8Array, ext: ShPaletteExt | null, zstdDecompress: ZstdDecompress): ShPalette;
/**
 * Materialize the per-splat SH-rest vector (length `coefCount * 3`) for a given
 * splat index from a decoded palette. Returns `null` when the palette doesn't
 * cover the requested degree.
 *
 * `coefCount` = (3 if shDeg>=1) + (5 if shDeg>=2) + (7 if shDeg>=3).
 */
export declare function paletteShRestForSplat(palette: ShPalette, splatIndex: number, shDegreeUsed: number): Float32Array | null;
//# sourceMappingURL=palette.d.ts.map