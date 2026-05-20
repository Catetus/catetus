/**
 * Decode `SF_zstd_split_buffer`: per-bufferView zstd frames concatenated into a
 * single GLB BIN chunk, optionally byte-plane transposed for better
 * compressibility on interleaved data.
 *
 * Wire format mirrors the producer in `catetus-optimize` / `catetus-gltf`.
 * The reference JS implementation lives in
 * `packages/viewer/src/streaming/glb.ts::decompressZstdSplitBuffer` — this file
 * is a hard copy with no runtime changes so the polyfill behaves identically to
 * the Catetus viewer.
 */
/** Function signature shared by `fzstd.decompress` and Node's `zlib.zstdDecompressSync`. */
export type ZstdDecompress = (compressed: Uint8Array) => Uint8Array;
/** Per-view record under `SF_zstd_split_buffer.views`. */
export interface ZstdSplitView {
    compOffset: number;
    compLength: number;
    origOffset: number;
    origLength: number;
    stride?: number;
    splitApplied?: boolean;
}
/** Top-level `SF_zstd_split_buffer` extension shape. */
export interface ZstdSplitBufferExt {
    buffer?: number;
    uncompressedByteLength: number;
    views: ZstdSplitView[];
}
/**
 * Reverse `SF_zstd_split_buffer`: decompress each per-bufferView zstd frame and
 * un-transpose the byte planes back into the original interleaved layout. The
 * returned buffer is a drop-in replacement for the GLB's BIN chunk — every
 * accessor's `byteOffset` resolves to the same bytes it would on an
 * uncompressed asset.
 */
export declare function decompressZstdSplitBuffer(compressed: Uint8Array, ext: ZstdSplitBufferExt, zstdDecompress: ZstdDecompress): Uint8Array;
//# sourceMappingURL=zstd-split.d.ts.map