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
/**
 * Reverse `SF_zstd_split_buffer`: decompress each per-bufferView zstd frame and
 * un-transpose the byte planes back into the original interleaved layout. The
 * returned buffer is a drop-in replacement for the GLB's BIN chunk — every
 * accessor's `byteOffset` resolves to the same bytes it would on an
 * uncompressed asset.
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
//# sourceMappingURL=zstd-split.js.map