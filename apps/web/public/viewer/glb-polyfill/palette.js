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
/** VQ vector dimensionality used by the v1 sidecar (covers SH-rest degrees 1..3). */
export const VQ_DIM = 45;
/**
 * Decode a `.shpal` sidecar.
 *
 * @param compressed Raw bytes of the `.shpal` file.
 * @param ext Optional extension metadata for sanity-checking. Pass `null` to skip.
 * @param zstdDecompress Pure zstd frame decoder (e.g. `fzstd.decompress`).
 */
export function decodeShPaletteSidecar(compressed, ext, zstdDecompress) {
    const raw = zstdDecompress(compressed);
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const magic = dv.getUint32(0, true);
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
    let off = 20; // 16 header + 4-byte alignment pad
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
 * Materialize the per-splat SH-rest vector (length `coefCount * 3`) for a given
 * splat index from a decoded palette. Returns `null` when the palette doesn't
 * cover the requested degree.
 *
 * `coefCount` = (3 if shDeg>=1) + (5 if shDeg>=2) + (7 if shDeg>=3).
 */
export function paletteShRestForSplat(palette, splatIndex, shDegreeUsed) {
    if (shDegreeUsed <= 0 || palette.shDegree <= 0)
        return null;
    const used = Math.min(shDegreeUsed, palette.shDegree);
    let coefCount = 0;
    if (used >= 1)
        coefCount += 3;
    if (used >= 2)
        coefCount += 5;
    if (used >= 3)
        coefCount += 7;
    const out = new Float32Array(coefCount * 3);
    const idx = palette.indices[splatIndex];
    const cbBase = idx * VQ_DIM;
    // Codebook is stored channel-major to match Inria PLY's f_rest_X convention:
    //   codebook[c * coefCount + k]   for c ∈ {R, G, B}, k ∈ {0..coefCount-1}
    // Renderer expects interleaved [k][rgb]:
    //   out[k * 3 + c]
    // Transpose here so the shader receives the right layout.
    // NOTE: when coefCount < 15 (shDegree 1 or 2), the unused slots between
    // R-coefs and G-coefs in the source codebook are *still 15 floats apart*
    // because VQ_DIM is fixed at 45. Step by 15, not by coefCount.
    const stride = 15;
    for (let k = 0; k < coefCount; k++) {
        out[k * 3 + 0] = palette.codebook[cbBase + 0 * stride + k];
        out[k * 3 + 1] = palette.codebook[cbBase + 1 * stride + k];
        out[k * 3 + 2] = palette.codebook[cbBase + 2 * stride + k];
    }
    return out;
}
//# sourceMappingURL=palette.js.map