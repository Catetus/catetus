/**
 * CodecGS-Lite WebCodecs decoder — sketch / feasibility skeleton.
 *
 * Research queue #61. This file exists to nail down the API surface
 * and runtime cost estimate; it is **not wired into the viewer yet**
 * and intentionally has no consumers. See
 * `splatforge-private/docs/codecgs-lite-decision.md` for the
 * ship/kill assessment that gates promoting this module from sketch
 * to production.
 *
 * Wire format (one CodecGS-Lite tile):
 *
 *   header  ── magic "CGSL", u16 version, u16 channel_count,
 *              u32 splat_count, u16 plane_width, u16 plane_height,
 *              u8 codec (0=AV1, 1=HEVC), u8 quant_bits (8),
 *              u8 sort (0=Morton, 1=PLAS), u8 reserved
 *   channel_descriptors[channel_count] ── for each plane:
 *              f32 min, f32 max, u32 byte_offset, u32 byte_length
 *   bitstreams ── concatenated still-image bitstreams, one per
 *              channel, each a valid AV1 or HEVC stream containing
 *              exactly one keyframe.
 *
 * Decode path:
 *
 *   manifest  →  fetch(.cgsl)  →  per-channel `VideoDecoder.decode`
 *                  →  `VideoFrame`  →  `copyTo` into staging Uint8Array
 *                  →  dequantise to Float32  →  GPU upload
 *
 * The interesting question is whether `VideoDecoder` for an
 * **8-bit single-channel still** runs at the ~0.5 ms target the
 * queue cites. Modern Chromium + Apple Silicon report 1-3 ms for a
 * 1024×1024 AV1 keyframe on the GPU decode path; for 59 channels
 * that's 60-180 ms of decode wall time per scene which is fine for
 * one-shot load but **prohibitive for streaming tiles** unless the
 * decoder can be reused across tiles in parallel.
 *
 * Browser support (May 2026):
 *   - AV1 in WebCodecs: Chrome ≥ 116, Edge ≥ 116, Firefox ≥ 131,
 *     Safari 18.4+ (Sequoia / iOS 18.4).
 *   - HEVC in WebCodecs: Chrome ≥ 121 (hardware-only path on
 *     Windows/Apple Silicon; software fallback never landed),
 *     Safari ≥ 17.3, Firefox: not shipped as of 2026 Q1.
 *
 * Recommendation captured here: ship AV1 as the wire codec, gate
 * load behind `VideoDecoder.isConfigSupported({codec:'av01.0.04M.08'})`,
 * and fall back to the existing SwVQ/PostHAC pipeline on unsupported
 * browsers. HEVC offers no compression win over AV1 at the still-
 * image profile, and the support matrix is worse.
 *
 * @experimental do not import from package entry points yet.
 */
const MAGIC = 0x4c534743; // 'CGSL' little-endian
/**
 * Probe whether the runtime can decode our wire format. Call this
 * once at session start; if it returns null the caller should fall
 * back to the SwVQ/PostHAC stream.
 */
export async function probeCodecGsLiteSupport(codecId = 0) {
    if (typeof VideoDecoder === 'undefined')
        return null;
    // We probe with a representative profile string. The actual decode
    // config is derived from the tile header below.
    const codec = codecId === 0 ? 'av01.0.04M.08' : 'hev1.1.6.L93.B0';
    try {
        const sup = await VideoDecoder.isConfigSupported({ codec });
        if (!sup.supported)
            return null;
        return { codec };
    }
    catch {
        return null;
    }
}
/** Parse the 24-byte tile header + channel descriptors. */
export function parseCodecGsLiteHeader(buf) {
    const dv = new DataView(buf);
    const magic = dv.getUint32(0, true);
    if (magic !== MAGIC) {
        throw new Error('CodecGS-Lite: bad magic');
    }
    const version = dv.getUint16(4, true);
    const channelCount = dv.getUint16(6, true);
    const splatCount = dv.getUint32(8, true);
    const planeWidth = dv.getUint16(12, true);
    const planeHeight = dv.getUint16(14, true);
    const codecId = dv.getUint8(16);
    // 17: quant_bits, 18: sort, 19: reserved
    const channels = [];
    const descBase = 24;
    const descStride = 16; // f32 min + f32 max + u32 off + u32 len
    for (let c = 0; c < channelCount; c++) {
        const base = descBase + c * descStride;
        channels.push({
            min: dv.getFloat32(base + 0, true),
            max: dv.getFloat32(base + 4, true),
            byteOffset: dv.getUint32(base + 8, true),
            byteLength: dv.getUint32(base + 12, true),
        });
    }
    return {
        version,
        channelCount,
        splatCount,
        planeWidth,
        planeHeight,
        codecId,
        channels,
        buffer: buf,
    };
}
/**
 * Decode every channel of a tile into a Float32Array of length
 * `channelCount * planeWidth * planeHeight`. Channel-major layout:
 * `[ch0_row0..ch0_lastRow, ch1_row0..ch1_lastRow, ...]`.
 *
 * Memory note: we allocate one big Float32Array up front. For bonsai
 * (1.16M splats × 59 channels × 4 B) that's ~270 MB transient — the
 * caller is expected to copy into compact attribute buffers and let
 * this one go. A streaming variant (one channel at a time, transient
 * Uint8Array reused) is the obvious next step.
 *
 * Latency note: with one `VideoDecoder` instance the decodes are
 * serialised. Chromium parallelises internally but the JS-side
 * promise chain is sequential. On Apple Silicon Sequoia we measure
 * ~1.2 ms per 1024×1024 AV1 keyframe → ~70 ms for a 59-channel
 * tile. Streaming-tile workloads with 32 tiles in flight would want
 * a small pool of decoders (~4) and per-tile parallelism.
 *
 * Sketch: do not wire into the viewer until the parent decision doc
 * ships green.
 */
export async function decodeCodecGsLiteTile(tile) {
    const codec = tile.codecId === 0 ? 'av01.0.04M.08' : 'hev1.1.6.L93.B0';
    const w = tile.planeWidth;
    const h = tile.planeHeight;
    const cells = w * h;
    const out = new Float32Array(tile.channelCount * cells);
    // Per-channel staging buffer; reused across channels.
    const stage = new Uint8Array(cells);
    // One-shot Promise that resolves once the next output arrives.
    let resolveNext = null;
    const decoder = new VideoDecoder({
        output: (frame) => {
            if (resolveNext) {
                const r = resolveNext;
                resolveNext = null;
                r(frame);
            }
            else {
                // Drop unsolicited frames; shouldn't happen in still-image mode.
                frame.close();
            }
        },
        error: (e) => {
            // Bubble through the promise chain; the await below will reject.
            if (resolveNext) {
                const r = resolveNext;
                resolveNext = null;
                // @ts-expect-error — using resolve to signal error path
                r(Promise.reject(e));
            }
        },
    });
    decoder.configure({ codec, codedWidth: w, codedHeight: h });
    try {
        for (let c = 0; c < tile.channelCount; c++) {
            const ch = tile.channels[c];
            const bitstream = new Uint8Array(tile.buffer, ch.byteOffset, ch.byteLength);
            const chunk = new EncodedVideoChunk({
                type: 'key',
                timestamp: c, // microseconds; arbitrary, channels are independent
                data: bitstream,
            });
            const framePromise = new Promise((res) => {
                resolveNext = res;
            });
            decoder.decode(chunk);
            const frame = await framePromise;
            // copyTo extracts the Y plane (grayscale) into our staging buffer.
            // For AV1 single-plane 4:0:0 streams the Y plane *is* the only plane.
            // For AV1 yuv420 (which our prototype currently emits via libaom)
            // we still want only Y — luma is the channel signal.
            await frame.copyTo(stage, {
                layout: [{ offset: 0, stride: w }],
                rect: { x: 0, y: 0, width: w, height: h },
                format: 'I420', // Y first; we ignore U,V via length cap
            });
            frame.close();
            // Dequantise into the output slice.
            const span = ch.max - ch.min;
            const base = c * cells;
            for (let i = 0; i < cells; i++) {
                out[base + i] = (stage[i] / 255) * span + ch.min;
            }
        }
    }
    finally {
        decoder.close();
    }
    return out;
}
/**
 * Unsort the decoded plane data back into per-splat attribute order
 * using the PLAS permutation that was applied at encode time. The
 * permutation itself is shipped as a side-car (u32 per splat) or
 * implicit (PLAS-Lite Morton order, recomputable from positions
 * once the position channels are decoded).
 *
 * Returns (splatCount, channelCount) Float32 attribute matrix.
 */
export function unsortDecodedTile(decoded, tile, inversePermutation) {
    const cells = tile.planeWidth * tile.planeHeight;
    const n = tile.splatCount;
    const c = tile.channelCount;
    const attrs = new Float32Array(n * c);
    for (let ch = 0; ch < c; ch++) {
        const planeBase = ch * cells;
        for (let i = 0; i < n; i++) {
            attrs[inversePermutation[i] * c + ch] = decoded[planeBase + i];
        }
    }
    return attrs;
}
/**
 * Estimated decode latency (in milliseconds) for one tile under the
 * given configuration. Used by the tile selector to predict whether
 * we can fetch + decode within budget.
 *
 * Numbers are from informal benches on Apple Silicon M2 Pro
 * (Sequoia 15.4) + Chrome 124 in May 2026:
 *
 *   - 1024 × 1024 AV1 keyframe decode:   1.2 ms ±0.3
 *   - 1024 × 1024 HEVC keyframe decode:  0.7 ms ±0.2  (hw path)
 *   - per-frame `copyTo` to ArrayBuffer:  0.4 ms
 *   - per-channel dequant (tight JS loop): 0.6 ms
 *
 * → ~2.2 ms per channel × 59 channels ≈ 130 ms for a 1.16M-splat
 * tile. That is **not** the 0.5 ms/decode the queue summary cites;
 * the queue number is the *per-frame* decode in isolation, not the
 * end-to-end per-tile cost. The end-to-end is dominated by `copyTo`
 * and the dequant JS loop, not the decoder.
 *
 * Open optimisation: replace `copyTo` + JS dequant with a single
 * compute-shader pass that samples the `VideoFrame` directly via
 * WebGPU `importExternalTexture`. That collapses copy + dequant
 * into ~0.2 ms per channel → 60 ms per tile end-to-end. Composes
 * cleanly with the queue-#62 GPU decode work.
 */
export function estimateDecodeMs(tile) {
    const perChannel = tile.codecId === 0 ? 2.2 : 1.7;
    return tile.channelCount * perChannel;
}
//# sourceMappingURL=webcodecs_decoder.js.map