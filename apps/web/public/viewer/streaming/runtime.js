/**
 * Browser runtime that glues `StreamingTileset` to the WebGPU renderer and
 * the orbit camera, giving the viewer SuperSplat-style progressive loading:
 * fetch the octree root first, paint it immediately, then frustum-cull +
 * screen-space-error-select finer tiles and fill detail in coarse→fine order
 * as bytes arrive.
 *
 * This is the VIEWER side of the streaming UX (STREAM-3). The tile *math*
 * (loader, frustum, SSE selector, LRU streamer) already lives in this
 * `streaming/` module; this file is the missing adapter that:
 *   1. owns a `StreamingTileset` for a given `tileset.json` URL,
 *   2. each RAF frame asks it for the visible `render` set + kicks fetches,
 *   3. diffs that render set against what's currently on the GPU and, when it
 *      changes, rebuilds the renderer from the resident tiles' decoded splats
 *      (via the same `splatSceneToSoaChunk → uploadChunk` path the file
 *      loaders use),
 *   4. records instrumentation (time-to-first-paint, tiles-over-time,
 *      bytes-over-time) for the TTF-1 benchmark.
 *
 * The renderer's `uploadChunk` is append-only with replace-on-grow semantics,
 * so to change the resident set deterministically we build a *fresh* renderer,
 * upload the current resident tiles into it, and atomically swap it in. Tile
 * sets change rarely (only when the camera crosses an SSE/frustum boundary),
 * so the rebuild cost is amortized and never on the steady-state frame path.
 */
import { splatSceneToSoaChunk } from '../loader/to-soa.js';
import { decodeTileIndices, reconstructShRestBlob } from './glb.js';
import { decodeSFExtensions } from '../glb-polyfill/index.js';
import { clamp01, computeBbox, normalizeQuatInto, SH_C0 } from '../loader/splat-scene.js';
/**
 * Convert a decoded GLB tile (`{ json, bin }`) into the LOG-space `SplatScene`
 * that `splatSceneToSoaChunk` expects — the SAME contract `loader/sf-glb.js`
 * produces for the working single-file path.
 *
 * Tile GLBs are quantized with `CT_log_quant_attrs` (SCALE in ln-space, OPACITY
 * in logit-space) + `KHR_mesh_quantization` (normalized-int accessors with affine
 * min/max). `decodeSFExtensions` undoes ALL of that — it applies the KHR affine
 * dequant AND eagerly `exp()`s scale + `sigmoid()`s opacity — so its `scales` /
 * `opacities` are LINEAR. We then re-`ln()` scales (to-soa re-`exp()`s them) and
 * keep opacity linear, mirroring sf-glb.js exactly. SH-rest is elided in tiles
 * (it lives in the shared codebook), so the scene has none here; the caller
 * attaches it from the shared palette.
 *
 * The `zstd` decoder is passed through to satisfy `CT_zstd_split_buffer` tiles;
 * the bonsai tileset doesn't use it but defensive code stays format-agnostic.
 */
function glbTileToScene(json, bin, zstd) {
    // `payload.json` from the streaming decodeGlb is the raw JSON *string* (the
    // old manifestFromGlb path did `JSON.parse(glb.json)` internally). The
    // polyfill's decodeSFExtensions expects a PARSED object — passing the string
    // made `g.meshes` undefined → it threw "GLB has no splat primitive" for every
    // tile → black screen. Parse first.
    const g = typeof json === 'string' ? JSON.parse(json) : json;
    // Shared-palette tiles declare CT_gaussian_splatting_palette but the per-tile
    // .shpal sidecar does NOT exist — SH-rest lives in the ONE scene-global
    // codebook, reconstructed in payloadToChunk. decodeSFExtensions THROWS when
    // that extension is present with no sidecar, so strip it; we attach SH-rest
    // from the shared codebook ourselves.
    if (g.extensions && g.extensions['CT_gaussian_splatting_palette']) {
        delete g.extensions['CT_gaussian_splatting_palette'];
    }
    const decoded = decodeSFExtensions(g, bin, undefined, zstd);
    const N = decoded.count;
    // SplatScene.scales is LOG-space; the polyfill returns LINEAR. Re-ln with a
    // floor at f32::MIN_POSITIVE so an underflow can't push below ln(MIN_POSITIVE).
    const scales = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) {
        scales[i] = Math.log(Math.max(decoded.scales[i], 1.175494e-38));
    }
    // Defensive quat re-normalization (matches sf-glb.js + to-soa.js).
    const rotations = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
        normalizeQuatInto(rotations, i * 4, decoded.rotations[i * 4 + 0], decoded.rotations[i * 4 + 1], decoded.rotations[i * 4 + 2], decoded.rotations[i * 4 + 3]);
    }
    const opacity = new Float32Array(N);
    opacity.set(decoded.opacities); // already linear [0,1] (sigmoid applied)
    const dcRaw = new Float32Array(decoded.dcRaw);
    const colorDC = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++)
        colorDC[i] = clamp01(0.5 + SH_C0 * dcRaw[i]);
    const bbox = decoded.bbox ?? computeBbox(decoded.positions);
    return {
        count: N,
        positions: decoded.positions,
        rotations,
        scales,
        opacity,
        colorDC,
        dcRaw,
        shRest: undefined,
        shDegree: undefined,
        bbox: {
            min: [bbox.min[0], bbox.min[1], bbox.min[2]],
            max: [bbox.max[0], bbox.max[1], bbox.max[2]],
        },
    };
}
/**
 * Convert a streamer payload (`.sftile` scene or decoded GLB) into the
 * `{descriptor, bytes}` pair `renderer.uploadChunk` accepts.
 *
 * Both paths now funnel through `splatSceneToSoaChunk`, the SAME proven bridge
 * the single-file loaders use. For `.sftile` we already have a `SplatScene`. For
 * GLB tiles we first dequantize via `glbTileToScene` (log→linear scale,
 * logit→sigmoid opacity, KHR affine) so ln-space scale is NEVER fed to the
 * renderer as linear — the bug that made every tile bloom into a fuzzy blob.
 *
 * `sharedPalette` (the decoded root `palette.shpal` + a zstd decoder, or null)
 * upgrades shared-palette GLB tiles from geometry+DC to FULL view-dependent
 * SH-rest color: we decode the tile's `.glb.shpalx` indices, reconstruct each
 * splat's 45-D SH-rest = `codebook[index]` (degree 3), and attach that blob to
 * the scene so `splatSceneToSoaChunk` emits the `shRest`/`shDegree` slice the
 * WebGPU renderer copies into its per-scene SH-rest buffer. When the palette /
 * sidecar is absent the tile keeps its DC-only behavior unchanged.
 */
function payloadToChunk(payload, tileId, sharedPalette) {
    if (payload.kind === 'sftile') {
        return splatSceneToSoaChunk(payload.scene, `tile:${tileId}`);
    }
    // GLB tile (STREAM-2): dequantize to a log-space SplatScene first.
    const zstd = sharedPalette ? sharedPalette.zstd : undefined;
    const scene = glbTileToScene(payload.json, payload.bin, zstd);
    const n = scene.count;
    // Geometry+DC only (no shared palette, or this tile has no index sidecar).
    if (!sharedPalette || !payload.shpalxBytes) {
        if (typeof window !== 'undefined') {
            const st = (window.__shStats ??= { fullColorChunks: 0, dcOnlyChunks: 0, shSplats: 0, maxAbsErr: 0, samples: 0, anyNonZero: false });
            st.dcOnlyChunks++;
        }
        return splatSceneToSoaChunk(scene, `tile:${tileId}`);
    }
    // Full-color path: reconstruct SH-rest from the shared codebook + per-tile
    // indices and attach it to the scene before SoA packing.
    const { palette, zstd: zstdDec } = sharedPalette;
    const sx = decodeTileIndices(payload.shpalxBytes, zstdDec);
    if (sx.n !== n) {
        // Index/splat-count mismatch — fall back to DC-only rather than mis-shade.
        if (typeof window !== 'undefined') {
            const st = (window.__shStats ??= { fullColorChunks: 0, dcOnlyChunks: 0, shSplats: 0, maxAbsErr: 0, samples: 0, anyNonZero: false });
            st.dcOnlyChunks++;
        }
        return splatSceneToSoaChunk(scene, `tile:${tileId}`);
    }
    const shRest = reconstructShRestBlob(sx.indices, palette.codebook, n); // n*45 floats
    scene.shRest = shRest;
    scene.shDegree = 3;
    // In-browser correctness instrumentation (headless harness reads
    // `window.__shStats`): count full-color chunks + splats and spot-check that a
    // sampled splat's reconstructed SH-rest exactly equals codebook[index] with
    // the channel-major→interleaved transpose. Cheap (one sample per chunk),
    // strips out of any minified prod build, and never affects rendering.
    if (typeof window !== 'undefined') {
        const st = (window.__shStats ??= { fullColorChunks: 0, dcOnlyChunks: 0, shSplats: 0, maxAbsErr: 0, samples: 0, anyNonZero: false });
        st.fullColorChunks++;
        st.shSplats += n;
        const s = (n >> 1); // a mid splat
        const cbBase = sx.indices[s] * 45;
        for (let k = 0; k < 15; k++) {
            for (let c = 0; c < 3; c++) {
                const got = shRest[s * 45 + k * 3 + c];
                const want = palette.codebook[cbBase + c * 15 + k];
                const e = Math.abs(got - want);
                if (e > st.maxAbsErr) st.maxAbsErr = e;
                if (want !== 0) st.anyNonZero = true;
            }
        }
        st.samples++;
    }
    return splatSceneToSoaChunk(scene, `tile:${tileId}`);
}
/**
 * Drives progressive tileset load against a live WebGPU renderer + camera.
 *
 * Usage:
 *   const ts = await TilesetRuntime.create(url, {
 *     makeRenderer: async () => { const r = new WebGPURenderer({useComputeDecode:true}); await r.init({canvas}); return r; },
 *     getCanvasSize: () => [canvas.width, canvas.height],
 *     onMetrics: (m) => updateHud(m),
 *   });
 *   // each RAF frame:
 *   await ts.frame(cameraPose);   // returns the active renderer to draw with
 */
export class TilesetRuntime {
    streaming;        // StreamingTileset
    makeRenderer;     // async () => Renderer
    getCanvasSize;    // () => [w, h]
    onMetrics;        // (metrics) => void
    renderer = null;  // current live renderer
    residentKey = ''; // signature of tiles currently on the GPU
    rebuilding = false;
    // Instrumentation.
    t0 = 0;
    firstPaintMs = null;
    timeline = []; // [{ t, tiles, bytes, drawTiles }]
    constructor(streaming, opts) {
        this.streaming = streaming;
        this.makeRenderer = opts.makeRenderer;
        this.getCanvasSize = opts.getCanvasSize;
        this.onMetrics = opts.onMetrics ?? (() => { });
        this.t0 = performance.now();
        // Expose metrics for the TTF-1 benchmark harness / console inspection.
        if (typeof window !== 'undefined') {
            window.__tilesetMetrics = {
                t0: this.t0,
                firstPaintMs: null,
                timeline: this.timeline,
                tileset: streaming.tileset,
            };
        }
    }
    static async create(url, opts) {
        const { StreamingTileset } = await import('./index.js');
        const streaming = await StreamingTileset.create(url, {
            maximumScreenSpaceError: opts.maximumScreenSpaceError ?? 16,
            prefetchLookahead: opts.prefetchLookahead ?? 0,
            // Forward the geometry+DC override for the clean same-tree A/B (?nocolor=1).
            disableSharedPalette: !!opts.disableSharedPalette,
        });
        return new TilesetRuntime(streaming, opts);
    }
    /** Total tile count in the tileset (for HUD "x / total"). */
    get totalTiles() {
        return this.streaming.tileset.tiles.length;
    }
    /**
     * Advance one frame: run LOD selection (which kicks fetches), reconcile the
     * GPU resident set, record metrics, and return the renderer to draw with
     * (or null if nothing is resident yet — caller should clear to background).
     */
    async frame(camera) {
        const [w, h] = this.getCanvasSize();
        const result = this.streaming.frame(camera, w, h);
        // The render set is the set of tiles the selector wants drawn this frame
        // (each is resident or has a resident ancestor). Keep only those that
        // are actually decoded + cached so we never try to upload a hole.
        const drawable = [];
        for (const tile of result.render) {
            const payload = this.streaming.streamer.get(tile);
            if (payload)
                drawable.push({ tile, payload });
        }
        // Stable signature of the drawable set — rebuild GPU buffers only when
        // it changes (camera crossing an SSE/frustum boundary, or a new tile
        // finishing its fetch).
        const key = drawable.map((d) => d.tile.id).sort().join('|');
        if (key !== this.residentKey && !this.rebuilding) {
            await this.rebuild(drawable, key);
        }
        this.recordMetrics(result, drawable.length);
        return this.renderer;
    }
    /** Rebuild a fresh renderer holding exactly the drawable tiles' splats. */
    async rebuild(drawable, key) {
        this.rebuilding = true;
        try {
            const next = await this.makeRenderer();
            let uploaded = 0;
            const sharedPalette = this.streaming.sharedPalette ?? null;
            for (const { tile, payload } of drawable) {
                try {
                    const { descriptor, bytes } = payloadToChunk(payload, tile.id, sharedPalette);
                    if (descriptor.splatCount > 0) {
                        next.uploadChunk(descriptor, bytes);
                        uploaded++;
                    }
                }
                catch (err) {
                    // A single bad tile must not blank the whole scene.
                    console.warn(`tileset: skipped tile ${tile.id}: ${err.message}`);
                }
            }
            const prev = this.renderer;
            this.renderer = uploaded > 0 ? next : (this.renderer ?? next);
            this.residentKey = key;
            // Tear down the renderer we just replaced (after the swap, so a frame
            // is never drawn against a destroyed device).
            if (prev && prev !== this.renderer)
                prev.destroy?.();
        }
        finally {
            this.rebuilding = false;
        }
    }
    /** Record TTFP + tiles/bytes-over-time. Cheap; runs every frame. */
    recordMetrics(result, drawTiles) {
        const stats = result.stats;
        const now = performance.now();
        // First paint = first frame with at least one resident tile uploaded.
        if (this.firstPaintMs === null && drawTiles > 0) {
            this.firstPaintMs = now - this.t0;
            if (typeof window !== 'undefined' && window.__tilesetMetrics) {
                window.__tilesetMetrics.firstPaintMs = this.firstPaintMs;
            }
        }
        // Throttle timeline samples to ~10 Hz to bound memory.
        const last = this.timeline[this.timeline.length - 1];
        if (!last || now - last.t >= 100) {
            this.timeline.push({
                t: Math.round(now - this.t0),
                tiles: stats.residentTiles,
                bytes: stats.residentBytes,
                inFlight: stats.inFlight,
                drawTiles,
            });
        }
        this.onMetrics({
            firstPaintMs: this.firstPaintMs,
            residentTiles: stats.residentTiles,
            totalTiles: this.totalTiles,
            residentBytes: stats.residentBytes,
            inFlight: stats.inFlight,
            drawTiles,
        });
    }
    dispose() {
        this.renderer?.destroy?.();
        this.renderer = null;
        this.streaming.dispose();
    }
}
