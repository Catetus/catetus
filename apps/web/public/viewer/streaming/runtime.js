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
import { decodeGlb, manifestFromGlb } from './glb.js';
/**
 * Convert a streamer payload (`.sftile` scene or decoded GLB) into the
 * `{descriptor, bytes}` pair `renderer.uploadChunk` accepts.
 *
 * For `.sftile` we already have a `SplatScene`-shaped object from
 * `decodeSftile`, so `splatSceneToSoaChunk` does the SoA packing directly.
 * For GLB we reuse `manifestFromGlb` (its chunk descriptor + BIN bytes are
 * already in the renderer's expected layout).
 */
function payloadToChunk(payload, tileId) {
    if (payload.kind === 'sftile') {
        return splatSceneToSoaChunk(payload.scene, `tile:${tileId}`);
    }
    // GLB tile (STREAM-2). manifestFromGlb yields a one-chunk manifest whose
    // single chunk descriptor + BIN bytes match uploadChunk's contract.
    const { manifest, bin } = manifestFromGlb({ json: payload.json, bin: payload.bin });
    const chunk = manifest.chunks[0];
    return { descriptor: { ...chunk, uri: `tile:${tileId}` }, bytes: bin };
}
void decodeGlb; // referenced for symmetry / future direct-GLB paths
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
            for (const { tile, payload } of drawable) {
                try {
                    const { descriptor, bytes } = payloadToChunk(payload, tile.id);
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
