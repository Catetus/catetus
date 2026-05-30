/**
 * Public entry point for the streaming-tile adapter.
 *
 * The composition story is:
 *
 *   tileset.json     →    TilesetLoader  →  TileNode tree
 *   camera+viewProj  →    Frustum + SSE  →  TileSelection
 *   TileSelection    →    TileStreamer   →  resident GLB bytes
 *   resident bytes   →    ComputeDecodePipeline / WebGPURenderer
 *
 * `StreamingTileset` exposes the surface a runtime needs to compose all
 * four layers. It is renderer-agnostic at this layer — the WebGPU wiring
 * lives in `Viewer.loadTileset`.
 *
 * Determinism: every public method takes a {@link CameraPose} (or a
 * pre-extracted viewProj) by value; given the same camera + tileset, the
 * resulting render-set and fetch-set are byte-identical across runs.
 */
import { buildViewProj } from '../renderer/math.js';
import { extractFrustum } from './frustum.js';
import { loadTilesetJson, } from './tileset_loader.js';
import { selectVisibleTiles, } from './lod_selector.js';
import { TileStreamer, } from './tile_streamer.js';
import { decodeShPaletteSidecar } from './glb.js';
// Import the browser zstd decoder by RELATIVE vendor path rather than the bare
// `'fzstd'` specifier: the standalone preview-shell index.html (and the local
// perf harness) don't carry the `fzstd` importmap entry that the Astro
// `Base.astro` layout provides, but `/viewer/vendor/fzstd.js` is always served.
import { decompress as fzstdDecompress } from '../vendor/fzstd.js';
export { extractFrustum } from './frustum.js';
export { selectVisibleTiles, screenSpaceError } from './lod_selector.js';
// `screenSpaceError` re-export is convenience for the bench / docs.
export { TileStreamer } from './tile_streamer.js';
export { loadTilesetJson, parseTileset } from './tileset_loader.js';
export { decodeGlb, manifestFromGlb, decodeTileIndices, reconstructShRestBlob } from './glb.js';
/**
 * Synchronous zstd frame decoder used for the shared SH-rest palette + per-tile
 * index sidecars. `fzstd.decompress` is the same browser zstd the single-file
 * GLB path uses (`glb-polyfill`); it is resolved via the page importmap
 * (`"fzstd": "/viewer/vendor/fzstd.js"`). Synchronous, so `payloadToChunk`
 * stays sync.
 */
const zstdDecompress = (b) => fzstdDecompress(b);
/**
 * Fetch + decode the tileset-root shared SH-rest palette (`palette.shpal`),
 * resolved as a sibling of `tileset.json`. Shared-palette tilesets store SH-rest
 * ONCE here (a VQ45 codebook); each tile carries only u16 indices in a
 * `.glb.shpalx` sidecar. Returns `null` when the tileset has no such palette
 * (FP32 / DC-only tilesets), so the caller silently falls back to geometry+DC.
 */
async function loadSharedPalette(tilesetUrl) {
    try {
        // Resolve `palette.shpal` as a sibling of tileset.json. `tilesetUrl` may be
        // a server-absolute path (e.g. "/_fs_ts/tileset.json"), which `new URL()`
        // can't take as a base — anchor to the page origin first (mirrors
        // tileset_loader.resolveAgainst), then fall back to a trailing-segment swap.
        let palUrl;
        const pageHref = typeof globalThis !== 'undefined' && globalThis.location?.href;
        try {
            const absBase = pageHref ? new URL(tilesetUrl, pageHref).toString() : tilesetUrl;
            palUrl = new URL('palette.shpal', absBase).toString();
        }
        catch {
            const slash = tilesetUrl.lastIndexOf('/');
            palUrl = slash >= 0 ? tilesetUrl.slice(0, slash + 1) + 'palette.shpal' : 'palette.shpal';
        }
        const res = await fetch(palUrl);
        if (!res.ok) {
            if (typeof window !== 'undefined')
                window.__sharedPaletteDebug = { ok: false, reason: `palette.shpal HTTP ${res.status}`, palUrl };
            return null; // no shared palette → FP32/DC tileset
        }
        const bytes = new Uint8Array(await res.arrayBuffer());
        // shDegree 3: the shared codebook is always full SH-rest (VQ45).
        const palette = decodeShPaletteSidecar(bytes, { shDegree: 3 }, zstdDecompress);
        if (typeof window !== 'undefined')
            window.__sharedPaletteDebug = { ok: true, K: palette.K, N: palette.N, palUrl };
        return { palette, zstd: zstdDecompress };
    }
    catch (err) {
        // Surface the reason (silent fallback to geometry+DC, never crash).
        if (typeof window !== 'undefined')
            window.__sharedPaletteDebug = { ok: false, reason: String(err && err.message || err) };
        return null;
    }
}
/**
 * Streaming-tile runtime. Glues the loader, frustum, selector, and streamer
 * into the per-frame loop that {@link CatetusViewer} drives.
 */
export class StreamingTileset {
    tileset;
    streamer;
    opts;
    lastEye;
    /** Decoded shared SH-rest palette + zstd decoder, or null for FP32/DC tilesets. */
    sharedPalette = null;
    constructor(tileset, opts) {
        this.tileset = tileset;
        this.streamer = new TileStreamer(opts);
        this.opts = {
            maximumScreenSpaceError: opts.maximumScreenSpaceError ?? 16,
            prefetchLookahead: opts.prefetchLookahead ?? 0,
        };
    }
    /**
     * Async factory. Fetches `tileset.json` and immediately enqueues the root
     * tile fetch so the first frame has something to draw.
     *
     * @param url absolute or relative URL of `tileset.json`.
     */
    static async create(url, opts = {}) {
        const tileset = await loadTilesetJson(url);
        const ts = new StreamingTileset(tileset, opts);
        // Fetch + decode the shared SH-rest palette ONCE (sibling of tileset.json),
        // so streamed tiles can render full view-dependent color. Null when absent.
        // `tileset.url` is the resolved tileset.json URL (parseTileset sets it).
        // `opts.disableSharedPalette` (wired from `?nocolor=1`) forces the
        // geometry+DC path on the SAME tileset tree — used by the perf harness to
        // isolate the SH-rest decode cost from the tile-tree (a clean same-tree A/B).
        ts.sharedPalette = opts.disableSharedPalette
            ? null
            : await loadSharedPalette(tileset.url ?? url);
        // When a shared palette exists, tell the streamer to also fetch each
        // tile's `.glb.shpalx` index sidecar alongside the GLB.
        ts.streamer.fetchShpalx = !!ts.sharedPalette;
        // Kick off the root fetch — without this, the first frame would render
        // nothing while LOD selection waits on a resident ancestor.
        void ts.streamer.fetchTile(tileset.root, Number.MAX_SAFE_INTEGER);
        return ts;
    }
    /**
     * Compute the visible / fetchable tile sets for `camera` and start any
     * pending fetches. Returns the lists immediately — actual content arrives
     * asynchronously; the caller polls via {@link streamer.stateOf}.
     *
     * The render-set returned here is the set of tiles whose content is
     * either resident or whose nearest resident ancestor is renderable. The
     * caller should bind those tiles' instance buffers to the GPU.
     *
     * @param viewportWidth canvas width in pixels (used by the SSE math).
     * @param viewportHeight canvas height in pixels.
     */
    frame(camera, viewportWidth, viewportHeight) {
        const aspect = viewportWidth / Math.max(viewportHeight, 1);
        const { viewProj } = buildViewProj(camera, aspect);
        const frustum = extractFrustum(viewProj);
        // Build the "resident" set by querying the streamer for each tile.
        const resident = new Set();
        for (const t of this.tileset.tiles) {
            if (this.streamer.stateOf(t) === 'loaded')
                resident.add(t.id);
        }
        const sel = selectVisibleTiles(this.tileset.root, {
            eye: camera.position,
            fovY: camera.fovY,
            viewportHeight,
            maximumScreenSpaceError: this.opts.maximumScreenSpaceError,
            frustum,
            resident,
        });
        // Touch resident tiles so the LRU keeps them.
        this.streamer.touch(sel.render);
        // Kick off fetches. Priority = the SSE the selector wanted; if the
        // selector didn't surface one we fall back to inverse-depth so coarser
        // levels load first.
        const fetched = [];
        for (let i = 0; i < sel.fetch.length; i++) {
            const tile = sel.fetch[i];
            const priority = 1_000_000 - tile.depth * 1000 - i;
            void this.streamer.fetchTile(tile, priority);
            fetched.push(tile);
        }
        // Optional camera-velocity prefetch (queue #63). Cheap: extrude the
        // camera along its movement vector, re-extract the frustum, and run the
        // selector again at a lower priority.
        if (this.opts.prefetchLookahead > 0 && this.lastEye) {
            const dx = camera.position[0] - this.lastEye[0];
            const dy = camera.position[1] - this.lastEye[1];
            const dz = camera.position[2] - this.lastEye[2];
            const speed = Math.hypot(dx, dy, dz);
            if (speed > 1e-4) {
                const k = this.opts.prefetchLookahead;
                const eyeAhead = [
                    camera.position[0] + dx * k,
                    camera.position[1] + dy * k,
                    camera.position[2] + dz * k,
                ];
                const ahead = selectVisibleTiles(this.tileset.root, {
                    eye: eyeAhead,
                    fovY: camera.fovY,
                    viewportHeight,
                    maximumScreenSpaceError: this.opts.maximumScreenSpaceError,
                    frustum,
                    resident,
                });
                for (const t of ahead.fetch) {
                    if (this.streamer.stateOf(t) === 'unloaded') {
                        void this.streamer.fetchTile(t, -t.depth); // lowest priority
                    }
                }
            }
        }
        this.lastEye = [camera.position[0], camera.position[1], camera.position[2]];
        return {
            render: sel.render,
            fetched,
            stats: this.streamer.stats(),
        };
    }
    /**
     * Block until every queued fetch settles. Useful for headless / visual-
     * regression runs that need a deterministic resident set before reading
     * back the framebuffer.
     */
    async drainPending() {
        // Snapshot in-flight promises; new ones queued during the await won't
        // be re-awaited. Repeat until quiescent.
        // `Map` doesn't expose its values cheaply via Array.from in some old
        // environments; we copy through iteration to keep this dependency-free.
        while (true) {
            // Access the private inFlight map by stats.inFlight; we use the
            // public API in a loop and let microtask scheduling handle it.
            const before = this.streamer.stats().inFlight;
            if (before === 0)
                return;
            await new Promise((r) => setTimeout(r, 0));
        }
    }
    /** Release every tile + tear down. Idempotent. */
    dispose() {
        this.streamer.clear();
    }
}
//# sourceMappingURL=index.js.map