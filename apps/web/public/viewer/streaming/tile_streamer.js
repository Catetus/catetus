/**
 * Async GLB fetcher with LRU eviction for streamed tiles.
 *
 * Responsibilities:
 *   1. Track per-tile fetch state (UNLOADED, LOADING, LOADED, FAILED).
 *   2. Dedupe concurrent fetches of the same tile (return the in-flight
 *      promise rather than racing two requests).
 *   3. Enforce a byte budget (default 512 MB) with LRU eviction — least-
 *      recently-rendered tiles are released first.
 *   4. Re-issue camera-velocity-based prefetches (research queue #63):
 *      tiles likely to enter the frustum next frame are fetched at lower
 *      priority *before* they're strictly needed.
 *
 * The streamer does NOT touch GPU resources. The runtime (`StreamingTileset`)
 * is responsible for uploading the fetched bytes into the WebGPU compute
 * pipeline; the streamer only owns the JavaScript-side bytes.
 *
 * Determinism caveat: LRU ordering is by render-frame index, which is
 * deterministic given a fixed camera path. Network jitter can change the
 * order tiles *complete* in, but that only affects which tiles are
 * resident on a given frame — the eventual `render` set converges to the
 * same answer for the same camera + tileset. The viewer's deterministic
 * mode awaits all pending fetches before emitting `frameRendered`.
 */
import { decodeGlb } from './glb.js';
import { isSftile, decodeSftile } from './sftile.js';
export class TileStreamer {
    cache = new Map();
    inFlight = new Map();
    failures = new Map();
    queue = [];
    maxBytes;
    fetchImpl;
    maxConcurrency;
    residentBytes = 0;
    currentFrame = 0;
    evictionCount = 0;
    hitCount = 0;
    missCount = 0;
    /**
     * When true, GLB tiles also fetch a sibling `.glb.shpalx` index sidecar and
     * stash its raw bytes on the payload (`payload.shpalxBytes`). Set by
     * `StreamingTileset.create` only when the tileset has a shared SH-rest
     * palette; the runtime decodes + reconstructs full color from it.
     */
    fetchShpalx = false;
    constructor(opts = {}) {
        this.maxBytes = opts.maxBytes ?? 512 * 1024 * 1024;
        this.fetchImpl = opts.fetch ?? ((...a) => fetch(...a));
        this.maxConcurrency = opts.maxConcurrency ?? 8;
    }
    /** Total bytes currently held in the resident set. */
    get residentByteCount() {
        return this.residentBytes;
    }
    /** Number of tiles currently resident. */
    get residentTileCount() {
        return this.cache.size;
    }
    /** Bump the frame counter and refresh `lastUsed` on a set of tiles. */
    touch(tiles) {
        this.currentFrame++;
        for (const t of tiles) {
            const e = this.cache.get(t.id);
            if (e)
                e.lastUsed = this.currentFrame;
        }
    }
    /** Current state for `tile`. */
    stateOf(tile) {
        if (this.cache.has(tile.id))
            return 'loaded';
        if (this.inFlight.has(tile.id))
            return 'loading';
        if (this.failures.has(tile.id))
            return 'failed';
        return 'unloaded';
    }
    /** Resident payload for `tile`, or `undefined` if not loaded. */
    get(tile) {
        const p = this.cache.get(tile.id);
        if (p)
            this.hitCount++;
        else
            this.missCount++;
        return p;
    }
    /**
     * Request that `tile` be fetched. Returns a promise that resolves to the
     * loaded payload (or rejects on fetch failure). Subsequent calls for the
     * same tile while a fetch is in flight return the existing promise.
     *
     * `priority` is used purely for ordering when the concurrency cap is hit —
     * higher numbers go first. The selector passes the tile's SSE here.
     */
    fetchTile(tile, priority = 0) {
        const cached = this.cache.get(tile.id);
        if (cached) {
            this.hitCount++;
            return Promise.resolve(cached);
        }
        const live = this.inFlight.get(tile.id);
        if (live)
            return live;
        if (this.inFlight.size >= this.maxConcurrency) {
            // Park in the priority queue; we'll start it when something completes.
            return new Promise((resolve, reject) => {
                this.queue.push({ tile, priority });
                // Re-sort so the highest priority gets pulled first.
                this.queue.sort((a, b) => b.priority - a.priority);
                const start = () => {
                    this.actuallyFetch(tile).then(resolve, reject);
                };
                // Stash resolver alongside the queued entry so the drain code can
                // pick it back up. We use a marker field on the tile node — safe
                // because TileNode is owned by the tileset loader and we add one
                // additional property name space.
                tile.__resume = start;
            });
        }
        return this.actuallyFetch(tile);
    }
    actuallyFetch(tile) {
        this.missCount++;
        const p = (async () => {
            try {
                const res = await this.fetchImpl(tile.contentUrl);
                if (!res.ok) {
                    throw new Error(`tile_fetch_failed: HTTP ${res.status} for ${tile.contentUrl}`);
                }
                const ab = await res.arrayBuffer();
                const bytes = new Uint8Array(ab);
                // Format dispatch by magic: STREAM-1 emits `.sftile`; STREAM-2's
                // GLB tiles use the `glTF` magic. Branch so the same streamer
                // serves whichever tile payload the encoder produced.
                let payload;
                try {
                    if (isSftile(bytes)) {
                        const scene = decodeSftile(bytes);
                        payload = {
                            kind: 'sftile',
                            scene,
                            bytes: bytes.byteLength,
                            lastUsed: this.currentFrame,
                        };
                    }
                    else {
                        const glb = decodeGlb(bytes);
                        payload = {
                            kind: 'glb',
                            json: glb.json,
                            bin: glb.bin,
                            bytes: bytes.byteLength,
                            lastUsed: this.currentFrame,
                        };
                        // Shared-palette tilesets carry a per-tile `.glb.shpalx`
                        // index sidecar; fetch it alongside the GLB so the runtime
                        // can reconstruct full SH-rest color. Best-effort: a missing
                        // sidecar leaves the tile at geometry+DC (never fails).
                        if (typeof window !== 'undefined') {
                            const dbg = (window.__shpalxDebug ??= { glbTiles: 0, attempted: 0, ok: 0, lastUrl: null, lastErr: null });
                            dbg.glbTiles++;
                            dbg.fetchShpalxFlag = this.fetchShpalx;
                        }
                        if (this.fetchShpalx) {
                            try {
                                const sxUrl = `${tile.contentUrl}.shpalx`;
                                if (typeof window !== 'undefined') {
                                    window.__shpalxDebug.attempted++;
                                    window.__shpalxDebug.lastUrl = sxUrl;
                                }
                                const sxRes = await this.fetchImpl(sxUrl);
                                if (sxRes.ok) {
                                    const sxBytes = new Uint8Array(await sxRes.arrayBuffer());
                                    payload.shpalxBytes = sxBytes;
                                    payload.bytes += sxBytes.byteLength;
                                    if (typeof window !== 'undefined')
                                        window.__shpalxDebug.ok++;
                                }
                                else if (typeof window !== 'undefined') {
                                    window.__shpalxDebug.lastErr = `HTTP ${sxRes.status}`;
                                }
                            }
                            catch (e) {
                                // No sidecar → DC-only for this tile; do not fail.
                                if (typeof window !== 'undefined')
                                    window.__shpalxDebug.lastErr = String(e && e.message || e);
                            }
                        }
                    }
                }
                catch (err) {
                    throw new Error(`tile_fetch_failed: ${tile.id} ${err.message}`);
                }
                this.cache.set(tile.id, payload);
                this.residentBytes += payload.bytes;
                this.maybeEvict();
                return payload;
            }
            catch (err) {
                this.failures.set(tile.id, err);
                throw err;
            }
            finally {
                this.inFlight.delete(tile.id);
                this.drainQueue();
            }
        })();
        this.inFlight.set(tile.id, p);
        return p;
    }
    /**
     * After a fetch completes, pull the next-highest-priority queued entry
     * (if any) into flight.
     */
    drainQueue() {
        while (this.inFlight.size < this.maxConcurrency &&
            this.queue.length > 0) {
            const entry = this.queue.shift();
            const tile = entry.tile;
            const resume = tile.__resume;
            tile.__resume = undefined;
            if (resume)
                resume();
        }
    }
    /**
     * Drop least-recently-rendered tiles until `residentBytes <= maxBytes`.
     * Tiles touched on the *current* frame are pinned — evicting them would
     * cause the renderer to draw a hole this frame.
     */
    maybeEvict() {
        if (this.residentBytes <= this.maxBytes)
            return;
        const entries = [...this.cache.entries()];
        entries.sort((a, b) => a[1].lastUsed - b[1].lastUsed);
        for (const [id, e] of entries) {
            if (this.residentBytes <= this.maxBytes)
                break;
            if (e.lastUsed === this.currentFrame)
                continue;
            this.cache.delete(id);
            this.residentBytes -= e.bytes;
            this.evictionCount++;
        }
    }
    /** Force-evict every tile. Used by `StreamingTileset.dispose()`. */
    clear() {
        this.cache.clear();
        this.residentBytes = 0;
        this.inFlight.clear();
        this.queue.length = 0;
        this.failures.clear();
    }
    /** Diagnostics snapshot — used by bench harness and STREAMING.md. */
    stats() {
        return {
            residentTiles: this.cache.size,
            residentBytes: this.residentBytes,
            inFlight: this.inFlight.size,
            evictions: this.evictionCount,
            cacheHits: this.hitCount,
            cacheMisses: this.missCount,
        };
    }
}
//# sourceMappingURL=tile_streamer.js.map