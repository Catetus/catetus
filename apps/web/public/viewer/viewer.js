/**
 * `SplatForgeViewer` — the SDK entry point.
 *
 * Lifecycle:
 *   1. `new SplatForgeViewer(opts)` — picks a renderer kind, prepares emitter.
 *   2. `await viewer.load()` — fetches manifest, streams chunks, fires events.
 *   3. `viewer.dispose()` — releases GPU resources and detaches listeners.
 *
 * Errors are surfaced both by emitting an `error` event and by rejecting the
 * `load()` promise. Consumers may subscribe to `error` for non-fatal warnings
 * via the separate `warning` channel.
 */
import { Emitter } from './events.js';
import { parseManifest, } from './manifest.js';
import { fetchChunkBytes, resolveChunkUri, validateChecksum, } from './loader.js';
import { bboxCenter, orbitFrames, orbitPose } from './camera.js';
import { StatsOverlay } from './stats.js';
import { isWebGPUAvailable, WebGPURenderer } from './renderer/webgpu.js';
import { WebGL2Renderer } from './renderer/webgl2.js';
import { fetchProgressive } from './progressive/fetcher.js';
import { ProgressiveUploader } from './progressive/uploader.js';
import { StreamingTileset, } from './streaming/index.js';
import { manifestFromGlb } from './streaming/glb.js';
/**
 * Main viewer class. Construct, subscribe, then call {@link load}.
 */
export class SplatForgeViewer {
    emitter = new Emitter();
    opts;
    renderer;
    stats;
    firstRenderFired = false;
    disposed = false;
    autoRotateRaf = null;
    cachedManifest;
    constructor(options) {
        if (!options.canvas)
            throw new Error('ViewerOptions.canvas is required');
        if (!options.src)
            throw new Error('ViewerOptions.src is required');
        this.opts = {
            canvas: options.canvas,
            src: options.src,
            budget: options.budget ?? 'web-desktop',
            renderer: options.renderer ?? 'auto',
            cameraPath: options.cameraPath ?? 'orbit-8',
            deterministic: options.deterministic ?? false,
            seed: options.seed ?? 0xc0ffee,
            stats: options.stats ?? false,
            autoRotate: options.autoRotate ?? false,
            autoRotateSpeed: options.autoRotateSpeed ?? 10,
            autoRotateFraming: options.autoRotateFraming ?? 1.0,
            cameraBbox: options.cameraBbox,
            // Stage 7 (sf-154) fast path: WebGPU compute-decode + per-page sort +
            // multi-draw. Default ON so every page (hero, /optimize TryIt, explore,
            // compare) gets the renderer that handles >100M splats without freezing
            // the browser. The legacy CPU project/sort path stays reachable by
            // passing `useComputeDecode: false` explicitly (deterministic tests,
            // bench harness). When the renderer ends up on WebGL2 (no WebGPU in
            // the browser), this flag is ignored — WebGL2Renderer has no compute
            // pipeline and uses the CPU path regardless.
            useComputeDecode: options.useComputeDecode ?? true,
        };
    }
    /**
     * Resolve which bbox to use for camera-pose math. Honors `cameraBbox` when
     * supplied (visual-regression cases that need a stable frame across presets);
     * otherwise falls back to the loaded scene's manifest bbox.
     */
    cameraBboxFor(manifest) {
        return this.opts.cameraBbox ?? manifest.bbox;
    }
    /**
     * Subscribe to a viewer event.
     * @returns unsubscribe function.
     */
    on(event, fn) {
        return this.emitter.on(event, fn);
    }
    /** Unsubscribe a previously registered listener. */
    off(event, fn) {
        this.emitter.off(event, fn);
    }
    /** Subscribe for exactly one delivery. */
    once(event, fn) {
        return this.emitter.once(event, fn);
    }
    /**
     * Begin loading the manifest and stream chunks in priority order.
     *
     * Resolves when `complete` fires. Rejects with an Error whose message is the
     * stable error code (e.g. `chunk_not_found`) when any fatal error occurs.
     */
    async load() {
        this.emitter.emit('loadStart', { type: 'loadStart' });
        try {
            const renderer = await this.pickRenderer();
            this.renderer = renderer;
            await renderer.init({ canvas: this.opts.canvas });
            if (this.opts.stats) {
                const parent = this.opts.canvas.parentElement;
                if (parent) {
                    if (getComputedStyle(parent).position === 'static') {
                        parent.style.position = 'relative';
                    }
                    this.stats = new StatsOverlay({ anchor: parent });
                }
            }
            const manifest = await this.fetchManifest();
            this.cachedManifest = manifest;
            this.emitter.emit('manifestLoaded', {
                type: 'manifestLoaded',
                chunkCount: manifest.chunks.length,
            });
            await this.streamChunks(manifest);
            await this.runCameraPath(manifest);
            this.emitter.emit('complete', { type: 'complete' });
            if (this.opts.autoRotate && !this.opts.deterministic) {
                this.startAutoRotate();
            }
        }
        catch (err) {
            const { code, message } = normalizeError(err);
            this.emitter.emit('error', { type: 'error', code, message });
            throw new Error(code);
        }
    }
    /**
     * Stream a `.mgs2` progressive bitstream from `url` and render it
     * coarse-to-fine while bytes arrive.
     *
     * Each progressive batch becomes a synthetic SoA chunk (see
     * `progressive/uploader.ts`) and is fed to the renderer's `uploadChunk`.
     * The first batch lands well before the byte budget is exhausted, so a
     * watchable preview appears at ~5 % of the bitstream and refines
     * monotonically as additional bytes arrive.
     *
     * Camera path semantics differ from {@link load}: `loadProgressive`
     * renders a single frame after every batch arrives (so observers see the
     * scene grow), then optionally drives the configured `cameraPath` once
     * the full bitstream has been received.
     *
     * @param url URL of the `.mgs2` bitstream.
     * @param opts Optional batch-size override + AbortSignal forwarding to
     *   {@link fetchProgressive}.
     */
    async loadProgressive(url, opts = {}) {
        this.emitter.emit('loadStart', { type: 'loadStart' });
        try {
            const renderer = await this.pickRenderer();
            this.renderer = renderer;
            await renderer.init({ canvas: this.opts.canvas });
            if (this.opts.stats) {
                const parent = this.opts.canvas.parentElement;
                if (parent) {
                    if (getComputedStyle(parent).position === 'static') {
                        parent.style.position = 'relative';
                    }
                    this.stats = new StatsOverlay({ anchor: parent });
                }
            }
            let uploader;
            let totalBatches = 0;
            let manifest;
            const aspect = this.opts.canvas.width > 0
                ? this.opts.canvas.width / Math.max(this.opts.canvas.height, 1)
                : 16 / 9;
            for await (const ev of fetchProgressive(url, opts)) {
                if (ev.kind === 'header') {
                    // Synthesize a manifest-like record now — the bbox is unknown
                    // until the first batch lands; default to a unit cube and update
                    // as batches arrive.
                    manifest = {
                        splatCount: ev.nSplats,
                        bbox: { min: [-1, -1, -1], max: [1, 1, 1] },
                        chunks: [],
                        shDegree: 0,
                    };
                    this.cachedManifest = manifest;
                    this.emitter.emit('manifestLoaded', {
                        type: 'manifestLoaded',
                        chunkCount: 0, // unknown — batches are emitted as they arrive
                    });
                    uploader = new ProgressiveUploader({
                        renderer,
                        fieldOffsets: ev.fieldOffsets,
                        onBatchUploaded: (batchIndex, totalSplats) => {
                            this.emitter.emit('chunkLoaded', {
                                type: 'chunkLoaded',
                                chunkIndex: batchIndex,
                                byteLength: totalSplats, // proxy: cumulative splat count
                            });
                        },
                    });
                }
                else if (ev.kind === 'chunk') {
                    if (!uploader)
                        throw new Error('progressive_protocol: chunk before header');
                    uploader.addBatch(ev.bytes, ev.splatsAdded);
                    totalBatches += 1;
                    // Update the cached manifest's bbox to the cumulative union so
                    // any consumer using `cachedManifest.bbox` for framing sees the
                    // scene grow in screen-space rather than zooming on the origin.
                    if (manifest) {
                        manifest.bbox = uploader.currentBbox;
                    }
                    // Render a frame after every batch so the canvas reflects the
                    // growing scene.
                    const camera = orbitPose(this.cameraBboxFor(manifest), 0, aspect);
                    await renderer.renderFrame(camera);
                    if (!this.firstRenderFired) {
                        this.firstRenderFired = true;
                        this.emitter.emit('firstRender', { type: 'firstRender' });
                    }
                }
                // 'done' event needs no action.
            }
            if (!uploader)
                throw new Error('progressive_protocol: stream ended before header');
            // Run the configured camera path once the full bitstream is in.
            if (manifest)
                await this.runCameraPath(manifest);
            this.emitter.emit('complete', { type: 'complete' });
            void totalBatches;
            if (this.opts.autoRotate && !this.opts.deterministic) {
                this.startAutoRotate();
            }
        }
        catch (err) {
            const { code, message } = normalizeError(err);
            this.emitter.emit('error', { type: 'error', code, message });
            throw new Error(code);
        }
    }
    /** Release GPU resources and drop all listeners. Idempotent. */
    dispose() {
        if (this.disposed)
            return;
        this.disposed = true;
        this.stopAutoRotate();
        this.streaming?.dispose();
        this.streaming = undefined;
        this.renderer?.destroy();
        this.renderer = undefined;
        this.stats?.destroy();
        this.stats = undefined;
        this.emitter.removeAll();
    }
    /* --------------------------------------------------------------- */
    /* Streaming tileset (queue #51)                                   */
    /* --------------------------------------------------------------- */
    streaming;
    /** Tile id → list of (descriptor, bytes) ready to upload. Cached so we
     *  don't re-parse the GLB every frame. */
    tileChunks = new Map();
    /** Set of tile ids currently uploaded to the renderer. */
    uploadedTiles = new Set();
    /**
     * Load a Cesium 3D Tiles 1.1 tileset and render it through the streaming
     * adapter. Each frame the adapter:
     *
     *   1. Selects visible tiles at the appropriate LOD (Cesium SSE).
     *   2. Requests missing tiles via the LRU-bounded streamer.
     *   3. Uploads any newly-arrived tiles into the renderer.
     *   4. Renders the resident set with the existing WebGPU pipeline.
     *
     * Returns once the `cameraPath` has finished. Subsequent frames (e.g.
     * `autoRotate`) continue to drive the streamer on every `renderFrame`.
     *
     * WebGPU-only. On WebGL2 the viewer falls back to rendering the tileset's
     * root tile at a single LOD — the streaming-path determinism budget
     * doesn't admit the per-frame upload churn on WebGL2.
     */
    async loadTileset(url, opts = {}) {
        this.emitter.emit('loadStart', { type: 'loadStart' });
        try {
            const renderer = await this.pickRenderer();
            this.renderer = renderer;
            await renderer.init({ canvas: this.opts.canvas });
            if (this.opts.stats) {
                const parent = this.opts.canvas.parentElement;
                if (parent) {
                    if (getComputedStyle(parent).position === 'static') {
                        parent.style.position = 'relative';
                    }
                    this.stats = new StatsOverlay({ anchor: parent });
                }
            }
            const streaming = await StreamingTileset.create(url, opts);
            this.streaming = streaming;
            // Use the root tile's AABB as a synthetic manifest bbox for camera math.
            const rootAabb = streaming.tileset.root.aabb;
            const manifestLike = {
                splatCount: 0,
                bbox: { min: rootAabb.min, max: rootAabb.max },
                chunks: [],
                shDegree: 0,
            };
            this.cachedManifest = manifestLike;
            this.emitter.emit('manifestLoaded', {
                type: 'manifestLoaded',
                chunkCount: streaming.tileset.tiles.length,
            });
            // First, wait for the root tile to land so we always have something
            // resident, then drive the deterministic camera path.
            await streaming.drainPending();
            // WebGL2 fallback: only render the root tile.
            if (renderer.kind !== 'webgpu') {
                const root = streaming.tileset.root;
                const payload = streaming.streamer.get(root);
                if (payload) {
                    this.uploadTileToRenderer(renderer, root, payload.json, payload.bin);
                }
            }
            await this.runStreamingCameraPath(streaming, manifestLike);
            this.emitter.emit('complete', { type: 'complete' });
        }
        catch (err) {
            const { code, message } = normalizeError(err);
            this.emitter.emit('error', { type: 'error', code, message });
            throw new Error(code);
        }
    }
    /**
     * Streaming variant of {@link runCameraPath}. Per frame:
     *   1. Ask the StreamingTileset which tiles to render + fetch.
     *   2. Drain pending fetches (deterministic mode only).
     *   3. Make sure every render-tile is uploaded to the renderer.
     *   4. Render.
     */
    async runStreamingCameraPath(streaming, manifest) {
        const renderer = this.renderer;
        if (!renderer)
            return;
        const path = this.opts.cameraPath;
        if (path === 'static')
            return;
        const canvas = this.opts.canvas;
        const aspect = canvas.width > 0 ? canvas.width / Math.max(canvas.height, 1) : 16 / 9;
        const camBbox = this.cameraBboxFor(manifest);
        let yaws;
        if (typeof path === 'object' && path && path.type === 'custom') {
            const center = [
                (camBbox.min[0] + camBbox.max[0]) * 0.5,
                (camBbox.min[1] + camBbox.max[1]) * 0.5,
                (camBbox.min[2] + camBbox.max[2]) * 0.5,
            ];
            yaws = path.positions.map((p) => Math.atan2(p[0] - center[0], p[2] - center[2]));
        }
        else {
            const count = path === 'orbit-8' ? 8 : 8;
            yaws = orbitFrames(count);
        }
        for (let i = 0; i < yaws.length; i++) {
            const pose = orbitPose(camBbox, yaws[i], aspect);
            const report = streaming.frame(pose, canvas.width, canvas.height);
            // Deterministic mode: wait for the fetches we just queued so this
            // frame's render-set matches across runs.
            if (this.opts.deterministic) {
                await streaming.drainPending();
            }
            // Re-run selection with the now-resident set so we use any newly-
            // arrived tiles (otherwise the first frame is always low-LOD).
            const report2 = streaming.frame(pose, canvas.width, canvas.height);
            this.ensureTilesUploaded(renderer, streaming, report2.render);
            await renderer.renderFrame(pose);
            this.emitter.emit('frameRendered', {
                type: 'frameRendered',
                index: i,
                total: yaws.length,
            });
            // Unused-variable pacifier — the first report is reserved for tests
            // that want to inspect the as-yet-unfilled render set.
            void report;
            // Yield so the host can sample the canvas backing store.
            await new Promise((resolve) => {
                const raf = globalThis
                    .requestAnimationFrame;
                if (typeof raf === 'function')
                    raf(() => resolve());
                else
                    resolve();
            });
        }
    }
    /**
     * Make sure every tile in `render` has had its chunks uploaded to the
     * renderer. We never re-upload a tile — the underlying renderer is
     * append-only, and the LRU is at the *streamer* level (not the GPU). For
     * the v0.1 streaming-adapter the streamer's bytes cache absorbs the
     * eviction load; the GPU keeps decoded splats for tiles that were
     * uploaded earlier. This is a deliberate trade: simpler GPU state at the
     * cost of per-tile capacity headroom.
     *
     * The first tile encountered triggers an instance of the renderer's
     * compute-decode pipeline (when `useComputeDecode` is on), so we honor
     * the caller's `useComputeDecode` setting transparently.
     */
    ensureTilesUploaded(renderer, streaming, render) {
        for (const tile of render) {
            if (this.uploadedTiles.has(tile.id))
                continue;
            const payload = streaming.streamer.get(tile);
            if (!payload)
                continue;
            this.uploadTileToRenderer(renderer, tile, payload.json, payload.bin);
            this.uploadedTiles.add(tile.id);
        }
    }
    /**
     * Parse a single tile's GLB and feed each of its chunks into the
     * renderer's `uploadChunk` path. The renderer treats each chunk like any
     * other streaming chunk — the compute-decode pipeline appends to its
     * canonical splat buffer, the rasterizer picks the result up on the next
     * draw.
     */
    uploadTileToRenderer(renderer, tile, json, bin) {
        // GLBs from the Cesium tileset emitter carry an inline BIN chunk —
        // their JSON declares `buffers[0].uri = null`. The viewer's generic
        // `parseManifest` rejects that path (it expects an external buffer
        // URI). The streaming layer instead uses `manifestFromGlb` to
        // synthesize a single chunk that points at the GLB's BIN slice.
        const { manifest, bin: binSlice } = manifestFromGlb({ json, bin });
        const tileChunks = [];
        for (const chunk of manifest.chunks) {
            // For a one-chunk GLB the slice is the entire BIN payload.
            const slice = binSlice.subarray(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
            renderer.uploadChunk(chunk, slice);
            tileChunks.push({ descriptor: chunk, bytes: slice });
        }
        this.tileChunks.set(tile.id, tileChunks);
    }
    /** Diagnostic accessor for tests + bench. */
    get streamingTileset() {
        return this.streaming;
    }
    /** Single-frame render helper for the bench harness — drives the streaming
     *  loop once at `camera` and returns the report. */
    async streamingRenderFrame(camera) {
        if (!this.streaming || !this.renderer)
            return undefined;
        const canvas = this.opts.canvas;
        this.streaming.frame(camera, canvas.width, canvas.height);
        if (this.opts.deterministic) {
            await this.streaming.drainPending();
        }
        const report = this.streaming.frame(camera, canvas.width, canvas.height);
        this.ensureTilesUploaded(this.renderer, this.streaming, report.render);
        await this.renderer.renderFrame(camera);
        return report;
    }
    /**
     * Drive a continuous yaw orbit via `requestAnimationFrame`. Idempotent —
     * calling twice keeps the existing loop. Stops on `dispose()` or when the
     * canvas leaves the document.
     */
    startAutoRotate() {
        if (this.autoRotateRaf !== null)
            return;
        const raf = globalThis.requestAnimationFrame;
        if (typeof raf !== 'function')
            return;
        const speedRad = (this.opts.autoRotateSpeed * Math.PI) / 180;
        const t0 = performance.now();
        // Single in-flight frame guard. The pre-Stage-7 path used "fire-and-
        // forget" — `void renderer.renderFrame(pose)` and immediately scheduled
        // the next rAF. That works for the CPU rasterizer (each renderFrame
        // blocks the main thread synchronously, so the rAF cadence self-limits)
        // but the WebGPU compute path returns to the event loop after
        // `queue.submit()` without waiting for presentation. Successive rAF
        // ticks then pile submits into the GPU's swap chain faster than the
        // compositor can present them. On Apple Silicon the swap chain
        // backpressures after ~5 seconds and the page collapses from 60 fps to
        // <1 fps for the rest of its life (this was the "machine freeze"
        // symptom reproduced on the marketing hero with the bonsai scene).
        //
        // Fix: gate the rAF on the previous renderFrame's Promise settling.
        // We still drop a frame if the GPU is behind — there's no queueing —
        // but we never accumulate work, so the queue can drain between vsyncs.
        let inFlight = null;
        const step = (now) => {
            if (this.disposed || !this.renderer || !this.cachedManifest) {
                this.autoRotateRaf = null;
                return;
            }
            const canvas = this.opts.canvas;
            // Bail cheaply when the canvas has been detached or hidden.
            if (!canvas.isConnected) {
                this.autoRotateRaf = null;
                return;
            }
            // If the previous frame's submit hasn't returned yet, skip this
            // vsync. raf re-fires anyway and we'll catch up on the next tick.
            if (inFlight) {
                this.autoRotateRaf = raf(step);
                return;
            }
            const aspect = canvas.width > 0 ? canvas.width / Math.max(canvas.height, 1) : 16 / 9;
            const yaw = ((now - t0) / 1000) * speedRad;
            const camBbox = this.cameraBboxFor(this.cachedManifest);
            const pose = orbitPose(camBbox, yaw, aspect);
            // Apply the marketing framing multiplier by lerping the eye toward the
            // target. We do this here rather than inside `orbitPose` so the
            // deterministic camera paths used by SPEC-0009 keep their canonical
            // distance — only the live auto-rotate is reframed.
            const framing = this.opts.autoRotateFraming;
            if (framing !== 1.0) {
                const center = bboxCenter(camBbox);
                pose.position = [
                    center[0] + (pose.position[0] - center[0]) * framing,
                    center[1] + (pose.position[1] - center[1]) * framing,
                    center[2] + (pose.position[2] - center[2]) * framing,
                ];
            }
            inFlight = this.renderer.renderFrame(pose).finally(() => {
                inFlight = null;
            });
            this.autoRotateRaf = raf(step);
        };
        this.autoRotateRaf = raf(step);
    }
    stopAutoRotate() {
        if (this.autoRotateRaf === null)
            return;
        const caf = globalThis.cancelAnimationFrame;
        if (typeof caf === 'function')
            caf(this.autoRotateRaf);
        this.autoRotateRaf = null;
    }
    /* --------------------------------------------------------------- */
    /* Internals                                                       */
    /* --------------------------------------------------------------- */
    async pickRenderer() {
        const kind = this.opts.renderer;
        if (kind === 'webgl2') {
            return new WebGL2Renderer();
        }
        // 'webgpu' and 'auto' both prefer WebGPU and silently fall back to
        // WebGL2 when WebGPU is missing. Callers should not need to probe
        // `navigator.gpu` themselves — pre-Stage-7 the explicit `webgpu`
        // mode threw, which forced every embed site to write its own
        // capability gate. Now the SDK is the single source of truth.
        if (await isWebGPUAvailable()) {
            return new WebGPURenderer({ useComputeDecode: this.opts.useComputeDecode });
        }
        this.emitter.emit('warning', {
            type: 'warning',
            code: 'webgpu_unavailable',
            message: 'WebGPU not available; falling back to WebGL2 renderer',
        });
        return new WebGL2Renderer();
    }
    async fetchManifest() {
        let res;
        try {
            res = await fetch(this.opts.src);
        }
        catch (err) {
            throw new Error(`manifest_not_found: ${err.message}`);
        }
        if (res.status === 404) {
            throw new Error(`manifest_not_found: ${this.opts.src}`);
        }
        if (!res.ok) {
            throw new Error(`manifest_not_found: HTTP ${res.status}`);
        }
        const text = await res.text();
        return parseManifest(text);
    }
    async streamChunks(manifest) {
        const renderer = this.renderer;
        if (!renderer)
            throw new Error('renderer_init_failed: pre-stream check');
        const aspect = this.opts.canvas.width > 0
            ? this.opts.canvas.width / Math.max(this.opts.canvas.height, 1)
            : 16 / 9;
        const camera = orbitPose(this.cameraBboxFor(manifest), 0, aspect);
        for (let i = 0; i < manifest.chunks.length; i++) {
            const chunk = manifest.chunks[i];
            await this.streamOneChunk(renderer, chunk, i);
            if (!this.firstRenderFired) {
                await renderer.renderFrame(camera);
                this.firstRenderFired = true;
                this.emitter.emit('firstRender', { type: 'firstRender' });
            }
        }
    }
    /**
     * Execute the configured camera path: render each pose, emit
     * `frameRendered` between renders, and yield to the next animation frame so
     * the canvas backing store reflects every frame for any external observer.
     */
    async runCameraPath(manifest) {
        const renderer = this.renderer;
        if (!renderer)
            return;
        const path = this.opts.cameraPath;
        if (path === 'static')
            return; // initial firstRender already covers this.
        const aspect = this.opts.canvas.width > 0
            ? this.opts.canvas.width / Math.max(this.opts.canvas.height, 1)
            : 16 / 9;
        const camBbox = this.cameraBboxFor(manifest);
        let yaws;
        if (typeof path === 'object' && path && path.type === 'custom') {
            // Custom positions are absolute eye points; synthesize yaws from them by
            // projecting onto the XZ plane relative to bbox center.
            const center = [
                (camBbox.min[0] + camBbox.max[0]) * 0.5,
                (camBbox.min[1] + camBbox.max[1]) * 0.5,
                (camBbox.min[2] + camBbox.max[2]) * 0.5,
            ];
            yaws = path.positions.map((p) => Math.atan2(p[0] - center[0], p[2] - center[2]));
        }
        else {
            const count = path === 'orbit-8' ? 8 : 8;
            yaws = orbitFrames(count);
        }
        for (let i = 0; i < yaws.length; i++) {
            const pose = orbitPose(camBbox, yaws[i], aspect);
            await renderer.renderFrame(pose);
            this.emitter.emit('frameRendered', {
                type: 'frameRendered',
                index: i,
                total: yaws.length,
            });
            // Yield to the host so observers can sample the canvas before we draw
            // the next pose. Use rAF when available (browser), otherwise a microtask.
            await new Promise((resolve) => {
                const raf = globalThis
                    .requestAnimationFrame;
                if (typeof raf === 'function') {
                    raf(() => resolve());
                }
                else {
                    resolve();
                }
            });
        }
    }
    async streamOneChunk(renderer, chunk, index) {
        const url = resolveChunkUri(this.opts.src, chunk.uri);
        const bytes = await fetchChunkBytes(url, chunk);
        if (chunk.checksum) {
            const r = await validateChecksum(bytes, chunk.checksum);
            if (!r.ok && r.reason === 'mismatch') {
                throw new Error(`checksum_failed: chunk ${index} ${chunk.uri}`);
            }
            if (!r.ok && r.reason === 'unsupported') {
                this.emitter.emit('warning', {
                    type: 'warning',
                    code: 'checksum_unsupported',
                    message: `Skipping checksum for chunk ${index} (SubtleCrypto missing or non-SHA-256)`,
                });
            }
        }
        renderer.uploadChunk(chunk, bytes);
        this.emitter.emit('chunkLoaded', {
            type: 'chunkLoaded',
            chunkIndex: index,
            byteLength: bytes.byteLength,
        });
    }
}
/** Map an arbitrary thrown value to `{ code, message }`. */
function normalizeError(err) {
    const message = err instanceof Error ? err.message : String(err);
    const head = message.split(':', 1)[0] ?? '';
    const known = [
        'manifest_not_found',
        'manifest_invalid',
        'chunk_not_found',
        'checksum_failed',
        'renderer_unavailable',
        'renderer_init_failed',
    ];
    for (const code of known) {
        if (head === code)
            return { code, message };
    }
    return { code: 'unknown', message };
}
//# sourceMappingURL=viewer.js.map