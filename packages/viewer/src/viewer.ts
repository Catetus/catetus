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
import {
  parseManifest,
  type ChunkDescriptor,
  type Manifest,
} from './manifest.js';
import {
  fetchChunkBytes,
  resolveChunkUri,
  validateChecksum,
} from './loader.js';
import { bboxCenter, orbitFrames, orbitPose } from './camera.js';
import { StatsOverlay } from './stats.js';
import type { Renderer } from './renderer/base.js';
import { isWebGPUAvailable, WebGPURenderer } from './renderer/webgpu.js';
import { WebGL2Renderer } from './renderer/webgl2.js';
import {
  StreamingTileset,
  type FrameReport,
  type StreamingTilesetOptions,
  type TileNode,
} from './streaming/index.js';
import { manifestFromGlb } from './streaming/glb.js';
import type {
  RendererKind,
  ViewerErrorCode,
  ViewerEvent,
  ViewerOptions,
} from './types.js';

/** Internal emitter map keyed by event `type`. */
type EmitterMap = {
  loadStart: Extract<ViewerEvent, { type: 'loadStart' }>;
  manifestLoaded: Extract<ViewerEvent, { type: 'manifestLoaded' }>;
  chunkLoaded: Extract<ViewerEvent, { type: 'chunkLoaded' }>;
  firstRender: Extract<ViewerEvent, { type: 'firstRender' }>;
  frameRendered: Extract<ViewerEvent, { type: 'frameRendered' }>;
  qualityChanged: Extract<ViewerEvent, { type: 'qualityChanged' }>;
  complete: Extract<ViewerEvent, { type: 'complete' }>;
  warning: Extract<ViewerEvent, { type: 'warning' }>;
  error: Extract<ViewerEvent, { type: 'error' }>;
};

/** Subscribable event type name. */
export type ViewerEventName = keyof EmitterMap;

/**
 * Main viewer class. Construct, subscribe, then call {@link load}.
 */
export class SplatForgeViewer {
  private readonly emitter = new Emitter<EmitterMap>();
  private readonly opts: Required<
    Omit<ViewerOptions, 'cameraPath' | 'budget' | 'renderer' | 'cameraBbox' | 'useComputeDecode'>
  > & {
    cameraPath: NonNullable<ViewerOptions['cameraPath']>;
    budget: NonNullable<ViewerOptions['budget']>;
    renderer: RendererKind;
    cameraBbox?: ViewerOptions['cameraBbox'];
    useComputeDecode: boolean;
  };
  private renderer?: Renderer;
  private stats?: StatsOverlay;
  private firstRenderFired = false;
  private disposed = false;
  private autoRotateRaf: number | null = null;
  private cachedManifest?: Manifest;

  constructor(options: ViewerOptions) {
    if (!options.canvas) throw new Error('ViewerOptions.canvas is required');
    if (!options.src) throw new Error('ViewerOptions.src is required');
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
      useComputeDecode: options.useComputeDecode ?? false,
    };
  }

  /**
   * Resolve which bbox to use for camera-pose math. Honors `cameraBbox` when
   * supplied (visual-regression cases that need a stable frame across presets);
   * otherwise falls back to the loaded scene's manifest bbox.
   */
  private cameraBboxFor(manifest: Manifest): Manifest['bbox'] {
    return this.opts.cameraBbox ?? manifest.bbox;
  }

  /**
   * Subscribe to a viewer event.
   * @returns unsubscribe function.
   */
  on<K extends ViewerEventName>(
    event: K,
    fn: (payload: EmitterMap[K]) => void,
  ): () => void {
    return this.emitter.on(event, fn);
  }

  /** Unsubscribe a previously registered listener. */
  off<K extends ViewerEventName>(
    event: K,
    fn: (payload: EmitterMap[K]) => void,
  ): void {
    this.emitter.off(event, fn);
  }

  /** Subscribe for exactly one delivery. */
  once<K extends ViewerEventName>(
    event: K,
    fn: (payload: EmitterMap[K]) => void,
  ): () => void {
    return this.emitter.once(event, fn);
  }

  /**
   * Begin loading the manifest and stream chunks in priority order.
   *
   * Resolves when `complete` fires. Rejects with an Error whose message is the
   * stable error code (e.g. `chunk_not_found`) when any fatal error occurs.
   */
  async load(): Promise<void> {
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
    } catch (err) {
      const { code, message } = normalizeError(err);
      this.emitter.emit('error', { type: 'error', code, message });
      throw new Error(code);
    }
  }

  /** Release GPU resources and drop all listeners. Idempotent. */
  dispose(): void {
    if (this.disposed) return;
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

  private streaming?: StreamingTileset;
  /** Tile id → list of (descriptor, bytes) ready to upload. Cached so we
   *  don't re-parse the GLB every frame. */
  private readonly tileChunks = new Map<string, Array<{ descriptor: ChunkDescriptor; bytes: Uint8Array }>>();
  /** Set of tile ids currently uploaded to the renderer. */
  private readonly uploadedTiles = new Set<string>();

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
  async loadTileset(url: string, opts: StreamingTilesetOptions = {}): Promise<void> {
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
      const manifestLike: Manifest = {
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
    } catch (err) {
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
  private async runStreamingCameraPath(
    streaming: StreamingTileset,
    manifest: Manifest,
  ): Promise<void> {
    const renderer = this.renderer;
    if (!renderer) return;
    const path = this.opts.cameraPath;
    if (path === 'static') return;

    const canvas = this.opts.canvas;
    const aspect = canvas.width > 0 ? canvas.width / Math.max(canvas.height, 1) : 16 / 9;
    const camBbox = this.cameraBboxFor(manifest);
    let yaws: number[];
    if (typeof path === 'object' && path && path.type === 'custom') {
      const center = [
        (camBbox.min[0] + camBbox.max[0]) * 0.5,
        (camBbox.min[1] + camBbox.max[1]) * 0.5,
        (camBbox.min[2] + camBbox.max[2]) * 0.5,
      ];
      yaws = path.positions.map((p) => Math.atan2(p[0] - center[0], p[2] - center[2]));
    } else {
      const count = path === 'orbit-8' ? 8 : 8;
      yaws = orbitFrames(count);
    }

    for (let i = 0; i < yaws.length; i++) {
      const pose = orbitPose(camBbox, yaws[i]!, aspect);
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
      await new Promise<void>((resolve) => {
        const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => number })
          .requestAnimationFrame;
        if (typeof raf === 'function') raf(() => resolve());
        else resolve();
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
  private ensureTilesUploaded(
    renderer: Renderer,
    streaming: StreamingTileset,
    render: TileNode[],
  ): void {
    for (const tile of render) {
      if (this.uploadedTiles.has(tile.id)) continue;
      const payload = streaming.streamer.get(tile);
      if (!payload) continue;
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
  private uploadTileToRenderer(
    renderer: Renderer,
    tile: TileNode,
    json: string,
    bin: Uint8Array,
  ): void {
    // GLBs from the Cesium tileset emitter carry an inline BIN chunk —
    // their JSON declares `buffers[0].uri = null`. The viewer's generic
    // `parseManifest` rejects that path (it expects an external buffer
    // URI). The streaming layer instead uses `manifestFromGlb` to
    // synthesize a single chunk that points at the GLB's BIN slice.
    const { manifest, bin: binSlice } = manifestFromGlb({ json, bin });
    const tileChunks: Array<{ descriptor: ChunkDescriptor; bytes: Uint8Array }> = [];
    for (const chunk of manifest.chunks) {
      // For a one-chunk GLB the slice is the entire BIN payload.
      const slice = binSlice.subarray(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);
      renderer.uploadChunk(chunk, slice);
      tileChunks.push({ descriptor: chunk, bytes: slice });
    }
    this.tileChunks.set(tile.id, tileChunks);
  }

  /** Diagnostic accessor for tests + bench. */
  get streamingTileset(): StreamingTileset | undefined {
    return this.streaming;
  }

  /** Single-frame render helper for the bench harness — drives the streaming
   *  loop once at `camera` and returns the report. */
  async streamingRenderFrame(camera: import('./camera.js').CameraPose): Promise<FrameReport | undefined> {
    if (!this.streaming || !this.renderer) return undefined;
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
  private startAutoRotate(): void {
    if (this.autoRotateRaf !== null) return;
    const raf = (
      globalThis as { requestAnimationFrame?: (cb: (t: number) => void) => number }
    ).requestAnimationFrame;
    if (typeof raf !== 'function') return;
    const speedRad = (this.opts.autoRotateSpeed * Math.PI) / 180;
    const t0 = performance.now();
    const step = (now: number): void => {
      if (this.disposed || !this.renderer || !this.cachedManifest) return;
      const canvas = this.opts.canvas;
      // Bail cheaply when the canvas has been detached or hidden.
      if (!canvas.isConnected) {
        this.autoRotateRaf = null;
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
      // Fire-and-forget — we'd rather drop a frame than back up the queue.
      void this.renderer.renderFrame(pose);
      this.autoRotateRaf = raf(step);
    };
    this.autoRotateRaf = raf(step);
  }

  private stopAutoRotate(): void {
    if (this.autoRotateRaf === null) return;
    const caf = (
      globalThis as { cancelAnimationFrame?: (id: number) => void }
    ).cancelAnimationFrame;
    if (typeof caf === 'function') caf(this.autoRotateRaf);
    this.autoRotateRaf = null;
  }

  /* --------------------------------------------------------------- */
  /* Internals                                                       */
  /* --------------------------------------------------------------- */

  private async pickRenderer(): Promise<Renderer> {
    const kind = this.opts.renderer;
    if (kind === 'webgpu') {
      if (!(await isWebGPUAvailable())) {
        throw new Error('renderer_unavailable: WebGPU requested but missing');
      }
      return new WebGPURenderer({ useComputeDecode: this.opts.useComputeDecode });
    }
    if (kind === 'webgl2') {
      return new WebGL2Renderer();
    }
    // auto
    if (await isWebGPUAvailable()) {
      return new WebGPURenderer({ useComputeDecode: this.opts.useComputeDecode });
    }
    return new WebGL2Renderer();
  }

  private async fetchManifest(): Promise<Manifest> {
    let res: Response;
    try {
      res = await fetch(this.opts.src);
    } catch (err) {
      throw new Error(`manifest_not_found: ${(err as Error).message}`);
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

  private async streamChunks(manifest: Manifest): Promise<void> {
    const renderer = this.renderer;
    if (!renderer) throw new Error('renderer_init_failed: pre-stream check');

    const aspect = this.opts.canvas.width > 0
      ? this.opts.canvas.width / Math.max(this.opts.canvas.height, 1)
      : 16 / 9;
    const camera = orbitPose(this.cameraBboxFor(manifest), 0, aspect);

    for (let i = 0; i < manifest.chunks.length; i++) {
      const chunk = manifest.chunks[i]!;
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
  private async runCameraPath(manifest: Manifest): Promise<void> {
    const renderer = this.renderer;
    if (!renderer) return;
    const path = this.opts.cameraPath;
    if (path === 'static') return; // initial firstRender already covers this.

    const aspect = this.opts.canvas.width > 0
      ? this.opts.canvas.width / Math.max(this.opts.canvas.height, 1)
      : 16 / 9;

    const camBbox = this.cameraBboxFor(manifest);
    let yaws: number[];
    if (typeof path === 'object' && path && path.type === 'custom') {
      // Custom positions are absolute eye points; synthesize yaws from them by
      // projecting onto the XZ plane relative to bbox center.
      const center = [
        (camBbox.min[0] + camBbox.max[0]) * 0.5,
        (camBbox.min[1] + camBbox.max[1]) * 0.5,
        (camBbox.min[2] + camBbox.max[2]) * 0.5,
      ];
      yaws = path.positions.map((p) =>
        Math.atan2(p[0] - center[0], p[2] - center[2]),
      );
    } else {
      const count = path === 'orbit-8' ? 8 : 8;
      yaws = orbitFrames(count);
    }

    for (let i = 0; i < yaws.length; i++) {
      const pose = orbitPose(camBbox, yaws[i]!, aspect);
      await renderer.renderFrame(pose);
      this.emitter.emit('frameRendered', {
        type: 'frameRendered',
        index: i,
        total: yaws.length,
      });
      // Yield to the host so observers can sample the canvas before we draw
      // the next pose. Use rAF when available (browser), otherwise a microtask.
      await new Promise<void>((resolve) => {
        const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => number })
          .requestAnimationFrame;
        if (typeof raf === 'function') {
          raf(() => resolve());
        } else {
          resolve();
        }
      });
    }
  }

  private async streamOneChunk(
    renderer: Renderer,
    chunk: ChunkDescriptor,
    index: number,
  ): Promise<void> {
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
function normalizeError(err: unknown): { code: ViewerErrorCode; message: string } {
  const message = err instanceof Error ? err.message : String(err);
  const head = message.split(':', 1)[0] ?? '';
  const known: readonly ViewerErrorCode[] = [
    'manifest_not_found',
    'manifest_invalid',
    'chunk_not_found',
    'checksum_failed',
    'renderer_unavailable',
    'renderer_init_failed',
  ];
  for (const code of known) {
    if (head === code) return { code, message };
  }
  return { code: 'unknown', message };
}
