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
import { orbitFrames, orbitPose } from './camera.js';
import { StatsOverlay } from './stats.js';
import type { Renderer } from './renderer/base.js';
import { isWebGPUAvailable, WebGPURenderer } from './renderer/webgpu.js';
import { WebGL2Renderer } from './renderer/webgl2.js';
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
    Omit<ViewerOptions, 'cameraPath' | 'budget' | 'renderer'>
  > & {
    cameraPath: NonNullable<ViewerOptions['cameraPath']>;
    budget: NonNullable<ViewerOptions['budget']>;
    renderer: RendererKind;
  };
  private renderer?: Renderer;
  private stats?: StatsOverlay;
  private firstRenderFired = false;
  private disposed = false;

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
    };
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
      this.emitter.emit('manifestLoaded', {
        type: 'manifestLoaded',
        chunkCount: manifest.chunks.length,
      });

      await this.streamChunks(manifest);
      await this.runCameraPath(manifest);
      this.emitter.emit('complete', { type: 'complete' });
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
    this.renderer?.destroy();
    this.renderer = undefined;
    this.stats?.destroy();
    this.stats = undefined;
    this.emitter.removeAll();
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
      return new WebGPURenderer();
    }
    if (kind === 'webgl2') {
      return new WebGL2Renderer();
    }
    // auto
    if (await isWebGPUAvailable()) return new WebGPURenderer();
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
    const camera = orbitPose(manifest.bbox, 0, aspect);

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

    let yaws: number[];
    if (typeof path === 'object' && path && path.type === 'custom') {
      // Custom positions are absolute eye points; synthesize yaws from them by
      // projecting onto the XZ plane relative to bbox center.
      const center = [
        (manifest.bbox.min[0] + manifest.bbox.max[0]) * 0.5,
        (manifest.bbox.min[1] + manifest.bbox.max[1]) * 0.5,
        (manifest.bbox.min[2] + manifest.bbox.max[2]) * 0.5,
      ];
      yaws = path.positions.map((p) =>
        Math.atan2(p[0] - center[0], p[2] - center[2]),
      );
    } else {
      const count = path === 'orbit-8' ? 8 : 8;
      yaws = orbitFrames(count);
    }

    for (let i = 0; i < yaws.length; i++) {
      const pose = orbitPose(manifest.bbox, yaws[i]!, aspect);
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
