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

import type { CameraPose } from '../camera.js';
import { buildViewProj } from '../renderer/math.js';
import { extractFrustum, type Frustum } from './frustum.js';
import {
  loadTilesetJson,
  type Tileset,
  type TileNode,
} from './tileset_loader.js';
import {
  selectVisibleTiles,
  type SelectorOptions,
  type TileSelection,
} from './lod_selector.js';
import {
  TileStreamer,
  type LoadedTilePayload,
  type StreamerStats,
  type TileStreamerOptions,
} from './tile_streamer.js';

export type {
  TileNode,
  Tileset,
  TileSelection,
  SelectorOptions,
  LoadedTilePayload,
  StreamerStats,
  Frustum,
};
export { extractFrustum } from './frustum.js';
export { selectVisibleTiles, screenSpaceError } from './lod_selector.js';
// `screenSpaceError` re-export is convenience for the bench / docs.
export { TileStreamer } from './tile_streamer.js';
export { loadTilesetJson, parseTileset } from './tileset_loader.js';
export { decodeGlb, manifestFromGlb } from './glb.js';

/** Knobs for {@link StreamingTileset.create}. */
export interface StreamingTilesetOptions extends TileStreamerOptions {
  /** Cesium SSE cutoff. Defaults to 16. */
  maximumScreenSpaceError?: number;
  /**
   * When > 0, the tileset preloads tiles likely to enter the frustum next
   * frame by extruding the camera position along its velocity vector.
   * Disabled when the camera is static. (Research queue #63.)
   */
  prefetchLookahead?: number;
}

/** Per-frame report returned by {@link StreamingTileset.frame}. */
export interface FrameReport {
  /** Tiles to draw this frame, in coarse-to-fine order. */
  render: TileNode[];
  /** Tiles whose fetches were kicked off this frame. */
  fetched: TileNode[];
  /** Streamer LRU snapshot. */
  stats: StreamerStats;
}

/**
 * Streaming-tile runtime. Glues the loader, frustum, selector, and streamer
 * into the per-frame loop that {@link SplatForgeViewer} drives.
 */
export class StreamingTileset {
  readonly tileset: Tileset;
  readonly streamer: TileStreamer;
  private readonly opts: Required<
    Pick<StreamingTilesetOptions, 'maximumScreenSpaceError' | 'prefetchLookahead'>
  >;
  private lastEye: [number, number, number] | undefined;

  private constructor(tileset: Tileset, opts: StreamingTilesetOptions) {
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
  static async create(
    url: string,
    opts: StreamingTilesetOptions = {},
  ): Promise<StreamingTileset> {
    const tileset = await loadTilesetJson(url);
    const ts = new StreamingTileset(tileset, opts);
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
  frame(
    camera: CameraPose,
    viewportWidth: number,
    viewportHeight: number,
  ): FrameReport {
    const aspect = viewportWidth / Math.max(viewportHeight, 1);
    const { viewProj } = buildViewProj(camera, aspect);
    const frustum = extractFrustum(viewProj);

    // Build the "resident" set by querying the streamer for each tile.
    const resident = new Set<string>();
    for (const t of this.tileset.tiles) {
      if (this.streamer.stateOf(t) === 'loaded') resident.add(t.id);
    }

    const sel = selectVisibleTiles(this.tileset.root, {
      eye: camera.position as [number, number, number],
      fovY: camera.fovY,
      viewportHeight,
      maximumScreenSpaceError: this.opts.maximumScreenSpaceError,
      frustum,
      resident,
    } satisfies SelectorOptions);

    // Touch resident tiles so the LRU keeps them.
    this.streamer.touch(sel.render);

    // Kick off fetches. Priority = the SSE the selector wanted; if the
    // selector didn't surface one we fall back to inverse-depth so coarser
    // levels load first.
    const fetched: TileNode[] = [];
    for (let i = 0; i < sel.fetch.length; i++) {
      const tile = sel.fetch[i]!;
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
        const eyeAhead: [number, number, number] = [
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
  async drainPending(): Promise<void> {
    // Snapshot in-flight promises; new ones queued during the await won't
    // be re-awaited. Repeat until quiescent.
    // `Map` doesn't expose its values cheaply via Array.from in some old
    // environments; we copy through iteration to keep this dependency-free.
    while (true) {
      // Access the private inFlight map by stats.inFlight; we use the
      // public API in a loop and let microtask scheduling handle it.
      const before = this.streamer.stats().inFlight;
      if (before === 0) return;
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }

  /** Release every tile + tear down. Idempotent. */
  dispose(): void {
    this.streamer.clear();
  }
}
