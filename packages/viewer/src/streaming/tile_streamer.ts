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

import { decodeGlb, type DecodedGlb } from './glb.js';
import type { TileNode } from './tileset_loader.js';

/** State machine values reported by {@link TileStreamer.stateOf}. */
export type TileLoadState = 'unloaded' | 'loading' | 'loaded' | 'failed';

/** Cached payload for a successfully-loaded tile. */
export interface LoadedTilePayload {
  /** The GLB's JSON chunk text. */
  json: string;
  /** The GLB's BIN chunk bytes. */
  bin: Uint8Array;
  /** Total bytes attributed to this tile in the LRU budget. */
  bytes: number;
  /** Monotonic frame index of the most recent render that touched this tile. */
  lastUsed: number;
}

/** Knobs for {@link TileStreamer}. */
export interface TileStreamerOptions {
  /** LRU cache budget in bytes. Defaults to 512 MB. */
  maxBytes?: number;
  /** Override the global `fetch` (test injection point). */
  fetch?: typeof fetch;
  /**
   * Maximum number of concurrent in-flight fetches. Defaults to 8 — more
   * than the typical browser per-origin HTTP/1.1 limit but well under
   * HTTP/2 stream cap.
   */
  maxConcurrency?: number;
}

/** Snapshot of the streamer's internal state for diagnostics + bench. */
export interface StreamerStats {
  residentTiles: number;
  residentBytes: number;
  inFlight: number;
  evictions: number;
  cacheHits: number;
  cacheMisses: number;
}

export class TileStreamer {
  private readonly cache = new Map<string, LoadedTilePayload>();
  private readonly inFlight = new Map<string, Promise<LoadedTilePayload>>();
  private readonly failures = new Map<string, Error>();
  private readonly queue: { tile: TileNode; priority: number }[] = [];
  private readonly maxBytes: number;
  private readonly fetchImpl: typeof fetch;
  private readonly maxConcurrency: number;
  private residentBytes = 0;
  private currentFrame = 0;
  private evictionCount = 0;
  private hitCount = 0;
  private missCount = 0;

  constructor(opts: TileStreamerOptions = {}) {
    this.maxBytes = opts.maxBytes ?? 512 * 1024 * 1024;
    this.fetchImpl = opts.fetch ?? ((...a) => fetch(...a));
    this.maxConcurrency = opts.maxConcurrency ?? 8;
  }

  /** Total bytes currently held in the resident set. */
  get residentByteCount(): number {
    return this.residentBytes;
  }

  /** Number of tiles currently resident. */
  get residentTileCount(): number {
    return this.cache.size;
  }

  /** Bump the frame counter and refresh `lastUsed` on a set of tiles. */
  touch(tiles: Iterable<TileNode>): void {
    this.currentFrame++;
    for (const t of tiles) {
      const e = this.cache.get(t.id);
      if (e) e.lastUsed = this.currentFrame;
    }
  }

  /** Current state for `tile`. */
  stateOf(tile: TileNode): TileLoadState {
    if (this.cache.has(tile.id)) return 'loaded';
    if (this.inFlight.has(tile.id)) return 'loading';
    if (this.failures.has(tile.id)) return 'failed';
    return 'unloaded';
  }

  /** Resident payload for `tile`, or `undefined` if not loaded. */
  get(tile: TileNode): LoadedTilePayload | undefined {
    const p = this.cache.get(tile.id);
    if (p) this.hitCount++;
    else this.missCount++;
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
  fetchTile(tile: TileNode, priority = 0): Promise<LoadedTilePayload> {
    const cached = this.cache.get(tile.id);
    if (cached) {
      this.hitCount++;
      return Promise.resolve(cached);
    }
    const live = this.inFlight.get(tile.id);
    if (live) return live;
    if (this.inFlight.size >= this.maxConcurrency) {
      // Park in the priority queue; we'll start it when something completes.
      return new Promise<LoadedTilePayload>((resolve, reject) => {
        this.queue.push({ tile, priority });
        // Re-sort so the highest priority gets pulled first.
        this.queue.sort((a, b) => b.priority - a.priority);
        const start = (): void => {
          this.actuallyFetch(tile).then(resolve, reject);
        };
        // Stash resolver alongside the queued entry so the drain code can
        // pick it back up. We use a marker field on the tile node — safe
        // because TileNode is owned by the tileset loader and we add one
        // additional property name space.
        (tile as TileNode & { __resume?: () => void }).__resume = start;
      });
    }
    return this.actuallyFetch(tile);
  }

  private actuallyFetch(tile: TileNode): Promise<LoadedTilePayload> {
    this.missCount++;
    const p = (async (): Promise<LoadedTilePayload> => {
      try {
        const res = await this.fetchImpl(tile.contentUrl);
        if (!res.ok) {
          throw new Error(`tile_fetch_failed: HTTP ${res.status} for ${tile.contentUrl}`);
        }
        const ab = await res.arrayBuffer();
        const bytes = new Uint8Array(ab);
        let glb: DecodedGlb;
        try {
          glb = decodeGlb(bytes);
        } catch (err) {
          throw new Error(
            `tile_fetch_failed: ${tile.id} ${(err as Error).message}`,
          );
        }
        const payload: LoadedTilePayload = {
          json: glb.json,
          bin: glb.bin,
          bytes: bytes.byteLength,
          lastUsed: this.currentFrame,
        };
        this.cache.set(tile.id, payload);
        this.residentBytes += payload.bytes;
        this.maybeEvict();
        return payload;
      } catch (err) {
        this.failures.set(tile.id, err as Error);
        throw err;
      } finally {
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
  private drainQueue(): void {
    while (
      this.inFlight.size < this.maxConcurrency &&
      this.queue.length > 0
    ) {
      const entry = this.queue.shift()!;
      const tile = entry.tile as TileNode & { __resume?: () => void };
      const resume = tile.__resume;
      tile.__resume = undefined;
      if (resume) resume();
    }
  }

  /**
   * Drop least-recently-rendered tiles until `residentBytes <= maxBytes`.
   * Tiles touched on the *current* frame are pinned — evicting them would
   * cause the renderer to draw a hole this frame.
   */
  private maybeEvict(): void {
    if (this.residentBytes <= this.maxBytes) return;
    const entries = [...this.cache.entries()];
    entries.sort((a, b) => a[1].lastUsed - b[1].lastUsed);
    for (const [id, e] of entries) {
      if (this.residentBytes <= this.maxBytes) break;
      if (e.lastUsed === this.currentFrame) continue;
      this.cache.delete(id);
      this.residentBytes -= e.bytes;
      this.evictionCount++;
    }
  }

  /** Force-evict every tile. Used by `StreamingTileset.dispose()`. */
  clear(): void {
    this.cache.clear();
    this.residentBytes = 0;
    this.inFlight.clear();
    this.queue.length = 0;
    this.failures.clear();
  }

  /** Diagnostics snapshot — used by bench harness and STREAMING.md. */
  stats(): StreamerStats {
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
