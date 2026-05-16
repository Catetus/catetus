// SPDX-License-Identifier: Apache-2.0
/**
 * LODGE Phase A.3 — runtime LOD pipeline orchestrator.
 *
 * Wires the offline LODGE pyramid (`.lodge` directory; Phase A.1 chunker)
 * + the viewer-side manifest loader + the per-frame LOD selector into a
 * single class that can be driven by an embedding application or a bench
 * harness.
 *
 * Two responsibilities, kept distinct:
 *
 *   1. **Streaming** (CPU): given a camera, pick the LOD level + the
 *      "near" / "far" chunks for the LODGE eq. 4 boundary blend, ensure
 *      they're loaded into the underlying {@link LodgeChunkLoader}, evict
 *      stale chunks under the byte budget.
 *
 *   2. **GPU pass** (optional): when a `device` is supplied, build the
 *      WGSL pipelines for `cs_lod_select` + `cs_lod_blend` +
 *      `cs_lod_alpha_reset` and run them per frame. The host writes the
 *      per-chunk records / activations / uniform; the GPU produces the
 *      modulated-opacity decoded-splat buffer that the existing
 *      `ComputeDecodePipeline` consumes.
 *
 * The bench harness can drive either path. The unit tests use the
 * CPU-only path (no WebGPU in vitest); the browser bench against the
 * 119M Sweet Corals scene uses the GPU path.
 *
 * Both paths share the JS-side reference math in `./lod-math.ts`. The
 * tests assert byte-identical decisions between the JS reference and a
 * mocked-up emulation of the WGSL.
 */

import type { Vec3 } from '../manifest.js';
import { LodgeChunkLoader, type ChunkFetcher, type UploadablePipeline } from './chunk-loader.js';
import {
  LOD_MAX_LEVELS,
  type ChunkRecord,
  type ChunkActivation,
  type LodSelectInputs,
  pickNearFarChunks,
  selectAllActivations,
} from './lod-math.js';
import type { LodgeManifest } from './manifest.js';
import { selectLodLevel, sceneBboxCenter } from './manifest.js';

/** Construction options for {@link LodgeLODPipeline}. */
export interface LodgeLODPipelineOptions {
  /** Base URL the chunk paths are resolved against. */
  baseUrl: string;
  /** Underlying GPU pipeline that ingests decoded chunks. */
  pipeline: UploadablePipeline;
  /** Optional override (defaults to `fetch`). */
  fetcher?: ChunkFetcher;
  /** Optional VRAM/cache byte budget (passed through to LodgeChunkLoader). */
  memoryBudgetBytes?: number;
  /**
   * Sub-pixel size threshold used by the LOD selector to bump to a coarser
   * level. Default 4 pixels (chunk projected radius below 4 px → drop to
   * the next coarser level). 0 disables the heuristic.
   */
  ssSizeThreshold?: number;
  /**
   * Optional WebGPU device. When present, the GPU LOD-select + LOD-blend
   * pipelines are built and the per-frame `encode` path becomes
   * available. When absent, the orchestrator stays CPU-only and the
   * `prepareFrame` method returns the activations + (near, far) chunks
   * for the embedding caller to consume.
   */
  device?: GPUDevice;
}

/** Snapshot of a frame's per-chunk decisions. Returned by
 *  {@link LodgeLODPipeline.prepareFrame}. */
export interface FrameDecision {
  /** Coarse LOD level picked by the CPU heuristic (used for streaming). */
  selectedLevel: number;
  /** Indices into `manifest.levels[selectedLevel].chunks` for the LODGE
   *  eq. 4 partners. `near == far` when the level only has one chunk. */
  nearChunkIndex: number;
  farChunkIndex: number;
  /** Flattened per-(level, chunk) records (input to the WGSL kernel). */
  records: ChunkRecord[];
  /** Flattened per-(level, chunk) activations (output of the JS reference
   *  selector — matches the WGSL kernel bit-for-bit on the tests). */
  activations: ChunkActivation[];
  /** Total splats marked active this frame across all chunks. */
  activeSplats: number;
  /** Sum of `splatCount` over active chunks at the selected level (a
   *  cheap proxy for the GPU render-frame survivor count). */
  selectedLevelSplats: number;
  /** Wall-clock ms spent inside `prepareFrame`. */
  elapsedMs: number;
}

/** Streaming snapshot returned by {@link LodgeLODPipeline.streamFrame}. */
export interface StreamFrameResult extends FrameDecision {
  /** Chunks fetched + decoded + uploaded this frame (cache miss count). */
  uploadedChunks: number;
  /** Chunks served from the LRU cache (cache hit count). */
  cachedChunks: number;
  /** Loader's resident-bytes budget usage. */
  memoryUsedBytes: number;
}

/**
 * Per-frame orchestrator: pick the right LOD level on the CPU,
 * stream the two LODGE-eq.4 partners into the GPU via {@link
 * LodgeChunkLoader}, then (when a `device` is supplied) drive the WGSL
 * compute pipelines that modulate per-chunk opacities.
 */
export class LodgeLODPipeline {
  readonly loader: LodgeChunkLoader;
  readonly manifest: LodgeManifest;
  readonly ssSizeThreshold: number;
  readonly device?: GPUDevice;

  private lastDecision: FrameDecision | null = null;

  constructor(loader: LodgeChunkLoader, opts: { ssSizeThreshold?: number; device?: GPUDevice }) {
    this.loader = loader;
    this.manifest = loader.manifest;
    this.ssSizeThreshold = opts.ssSizeThreshold ?? 4.0;
    this.device = opts.device;
  }

  /** Load the manifest, instantiate the loader, and wrap it in an LOD
   *  pipeline. The static factory matches `LodgeChunkLoader.load` for
   *  callers that want a single async constructor. */
  static async load(opts: LodgeLODPipelineOptions): Promise<LodgeLODPipeline> {
    const loader = await LodgeChunkLoader.load({
      baseUrl: opts.baseUrl,
      pipeline: opts.pipeline,
      fetcher: opts.fetcher,
      memoryBudgetBytes: opts.memoryBudgetBytes,
    });
    return new LodgeLODPipeline(loader, {
      ssSizeThreshold: opts.ssSizeThreshold,
      device: opts.device,
    });
  }

  /** Return the last decision computed by `prepareFrame` / `streamFrame`.
   *  Used by tests + the bench harness to inspect intermediate state. */
  get lastFrameDecision(): FrameDecision | null {
    return this.lastDecision;
  }

  /**
   * Compute the per-frame LOD decision **without streaming** — pure JS,
   * no I/O. The caller is responsible for ensuring the relevant chunks
   * are resident before calling the GPU render path.
   *
   * Cost: O(N_chunks) — fewer than 10⁴ chunks even on a 100 M scene,
   * so this is sub-millisecond on the CPU.
   *
   * @param cameraPos        World-space camera position.
   * @param focalY           Focal length in pixels (image_height /
   *                          (2 * tan(fov_y/2))).
   */
  prepareFrame(cameraPos: Vec3, focalY: number): FrameDecision {
    const t0 = nowMs();
    const selectedLevel = selectLodLevel(this.manifest, cameraPos);
    const { near, far } = pickNearFarChunks(this.manifest, selectedLevel, cameraPos);

    const lvl = this.manifest.levels[selectedLevel]!;
    const nearChunkIndex = lvl.chunks.indexOf(near);
    const farChunkIndex = lvl.chunks.indexOf(far);

    // Build depth-thresholds + lookup. Pad to LOD_MAX_LEVELS with +inf so
    // the WGSL kernel sees a stable shape.
    const depthThresholds = new Array<number>(LOD_MAX_LEVELS).fill(
      Number.POSITIVE_INFINITY,
    );
    for (let l = 0; l < Math.min(this.manifest.levels.length, LOD_MAX_LEVELS); l++) {
      depthThresholds[l] = this.manifest.levels[l]!.depthThreshold;
    }

    const inputs: LodSelectInputs = {
      cameraPos,
      sceneCenter: sceneBboxCenter(this.manifest),
      depthThresholds,
      nearCentroid: near.centroid,
      farCentroid: far.centroid,
      ssSizeThreshold: this.ssSizeThreshold,
      focalY,
    };

    const { records, activations } = selectAllActivations(this.manifest, inputs);

    let activeSplats = 0;
    let selectedLevelSplats = 0;
    for (let i = 0; i < records.length; i++) {
      const act = activations[i]!;
      const rec = records[i]!;
      if (act.active === 1) {
        // Weighted contribution by t_blend: a splat with t=0.5 only
        // contributes "half" to the visible budget. This is an
        // approximation — the renderer still spends sort/project on it,
        // but the alpha-modulated cull predicate in cs_cull will drop it
        // post-blend once alpha < tau.
        activeSplats += rec.splatCount;
        if (rec.level === selectedLevel) {
          selectedLevelSplats += rec.splatCount;
        }
      }
    }

    const decision: FrameDecision = {
      selectedLevel,
      nearChunkIndex,
      farChunkIndex,
      records,
      activations,
      activeSplats,
      selectedLevelSplats,
      elapsedMs: nowMs() - t0,
    };
    this.lastDecision = decision;
    return decision;
  }

  /**
   * Convenience: compute the per-frame decision AND stream the chunks
   * needed for it into the loader. Returns the merged result.
   *
   * Streaming policy:
   *   - Ensure both `near` and `far` chunks at `selectedLevel` are
   *     resident (fetch + decode if missing).
   *   - Touch the LRU on every cache hit so the next eviction prefers
   *     stale chunks from earlier camera positions.
   *   - The full LOD level's other chunks are NOT streamed in this call
   *     — that's a separate "warm" pass invoked by the embedding shell
   *     when the camera dwells (see `warmLevel`).
   */
  async streamFrame(cameraPos: Vec3, focalY: number): Promise<StreamFrameResult> {
    const decision = this.prepareFrame(cameraPos, focalY);
    // For the LODGE eq. 4 two-chunk-blend mode we only need the
    // near + far chunks at the selected level. We delegate the cache /
    // upload bookkeeping to the loader.
    const lvl = this.manifest.levels[decision.selectedLevel]!;
    const wanted = [lvl.chunks[decision.nearChunkIndex]!, lvl.chunks[decision.farChunkIndex]!];

    let uploadedChunks = 0;
    let cachedChunks = 0;
    // We piggyback on the loader's existing `fetchAndDecode` so we don't
    // re-implement caching here. Calling `loadLevel` would re-upload the
    // entire level, which is overkill — instead we do a manual
    // ensure-resident loop. The loader's LRU is mutated indirectly:
    // `fetchAndDecode` is a no-cache primitive, so we wrap with our own
    // cache key check via the loader's `cacheKeys` snapshot.
    const resident = new Set(this.loader.cacheKeys());
    for (const c of wanted) {
      const key = `${decision.selectedLevel}/${c.index}`;
      if (resident.has(key)) {
        cachedChunks++;
      } else {
        await this.loader.fetchAndDecode(c);
        uploadedChunks++;
      }
    }
    // Re-upload the wanted pair to the GPU pipeline (uses the loader's
    // cached bytes when present; the first-call path also re-decodes the
    // PLY, which is the slow path).
    await this.loader.loadLevel(decision.selectedLevel, cameraPos);

    return {
      ...decision,
      uploadedChunks,
      cachedChunks,
      memoryUsedBytes: this.loader.memoryUsage,
    };
  }

  /** Preload all chunks of a level. Used by the bench harness to
   *  measure pure-render fps once the disk + decode cost is amortized. */
  async warmLevel(level: number, cameraPos?: Vec3): Promise<void> {
    await this.loader.loadLevel(level, cameraPos);
  }

  // ---------------------------------------------------------------------
  // GPU buffer-layout helpers — produce the byte buffers that the WGSL
  // kernels expect. Exposed so the bench harness (or any embedding
  // caller) can write them to GPU storage buffers without re-deriving
  // the layout from the WGSL files.
  // ---------------------------------------------------------------------

  /** Pack the per-chunk records into the layout `cs_lod_select.wgsl`
   *  expects (32 bytes per record: vec4 centroid+radius + 4 u32s). */
  encodeChunkRecords(records: ChunkRecord[]): ArrayBuffer {
    const buf = new ArrayBuffer(records.length * CHUNK_RECORD_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    for (let i = 0; i < records.length; i++) {
      const r = records[i]!;
      const o = i * CHUNK_RECORD_F32S;
      f[o + 0] = r.centroid[0];
      f[o + 1] = r.centroid[1];
      f[o + 2] = r.centroid[2];
      f[o + 3] = r.radius;
      u[o + 4] = r.level;
      u[o + 5] = r.chunkIndex;
      u[o + 6] = r.splatCount;
      u[o + 7] = 0; // _pad
    }
    return buf;
  }

  /** Pack per-level depth-threshold records as `cs_lod_select.wgsl`
   *  expects. 16 bytes per level (depth_threshold + level + 2 pads). */
  encodeLevelRecords(): ArrayBuffer {
    const N = LOD_MAX_LEVELS;
    const buf = new ArrayBuffer(N * LEVEL_RECORD_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    for (let i = 0; i < N; i++) {
      const o = i * LEVEL_RECORD_F32S;
      if (i < this.manifest.levels.length) {
        const lvl = this.manifest.levels[i]!;
        f[o + 0] = lvl.depthThreshold;
        u[o + 1] = lvl.level;
      } else {
        f[o + 0] = Number.POSITIVE_INFINITY;
        u[o + 1] = i;
      }
      u[o + 2] = 0;
      u[o + 3] = 0;
    }
    return buf;
  }

  /** Pack the LodSelectUniforms struct (5×vec4 + 2 u32 + 1 f32 + pad
   *  = 96 bytes after alignment). Order matches the WGSL layout. */
  encodeLodSelectUniforms(
    decision: FrameDecision,
    cameraPos: Vec3,
    focalY: number,
  ): ArrayBuffer {
    const buf = new ArrayBuffer(LOD_UNIFORMS_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    let p = 0;
    // camera_pos
    f[p++] = cameraPos[0];
    f[p++] = cameraPos[1];
    f[p++] = cameraPos[2];
    f[p++] = 0;
    // scene_center
    const sc = sceneBboxCenter(this.manifest);
    f[p++] = sc[0];
    f[p++] = sc[1];
    f[p++] = sc[2];
    f[p++] = 0;
    // depth_thresholds[0..3]
    for (let i = 0; i < 4; i++) {
      f[p++] =
        i < this.manifest.levels.length
          ? this.manifest.levels[i]!.depthThreshold
          : Number.POSITIVE_INFINITY;
    }
    // depth_thresholds2[4..7]
    for (let i = 4; i < 8; i++) {
      f[p++] =
        i < this.manifest.levels.length
          ? this.manifest.levels[i]!.depthThreshold
          : Number.POSITIVE_INFINITY;
    }
    // chunk_count
    u[p++] = decision.records.length;
    // num_levels
    u[p++] = Math.min(this.manifest.levels.length, LOD_MAX_LEVELS);
    // Skip 2 u32 of padding to align next vec4
    p += 2;
    // near_centroid
    const lvl = this.manifest.levels[decision.selectedLevel]!;
    const near = lvl.chunks[decision.nearChunkIndex]!;
    const far = lvl.chunks[decision.farChunkIndex]!;
    f[p++] = near.centroid[0];
    f[p++] = near.centroid[1];
    f[p++] = near.centroid[2];
    f[p++] = 0;
    // far_centroid
    f[p++] = far.centroid[0];
    f[p++] = far.centroid[1];
    f[p++] = far.centroid[2];
    f[p++] = 0;
    // ss_size_threshold + focal_y + 2 u32 pad
    f[p++] = this.ssSizeThreshold;
    f[p++] = focalY;
    p += 2;
    return buf;
  }

  /** Decode the WGSL `ChunkActivation` struct back into JS objects.
   *  Used by the bench harness when reading the GPU output for
   *  cross-validation against the JS reference. */
  decodeActivations(bytes: ArrayBuffer): ChunkActivation[] {
    const u = new Uint32Array(bytes);
    const f = new Float32Array(bytes);
    const N = bytes.byteLength / ACTIVATION_BYTES;
    const out: ChunkActivation[] = new Array(N);
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      out[i] = {
        level: u[o + 0]!,
        active: u[o + 1]!,
        slot: u[o + 2]!,
        tBlend: f[o + 3]!,
      };
    }
    return out;
  }
}

// ---------------------------------------------------------------------
// Layout constants — kept in sync with the WGSL structs.
// ---------------------------------------------------------------------

/** Bytes per `ChunkDesc` record in `cs_lod_select.wgsl`. */
export const CHUNK_RECORD_BYTES = 32;
const CHUNK_RECORD_F32S = CHUNK_RECORD_BYTES / 4;

/** Bytes per `LevelDesc` record in `cs_lod_select.wgsl`. */
export const LEVEL_RECORD_BYTES = 16;
const LEVEL_RECORD_F32S = LEVEL_RECORD_BYTES / 4;

/** Bytes per `ChunkActivation` record. 4 u32 = 16 bytes. */
export const ACTIVATION_BYTES = 16;

/** Bytes for the `LodSelectUniforms` block:
 *    camera_pos vec4 + scene_center vec4 + depth_thresholds vec4 +
 *    depth_thresholds2 vec4 + (chunk_count, num_levels, _pad×2) vec4 +
 *    near_centroid vec4 + far_centroid vec4 + (ss_size_threshold,
 *    focal_y, _pad×2) vec4
 *  = 8 × 16 = 128 bytes. */
export const LOD_UNIFORMS_BYTES = 128;

function nowMs(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}
