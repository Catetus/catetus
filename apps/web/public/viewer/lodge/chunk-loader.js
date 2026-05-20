/**
 * Per-chunk LODGE loader — fetch chunk PLYs on demand, decode to canonical
 * SoA bytes, and stream the result to a {@link ComputeDecodePipeline}
 * via `uploadChunk()`.
 *
 * Phase A.2 responsibilities (in this file):
 *
 *   - HTTP / file:// fetch of chunk PLYs by relative path against a
 *     configurable `baseUrl` (so the same loader works for `cdn://`,
 *     `https://`, `file://`, and the test harness's blob URLs).
 *   - PLY → SoA decode via {@link decodePlyToSoa}.
 *   - GPU upload via the existing `ComputeDecodePipeline.uploadChunk`.
 *   - In-memory LRU eviction tied to a configurable byte budget
 *     (default 8 GiB working set) so the loader can keep loaded chunks
 *     bounded under a typical 16-GiB iGPU shared-memory ceiling.
 *
 * Phase A.3 will add a WGSL kernel that picks per-Gaussian LOD bands per
 * camera distance and removes the need for an explicit "active level"
 * choice on the CPU. The TS-side {@link LodgeChunkLoader.loadLevel}
 * remains useful as a coarse warm-up + as the fallback when the compute
 * pass is gated off.
 *
 * Note: this loader does NOT itself drive the renderer's `encode()`
 * loop. It only mutates the pipeline's decoded-splat buffer. Callers
 * should call `loadLevel()` once when the camera enters a new LOD band
 * and then run `encode()` per frame as usual.
 */
import { resolveChunkUri } from '../loader.js';
import { decodePlyToSoa } from './ply.js';
import { chunksByCameraProximity, parseLodgeManifest, selectLodLevel, } from './manifest.js';
const DEFAULT_BUDGET = 8 * 1024 * 1024 * 1024; // 8 GiB
const DEFAULT_FETCHER = async (url) => {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`lodge_chunk_fetch_failed: HTTP ${res.status} for ${url}`);
    }
    const ab = await res.arrayBuffer();
    return new Uint8Array(ab);
};
/** Loader state machine. One per scene; multiple loaders can coexist
 *  inside a single page (e.g. multi-scene comparison harness). */
export class LodgeChunkLoader {
    manifest;
    baseUrl;
    pipeline;
    fetcher;
    memoryBudgetBytes;
    /** Insertion-order map → built-in LRU semantics. */
    lru = new Map();
    memoryUsedBytes = 0;
    /** Last level loaded by `loadLevel`, or -1 before the first call. */
    currentLevel = -1;
    constructor(manifest, opts) {
        this.manifest = manifest;
        // Normalise baseUrl with a trailing slash so `new URL(rel, base)`
        // treats `base` as a directory.
        this.baseUrl = opts.baseUrl.endsWith('/') ? opts.baseUrl : opts.baseUrl + '/';
        this.pipeline = opts.pipeline;
        this.fetcher = opts.fetcher ?? DEFAULT_FETCHER;
        this.memoryBudgetBytes = opts.memoryBudgetBytes ?? DEFAULT_BUDGET;
    }
    /** Load a {@link LodgeManifest} from `<baseUrl>manifest.json`. */
    static async load(opts) {
        const url = resolveChunkUri(opts.baseUrl.endsWith('/') ? opts.baseUrl : opts.baseUrl + '/', 'manifest.json');
        const fetcher = opts.fetcher ?? DEFAULT_FETCHER;
        const bytes = await fetcher(url);
        const text = new TextDecoder('utf-8').decode(bytes);
        return new LodgeChunkLoader(parseLodgeManifest(text), opts);
    }
    /** Memory used by cached decoded bytes (NOT GPU memory — host side). */
    get memoryUsage() {
        return this.memoryUsedBytes;
    }
    /** Active level (last loaded), or -1 before any `loadLevel` call. */
    get activeLevel() {
        return this.currentLevel;
    }
    /** Cached entries (read-only snapshot, useful for tests). */
    cacheKeys() {
        return Array.from(this.lru.keys());
    }
    /**
     * Pick a LOD level for a camera position (Phase A.2 TS-side heuristic).
     * Equivalent to `selectLodLevel(loader.manifest, cameraPos)`.
     */
    pickLevel(cameraPos) {
        return selectLodLevel(this.manifest, cameraPos);
    }
    /**
     * Ensure every chunk of `level` is decoded into the GPU pipeline. Chunks
     * already resident in the LRU are re-uploaded without re-decoding; new
     * chunks are fetched + decoded + uploaded; evicted chunks are dropped
     * from the cache to honour the byte budget.
     *
     * The pipeline's decoded-splat buffer is **reset to zero before each
     * level switch** (we re-upload from scratch). This is the simplest
     * correct semantics until Phase A.3 introduces per-chunk active sets.
     *
     * @param level Index into `manifest.levels`.
     * @param cameraPos Optional — when supplied, chunks are uploaded in
     *   ascending distance-to-camera order (so the camera-near chunks land
     *   in the GPU buffer first).
     */
    async loadLevel(level, cameraPos) {
        const t0 = nowMs();
        const lvl = this.manifest.levels[level];
        if (!lvl) {
            throw new Error(`lodge_level_invalid: ${level} (have ${this.manifest.levels.length})`);
        }
        // Order chunks by camera proximity if requested.
        const chunks = cameraPos
            ? chunksByCameraProximity(this.manifest, level, cameraPos)
            : lvl.chunks;
        // Reset the GPU pipeline by allocating a fresh splat-offset cursor.
        // The pipeline's existing API doesn't expose a `reset()`; instead we
        // call `resetPipeline` which clones the strategy used by callers
        // doing a full re-upload between renders. The fallback is to do
        // nothing here — uploadChunk will still error on capacity overflow.
        resetPipeline(this.pipeline);
        let uploaded = 0;
        let cached = 0;
        for (const c of chunks) {
            const key = cacheKey(level, c.index);
            const entry = this.lru.get(key);
            let bytes;
            let splatCount;
            if (entry) {
                // Touch — re-insert at the tail.
                this.lru.delete(key);
                this.lru.set(key, entry);
                bytes = entry.bytes;
                splatCount = entry.splatCount;
                cached++;
            }
            else {
                const decoded = await this.fetchAndDecode(c);
                bytes = decoded.bytes;
                splatCount = decoded.splatCount;
                const fresh = {
                    key,
                    bytes,
                    splatCount,
                    byteLength: bytes.byteLength,
                };
                this.lru.set(key, fresh);
                this.memoryUsedBytes += fresh.byteLength;
                this.evictToBudget(key);
                uploaded++;
            }
            // Build a ChunkDescriptor for the pipeline that points at the
            // freshly-decoded SoA buffer. The layout is the standard one
            // emitted by `decodePlyToSoa` (positions | rotations | scales |
            // opacities | colorDC, all f32).
            const desc = buildPipelineDescriptor(c, splatCount, bytes.byteLength);
            this.pipeline.uploadChunk(desc, bytes);
            // WebGPU storage-buffer bind offsets must be 256-byte aligned.
            // Each decoded splat is 64 bytes (BYTES_PER_DECODED_SPLAT), so the
            // pipeline's `decodedSplats` cursor needs to land on a multiple of
            // 4 splats before the NEXT chunk binds its destination view. The
            // single-chunk real-scene bench is fine (decodedSplats starts at 0
            // and binds once), but multi-chunk LODGE streaming hits arbitrary
            // splat counts. Round up here so the next uploadChunk's
            // `decodedSplats * 64` offset is 256-aligned.
            alignDecodedSplats(this.pipeline);
        }
        this.currentLevel = level;
        return {
            level,
            uploadedChunks: uploaded,
            cachedChunks: cached,
            splats: this.pipeline.splatCount,
            memoryUsedBytes: this.memoryUsedBytes,
            elapsedMs: nowMs() - t0,
        };
    }
    /**
     * Fetch a single chunk PLY, decode it, and return the canonical SoA
     * bytes. Exposed for tests + integration scripts (the loader uses it
     * internally inside `loadLevel`).
     */
    async fetchAndDecode(chunk) {
        const url = resolveChunkUri(this.baseUrl, chunk.path);
        const plyBytes = await this.fetcher(url);
        const decoded = decodePlyToSoa(plyBytes);
        if (decoded.splatCount !== chunk.splatCount) {
            throw new Error(`lodge_chunk_count_mismatch: ${chunk.path} expected ${chunk.splatCount}, got ${decoded.splatCount}`);
        }
        return { bytes: decoded.bytes, splatCount: decoded.splatCount };
    }
    /**
     * Evict LRU entries until total cached bytes are <= budget, keeping
     * `preserveKey` resident.
     */
    evictToBudget(preserveKey) {
        if (this.memoryUsedBytes <= this.memoryBudgetBytes)
            return;
        // Map insertion order is iteration order; the OLDEST entry is first.
        for (const [key, entry] of this.lru) {
            if (this.memoryUsedBytes <= this.memoryBudgetBytes)
                break;
            if (key === preserveKey)
                continue;
            this.lru.delete(key);
            this.memoryUsedBytes -= entry.byteLength;
        }
    }
}
function cacheKey(level, chunkIndex) {
    return `${level}/${chunkIndex}`;
}
function nowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}
/**
 * Reset the pipeline's decoded-splat cursor between full-level
 * re-uploads. The cursor is a private field on `ComputeDecodePipeline`;
 * we mutate it through a structurally-typed cast so we don't import the
 * concrete class (keeps the loader pure-TS friendly to tests).
 *
 * If the pipeline doesn't expose a settable `decodedSplats`, the loader
 * will eventually trip its capacity check — surface that to the caller
 * as a build-time configuration error rather than silently overrun.
 */
function resetPipeline(p) {
    const target = p;
    if ('decodedSplats' in target) {
        target.decodedSplats = 0;
    }
    if (Array.isArray(target.chunks)) {
        target.chunks.length = 0;
    }
}
/**
 * Round the pipeline's `decodedSplats` cursor up to a multiple of 4 so the
 * next chunk's GPU destination view starts on a 256-byte boundary. Each
 * decoded splat is 64 bytes; WebGPU requires storage-buffer bind offsets
 * to be 256-byte-aligned, so the cursor must be 4-aligned in splat units.
 *
 * The padding splats remain at zero in the destination buffer (the GPU
 * `cs_decode` only writes the survivors), so they're invisible to the
 * downstream cull/project/sort/gather flow which gates on `alpha >= tau`
 * (the zero-alpha padding splats fail this predicate and never reach
 * rasterization).
 */
function alignDecodedSplats(p) {
    const target = p;
    if (typeof target.decodedSplats !== 'number')
        return;
    const cur = target.decodedSplats;
    const aligned = (cur + 3) & ~3;
    if (aligned !== cur) {
        target.decodedSplats = aligned;
    }
}
function buildPipelineDescriptor(chunk, splatCount, byteLength) {
    // SoA byte layout matches decodePlyToSoa.
    const posBytes = splatCount * 12;
    const rotBytes = splatCount * 16;
    const scaleBytes = splatCount * 12;
    const opBytes = splatCount * 4;
    const dcBytes = splatCount * 12;
    return {
        uri: `lodge:${chunk.path}`,
        byteOffset: 0,
        byteLength,
        splatCount,
        bbox: { min: chunk.bbox[0], max: chunk.bbox[1] },
        lod: 0, // legacy field; the LODGE-specific level lives in the
        // LodgeLevelDescriptor (not the per-chunk descriptor) and is
        // tracked by `loadLevel` separately.
        checksum: chunk.blake3,
        loadPriority: chunk.index,
        attributeLayout: {
            positions: { byteOffset: 0, byteLength: posBytes, componentType: 5126 },
            rotations: { byteOffset: posBytes, byteLength: rotBytes, componentType: 5126 },
            scales: {
                byteOffset: posBytes + rotBytes,
                byteLength: scaleBytes,
                componentType: 5126,
            },
            opacities: {
                byteOffset: posBytes + rotBytes + scaleBytes,
                byteLength: opBytes,
                componentType: 5126,
            },
            colorDC: {
                byteOffset: posBytes + rotBytes + scaleBytes + opBytes,
                byteLength: dcBytes,
                componentType: 5126,
            },
        },
    };
}
//# sourceMappingURL=chunk-loader.js.map