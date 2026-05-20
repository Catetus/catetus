/**
 * LODGE hierarchical-LOD runtime — Phase A.2.
 *
 * See `docs/perf/lodge-lod-spec.md` for the full design memo. This module
 * wires the offline chunker output (the `catetus lodge build`
 * directory) into the viewer's `ComputeDecodePipeline`.
 *
 * Public surface:
 *   - {@link parseLodgeManifest}   — JSON → typed manifest.
 *   - {@link selectLodLevel}       — TS-side LOD picker (CPU heuristic).
 *   - {@link chunksByCameraProximity}, {@link sceneBboxCenter},
 *     {@link sceneBboxRadius} — helpers.
 *   - {@link LodgeChunkLoader}     — fetch + decode + GPU upload + LRU.
 *   - {@link decodePlyToSoa}       — PLY → canonical SoA bytes (also
 *     usable standalone).
 */
export { parseLodgeManifest, selectLodLevel, chunksByCameraProximity, sceneBboxCenter, sceneBboxRadius, LODGE_MANIFEST_VERSION, } from './manifest.js';
export { LodgeChunkLoader } from './chunk-loader.js';
export { decodePlyToSoa } from './ply.js';
// Phase A.3 — per-frame LOD selection + boundary-blend math.
export { LOD_SLOT_NEAR, LOD_SLOT_FAR, LOD_MAX_LEVELS, boundaryBlendT, selectChunkActivation, selectAllActivations, pickNearFarChunks, } from './lod-math.js';
export { LodgeLODPipeline, CHUNK_RECORD_BYTES, LEVEL_RECORD_BYTES, ACTIVATION_BYTES, LOD_UNIFORMS_BYTES, } from './lod-pipeline.js';
//# sourceMappingURL=index.js.map