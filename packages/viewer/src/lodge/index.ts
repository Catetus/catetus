/**
 * LODGE hierarchical-LOD runtime — Phase A.2.
 *
 * See `docs/perf/lodge-lod-spec.md` for the full design memo. This module
 * wires the offline chunker output (the `splatforge lodge build`
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
export {
  parseLodgeManifest,
  selectLodLevel,
  chunksByCameraProximity,
  sceneBboxCenter,
  sceneBboxRadius,
  LODGE_MANIFEST_VERSION,
} from './manifest.js';
export type {
  LodgeManifest,
  LodgeLevelDescriptor,
  LodgeChunkDescriptor,
  LodgeAabb,
} from './manifest.js';

export { LodgeChunkLoader } from './chunk-loader.js';
export type {
  LodgeChunkLoaderOptions,
  ChunkFetcher,
  UploadablePipeline,
  LoadLevelResult,
} from './chunk-loader.js';

export { decodePlyToSoa } from './ply.js';
export type { DecodedPlyChunk } from './ply.js';
