/**
 * Public entry point for `@catetus/viewer`.
 *
 * @packageDocumentation
 */
export { CatetusViewer } from './viewer.js';
// Phase 2a — file-format loaders (PLY / .splat / SOG / SF .glb + sidecars).
// Returns the canonical SplatScene; pair with `splatSceneToSoaChunk` to feed
// the WebGPU compute-decode renderer.
export { loadFromFiles, loadFromUrl, } from './loader/dispatch.js';
export { loadPly, isLikely3DGSPly } from './loader/ply.js';
export { loadSplat } from './loader/splat.js';
export { loadSogContainer } from './loader/sog-container.js';
export { loadSfGlb } from './loader/sf-glb.js';
export { shRestCoefCount, computeBbox, normalizeQuatInto, sigmoid, clamp01, SH_C0, } from './loader/splat-scene.js';
export { splatSceneToSoaChunk } from './loader/to-soa.js';
export { ComputeDecodePipeline } from './webgpu/index.js';
export { RadixSort, createRadixSortPipelines } from './webgpu/radix_sort.js';
export { orbitFrames, orbitPose, bboxCenter, bboxRadius } from './camera.js';
export { parseManifest } from './manifest.js';
export { StreamingTileset, TileStreamer, loadTilesetJson, parseTileset, decodeGlb, manifestFromGlb, extractFrustum, selectVisibleTiles, screenSpaceError, } from './streaming/index.js';
export { fetchProgressive, parsePlyHeader, MGS2_MAGIC, MGS2_VERSION, MGS2_PREFIX_LEN, } from './progressive/fetcher.js';
export { ProgressiveUploader, buildSoaChunk, SOA_BYTES_PER_SPLAT, } from './progressive/uploader.js';
export { parseLodgeManifest, selectLodLevel, chunksByCameraProximity, sceneBboxCenter, sceneBboxRadius, LODGE_MANIFEST_VERSION, LodgeChunkLoader, decodePlyToSoa, 
// Phase A.3 — runtime LOD selector + boundary blend.
LOD_SLOT_NEAR, LOD_SLOT_FAR, LOD_MAX_LEVELS, boundaryBlendT, selectChunkActivation, selectAllActivations, pickNearFarChunks, LodgeLODPipeline, CHUNK_RECORD_BYTES, LEVEL_RECORD_BYTES, ACTIVATION_BYTES, LOD_UNIFORMS_BYTES, } from './lodge/index.js';
// WebXR bridge — see ./webxr/WebXRSplatViewer.ts.
export { WebXRSplatViewer, rowMajorToColMajor4, isXRSessionSupported, clampLodForXR, COMFORT, FrameBudgetTracker, } from './webxr/index.js';
//# sourceMappingURL=index.js.map