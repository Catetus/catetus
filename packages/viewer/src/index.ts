/**
 * Public entry point for `@splatforge/viewer`.
 *
 * @packageDocumentation
 */
export { SplatForgeViewer, type ViewerEventName } from './viewer.js';
export type {
  ViewerOptions,
  ViewerEvent,
  ViewerErrorCode,
  RendererKind,
  BudgetPreset,
  CameraPath,
} from './types.js';
export { ComputeDecodePipeline } from './webgpu/index.js';
export { RadixSort, createRadixSortPipelines } from './webgpu/radix_sort.js';
export { orbitFrames, orbitPose, bboxCenter, bboxRadius } from './camera.js';
export type { CameraPose } from './camera.js';
export { parseManifest } from './manifest.js';
export type { Manifest, ChunkDescriptor, Bbox, Vec3 } from './manifest.js';
export {
  StreamingTileset,
  TileStreamer,
  loadTilesetJson,
  parseTileset,
  decodeGlb,
  manifestFromGlb,
  extractFrustum,
  selectVisibleTiles,
  screenSpaceError,
} from './streaming/index.js';
export type {
  StreamingTilesetOptions,
  FrameReport,
  TileNode,
  Tileset,
  TileSelection,
  SelectorOptions,
  LoadedTilePayload,
  StreamerStats,
  Frustum,
} from './streaming/index.js';
export {
  fetchProgressive,
  parsePlyHeader,
  MGS2_MAGIC,
  MGS2_VERSION,
  MGS2_PREFIX_LEN,
} from './progressive/fetcher.js';
export type {
  ProgressiveEvent,
  ProgressiveHeaderEvent,
  ProgressiveChunkEvent,
  ProgressiveDoneEvent,
  PlyFieldOffsets,
  FetchProgressiveOptions,
} from './progressive/fetcher.js';
export {
  ProgressiveUploader,
  buildSoaChunk,
  SOA_BYTES_PER_SPLAT,
} from './progressive/uploader.js';
export type { ProgressiveUploaderInit } from './progressive/uploader.js';
export {
  parseLodgeManifest,
  selectLodLevel,
  chunksByCameraProximity,
  sceneBboxCenter,
  sceneBboxRadius,
  LODGE_MANIFEST_VERSION,
  LodgeChunkLoader,
  decodePlyToSoa,
  // Phase A.3 — runtime LOD selector + boundary blend.
  LOD_SLOT_NEAR,
  LOD_SLOT_FAR,
  LOD_MAX_LEVELS,
  boundaryBlendT,
  selectChunkActivation,
  selectAllActivations,
  pickNearFarChunks,
  LodgeLODPipeline,
  CHUNK_RECORD_BYTES,
  LEVEL_RECORD_BYTES,
  ACTIVATION_BYTES,
  LOD_UNIFORMS_BYTES,
} from './lodge/index.js';
export type {
  LodgeManifest,
  LodgeLevelDescriptor,
  LodgeChunkDescriptor,
  LodgeAabb,
  LodgeChunkLoaderOptions,
  ChunkFetcher,
  UploadablePipeline,
  LoadLevelResult,
  DecodedPlyChunk,
  // Phase A.3.
  ChunkActivation,
  ChunkRecord,
  LodSelectInputs,
  LodgeLODPipelineOptions,
  FrameDecision,
  StreamFrameResult,
} from './lodge/index.js';

// WebXR bridge — see ./webxr/WebXRSplatViewer.ts.
export {
  WebXRSplatViewer,
  rowMajorToColMajor4,
  isXRSessionSupported,
  clampLodForXR,
  COMFORT,
  FrameBudgetTracker,
} from './webxr/index.js';
export type {
  WebXRSplatViewerDeps,
  XREyeRenderArgs,
  SessionStartInfo,
  LodgeLevelHandle,
  XRSessionModeLite,
  XRSessionLite,
  XRSystemLite,
  XRFrameLite,
  XRViewerPoseLite,
  XRViewLite,
  XRWebGLLayerLite,
  XRReferenceSpaceLite,
  XRReferenceSpaceTypeLite,
  XRRigidTransformLite,
  XRViewportLite,
  XREyeLite,
} from './webxr/index.js';
