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
