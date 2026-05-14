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
export { orbitFrames, orbitPose, bboxCenter, bboxRadius } from './camera.js';
export type { CameraPose } from './camera.js';
export { parseManifest } from './manifest.js';
export type { Manifest, ChunkDescriptor, Bbox, Vec3 } from './manifest.js';
