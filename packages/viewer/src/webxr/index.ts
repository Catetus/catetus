/**
 * Public surface for the WebXR splat-viewer bridge.
 *
 * See {@link WebXRSplatViewer} for lifecycle docs.
 */
export {
  WebXRSplatViewer,
  rowMajorToColMajor4,
  isXRSessionSupported,
  type WebXRSplatViewerDeps,
  type XREyeRenderArgs,
  type SessionStartInfo,
} from './WebXRSplatViewer.js';
export {
  clampLodForXR,
  COMFORT,
  FrameBudgetTracker,
  type LodgeLevelHandle,
} from './comfort.js';
export type {
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
} from './types.js';
