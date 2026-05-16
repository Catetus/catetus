/**
 * Minimal WebXR Device API types.
 *
 * Mirrors the subset of the WebXR Device API
 * (https://www.w3.org/TR/webxr/) consumed by {@link WebXRSplatViewer} and
 * its unit tests. We declare these locally instead of pulling
 * `@types/webxr` so the viewer package keeps zero runtime deps and so the
 * Vitest mocks can plug in a structurally-typed stub.
 *
 * The shapes match the live browser API one-for-one — `XRView.transform`
 * has a row-major `matrix: Float32Array` of length 16, `XRRigidTransform`
 * exposes `position`/`orientation` DOMPointReadOnlys, etc.
 */

/**
 * `XRSessionMode` per spec.
 *
 * The viewer accepts `immersive-vr` (Quest 3, Vision Pro PSVR mode,
 * desktop Chrome/Edge with a headset) and `immersive-ar` (Quest 3
 * passthrough, Vision Pro AR).
 */
export type XRSessionModeLite = 'immersive-vr' | 'immersive-ar' | 'inline';

/** Reference-space type the renderer asks for. `local-floor` is the
 *  canonical comfort default (origin under the headset, +Y up,
 *  feet-on-floor). */
export type XRReferenceSpaceTypeLite =
  | 'viewer'
  | 'local'
  | 'local-floor'
  | 'bounded-floor'
  | 'unbounded';

/** Eye identifier — `'none'` is reported by inline (mono) sessions. */
export type XREyeLite = 'left' | 'right' | 'none';

/** Read-only DOMPoint — only `x/y/z/w` matter to us. */
export interface XRPointReadOnly {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly w: number;
}

/** Rigid transform exposed on `XRView.transform` and `XRViewerPose`. */
export interface XRRigidTransformLite {
  readonly position: XRPointReadOnly;
  readonly orientation: XRPointReadOnly;
  /** Row-major 4x4, as the browser ships. Length 16. */
  readonly matrix: Float32Array;
  /** Inverse of `matrix`. The browser pre-computes this; we use it as the
   *  view matrix when blitting to the per-eye framebuffer. */
  readonly inverse: XRRigidTransformLite;
}

/** One per-eye view inside an `XRViewerPose`. */
export interface XRViewLite {
  readonly eye: XREyeLite;
  /** Eye-to-head pose. `transform.inverse.matrix` is the view matrix. */
  readonly transform: XRRigidTransformLite;
  /** Projection matrix (row-major fp32, length 16). */
  readonly projectionMatrix: Float32Array;
}

/** Pose returned by `XRFrame.getViewerPose(referenceSpace)`. */
export interface XRViewerPoseLite {
  readonly transform: XRRigidTransformLite;
  readonly views: ReadonlyArray<XRViewLite>;
}

/** Reference space — opaque token to the JS side. */
export interface XRReferenceSpaceLite {
  /** Distinguishing tag for tests + logs. The browser doesn't expose this
   *  directly; we set it on the mock and on the value returned by
   *  `requestReferenceSpace`. */
  readonly type?: XRReferenceSpaceTypeLite;
}

/** WebGL layer sub-image for one eye. */
export interface XRViewportLite {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** The XR-owned WebGL framebuffer + per-eye viewport mapping. */
export interface XRWebGLLayerLite {
  readonly framebuffer: WebGLFramebuffer | null;
  /** Highest reduction `1.0`; 0 disables. Set by viewer before frame loop. */
  fixedFoveation: number | null;
  getViewport(view: XRViewLite): XRViewportLite | null;
}

/** A single XR animation frame. */
export interface XRFrameLite {
  readonly session: XRSessionLite;
  getViewerPose(refSpace: XRReferenceSpaceLite): XRViewerPoseLite | null;
}

/** Callback signature for `XRSession.requestAnimationFrame`. */
export type XRFrameRequestCallbackLite = (
  time: number,
  frame: XRFrameLite,
) => void;

/** Subset of `XRSession` we touch. */
export interface XRSessionLite {
  readonly mode?: XRSessionModeLite;
  requestReferenceSpace(
    type: XRReferenceSpaceTypeLite,
  ): Promise<XRReferenceSpaceLite>;
  requestAnimationFrame(cb: XRFrameRequestCallbackLite): number;
  end(): Promise<void>;
  updateRenderState(state: {
    baseLayer?: XRWebGLLayerLite;
    depthFar?: number;
    depthNear?: number;
  }): void;
  addEventListener(type: 'end', listener: () => void): void;
}

/** The `navigator.xr` global. */
export interface XRSystemLite {
  isSessionSupported(mode: XRSessionModeLite): Promise<boolean>;
  requestSession(
    mode: XRSessionModeLite,
    init?: { optionalFeatures?: string[]; requiredFeatures?: string[] },
  ): Promise<XRSessionLite>;
}
