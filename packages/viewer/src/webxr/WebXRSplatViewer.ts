/**
 * WebXRSplatViewer — drives a WebXR session against the SplatForge
 * compute-decode renderer.
 *
 * Responsibilities, in lifecycle order:
 *
 *   1. **Session start**: call `navigator.xr.requestSession` for
 *      `immersive-vr` or `immersive-ar`, configure an `XRWebGLLayer`
 *      bound to a host WebGL2 canvas (the WebGPU compute path stays in
 *      its own GPUDevice; the XR system requires a WebGL framebuffer
 *      to write to, so the splat compute output is blitted via a
 *      readback or a shared GPUExternalTexture in a follow-up).
 *   2. **LOD comfort clamp**: forward the session mode to
 *      {@link clampLodForXR} to enforce L4 (7M) max in VR/AR.
 *   3. **Frame loop**: for each `XRFrame`, fetch the `XRViewerPose`,
 *      convert each `XRView`'s row-major `projectionMatrix` + inverse
 *      transform to column-major view/projection matrices, and call
 *      the renderer per eye with the appropriate viewport.
 *   4. **Foveation**: set `XRWebGLLayer.fixedFoveation = 1.0` so the
 *      headset reduces shading rate in the periphery.
 *
 * The actual splat rendering happens through `renderEye()`; this class
 * is the bridge, not the renderer. In tests the renderer is replaced
 * by a recording mock so we can assert per-eye view-matrix correctness.
 */
import {
  clampLodForXR,
  COMFORT,
  FrameBudgetTracker,
  type LodgeLevelHandle,
} from './comfort.js';
import type {
  XRFrameLite,
  XRReferenceSpaceLite,
  XRSessionLite,
  XRSessionModeLite,
  XRSystemLite,
  XRViewLite,
  XRWebGLLayerLite,
} from './types.js';

/** Per-eye render input passed to the host renderer. */
export interface XREyeRenderArgs {
  eye: 'left' | 'right';
  /** Column-major 4x4 view matrix (length 16). */
  view: Float32Array;
  /** Column-major 4x4 projection matrix (length 16). */
  projection: Float32Array;
  /** Pixel viewport on the XR layer's framebuffer. */
  viewport: { x: number; y: number; width: number; height: number };
  /** Frame epoch ms reported by the XR runtime. */
  time: number;
}

/** Host-supplied dependencies. Keeps this module testable & renderer-agnostic. */
export interface WebXRSplatViewerDeps {
  /** Usually `navigator.xr`. Pass in for tests. */
  xr: XRSystemLite;
  /** Available LOD pyramid handles for the loaded scene. */
  levels: ReadonlyArray<LodgeLevelHandle>;
  /** Construct an `XRWebGLLayer` bound to `session`. Tests stub this. */
  createXRWebGLLayer(session: XRSessionLite): XRWebGLLayerLite;
  /** Render one eye. Called twice per frame in stereo. */
  renderEye(args: XREyeRenderArgs): void;
  /** Optional hook called once after the session is ready. */
  onSessionStart?: (info: SessionStartInfo) => void;
  /** Optional hook called when the session ends (user removes headset). */
  onSessionEnd?: () => void;
}

export interface SessionStartInfo {
  mode: XRSessionModeLite;
  selectedLevel: LodgeLevelHandle;
  refSpaceType: 'local-floor' | 'local';
}

/**
 * Convert a row-major fp32 matrix (as the WebXR API ships) to a
 * column-major fp32 matrix (the layout {@link Float32Array} consumers
 * like WGSL `mat4x4` and GLSL `uniformMatrix4fv(..., transpose=false)`
 * expect).
 *
 * Exposed for unit tests.
 */
export function rowMajorToColMajor4(rm: Float32Array): Float32Array {
  if (rm.length !== 16) {
    throw new Error(`rowMajorToColMajor4: expected length 16, got ${rm.length}`);
  }
  const cm = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      // Row-major index = r*4 + c. Column-major index = c*4 + r.
      cm[c * 4 + r] = rm[r * 4 + c]!;
    }
  }
  return cm;
}

/** Probe support without instantiating a session. */
export async function isXRSessionSupported(
  xr: XRSystemLite | undefined,
  mode: XRSessionModeLite,
): Promise<boolean> {
  if (!xr) return false;
  try {
    return await xr.isSessionSupported(mode);
  } catch {
    return false;
  }
}

/**
 * Bridges a WebXR session to the SplatForge renderer.
 *
 * Construct, then call {@link start} with the desired session mode.
 * The frame loop is owned by the XR runtime — we register an rAF
 * callback per frame inside {@link start}.
 */
export class WebXRSplatViewer {
  private readonly deps: WebXRSplatViewerDeps;
  private session: XRSessionLite | null = null;
  private refSpace: XRReferenceSpaceLite | null = null;
  private layer: XRWebGLLayerLite | null = null;
  private selectedLevel: LodgeLevelHandle | null = null;
  private mode: XRSessionModeLite | null = null;
  private readonly frameBudget: FrameBudgetTracker;
  private rafId: number | null = null;
  private lastFrameStartMs = 0;

  constructor(deps: WebXRSplatViewerDeps) {
    this.deps = deps;
    this.frameBudget = new FrameBudgetTracker({ startLevel: COMFORT.maxLodLevelImmersive });
  }

  /** Currently-targeted LOD level (post-comfort-clamp). */
  get currentLevel(): LodgeLevelHandle | null {
    return this.selectedLevel;
  }

  /** Active XR session mode (null when not running). */
  get currentMode(): XRSessionModeLite | null {
    return this.mode;
  }

  /** Begin an XR session. Throws if the mode is unsupported. */
  async start(mode: XRSessionModeLite): Promise<SessionStartInfo> {
    if (this.session) {
      throw new Error('WebXRSplatViewer.start: session already active');
    }
    const supported = await isXRSessionSupported(this.deps.xr, mode);
    if (!supported) {
      throw new Error(`WebXR session unsupported: ${mode}`);
    }
    // AR optionally wants local-floor + hit-test; VR wants local-floor.
    const refSpaceType: 'local-floor' | 'local' = 'local-floor';
    const init =
      mode === 'immersive-ar'
        ? { requiredFeatures: ['local-floor'], optionalFeatures: ['hit-test', 'anchors'] }
        : { requiredFeatures: ['local-floor'] };
    const session = await this.deps.xr.requestSession(mode, init);

    // Comfort clamp BEFORE we hit the frame loop — the renderer pre-uploads.
    const level = clampLodForXR(mode, this.deps.levels);
    this.selectedLevel = level;
    this.mode = mode;

    const layer = this.deps.createXRWebGLLayer(session);
    layer.fixedFoveation = COMFORT.foveationLevel;
    session.updateRenderState({ baseLayer: layer, depthNear: 0.05, depthFar: 1000 });

    const refSpace = await session.requestReferenceSpace(refSpaceType);

    session.addEventListener('end', () => this.handleEnd());

    this.session = session;
    this.layer = layer;
    this.refSpace = refSpace;

    const info: SessionStartInfo = { mode, selectedLevel: level, refSpaceType };
    this.deps.onSessionStart?.(info);

    // Kick the rAF loop.
    this.lastFrameStartMs = performance.now();
    this.rafId = session.requestAnimationFrame((t, f) => this.onFrame(t, f));
    return info;
  }

  /** End the active session. Idempotent. */
  async end(): Promise<void> {
    if (!this.session) return;
    const s = this.session;
    this.session = null;
    await s.end();
  }

  /**
   * Per-frame callback. Public for tests — production code never calls
   * this directly; the XR runtime invokes it through requestAnimationFrame.
   */
  onFrame(time: number, frame: XRFrameLite): void {
    if (!this.session || !this.refSpace || !this.layer) return;

    // Frame-time accounting drives the promote/demote tracker.
    const dt = time - this.lastFrameStartMs;
    this.lastFrameStartMs = time;
    if (dt > 0 && dt < 100) {
      this.frameBudget.push(dt);
    }
    const proposed = this.frameBudget.decide();
    // Caller is responsible for switching the loaded LOD if `proposed`
    // differs from `currentLevel`. We surface it via `currentLevel`.
    if (this.selectedLevel && proposed !== this.selectedLevel.level) {
      const hit = this.deps.levels.find((l) => l.level === proposed);
      if (hit) this.selectedLevel = hit;
    }

    const pose = frame.getViewerPose(this.refSpace);
    if (pose) {
      for (const view of pose.views) {
        this.renderView(view, time);
      }
    }

    // Re-arm.
    if (this.session) {
      this.rafId = this.session.requestAnimationFrame((t, f) => this.onFrame(t, f));
    }
  }

  private renderView(view: XRViewLite, time: number): void {
    if (!this.layer) return;
    const viewport = this.layer.getViewport(view);
    if (!viewport) return;

    // XR ships row-major fp32; our renderer math is column-major.
    const viewMatrix = rowMajorToColMajor4(view.transform.inverse.matrix);
    const projMatrix = rowMajorToColMajor4(view.projectionMatrix);

    // `eye` may be `'none'` in inline / emulator inline mode — we still want
    // to render that case, treating it as left.
    const eye: 'left' | 'right' = view.eye === 'right' ? 'right' : 'left';
    this.deps.renderEye({
      eye,
      view: viewMatrix,
      projection: projMatrix,
      viewport,
      time,
    });
  }

  private handleEnd(): void {
    this.session = null;
    this.layer = null;
    this.refSpace = null;
    if (this.rafId != null) this.rafId = null;
    this.deps.onSessionEnd?.();
  }
}
