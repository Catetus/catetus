/**
 * Unit tests for the WebXR splat-viewer bridge.
 *
 * Covered:
 *   - row-major (WebXR) -> column-major (renderer) matrix conversion
 *   - per-eye render gets distinct view matrices (left vs right)
 *   - comfort clamp forces L4 (level=4) max in immersive-vr/ar
 *   - frame-budget tracker promotes/demotes per the policy
 */
import { describe, expect, it, vi } from 'vitest';
import {
  clampLodForXR,
  COMFORT,
  FrameBudgetTracker,
  isXRSessionSupported,
  rowMajorToColMajor4,
  WebXRSplatViewer,
  type LodgeLevelHandle,
  type XREyeRenderArgs,
} from '../../webxr/index.js';
import type {
  XRFrameLite,
  XRReferenceSpaceLite,
  XRRigidTransformLite,
  XRSessionLite,
  XRSystemLite,
  XRViewLite,
  XRViewerPoseLite,
  XRWebGLLayerLite,
} from '../../webxr/types.js';

const FULL_PYRAMID: LodgeLevelHandle[] = [
  { level: 0, splatCount: 119_824_300 },
  { level: 1, splatCount: 54_200_000 },
  { level: 2, splatCount: 28_000_000 },
  { level: 3, splatCount: 13_318_000 },
  { level: 4, splatCount: 7_089_000 },
  { level: 5, splatCount: 3_200_000 },
];

function rigid(matrix: Float32Array): XRRigidTransformLite {
  // Tests don't exercise position/orientation directly. Provide a stub
  // inverse that returns the same matrix (the test asserts the *renderer*
  // receives the inverse; the conversion is what's interesting).
  const t: XRRigidTransformLite = {
    position: { x: 0, y: 0, z: 0, w: 1 },
    orientation: { x: 0, y: 0, z: 0, w: 1 },
    matrix,
    get inverse() {
      return t;
    },
  } as XRRigidTransformLite;
  return t;
}

function view(eye: 'left' | 'right', viewMatRowMajor: Float32Array): XRViewLite {
  // Distinct projection matrices per eye so we can also assert separation.
  const proj = new Float32Array(16);
  proj[0] = eye === 'left' ? 1.5 : 1.7;
  proj[5] = 1.6;
  proj[10] = -1.0;
  proj[11] = -0.1;
  proj[14] = -0.1;
  return {
    eye,
    transform: rigid(viewMatRowMajor),
    projectionMatrix: proj,
  };
}

describe('rowMajorToColMajor4', () => {
  it('transposes a 4x4 row-major matrix to column-major', () => {
    const rm = new Float32Array([
      // row 0
      1, 2, 3, 4,
      // row 1
      5, 6, 7, 8,
      // row 2
      9, 10, 11, 12,
      // row 3
      13, 14, 15, 16,
    ]);
    const cm = rowMajorToColMajor4(rm);
    // Column-major: column 0 is rm rows 0..3 column 0 = (1, 5, 9, 13).
    expect(Array.from(cm.slice(0, 4))).toEqual([1, 5, 9, 13]);
    expect(Array.from(cm.slice(4, 8))).toEqual([2, 6, 10, 14]);
    expect(Array.from(cm.slice(8, 12))).toEqual([3, 7, 11, 15]);
    expect(Array.from(cm.slice(12, 16))).toEqual([4, 8, 12, 16]);
  });

  it('throws on wrong length', () => {
    expect(() => rowMajorToColMajor4(new Float32Array(15))).toThrowError();
  });
});

describe('clampLodForXR', () => {
  it('returns L4 (level=4) in immersive-vr for the full Sweet Corals pyramid', () => {
    const pick = clampLodForXR('immersive-vr', FULL_PYRAMID);
    expect(pick.level).toBe(4);
    expect(pick.splatCount).toBe(7_089_000);
  });

  it('returns L4 in immersive-ar as well (same comfort floor)', () => {
    const pick = clampLodForXR('immersive-ar', FULL_PYRAMID);
    expect(pick.level).toBe(COMFORT.maxLodLevelImmersive);
  });

  it('never returns L0/L1/L2/L3 in immersive-vr', () => {
    const pick = clampLodForXR('immersive-vr', FULL_PYRAMID);
    expect(pick.level).toBeGreaterThanOrEqual(COMFORT.maxLodLevelImmersive);
  });

  it('falls back to coarsest level if pyramid is truncated above the cap', () => {
    const truncated: LodgeLevelHandle[] = [
      { level: 0, splatCount: 119_824_300 },
      { level: 1, splatCount: 54_200_000 },
      { level: 2, splatCount: 28_000_000 },
    ];
    const pick = clampLodForXR('immersive-vr', truncated);
    expect(pick.level).toBe(2);
  });

  it('respects defaultLevel in inline mode', () => {
    const pick = clampLodForXR('inline', FULL_PYRAMID, 3);
    expect(pick.level).toBe(3);
  });
});

describe('FrameBudgetTracker', () => {
  it('demotes immediately on a slow frame', () => {
    const t = new FrameBudgetTracker({ startLevel: 4, minLevel: 3, maxLevel: 5 });
    t.push(20); // 50 Hz — sickness territory
    expect(t.decide()).toBe(5);
  });

  it('promotes after a full window of headroom frames', () => {
    const t = new FrameBudgetTracker({ startLevel: 4, minLevel: 3, maxLevel: 5 });
    for (let i = 0; i < COMFORT.promoteWindow; i++) {
      t.push(10); // 100 Hz
    }
    expect(t.decide()).toBe(3);
  });

  it('clamps promotion to minLevel', () => {
    const t = new FrameBudgetTracker({ startLevel: 3, minLevel: 3, maxLevel: 5 });
    for (let i = 0; i < COMFORT.promoteWindow * 2; i++) t.push(10);
    t.decide();
    expect(t.level).toBe(3);
  });
});

describe('isXRSessionSupported', () => {
  it('returns false when navigator.xr is undefined', async () => {
    await expect(isXRSessionSupported(undefined, 'immersive-vr')).resolves.toBe(false);
  });

  it('returns true when the system reports support', async () => {
    const xr: XRSystemLite = {
      isSessionSupported: vi.fn().mockResolvedValue(true),
      requestSession: vi.fn(),
    };
    await expect(isXRSessionSupported(xr, 'immersive-vr')).resolves.toBe(true);
  });
});

/**
 * Build a fully-stubbed XR session that yields a single canned pose with
 * distinct left/right view matrices, then ends.
 */
function makeFakeSession(opts: {
  leftViewRowMajor: Float32Array;
  rightViewRowMajor: Float32Array;
  framesToServe: number;
}): {
  xr: XRSystemLite;
  layer: XRWebGLLayerLite;
  rafCalls: number;
  endCalled: { current: boolean };
} {
  const refSpace: XRReferenceSpaceLite = { type: 'local-floor' };

  let endCb: (() => void) | null = null;
  const endCalled = { current: false };

  const layer: XRWebGLLayerLite = {
    framebuffer: null,
    fixedFoveation: null,
    getViewport: (v) => ({
      x: v.eye === 'right' ? 1024 : 0,
      y: 0,
      width: 1024,
      height: 1024,
    }),
  };

  const session: XRSessionLite = {
    mode: 'immersive-vr',
    requestReferenceSpace: vi.fn().mockResolvedValue(refSpace),
    requestAnimationFrame: vi.fn(),
    end: vi.fn().mockImplementation(async () => {
      endCalled.current = true;
      endCb?.();
    }),
    updateRenderState: vi.fn(),
    addEventListener: (type, listener) => {
      if (type === 'end') endCb = listener;
    },
  };

  let frameTime = 0;
  let frameIdx = 0;
  let rafCalls = 0;
  (session.requestAnimationFrame as ReturnType<typeof vi.fn>).mockImplementation(
    (cb) => {
      rafCalls++;
      if (frameIdx >= opts.framesToServe) return rafCalls;
      frameIdx++;
      frameTime += 12; // ~83 Hz
      const pose: XRViewerPoseLite = {
        transform: rigid(new Float32Array(16)),
        views: [view('left', opts.leftViewRowMajor), view('right', opts.rightViewRowMajor)],
      };
      const frame: XRFrameLite = {
        session,
        getViewerPose: () => pose,
      };
      // Synchronous invocation to keep tests deterministic.
      queueMicrotask(() => cb(frameTime, frame));
      return rafCalls;
    },
  );

  const xr: XRSystemLite = {
    isSessionSupported: vi.fn().mockResolvedValue(true),
    requestSession: vi.fn().mockResolvedValue(session),
  };

  return { xr, layer, get rafCalls() { return rafCalls; }, endCalled } as unknown as {
    xr: XRSystemLite;
    layer: XRWebGLLayerLite;
    rafCalls: number;
    endCalled: { current: boolean };
  };
}

describe('WebXRSplatViewer.start', () => {
  it('clamps to L4 when entering immersive-vr', async () => {
    const fake = makeFakeSession({
      leftViewRowMajor: new Float32Array(16),
      rightViewRowMajor: new Float32Array(16),
      framesToServe: 0,
    });
    const renderEye = vi.fn();
    const viewer = new WebXRSplatViewer({
      xr: fake.xr,
      levels: FULL_PYRAMID,
      createXRWebGLLayer: () => fake.layer,
      renderEye,
    });
    const info = await viewer.start('immersive-vr');
    expect(info.selectedLevel.level).toBe(4);
    expect(viewer.currentLevel?.level).toBe(4);
    await viewer.end();
  });

  it('renders both eyes with distinct view matrices in column-major order', async () => {
    // Distinguishable row-major matrices: left has 1 in (0,3), right has 2.
    const leftRm = new Float32Array(16);
    leftRm[3] = 1;
    const rightRm = new Float32Array(16);
    rightRm[3] = 2;

    const fake = makeFakeSession({
      leftViewRowMajor: leftRm,
      rightViewRowMajor: rightRm,
      framesToServe: 1,
    });
    const captured: XREyeRenderArgs[] = [];
    const viewer = new WebXRSplatViewer({
      xr: fake.xr,
      levels: FULL_PYRAMID,
      createXRWebGLLayer: () => fake.layer,
      renderEye: (a) => captured.push(a),
    });
    await viewer.start('immersive-vr');
    // Let the queued microtask + the next rAF fire.
    await new Promise((r) => setTimeout(r, 10));
    await viewer.end();

    expect(captured.length).toBeGreaterThanOrEqual(2);
    const left = captured.find((c) => c.eye === 'left');
    const right = captured.find((c) => c.eye === 'right');
    expect(left).toBeDefined();
    expect(right).toBeDefined();

    // Column-major: row-major (0,3) lands at column-major index 12.
    expect(left!.view[12]).toBe(1);
    expect(right!.view[12]).toBe(2);
    // And the viewports differ.
    expect(left!.viewport.x).toBe(0);
    expect(right!.viewport.x).toBe(1024);
  });

  it('throws when the requested mode is unsupported', async () => {
    const xr: XRSystemLite = {
      isSessionSupported: vi.fn().mockResolvedValue(false),
      requestSession: vi.fn(),
    };
    const viewer = new WebXRSplatViewer({
      xr,
      levels: FULL_PYRAMID,
      createXRWebGLLayer: () => ({
        framebuffer: null,
        fixedFoveation: null,
        getViewport: () => null,
      }),
      renderEye: () => {},
    });
    await expect(viewer.start('immersive-vr')).rejects.toThrow(/unsupported/);
  });
});
