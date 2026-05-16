/**
 * VR/AR comfort-tuning policy.
 *
 * Spatial-computing devices reproject (async timewarp / ATW) between
 * application frames, but sustained dropped frames below the headset's
 * native refresh (90 Hz on Quest 3, 100/120 Hz on Vision Pro) cause
 * judder and motion sickness. We therefore clamp LOD selection
 * aggressively when entering an immersive session:
 *
 *   - VR / AR floor LOD = `L4` (Sweet Corals = 7M splats). Reliably
 *     rendered at 80 Hz+ on a Quest 3 (Snapdragon XR2 Gen 2, ~2.5 TFLOPS
 *     mobile) with foveation.
 *   - Promote to `L3` (13M) only when the previous N frames sustained
 *     >= 72 Hz with headroom (frame time <= 11.5 ms).
 *   - Never load `L0`/`L1`/`L2` in VR — these are 28M-119M splats and
 *     fall well below 72 Hz on every shipping headset GPU. Sub-72 Hz in
 *     VR is sickness territory.
 *
 * The XR-emulator path (`mode === 'inline'`) skips the clamp so desktop
 * dev with the WebXR API Emulator can still exercise the full chain.
 */
import type { XRSessionModeLite } from './types.js';

/**
 * LOD levels from the LODGE Sweet Corals pyramid, as published.
 * Numbers match `apps/web/src/pages/scale.astro`:
 *
 *   L0 = 119,824,300  L1 = 54,200,000  L2 = 28,000,000
 *   L3 = 13,318,000   L4 =  7,089,000  L5 =  3,200,000
 */
export interface LodgeLevelHandle {
  /** 0..5 — finer is smaller integer. */
  level: number;
  splatCount: number;
}

/** Comfort policy thresholds. Exposed for tests. */
export const COMFORT = {
  /** Cap LOD in immersive-vr/ar to this level (4 = L4 = 7M). */
  maxLodLevelImmersive: 4,
  /** Promote to `maxLodLevelImmersive - 1` if frame-time <= this (ms). */
  promoteFrameMsThreshold: 11.5, // ~87 Hz
  /** Demote back to floor if frame-time exceeds this (ms). */
  demoteFrameMsThreshold: 13.9, // ~72 Hz
  /** Min consecutive good frames before promoting. */
  promoteWindow: 60,
  /** Foveation level for `XRWebGLLayer.fixedFoveation` (0..1). */
  foveationLevel: 1.0,
} as const;

/**
 * Pick the LOD level to load given the XR session mode + available levels.
 *
 * Strict rule: in immersive sessions, return the *coarsest* LOD whose
 * `level` is `>= maxLodLevelImmersive` — i.e. never load anything finer
 * than L4. If no level satisfies, return the coarsest available (this
 * lets unit tests construct truncated pyramids).
 *
 * In `inline` mode (desktop dev, emulator dev) we fall through to the
 * caller's previously-selected `defaultLevel`.
 */
export function clampLodForXR(
  mode: XRSessionModeLite,
  levels: ReadonlyArray<LodgeLevelHandle>,
  defaultLevel?: number,
): LodgeLevelHandle {
  if (levels.length === 0) {
    throw new Error('clampLodForXR: no levels provided');
  }
  if (mode === 'inline') {
    if (defaultLevel != null) {
      const hit = levels.find((l) => l.level === defaultLevel);
      if (hit) return hit;
    }
    // Fall through: pick coarsest.
    return [...levels].sort((a, b) => b.level - a.level)[0]!;
  }
  // Immersive: pick the finest level that is still >= the immersive cap.
  // Sorted ascending by level (coarser = higher idx).
  const sorted = [...levels].sort((a, b) => a.level - b.level);
  const allowed = sorted.filter((l) => l.level >= COMFORT.maxLodLevelImmersive);
  if (allowed.length === 0) {
    // Pyramid truncated above the cap; return coarsest available.
    return sorted[sorted.length - 1]!;
  }
  // The finest *allowed* = smallest level number among `allowed`.
  return allowed[0]!;
}

/**
 * Frame-time tracker for the promote/demote rule.
 *
 * Caller pushes the wall-clock ms each frame; {@link decide} returns the
 * proposed LOD level, never finer than `maxLodLevelImmersive - 1`.
 */
export class FrameBudgetTracker {
  private samples: number[] = [];
  private currentLevel: number;
  private readonly minLevel: number;
  private readonly maxLevel: number;

  constructor(opts: {
    /** Starting LOD level (e.g. 4). */
    startLevel: number;
    /** Finest level the tracker may ever promote to (e.g. 3). */
    minLevel?: number;
    /** Coarsest level (highest integer) the tracker will demote to. */
    maxLevel?: number;
  }) {
    this.currentLevel = opts.startLevel;
    this.minLevel = opts.minLevel ?? COMFORT.maxLodLevelImmersive - 1;
    this.maxLevel = opts.maxLevel ?? 5;
  }

  push(frameMs: number): void {
    this.samples.push(frameMs);
    if (this.samples.length > COMFORT.promoteWindow) {
      this.samples.shift();
    }
  }

  /** Return the level the renderer should target *now*. */
  decide(): number {
    if (this.samples.length === 0) return this.currentLevel;
    const last = this.samples[this.samples.length - 1]!;
    // Demote on the spot if the latest frame stuttered.
    if (last > COMFORT.demoteFrameMsThreshold) {
      this.currentLevel = Math.min(this.maxLevel, this.currentLevel + 1);
      return this.currentLevel;
    }
    // Promote only after a full window of headroom.
    if (this.samples.length >= COMFORT.promoteWindow) {
      const max = Math.max(...this.samples);
      if (max <= COMFORT.promoteFrameMsThreshold) {
        this.currentLevel = Math.max(this.minLevel, this.currentLevel - 1);
        // Reset window so we don't insta-promote again.
        this.samples = [];
      }
    }
    return this.currentLevel;
  }

  get level(): number {
    return this.currentLevel;
  }
}
