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
};
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
export function clampLodForXR(mode, levels, defaultLevel) {
    if (levels.length === 0) {
        throw new Error('clampLodForXR: no levels provided');
    }
    if (mode === 'inline') {
        if (defaultLevel != null) {
            const hit = levels.find((l) => l.level === defaultLevel);
            if (hit)
                return hit;
        }
        // Fall through: pick coarsest.
        return [...levels].sort((a, b) => b.level - a.level)[0];
    }
    // Immersive: pick the finest level that is still >= the immersive cap.
    // Sorted ascending by level (coarser = higher idx).
    const sorted = [...levels].sort((a, b) => a.level - b.level);
    const allowed = sorted.filter((l) => l.level >= COMFORT.maxLodLevelImmersive);
    if (allowed.length === 0) {
        // Pyramid truncated above the cap; return coarsest available.
        return sorted[sorted.length - 1];
    }
    // The finest *allowed* = smallest level number among `allowed`.
    return allowed[0];
}
/**
 * Frame-time tracker for the promote/demote rule.
 *
 * Caller pushes the wall-clock ms each frame; {@link decide} returns the
 * proposed LOD level, never finer than `maxLodLevelImmersive - 1`.
 */
export class FrameBudgetTracker {
    samples = [];
    currentLevel;
    minLevel;
    maxLevel;
    constructor(opts) {
        this.currentLevel = opts.startLevel;
        this.minLevel = opts.minLevel ?? COMFORT.maxLodLevelImmersive - 1;
        this.maxLevel = opts.maxLevel ?? 5;
    }
    push(frameMs) {
        this.samples.push(frameMs);
        if (this.samples.length > COMFORT.promoteWindow) {
            this.samples.shift();
        }
    }
    /** Return the level the renderer should target *now*. */
    decide() {
        if (this.samples.length === 0)
            return this.currentLevel;
        const last = this.samples[this.samples.length - 1];
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
    get level() {
        return this.currentLevel;
    }
}
//# sourceMappingURL=comfort.js.map