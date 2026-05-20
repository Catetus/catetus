import type { ZstdDecompress } from './zstd-split.js';
/** Parsed sidecar header. */
export interface V5TailHeader {
    variant: number;
    flags: number;
    nSplats: number;
    kSelected: number;
    shRestCoefs: number;
    nCells: number;
}
/** Fully-decoded sidecar payload — per-group residuals already de-Morton-
 *  permuted into ascending-SF order on the selected subset. */
export interface DecodedV5Tail {
    header: V5TailHeader;
    /** Indices into the full `[0, nSplats)` splat array, ascending. */
    selIdx: Uint32Array;
    /** `(K, 3)` row-major position residual. */
    pos: Float32Array;
    /** `(K, 4)` row-major rotation residual (raw, un-normalized). */
    rot: Float32Array;
    /** `(K,)` opacity residual in logit-space. */
    opa: Float32Array;
    /** `(K, 3)` log-scale residual. */
    sca: Float32Array;
    /** `(K, 3)` DC color residual. */
    dc: Float32Array;
    /** `(K, sh_rest_coefs * 3)` SH-rest residual. */
    shr: Float32Array;
}
/** Splat scene attributes the apply path needs as typed arrays. */
export interface ApplyTargetScene {
    /** `count * 3` linear-space positions. */
    positions: Float32Array;
    /** `count * 4` quaternions (any normalization — additive residual is raw). */
    rotations: Float32Array;
    /** `count * 3` linear-space scales. The apply path round-trips through
     *  `ln(scale) + sca_res → exp(...)`. */
    scales: Float32Array;
    /** `count` linear-space opacities in [0,1]. Round-tripped through
     *  `sigmoid(logit(opacity) + opa_res)`. */
    opacities: Float32Array;
    /** `count * 3` raw DC SH coefficients (i.e. the value the polyfill returns
     *  in `DecodedSplats.dcRaw` before the optional SH_C0 bake). Field name
     *  matches `DecodedSplats.dcRaw` exactly so the apply call can pass it
     *  straight through. */
    dcRaw: Float32Array;
    /** `count * shRestCoefs * 3` flat per-splat SH-rest residual targets.
     *  May be `null` when the source scene has no SH-rest (V5.2 sidecars
     *  emitted from such a scene silently skip the shr group). */
    shRest: Float32Array | null;
    /** Number of SH-rest coefficient slots per channel (e.g. 15 for degree 3). */
    shRestCoefs: number;
}
/**
 * Parse a V5.2 sidecar byte slice. Returns the per-group residuals already
 * de-Morton-permuted into ascending-SF order (so `out.pos[k*3..k*3+3]` is
 * the residual for splat `out.selIdx[k]`).
 *
 * Throws on bad magic / version / variant / truncation / malformed groups.
 */
export declare function decodeV5TailBytes(bytes: Uint8Array, zstdDecompress?: ZstdDecompress): DecodedV5Tail;
/**
 * Apply a decoded V5.2 sidecar to a splat scene (in-place mutation of the
 * scene's typed arrays). Returns the number of splats actually modified.
 *
 * Coordinate conventions mirror the Rust apply path:
 *   * `opacity`: residual is logit-space → round-trip through logit + sigmoid.
 *   * `scale`:   residual is log-space → round-trip through ln + exp.
 *   * `position` / `rotation` / `dc` / `sh_rest`: linear additive.
 *
 * The caller is responsible for re-normalizing rotation quats (if it cares)
 * AFTER this returns. We don't normalize here because the V5.2 prototype
 * adds the un-normalized PLY residual, matching the Rust apply path.
 */
export declare function applyV5TailToScene(scene: ApplyTargetScene, decoded: DecodedV5Tail): number;
//# sourceMappingURL=v5tail.d.ts.map