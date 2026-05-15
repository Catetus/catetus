/**
 * Public type surface for `@splatforge/viewer`.
 *
 * These types form the stable contract between the viewer SDK and its
 * consumers (the `splatforge preview` shell, the visual-regression harness in
 * SPEC-0009, and downstream embedders).
 */

/**
 * Renderer backend selector.
 *
 * - `webgpu` — force WebGPU. Throws `renderer_unavailable` if unsupported.
 * - `webgl2` — force WebGL2 fallback path.
 * - `auto`   — probe WebGPU first, fall back to WebGL2.
 */
export type RendererKind = 'webgpu' | 'webgl2' | 'auto';

/**
 * Performance/quality budget preset. Drives default LOD bias, max splat count,
 * and tile prefetch depth inside the viewer.
 */
export type BudgetPreset =
  | 'web-mobile'
  | 'web-desktop'
  | 'quest-browser'
  | 'visionos-preview';

/**
 * Deterministic camera path. Used by the headless visual-regression harness.
 *
 * - `orbit`    — continuous orbit around scene bbox center.
 * - `orbit-8`  — 8 discrete yaw frames (every 45 deg). Default for tests.
 * - `static`   — single canonical pose.
 * - `custom`   — caller-supplied positions.
 */
export type CameraPath =
  | 'orbit'
  | 'orbit-8'
  | 'static'
  | { type: 'custom'; positions: [number, number, number][] };

/**
 * Construction options for {@link SplatForgeViewer}.
 */
export interface ViewerOptions {
  /** Target `<canvas>` element. The viewer owns its drawing buffer. */
  canvas: HTMLCanvasElement;
  /** URL to the root glTF (or .gltf JSON / .spz binary) asset. */
  src: string;
  /** Performance budget preset. Defaults to `web-desktop`. */
  budget?: BudgetPreset;
  /** Renderer backend. Defaults to `auto`. */
  renderer?: RendererKind;
  /** Deterministic camera path used in headless / CI mode. */
  cameraPath?: CameraPath;
  /** Enable deterministic mode (fixed RNG seed + frozen clock). */
  deterministic?: boolean;
  /** Seed for deterministic RNG. Defaults to `0xC0FFEE`. */
  seed?: number;
  /** Mount a tiny on-canvas FPS / memory overlay. Defaults to `false`. */
  stats?: boolean;
  /**
   * Continuously orbit after the deterministic camera path completes. Used by
   * the marketing hero where we want a live, breathing preview rather than a
   * frozen final frame. Ignored in headless / regression mode.
   */
  autoRotate?: boolean;
  /** Auto-rotate angular speed in degrees per second. Defaults to `10`. */
  autoRotateSpeed?: number;
  /**
   * Multiplier applied to the orbit radius the viewer otherwise picks from
   * the scene bbox half-diagonal. `1.0` is the canonical
   * fits-the-whole-bbox framing used by the deterministic camera paths;
   * values below `1.0` push the camera closer (useful for hero shots that
   * want to fill more pixels). Defaults to `1.0`.
   *
   * Only consulted by `autoRotate`. The deterministic `orbit-8` / `static`
   * paths intentionally ignore this so visual-regression frames stay stable.
   */
  autoRotateFraming?: number;
  /**
   * Optional override for the bbox the viewer uses to position the camera.
   * When provided, all camera-path math uses this bbox in place of the
   * bbox loaded from the asset's manifest.
   *
   * Use case: visual-regression. If a preset prunes geometry (e.g.
   * `size-min` killing low-opacity floaters), the manifest's bbox shrinks
   * and the auto-framed camera flies in close — comparing it to the
   * baseline's distant camera then measures the *reframing*, not the
   * preset's actual quality. Passing the input scene's bbox here pins the
   * camera so every preset is rendered from the same viewpoint.
   */
  cameraBbox?: { min: [number, number, number]; max: [number, number, number] };
  /**
   * Use the WGSL compute-decode + GPU radix-sort pipeline (queue #62) when on
   * the WebGPU backend. Defaults to `false` — the CPU decode/sort path stays
   * as the conservative default. Has no effect under the WebGL2 backend.
   *
   * The compute path moves dequantization, projection, depth-key generation,
   * and back-to-front sorting entirely onto the GPU. This is the path that
   * unlocks the v2 "30 GB scenes at 60 fps on mobile" target by eliminating
   * the per-frame CPU→GPU vertex-buffer upload that dominates at >1M splats.
   *
   * Behaviour is deterministic for a fixed input + camera; the visual-
   * regression harness in SPEC-0009 covers parity vs the CPU path.
   */
  useComputeDecode?: boolean;
}

/**
 * Discriminated event union emitted by {@link SplatForgeViewer}.
 *
 * Use {@link SplatForgeViewer.on} with the literal `type` string to subscribe.
 */
export type ViewerEvent =
  | { type: 'loadStart' }
  | { type: 'manifestLoaded'; chunkCount: number }
  | { type: 'chunkLoaded'; chunkIndex: number; byteLength: number }
  | { type: 'firstRender' }
  | { type: 'frameRendered'; index: number; total: number }
  | { type: 'qualityChanged'; level: number }
  | { type: 'complete' }
  | { type: 'warning'; code: string; message: string }
  | { type: 'error'; code: ViewerErrorCode; message: string };

/**
 * Stable error codes. SPEC-0008 acceptance tests assert on `chunk_not_found`.
 */
export type ViewerErrorCode =
  | 'manifest_not_found'
  | 'manifest_invalid'
  | 'chunk_not_found'
  | 'checksum_failed'
  | 'renderer_unavailable'
  | 'renderer_init_failed'
  | 'unknown';
