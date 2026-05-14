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
