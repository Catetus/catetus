/**
 * Renderer backend abstraction. Both the WebGPU and WebGL2 implementations
 * conform to this surface so {@link SplatForgeViewer} can swap them
 * transparently.
 */
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor, SoaAttributeLayout } from '../manifest.js';

/** One-time renderer initialization options. */
export interface RendererInitOptions {
  canvas: HTMLCanvasElement;
  /** Clear color. Defaults to opaque black. */
  clearColor?: [number, number, number, number];
}

/**
 * Renderer backend interface. Implementations are expected to be stateful and
 * single-canvas: they own GPU resources from `init` through `destroy`.
 */
export interface Renderer {
  /** Friendly name reported in error events. */
  readonly kind: 'webgpu' | 'webgl2';
  /** Acquire the device / context and create static pipelines. */
  init(opts: RendererInitOptions): Promise<void>;
  /** Stage a decoded chunk's splat array onto the GPU. */
  uploadChunk(descriptor: ChunkDescriptor, bytes: Uint8Array): void;
  /** Issue a single frame for `camera`. Resolves after submit. */
  renderFrame(camera: CameraPose): Promise<void>;
  /** Read back the current framebuffer as RGBA bytes (used by SPEC-0009). */
  readPixels(): Promise<Uint8Array>;
  /** Tear down all GPU resources. */
  destroy(): void;
}

/**
 * Decoded splat record (32 floats / 128 bytes). The on-wire layout from
 * `KHR_gaussian_splatting` is compact and quantized; this is the canonical
 * decoded form fed to the GPU.
 */
export interface DecodedSplat {
  position: [number, number, number];
  scale: [number, number, number];
  rotation: [number, number, number, number]; // quaternion (x,y,z,w)
  opacity: number;
  colorDC: [number, number, number];
}

/**
 * Decode a chunk based on its descriptor: SoA when `attributeLayout` is set
 * (the wire format emitted by `splatforge convert`/`optimize`), otherwise the
 * legacy interleaved AoS layout used by hand-crafted test fixtures.
 */
export function decodeChunkBytes(
  bytes: Uint8Array,
  descriptor: ChunkDescriptor,
): DecodedSplat[] {
  if (descriptor.attributeLayout) {
    return decodeSplatsSoa(bytes, descriptor.attributeLayout, descriptor.splatCount);
  }
  return decodeSplats(bytes);
}

/**
 * Decode splats from a structure-of-arrays binary chunk. Attribute order on
 * disk (POSITION, _ROTATION, _SCALE, _OPACITY, _COLOR_DC) follows
 * `KHR_gaussian_splatting`; we re-interleave to `DecodedSplat`.
 *
 * Quaternion convention on the wire is (x, y, z, w) which already matches our
 * runtime convention — no axis flip needed.
 */
export function decodeSplatsSoa(
  bytes: Uint8Array,
  layout: SoaAttributeLayout,
  splatCount: number,
): DecodedSplat[] {
  if (splatCount === 0) return [];
  const buf = bytes.buffer;
  const base = bytes.byteOffset;
  const view = (slice: { byteOffset: number; byteLength: number }, comps: number): Float32Array =>
    new Float32Array(buf, base + slice.byteOffset, splatCount * comps);
  const pos = view(layout.positions, 3);
  const rot = view(layout.rotations, 4);
  const scl = view(layout.scales, 3);
  const op = view(layout.opacities, 1);
  const dc = view(layout.colorDC, 3);
  const out: DecodedSplat[] = new Array(splatCount);
  for (let i = 0; i < splatCount; i++) {
    out[i] = {
      position: [pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!],
      rotation: [rot[i * 4]!, rot[i * 4 + 1]!, rot[i * 4 + 2]!, rot[i * 4 + 3]!],
      scale: [scl[i * 3]!, scl[i * 3 + 1]!, scl[i * 3 + 2]!],
      opacity: op[i]!,
      colorDC: [dc[i * 3]!, dc[i * 3 + 1]!, dc[i * 3 + 2]!],
    };
  }
  return out;
}

/**
 * Naive bytes-to-splats decoder. The optimized packer produces a 32-byte
 * fixed-point layout we'll wire in once SPEC-0007 lands; for now we parse a
 * float32 layout so unit tests can round-trip easily.
 *
 * Layout (per splat, 14 float32 = 56 bytes):
 *   px py pz  sx sy sz  qx qy qz qw  opacity  cr cg cb
 */
export function decodeSplats(bytes: Uint8Array): DecodedSplat[] {
  const stride = 14 * 4;
  const count = Math.floor(bytes.byteLength / stride);
  if (count === 0) return [];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out: DecodedSplat[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    out[i] = {
      position: [dv.getFloat32(o + 0, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)],
      scale: [dv.getFloat32(o + 12, true), dv.getFloat32(o + 16, true), dv.getFloat32(o + 20, true)],
      rotation: [
        dv.getFloat32(o + 24, true),
        dv.getFloat32(o + 28, true),
        dv.getFloat32(o + 32, true),
        dv.getFloat32(o + 36, true),
      ],
      opacity: dv.getFloat32(o + 40, true),
      colorDC: [dv.getFloat32(o + 44, true), dv.getFloat32(o + 48, true), dv.getFloat32(o + 52, true)],
    };
  }
  return out;
}

/**
 * Sort `indices` so splats are drawn back-to-front (largest depth first).
 *
 * Determinism note: equal depths fall back to ascending splat index so two
 * runs on the same inputs produce the same draw order. The sort is in-place
 * via a paired index/depth array so it remains O(n log n) on big scenes.
 */
export function sortBackToFront(
  splats: DecodedSplat[],
  cam: CameraPose,
  indices: Uint32Array,
): void {
  const n = indices.length;
  if (n <= 1) return;
  const depths = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = splats[indices[i]!];
    if (!s) {
      depths[i] = 0;
      continue;
    }
    const dx = s.position[0] - cam.position[0];
    const dy = s.position[1] - cam.position[1];
    const dz = s.position[2] - cam.position[2];
    depths[i] = dx * dx + dy * dy + dz * dz;
  }
  // Build paired array, sort, copy back. Pair-based sort avoids the O(n²)
  // insertion sort that doesn't scale past ~10K splats.
  const pairs: number[][] = new Array(n);
  for (let i = 0; i < n; i++) pairs[i] = [depths[i]!, indices[i]!];
  pairs.sort((a, b) => (b[0]! - a[0]!) || (a[1]! - b[1]!));
  for (let i = 0; i < n; i++) indices[i] = pairs[i]![1]!;
}
