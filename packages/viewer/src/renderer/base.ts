/**
 * Renderer backend abstraction. Both the WebGPU and WebGL2 implementations
 * conform to this surface so {@link SplatForgeViewer} can swap them
 * transparently.
 */
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor } from '../manifest.js';

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
 * Back-to-front insertion sort. Stable, deterministic, O(n²) — fine for the
 * tiny synthetic test corpus and for early-Phase-2 fixtures. Bigger scenes
 * swap this for radix sort on a typed array of depths.
 */
export function sortBackToFront(
  splats: DecodedSplat[],
  cam: CameraPose,
  indices: Uint32Array,
): void {
  // Compute depths first, then insertion-sort indices by descending depth.
  const depths = new Float32Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
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
  for (let i = 1; i < indices.length; i++) {
    const di = depths[i]!;
    const ii = indices[i]!;
    let j = i - 1;
    while (j >= 0 && depths[j]! < di) {
      depths[j + 1] = depths[j]!;
      indices[j + 1] = indices[j]!;
      j--;
    }
    depths[j + 1] = di;
    indices[j + 1] = ii;
  }
}
