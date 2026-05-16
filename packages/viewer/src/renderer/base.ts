/**
 * Renderer backend abstraction. Both the WebGPU and WebGL2 implementations
 * conform to this surface so {@link SplatForgeViewer} can swap them
 * transparently.
 */
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor, SoaAttributeLayout, SoaAttributeSlice } from '../manifest.js';

/**
 * SH degree-1 band-1 normalization constant (matches the 3DGS reference
 * implementation): `Y_1^m(d) = SH_C1 * d_axis` for d normalized.
 */
export const SH_C1 = 0.4886025119029199;

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
  /**
   * SH degree-1 coefficients, 3 coefs × 3 channels = 9 floats, laid out as
   * `[c0.r, c0.g, c0.b, c1.r, c1.g, c1.b, c2.r, c2.g, c2.b]`. Undefined when
   * the chunk only carries DC (degree 0). Coefficient order matches the
   * `KHR_gaussian_splatting:SH_DEGREE_1_COEF_{0,1,2}` accessors emitted by
   * the Rust encoder.
   */
  sh1?: Float32Array;
}

/**
 * Evaluate the view-dependent color contribution of degree-1 SH for `splat`
 * given the (normalized) splat-to-camera direction `dir = (x, y, z)`.
 *
 * Reference (3DGS Eq. 6): `c = SH_C0*sh0 - SH_C1*y*sh1 + SH_C1*z*sh2 - SH_C1*x*sh3`.
 * The DC term is already pre-baked into `splat.colorDC` (with SplatForge's
 * convention of passing f_dc through directly as RGB rather than applying
 * `SH_C0 + 0.5`), so this function returns only the *additive* degree-1 part.
 *
 * Sign convention here matches gsplat / antimatter15:
 *   `+SH_C1*y*sh[0] +SH_C1*z*sh[1] +SH_C1*x*sh[2]`
 * with `dir = normalize(splat.position - camera.position)` (splat→camera
 * direction). Different reference implementations vary the signs; the test
 * pins the chosen convention so a regression surfaces immediately.
 *
 * Returns `[r, g, b]` to add to `colorDC` before clamping into [0, 1].
 */
export function evaluateSh1(
  sh1: Float32Array,
  dirX: number,
  dirY: number,
  dirZ: number,
): [number, number, number] {
  const ky = SH_C1 * dirY;
  const kz = SH_C1 * dirZ;
  const kx = SH_C1 * dirX;
  // Coefficient layout: [c0=Y_1^-1, c1=Y_1^0, c2=Y_1^1] (rgb-major).
  return [
    ky * sh1[0]! + kz * sh1[3]! + kx * sh1[6]!,
    ky * sh1[1]! + kz * sh1[4]! + kx * sh1[7]!,
    ky * sh1[2]! + kz * sh1[5]! + kx * sh1[8]!,
  ];
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
 * Each attribute is decoded according to its accessor `componentType` /
 * `normalized` / `min` / `max` — SPEC-0013 (`KHR_mesh_quantization`) permits
 * POSITION → u16, _SCALE/_OPACITY/_COLOR_DC → u8 with normalized=true.
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
  const pos = decodeAttribute(bytes, layout.positions, splatCount, 3);
  const rot = decodeAttribute(bytes, layout.rotations, splatCount, 4);
  const scl = decodeAttribute(bytes, layout.scales, splatCount, 3);
  const op = decodeAttribute(bytes, layout.opacities, splatCount, 1);
  const dc = decodeAttribute(bytes, layout.colorDC, splatCount, 3);
  // Optional SH degree-1 coefficients. Each is a VEC3 FLOAT accessor of
  // length `splatCount`. We re-pack into a single 9-float interleaved slice
  // per splat so the per-instance attribute build stays a single pass.
  const sh1Slices: (SoaAttributeSlice | undefined)[] = [
    layout.sh1Coef0,
    layout.sh1Coef1,
    layout.sh1Coef2,
  ];
  let sh1: Float32Array | undefined;
  if (sh1Slices.every((s) => s !== undefined)) {
    const c0 = decodeAttribute(bytes, sh1Slices[0]!, splatCount, 3);
    const c1 = decodeAttribute(bytes, sh1Slices[1]!, splatCount, 3);
    const c2 = decodeAttribute(bytes, sh1Slices[2]!, splatCount, 3);
    sh1 = new Float32Array(splatCount * 9);
    for (let i = 0; i < splatCount; i++) {
      const o = i * 9;
      sh1[o + 0] = c0[i * 3]!;
      sh1[o + 1] = c0[i * 3 + 1]!;
      sh1[o + 2] = c0[i * 3 + 2]!;
      sh1[o + 3] = c1[i * 3]!;
      sh1[o + 4] = c1[i * 3 + 1]!;
      sh1[o + 5] = c1[i * 3 + 2]!;
      sh1[o + 6] = c2[i * 3]!;
      sh1[o + 7] = c2[i * 3 + 1]!;
      sh1[o + 8] = c2[i * 3 + 2]!;
    }
  }
  const out: DecodedSplat[] = new Array(splatCount);
  for (let i = 0; i < splatCount; i++) {
    const splat: DecodedSplat = {
      position: [pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!],
      rotation: [rot[i * 4]!, rot[i * 4 + 1]!, rot[i * 4 + 2]!, rot[i * 4 + 3]!],
      scale: [scl[i * 3]!, scl[i * 3 + 1]!, scl[i * 3 + 2]!],
      opacity: op[i]!,
      colorDC: [dc[i * 3]!, dc[i * 3 + 1]!, dc[i * 3 + 2]!],
    };
    if (sh1) {
      splat.sh1 = sh1.subarray(i * 9, i * 9 + 9);
    }
    out[i] = splat;
  }
  return out;
}

const FLOAT_CT = 5126;
const UBYTE_CT = 5121;
const USHORT_CT = 5123;

/**
 * Decode one SoA attribute into a flat `Float32Array` of length
 * `splatCount * comps`. Handles FLOAT pass-through plus the
 * `KHR_mesh_quantization` integer variants (UNSIGNED_BYTE, UNSIGNED_SHORT),
 * dequantizing against the accessor's per-component min/max when normalized.
 */
function decodeAttribute(
  bytes: Uint8Array,
  slice: {
    byteOffset: number;
    byteLength: number;
    componentType?: number;
    normalized?: boolean;
    min?: number[];
    max?: number[];
  },
  splatCount: number,
  comps: number,
): Float32Array {
  const total = splatCount * comps;
  const buf = bytes.buffer;
  const base = bytes.byteOffset + slice.byteOffset;
  const ct = slice.componentType ?? FLOAT_CT;

  if (ct === FLOAT_CT) {
    // Zero-copy float view when the byte offset is 4-aligned; copy otherwise.
    if ((base & 3) === 0) {
      return new Float32Array(buf, base, total);
    }
    const dv = new DataView(buf, base, total * 4);
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = dv.getFloat32(i * 4, true);
    return out;
  }

  if (ct === USHORT_CT) {
    const src = new Uint16Array(buf, base, total);
    const out = new Float32Array(total);
    if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const lo = slice.min[k]!;
        const hi = slice.max[k]!;
        out[i] = lo + (src[i]! / 65535) * (hi - lo);
      }
    } else if (slice.normalized) {
      for (let i = 0; i < total; i++) out[i] = src[i]! / 65535;
    } else {
      for (let i = 0; i < total; i++) out[i] = src[i]!;
    }
    return out;
  }

  if (ct === UBYTE_CT) {
    const src = new Uint8Array(buf, base, total);
    const out = new Float32Array(total);
    if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const lo = slice.min[k]!;
        const hi = slice.max[k]!;
        out[i] = lo + (src[i]! / 255) * (hi - lo);
      }
    } else if (slice.normalized) {
      for (let i = 0; i < total; i++) out[i] = src[i]! / 255;
    } else {
      for (let i = 0; i < total; i++) out[i] = src[i]!;
    }
    return out;
  }

  throw new Error(`unsupported componentType: ${ct}`);
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
