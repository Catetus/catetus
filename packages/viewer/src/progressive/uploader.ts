// SPDX-License-Identifier: Apache-2.0
/**
 * Progressive `.mgs2` uploader.
 *
 * Converts batches of raw PLY records (the AoS bytes emitted by
 * {@link fetchProgressive}) into the SoA chunk format the viewer's
 * `Renderer.uploadChunk(...)` expects, and feeds each batch to the renderer
 * as it arrives.
 *
 * Why SoA: the WebGPU compute-decode pipeline (`packages/viewer/src/webgpu`)
 * reads splats attribute-by-attribute (POSITION, ROTATION, SCALE, OPACITY,
 * COLOR_DC). The PLY-record layout interleaves those plus a long tail of
 * `f_rest_*` / SH coefficients we don't need at all for rendering. Converting
 * once at upload time keeps the renderer's hot path bit-identical to the
 * production glTF chunk path.
 *
 * What gets converted per record:
 *   - POSITION  : `[x, y, z]`                              (PLY f32 → f32)
 *   - ROTATION  : `[rot_1, rot_2, rot_3, rot_0]`           (PLY w,x,y,z → IR x,y,z,w)
 *                 then normalized to unit quaternion (matches Rust).
 *   - SCALE     : `[exp(scale_0), exp(scale_1), exp(scale_2)]` (log-scale → linear)
 *   - OPACITY   : `sigmoid(opacity_logit)`                 (logit → [0,1])
 *   - COLOR_DC  : `[f_dc_0, f_dc_1, f_dc_2]`               (PLY f32 → f32, passthrough)
 *
 * The per-batch SoA chunk is built as five tightly-packed Float32Array
 * buffers concatenated in this order:
 *     POSITION (12B/splat) | ROTATION (16B/splat) | SCALE (12B/splat)
 *     | OPACITY (4B/splat) | COLOR_DC (12B/splat)
 *
 * Total: 56 bytes per splat. The chunk's `attributeLayout` reports each
 * slice's byteOffset within this concatenation.
 */

import type { Renderer } from '../renderer/base.js';
import type { Bbox, ChunkDescriptor, SoaAttributeLayout } from '../manifest.js';
import { decodeChunkBytes } from '../renderer/base.js';
import type { DecodedSplat } from '../renderer/base.js';
import type { PlyFieldOffsets } from './fetcher.js';

/** glTF FLOAT component type — every attribute we emit is float32. */
const FLOAT_CT = 5126;
/** Per-splat byte size of each SoA slice for our 56-byte chunk layout. */
const STRIDE_POS = 12;
const STRIDE_ROT = 16;
const STRIDE_SCL = 12;
const STRIDE_OP = 4;
const STRIDE_DC = 12;
/** Total per-splat byte size in the SoA chunk emitted by this uploader. */
export const SOA_BYTES_PER_SPLAT = STRIDE_POS + STRIDE_ROT + STRIDE_SCL + STRIDE_OP + STRIDE_DC; // 56

/**
 * Inverse-sigmoid → sigmoid: maps PLY's opacity logit into [0,1]. Hot path
 * lives inside a tight per-record loop; we inline the call site below rather
 * than relying on V8 to do it.
 */
function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Build a per-batch SoA-layout chunk from `count` PLY records.
 *
 * `records` is the raw little-endian PLY payload (whole records only).
 * `fieldOffsets` carries the byte offsets within a record for each field
 * we need. The fixed PLY stride is `fieldOffsets.recordSize`.
 *
 * Returns the concatenated chunk bytes plus a matching `SoaAttributeLayout`.
 */
export function buildSoaChunk(
  records: Uint8Array,
  count: number,
  off: PlyFieldOffsets,
): { bytes: Uint8Array; layout: SoaAttributeLayout; bbox: Bbox } {
  if (count * off.recordSize !== records.byteLength) {
    throw new Error(
      `progressive uploader: records (${records.byteLength} B) is not a whole number of records (count=${count}, stride=${off.recordSize})`,
    );
  }
  // Aligned views over the source payload — PLY columns are always 4-byte
  // f32, but the per-record stride is rarely a multiple of 4 alignment from
  // the source ArrayBuffer's start. Use DataView so unaligned reads work.
  const src = new DataView(records.buffer, records.byteOffset, records.byteLength);

  // Allocate the output chunk: 5 contiguous SoA slices in this order:
  //   POSITION | ROTATION | SCALE | OPACITY | COLOR_DC
  const posOffset = 0;
  const rotOffset = posOffset + count * STRIDE_POS;
  const sclOffset = rotOffset + count * STRIDE_ROT;
  const opOffset = sclOffset + count * STRIDE_SCL;
  const dcOffset = opOffset + count * STRIDE_OP;
  const totalBytes = dcOffset + count * STRIDE_DC;

  const out = new Uint8Array(totalBytes);
  const dvOut = new DataView(out.buffer, out.byteOffset, out.byteLength);

  // Per-batch bbox: helps the optional camera-framing math when the caller
  // wants to follow the progressively-revealed scene.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < count; i++) {
    const base = i * off.recordSize;
    // Position (passthrough).
    const x = src.getFloat32(base + off.x, true);
    const y = src.getFloat32(base + off.y, true);
    const z = src.getFloat32(base + off.z, true);
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    dvOut.setFloat32(posOffset + i * STRIDE_POS + 0, x, true);
    dvOut.setFloat32(posOffset + i * STRIDE_POS + 4, y, true);
    dvOut.setFloat32(posOffset + i * STRIDE_POS + 8, z, true);

    // Rotation: PLY (w, x, y, z) → IR (x, y, z, w), normalize defensively.
    const rw = src.getFloat32(base + off.rot0, true);
    const rx = src.getFloat32(base + off.rot1, true);
    const ry = src.getFloat32(base + off.rot2, true);
    const rz = src.getFloat32(base + off.rot3, true);
    const rnorm = Math.hypot(rx, ry, rz, rw) || 1.0;
    const inv = 1.0 / rnorm;
    dvOut.setFloat32(rotOffset + i * STRIDE_ROT + 0, rx * inv, true);
    dvOut.setFloat32(rotOffset + i * STRIDE_ROT + 4, ry * inv, true);
    dvOut.setFloat32(rotOffset + i * STRIDE_ROT + 8, rz * inv, true);
    dvOut.setFloat32(rotOffset + i * STRIDE_ROT + 12, rw * inv, true);

    // Scale (log → linear).
    const sx = Math.exp(src.getFloat32(base + off.scale0, true));
    const sy = Math.exp(src.getFloat32(base + off.scale1, true));
    const sz = Math.exp(src.getFloat32(base + off.scale2, true));
    dvOut.setFloat32(sclOffset + i * STRIDE_SCL + 0, sx, true);
    dvOut.setFloat32(sclOffset + i * STRIDE_SCL + 4, sy, true);
    dvOut.setFloat32(sclOffset + i * STRIDE_SCL + 8, sz, true);

    // Opacity (logit → sigmoid).
    const opacity = sigmoid(src.getFloat32(base + off.opacity, true));
    dvOut.setFloat32(opOffset + i * STRIDE_OP, opacity, true);

    // Color DC (passthrough).
    const cr = src.getFloat32(base + off.fDc0, true);
    const cg = src.getFloat32(base + off.fDc1, true);
    const cb = src.getFloat32(base + off.fDc2, true);
    dvOut.setFloat32(dcOffset + i * STRIDE_DC + 0, cr, true);
    dvOut.setFloat32(dcOffset + i * STRIDE_DC + 4, cg, true);
    dvOut.setFloat32(dcOffset + i * STRIDE_DC + 8, cb, true);
  }

  // Degenerate bbox when count == 0 — collapse to the origin so downstream
  // bbox-union code stays sane.
  if (count === 0) {
    minX = minY = minZ = 0;
    maxX = maxY = maxZ = 0;
  }

  const layout: SoaAttributeLayout = {
    positions: {
      byteOffset: posOffset,
      byteLength: count * STRIDE_POS,
      componentType: FLOAT_CT,
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
    rotations: {
      byteOffset: rotOffset,
      byteLength: count * STRIDE_ROT,
      componentType: FLOAT_CT,
    },
    scales: {
      byteOffset: sclOffset,
      byteLength: count * STRIDE_SCL,
      componentType: FLOAT_CT,
    },
    opacities: {
      byteOffset: opOffset,
      byteLength: count * STRIDE_OP,
      componentType: FLOAT_CT,
    },
    colorDC: {
      byteOffset: dcOffset,
      byteLength: count * STRIDE_DC,
      componentType: FLOAT_CT,
    },
  };

  return {
    bytes: out,
    layout,
    bbox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
  };
}

/**
 * Construct a synthetic {@link ChunkDescriptor} for one progressive batch.
 * Each batch is a new "chunk" from the renderer's perspective — append-only,
 * non-overlapping, no checksum.
 */
function makeBatchDescriptor(
  index: number,
  count: number,
  layout: SoaAttributeLayout,
  bbox: Bbox,
  byteLength: number,
): ChunkDescriptor {
  return {
    uri: `mgs2:batch:${index}`,
    byteOffset: 0,
    byteLength,
    splatCount: count,
    bbox,
    lod: 0,
    checksum: '',
    loadPriority: index,
    attributeLayout: layout,
  };
}

/**
 * Stateful uploader. Receives PLY-record batches in importance-descending
 * order and forwards them to the renderer as SoA chunks.
 *
 * The uploader also records per-batch `DecodedSplat[]` for the unit tests
 * (CPU-only path) so a headless harness without a real WebGPU device can
 * still validate progressive-decode correctness.
 *
 * Lifecycle:
 *   1. Construct with `(renderer, fieldOffsets)`.
 *   2. Call `addBatch(records, count)` once per `chunk` event.
 *   3. Inspect `totalSplats`, `decodedSplats`, or the events fired on the
 *      optional callback (`onBatchUploaded`) for progress / test gating.
 */
export interface ProgressiveUploaderInit {
  renderer: Renderer;
  fieldOffsets: PlyFieldOffsets;
  /**
   * Called after each batch has been pushed to the renderer. The callback
   * receives the cumulative splat count (`totalSplats`) and the bbox of the
   * batch just added — useful for emitting `chunkLoaded` events on the
   * viewer's event emitter.
   */
  onBatchUploaded?: (batchIndex: number, totalSplats: number, batchBbox: Bbox) => void;
  /**
   * When `true`, the uploader also keeps a CPU-decoded copy of every splat
   * (via `decodeChunkBytes`) in `decodedSplats`. Off by default to save RAM;
   * the test harness flips it on.
   */
  keepDecodedSplats?: boolean;
}

export class ProgressiveUploader {
  readonly renderer: Renderer;
  readonly fieldOffsets: PlyFieldOffsets;
  private readonly onBatch?: ProgressiveUploaderInit['onBatchUploaded'];
  private readonly keepDecoded: boolean;
  private batchIndex = 0;
  private _totalSplats = 0;
  /** Union bbox across every batch uploaded so far. */
  private bboxMin: [number, number, number] = [Infinity, Infinity, Infinity];
  private bboxMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  /** Accumulated CPU-decoded splats. Only populated when `keepDecodedSplats=true`. */
  readonly decodedSplats: DecodedSplat[] = [];

  constructor(init: ProgressiveUploaderInit) {
    this.renderer = init.renderer;
    this.fieldOffsets = init.fieldOffsets;
    this.onBatch = init.onBatchUploaded;
    this.keepDecoded = init.keepDecodedSplats ?? false;
  }

  /** Cumulative count of splats uploaded so far. */
  get totalSplats(): number {
    return this._totalSplats;
  }

  /** Current cumulative bbox (origin if no batches uploaded). */
  get currentBbox(): Bbox {
    if (!Number.isFinite(this.bboxMin[0])) {
      return { min: [0, 0, 0], max: [0, 0, 0] };
    }
    return {
      min: [this.bboxMin[0], this.bboxMin[1], this.bboxMin[2]],
      max: [this.bboxMax[0], this.bboxMax[1], this.bboxMax[2]],
    };
  }

  /**
   * Convert one progressive batch (PLY records) → SoA chunk and upload it.
   */
  addBatch(records: Uint8Array, count: number): ChunkDescriptor {
    if (count <= 0) {
      // No-op for empty batches; the fetcher should never emit them but we
      // tolerate it for callers that pipe through a transform.
      return {
        uri: `mgs2:batch:${this.batchIndex}`,
        byteOffset: 0,
        byteLength: 0,
        splatCount: 0,
        bbox: { min: [0, 0, 0], max: [0, 0, 0] },
        lod: 0,
        checksum: '',
        loadPriority: this.batchIndex,
      };
    }
    const { bytes, layout, bbox } = buildSoaChunk(records, count, this.fieldOffsets);
    const descriptor = makeBatchDescriptor(
      this.batchIndex,
      count,
      layout,
      bbox,
      bytes.byteLength,
    );
    this.renderer.uploadChunk(descriptor, bytes);
    if (this.keepDecoded) {
      const decoded = decodeChunkBytes(bytes, descriptor);
      for (const s of decoded) this.decodedSplats.push(s);
    }
    // Union bbox.
    if (count > 0) {
      if (bbox.min[0] < this.bboxMin[0]) this.bboxMin[0] = bbox.min[0];
      if (bbox.min[1] < this.bboxMin[1]) this.bboxMin[1] = bbox.min[1];
      if (bbox.min[2] < this.bboxMin[2]) this.bboxMin[2] = bbox.min[2];
      if (bbox.max[0] > this.bboxMax[0]) this.bboxMax[0] = bbox.max[0];
      if (bbox.max[1] > this.bboxMax[1]) this.bboxMax[1] = bbox.max[1];
      if (bbox.max[2] > this.bboxMax[2]) this.bboxMax[2] = bbox.max[2];
    }
    this._totalSplats += count;
    this.onBatch?.(this.batchIndex, this._totalSplats, bbox);
    this.batchIndex += 1;
    return descriptor;
  }
}
