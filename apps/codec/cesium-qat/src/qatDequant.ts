// Catetus QAT-PLY body reader + int8/int4 dequantization.
// All math is fp32 single-precision via Float32Array round-trips, matching the
// numpy reference: feat_fp32[n, c] = int8[n, c] * scales[c].

import type {
  QatPlyHeader,
  PlyProperty,
  QuantizedInt8Field,
  QuantizedInt4Field,
} from "./qatHeaderParser.js";

export interface ColumnLayout {
  readonly name: string;
  readonly byteOffsetInRow: number;
  readonly byteSize: number;
  readonly type: PlyProperty["type"];
}

/** Compute per-column byte offsets within a vertex row. */
export function computeColumnLayout(
  header: QatPlyHeader,
): ReadonlyMap<string, ColumnLayout> {
  const layout = new Map<string, ColumnLayout>();
  let offset = 0;
  for (const p of header.properties) {
    const size = sizeOf(p.type);
    layout.set(p.name, {
      name: p.name,
      byteOffsetInRow: offset,
      byteSize: size,
      type: p.type,
    });
    offset += size;
  }
  return layout;
}

function sizeOf(t: PlyProperty["type"]): number {
  switch (t) {
    case "char":
    case "uchar":
      return 1;
    case "short":
    case "ushort":
      return 2;
    case "int":
    case "uint":
    case "float":
      return 4;
    case "double":
      return 8;
  }
}

/**
 * Read a single fp32 column from the binary body into a Float32Array of length N.
 * If the column is missing but `header.constants` has it, fill with the constant.
 */
export function readFloatColumn(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
  name: string,
): Float32Array {
  const N = header.vertexCount;
  const out = new Float32Array(N);
  const col = layout.get(name);
  if (col) {
    if (col.type !== "float") {
      throw new Error(`Column ${name} is type ${col.type}, expected float`);
    }
    const stride = header.rowStride;
    const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
    let off = col.byteOffsetInRow;
    for (let i = 0; i < N; i++) {
      out[i] = view.getFloat32(off, true);
      off += stride;
    }
    return out;
  }
  const c = header.constants.get(name);
  if (c !== undefined) {
    out.fill(c);
    return out;
  }
  throw new Error(`PLY missing column '${name}' (no per-vertex property and no constant_field)`);
}

/** Read an int8 column (PLY type 'char' or 'uchar' both accepted) into Int8Array. */
export function readInt8Column(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
  name: string,
): Int8Array {
  const col = layout.get(name);
  if (!col) throw new Error(`PLY missing int8 column '${name}'`);
  if (col.byteSize !== 1) throw new Error(`Column ${name} is ${col.byteSize}B, expected 1B`);
  const N = header.vertexCount;
  const out = new Int8Array(N);
  const stride = header.rowStride;
  let off = col.byteOffsetInRow;
  for (let i = 0; i < N; i++) {
    // Reinterpret the raw byte as signed int8 regardless of declared char/uchar.
    const u = body[off];
    out[i] = u < 128 ? u : u - 256;
    off += stride;
  }
  return out;
}

/**
 * Decode an int8-quantized field into a dense (N, C) Float32Array (row-major).
 * Matches numpy: `out[n, c] = int8[n, c] * scales[c]`.
 *
 * Expects per-vertex properties named `${field.name}_q_0` .. `${field.name}_q_{C-1}`.
 */
export function decodeQuantizedInt8Field(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
  field: QuantizedInt8Field,
): Float32Array {
  const { name, channels, scales } = field;
  if (scales.length !== channels) {
    throw new Error(`${name}: scales length ${scales.length} != channels ${channels}`);
  }
  const N = header.vertexCount;
  const out = new Float32Array(N * channels);
  const stride = header.rowStride;

  // Pre-resolve byte offsets for each q column so the inner loop is tight.
  const offsets = new Int32Array(channels);
  for (let c = 0; c < channels; c++) {
    const col = layout.get(`${name}_q_${c}`);
    if (!col) throw new Error(`PLY missing '${name}_q_${c}'`);
    if (col.byteSize !== 1) {
      throw new Error(`${name}_q_${c} is ${col.byteSize}B, expected 1B`);
    }
    offsets[c] = col.byteOffsetInRow;
  }

  for (let n = 0; n < N; n++) {
    const rowBase = n * stride;
    const outBase = n * channels;
    for (let c = 0; c < channels; c++) {
      const u = body[rowBase + offsets[c]];
      const s = u < 128 ? u : u - 256;
      // fp32 mul: cast through Math.fround to match numpy float32 semantics.
      out[outBase + c] = Math.fround(s * scales[c]);
    }
  }
  return out;
}

/**
 * Decode an int4-quantized field into a dense (N, C) Float32Array.
 * Encoding: each byte holds two unsigned nibbles (low = even-indexed channel
 * 2i, high = 2i+1). Each unsigned nibble u in [0, 15] dequantizes as
 *   out[n, c] = (u - 8) * f_offset_scale[n].
 */
export function decodeQuantizedInt4Field(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
  field: QuantizedInt4Field,
  /** name of the per-anchor fp32 scale column (default `${field.name}_scale`). */
  scaleColumnName?: string,
): Float32Array {
  const { name, channels } = field;
  // ceil(C/2): odd channel counts pad the last byte's high nibble (writer
  // must zero it; readers MUST ignore it per spec section 3.2).
  const nBytes = (channels + 1) >> 1;
  const N = header.vertexCount;
  const out = new Float32Array(N * channels);
  const stride = header.rowStride;

  const offsets = new Int32Array(nBytes);
  for (let b = 0; b < nBytes; b++) {
    const col = layout.get(`${name}_q_${b}`);
    if (!col) throw new Error(`PLY missing '${name}_q_${b}'`);
    if (col.byteSize !== 1) {
      throw new Error(`${name}_q_${b} is ${col.byteSize}B, expected 1B`);
    }
    offsets[b] = col.byteOffsetInRow;
  }

  const scaleName = scaleColumnName ?? `${name}_scale`;
  const scales = readFloatColumn(header, body, layout, scaleName);

  for (let n = 0; n < N; n++) {
    const rowBase = n * stride;
    const outBase = n * channels;
    const s = scales[n];
    for (let b = 0; b < nBytes; b++) {
      const byte = body[rowBase + offsets[b]];
      const low = byte & 0xf;
      const high = (byte >> 4) & 0xf;
      const cLow = 2 * b;
      const cHigh = cLow + 1;
      out[outBase + cLow] = Math.fround((low - 8) * s);
      if (cHigh < channels) {
        out[outBase + cHigh] = Math.fround((high - 8) * s);
      }
      // else: odd-C tail nibble is unused per spec.
    }
  }
  return out;
}

/**
 * Read x/y/z position columns into a tightly packed Float32Array of length 3N
 * (xyz xyz xyz ...). Convenience for BufferGeometry / Babylon / Cesium.
 */
export function readPositions(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
): Float32Array {
  const N = header.vertexCount;
  const out = new Float32Array(N * 3);
  const xs = readFloatColumn(header, body, layout, "x");
  const ys = readFloatColumn(header, body, layout, "y");
  const zs = readFloatColumn(header, body, layout, "z");
  for (let i = 0; i < N; i++) {
    out[3 * i] = xs[i];
    out[3 * i + 1] = ys[i];
    out[3 * i + 2] = zs[i];
  }
  return out;
}

/**
 * Vanilla-Inria-3DGS DC SH coefficient -> sRGB-ish unit color.
 *   color = clamp(0.5 + SH_C0 * f_dc, 0, 1)  with  SH_C0 = 0.28209479177387814
 * This is the standard splat viewer mapping for the f_dc_{0,1,2} channels.
 */
export const SH_C0 = 0.28209479177387814;

export function readDcColors(
  header: QatPlyHeader,
  body: Uint8Array,
  layout: ReadonlyMap<string, ColumnLayout>,
): Float32Array | null {
  if (!layout.has("f_dc_0") && !header.constants.has("f_dc_0")) return null;
  const N = header.vertexCount;
  const out = new Float32Array(N * 3);
  const r = readFloatColumn(header, body, layout, "f_dc_0");
  const g = readFloatColumn(header, body, layout, "f_dc_1");
  const b = readFloatColumn(header, body, layout, "f_dc_2");
  for (let i = 0; i < N; i++) {
    out[3 * i] = clamp01(0.5 + SH_C0 * r[i]);
    out[3 * i + 1] = clamp01(0.5 + SH_C0 * g[i]);
    out[3 * i + 2] = clamp01(0.5 + SH_C0 * b[i]);
  }
  return out;
}

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}
