// Pure-TypeScript reference encoder for synthetic QAT-PLY fixtures.
//
// This is the inverse of the loader path and is used by all three plugin test
// suites to assert parity. Output is byte-identical to what the Python
// `save_ply` path would produce for the same inputs (modulo plyfile's own
// header line ordering, which we mirror).
//
// We do NOT try to be a full PLY writer. The fixture supports exactly the
// subset Catetus emits: vertex element, fp32 + int8 columns, header
// comments for constant_field and quantized_field.

export interface FixtureColumnFloat {
  readonly kind: "float";
  readonly name: string;
  readonly values: Float32Array; // length N
}

export interface FixtureColumnInt8 {
  readonly kind: "int8";
  readonly name: string;
  readonly values: Int8Array; // length N
}

export type FixtureColumn = FixtureColumnFloat | FixtureColumnInt8;

export interface FixtureSpec {
  readonly vertexCount: number;
  readonly columns: readonly FixtureColumn[];
  /** `comment constant_field <name> <hex>` lines. */
  readonly constants?: ReadonlyMap<string, number>;
  /** Per int8 quantized field. */
  readonly quantizedInt8?: readonly {
    name: string;
    channels: number;
    scales: Float32Array;
  }[];
  /** Per int4 quantized field — channels MUST be even, packed_per_byte=2. */
  readonly quantizedInt4?: readonly {
    name: string;
    channels: number;
  }[];
}

/** Float-to-hex (Python float.hex compatible) for constant_field literals. */
export function floatToHex(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0x0.0p+0" : "0x0.0p+0";
  // Use IEEE-754 fp64 bit pattern to extract sign, exponent, mantissa.
  const buf = new ArrayBuffer(8);
  new Float64Array(buf)[0] = x;
  const hi = new Uint32Array(buf)[1];
  const lo = new Uint32Array(buf)[0];
  const sign = hi >>> 31;
  let exp = (hi >>> 20) & 0x7ff;
  const mantHi = hi & 0xfffff;
  const mantLo = lo;
  let leading: string;
  let unbiased: number;
  if (exp === 0) {
    // Denormal: 0.<mantissa>p-1022
    leading = "0";
    unbiased = -1022;
  } else {
    leading = "1";
    unbiased = exp - 1023;
  }
  // Format mantissa as 13 hex digits (52 bits = 5 + 8 hex digits).
  let mantHex = (mantHi.toString(16).padStart(5, "0") + mantLo.toString(16).padStart(8, "0"));
  // Trim trailing zeros but keep at least one digit.
  mantHex = mantHex.replace(/0+$/, "") || "0";
  const sgnStr = sign ? "-" : "";
  const expStr = unbiased >= 0 ? `+${unbiased}` : `${unbiased}`;
  return `${sgnStr}0x${leading}.${mantHex}p${expStr}`;
}

/** Standard base64 of a fp32 Float32Array. */
export function floatArrayToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  if (typeof btoa === "function") {
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }
  // Node fallback
  const g = (globalThis as unknown) as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } };
  if (g.Buffer) return g.Buffer.from(bytes).toString("base64");
  throw new Error("No base64 encoder available");
}

/** Encode a synthetic QAT-PLY fixture to a Uint8Array. */
export function encodeFixture(spec: FixtureSpec): Uint8Array {
  const lines: string[] = [];
  lines.push("ply");
  lines.push("format binary_little_endian 1.0");
  // Constants
  if (spec.constants) {
    const names = Array.from(spec.constants.keys()).sort();
    for (const name of names) {
      lines.push(`comment constant_field ${name} ${floatToHex(spec.constants.get(name)!)}`);
    }
  }
  // Quantized field declarations
  for (const q of spec.quantizedInt8 ?? []) {
    if (q.scales.length !== q.channels) {
      throw new Error(`quantized int8 ${q.name}: scales length ${q.scales.length} != channels ${q.channels}`);
    }
    const b64 = floatArrayToBase64(q.scales);
    lines.push(`comment quantized_field ${q.name} int8 channels=${q.channels} scale_b64=${b64}`);
  }
  for (const q of spec.quantizedInt4 ?? []) {
    if ((q.channels & 1) !== 0) {
      throw new Error(`int4 channels must be even, got ${q.channels}`);
    }
    lines.push(
      `comment quantized_field ${q.name} int4 channels=${q.channels} packed_per_byte=2 scale_kind=per_anchor`,
    );
  }
  lines.push(`element vertex ${spec.vertexCount}`);
  // Property declarations: 'char' for int8 (matches plyfile 'i1'); 'float' for fp32.
  for (const c of spec.columns) {
    if (c.kind === "float") lines.push(`property float ${c.name}`);
    else if (c.kind === "int8") lines.push(`property char ${c.name}`);
  }
  lines.push("end_header");
  const headerText = lines.join("\n") + "\n";
  const headerBytes = new TextEncoder().encode(headerText);

  let rowStride = 0;
  for (const c of spec.columns) rowStride += c.kind === "float" ? 4 : 1;

  const bodyBytes = new Uint8Array(rowStride * spec.vertexCount);
  const view = new DataView(bodyBytes.buffer);
  const offsets: number[] = [];
  let cursor = 0;
  for (const c of spec.columns) {
    offsets.push(cursor);
    cursor += c.kind === "float" ? 4 : 1;
  }

  for (let n = 0; n < spec.vertexCount; n++) {
    const rowBase = n * rowStride;
    for (let ci = 0; ci < spec.columns.length; ci++) {
      const c = spec.columns[ci];
      const off = rowBase + offsets[ci];
      if (c.kind === "float") {
        view.setFloat32(off, c.values[n], true);
      } else {
        bodyBytes[off] = c.values[n] & 0xff;
      }
    }
  }

  const out = new Uint8Array(headerBytes.byteLength + bodyBytes.byteLength);
  out.set(headerBytes, 0);
  out.set(bodyBytes, headerBytes.byteLength);
  return out;
}

/**
 * Build a deterministic synthetic QAT-PLY fixture covering all 3 codec paths:
 *   - fp32 position columns (x, y, z)
 *   - constant_field columns (nx, ny, nz, opacity, scale_0/1/2, rot_0..3) set to known values
 *   - fp32 SH DC columns f_dc_{0,1,2} for color verification
 *   - int8 quantized field `f_anchor_feat` with channels=4 + per-channel scales
 *   - int4 quantized field `f_offset` with channels=6 + per-anchor scale column
 *
 * Returns the encoded bytes plus the ground-truth dequantized arrays for parity
 * assertions. Same inputs always yield byte-identical output.
 */
export interface SyntheticFixture {
  readonly bytes: Uint8Array;
  readonly N: number;
  readonly expectedPositions: Float32Array; // length 3N
  readonly expectedAnchorFeat: Float32Array; // (N, 4) row-major
  readonly expectedAnchorFeatChannels: number;
  readonly expectedOffset: Float32Array; // (N, 6) row-major
  readonly expectedOffsetChannels: number;
  readonly expectedDcColors: Float32Array; // length 3N, clamp01(0.5 + SH_C0 * f_dc)
}

const SH_C0 = 0.28209479177387814;

export function buildSyntheticFixture(N = 32): SyntheticFixture {
  // Deterministic LCG so the fixture is reproducible across platforms.
  let state = 0xdeadbeef >>> 0;
  const rand = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };

  const posX = new Float32Array(N);
  const posY = new Float32Array(N);
  const posZ = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    posX[i] = Math.fround((rand() - 0.5) * 2);
    posY[i] = Math.fround((rand() - 0.5) * 2);
    posZ[i] = Math.fround((rand() - 0.5) * 2);
  }
  const fDc0 = new Float32Array(N);
  const fDc1 = new Float32Array(N);
  const fDc2 = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    fDc0[i] = Math.fround(rand() - 0.5);
    fDc1[i] = Math.fround(rand() - 0.5);
    fDc2[i] = Math.fround(rand() - 0.5);
  }

  // int8 anchor_feat with 4 channels, per-channel scales.
  const featC = 4;
  const featScales = new Float32Array([0.001, 0.0025, 0.005, 0.01]);
  const featInt8: Int8Array[] = [];
  for (let c = 0; c < featC; c++) {
    const col = new Int8Array(N);
    for (let i = 0; i < N; i++) col[i] = Math.floor(rand() * 256) - 128;
    featInt8.push(col);
  }

  // int4 f_offset with 6 channels (3 bytes packed), per-anchor scale.
  const offC = 6;
  const offScale = new Float32Array(N);
  for (let i = 0; i < N; i++) offScale[i] = Math.fround(0.01 + rand() * 0.05);
  // Per-anchor signed nibbles in [-8, 7], stored unsigned [0, 15] = signed + 8.
  // We'll build the unsigned nibble array, pack to bytes, AND compute the
  // expected dequantized output up front.
  const offNibU = new Uint8Array(N * offC); // unsigned, length N*C
  for (let i = 0; i < N * offC; i++) offNibU[i] = Math.floor(rand() * 16);
  const expectedOffset = new Float32Array(N * offC);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < offC; c++) {
      const u = offNibU[n * offC + c];
      const s = u - 8;
      expectedOffset[n * offC + c] = Math.fround(s * offScale[n]);
    }
  }
  // Pack to bytes.
  const nBytes = offC / 2;
  const offPacked: Int8Array[] = [];
  for (let b = 0; b < nBytes; b++) offPacked.push(new Int8Array(N));
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < nBytes; b++) {
      const low = offNibU[n * offC + 2 * b] & 0xf;
      const high = offNibU[n * offC + 2 * b + 1] & 0xf;
      const byte = ((high << 4) | low) & 0xff;
      offPacked[b][n] = byte < 128 ? byte : byte - 256;
    }
  }

  // Constants (header-only).
  const constants = new Map<string, number>([
    ["nx", 0.0],
    ["ny", 0.0],
    ["nz", 0.0],
    ["opacity", -2.5],
    ["scale_0", -3.0],
    ["scale_1", -3.0],
    ["scale_2", -3.0],
    ["rot_0", 1.0],
    ["rot_1", 0.0],
    ["rot_2", 0.0],
    ["rot_3", 0.0],
  ]);

  const columns: FixtureColumn[] = [];
  columns.push({ kind: "float", name: "x", values: posX });
  columns.push({ kind: "float", name: "y", values: posY });
  columns.push({ kind: "float", name: "z", values: posZ });
  columns.push({ kind: "float", name: "f_dc_0", values: fDc0 });
  columns.push({ kind: "float", name: "f_dc_1", values: fDc1 });
  columns.push({ kind: "float", name: "f_dc_2", values: fDc2 });
  for (let c = 0; c < featC; c++) {
    columns.push({ kind: "int8", name: `f_anchor_feat_q_${c}`, values: featInt8[c] });
  }
  for (let b = 0; b < nBytes; b++) {
    columns.push({ kind: "int8", name: `f_offset_q_${b}`, values: offPacked[b] });
  }
  columns.push({ kind: "float", name: "f_offset_scale", values: offScale });

  const bytes = encodeFixture({
    vertexCount: N,
    columns,
    constants,
    quantizedInt8: [{ name: "f_anchor_feat", channels: featC, scales: featScales }],
    quantizedInt4: [{ name: "f_offset", channels: offC }],
  });

  const expectedPositions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    expectedPositions[3 * i] = posX[i];
    expectedPositions[3 * i + 1] = posY[i];
    expectedPositions[3 * i + 2] = posZ[i];
  }
  const expectedAnchorFeat = new Float32Array(N * featC);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < featC; c++) {
      expectedAnchorFeat[n * featC + c] = Math.fround(featInt8[c][n] * featScales[c]);
    }
  }
  const expectedDcColors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    expectedDcColors[3 * i] = clamp01(0.5 + SH_C0 * fDc0[i]);
    expectedDcColors[3 * i + 1] = clamp01(0.5 + SH_C0 * fDc1[i]);
    expectedDcColors[3 * i + 2] = clamp01(0.5 + SH_C0 * fDc2[i]);
  }

  return {
    bytes,
    N,
    expectedPositions,
    expectedAnchorFeat,
    expectedAnchorFeatChannels: featC,
    expectedOffset: expectedOffset,
    expectedOffsetChannels: offC,
    expectedDcColors,
  };
}

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}
