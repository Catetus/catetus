// Catetus QAT-PLY header + body parser. Pure TypeScript, no runtime deps.
//
// On-disk format (as emitted by the Catetus encoder):
//   ply
//   format binary_little_endian 1.0
//   comment constant_field <name> <float-as-hex>          -- repeatable
//   comment quantized_field <name> int8  channels=<C> scale_b64=<base64-fp32[C]>
//   comment quantized_field <name> int4  channels=<C> packed_per_byte=2 scale_kind=per_anchor
//   element vertex <N>
//   property float  <name>          -- per-anchor fp32 column
//   property uchar  <name>_q_<i>    -- per-anchor signed-int8 column (PLY uchar slot, reinterpreted)
//   property float  <name>_scale    -- per-anchor fp32 scale (int4 path only)
//   end_header
//   <N * row-size bytes of little-endian binary>
//
// The encoder uses PLY type 'i1' (int8) when writing quantized columns; some
// PLY writers downgrade signed-1-byte to 'uchar'. We accept either uchar/uint8
// or char/int8 and reinterpret as signed int8.

export interface ConstantField {
  readonly name: string;
  readonly value: number;
}

export interface QuantizedInt8Field {
  readonly kind: "int8";
  readonly name: string; // base field name, e.g. "f_anchor_feat"
  readonly channels: number; // C
  readonly scales: Float32Array; // length = C
}

export interface QuantizedInt4Field {
  readonly kind: "int4";
  readonly name: string; // e.g. "f_offset"
  readonly channels: number; // C (logical channel count, even)
  readonly packedPerByte: 2;
  readonly scaleKind: "per_anchor";
}

export type QuantizedField = QuantizedInt8Field | QuantizedInt4Field;

export type PlyPropType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

export interface PlyProperty {
  readonly name: string;
  readonly type: PlyPropType;
}

export interface QatPlyHeader {
  readonly vertexCount: number;
  readonly properties: readonly PlyProperty[];
  readonly rowStride: number; // bytes per vertex row
  readonly headerByteLength: number; // bytes of "<header>\nend_header\n"
  readonly constants: ReadonlyMap<string, number>;
  readonly quantized: ReadonlyMap<string, QuantizedField>;
  readonly comments: readonly string[];
}

const TYPE_SIZES: Record<PlyPropType, number> = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
};

const TYPE_ALIASES: Record<string, PlyPropType> = {
  char: "char",
  int8: "char",
  uchar: "uchar",
  uint8: "uchar",
  short: "short",
  int16: "short",
  ushort: "ushort",
  uint16: "ushort",
  int: "int",
  int32: "int",
  uint: "uint",
  uint32: "uint",
  float: "float",
  float32: "float",
  double: "double",
  float64: "double",
};

/** Decode a standard (no-pad) base64 string to a Uint8Array. */
export function decodeBase64(b64: string): Uint8Array {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) & 0xff;
    return out;
  }
  // Node fallback
  const g = (globalThis as unknown) as {
    Buffer?: { from(s: string, enc: string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number } };
  };
  if (g.Buffer) {
    const buf = g.Buffer.from(b64, "base64");
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength).slice();
  }
  throw new Error("No base64 decoder available (need atob or Buffer)");
}

/** Bit-exact float.hex roundtrip used by `constant_field` comments. */
export function parseFloatHex(s: string): number {
  // Python emits e.g. "0x1.4p3", "-0x0.0p+0", "inf", "-inf", "nan".
  const t = s.trim();
  const low = t.toLowerCase();
  if (low === "inf" || low === "+inf" || low === "infinity") return Infinity;
  if (low === "-inf" || low === "-infinity") return -Infinity;
  if (low === "nan" || low === "-nan") return NaN;
  const re =
    /^([+-]?)0x([0-9a-f]+)(?:\.([0-9a-f]+))?p([+-]?\d+)$/i;
  const m = re.exec(t);
  if (m === null) {
    const f = Number(t);
    if (!Number.isNaN(f)) return f;
    throw new Error(`Cannot parse float-hex literal: ${s}`);
  }
  const sign = m[1] === "-" ? -1 : 1;
  const intPart = m[2];
  const fracPart = m[3] ?? "";
  const exp = parseInt(m[4], 10);
  // mantissa = intPart.fracPart in base 16
  let mantissa = 0;
  for (const ch of intPart) mantissa = mantissa * 16 + parseInt(ch, 16);
  let frac = 0;
  let scale = 1 / 16;
  for (const ch of fracPart) {
    frac += parseInt(ch, 16) * scale;
    scale /= 16;
  }
  return sign * (mantissa + frac) * Math.pow(2, exp);
}

function findHeaderEnd(buf: Uint8Array): number {
  // PLY header is ASCII; find "end_header\n" (LF) or "end_header\r\n".
  const needle = "end_header";
  // Search in a reasonably-sized window (PLY headers are tiny).
  const limit = Math.min(buf.length, 1 << 20); // 1 MiB cap; QAT headers fit easily
  // Convert window to string via TextDecoder for ascii-only header.
  const td = new TextDecoder("ascii");
  const text = td.decode(buf.subarray(0, limit));
  const idx = text.indexOf(needle);
  if (idx < 0) throw new Error("PLY header missing end_header");
  // Skip past end_header line terminator.
  let end = idx + needle.length;
  if (text[end] === "\r") end++;
  if (text[end] === "\n") end++;
  else throw new Error("PLY header end_header missing trailing newline");
  return end;
}

/**
 * Parse the ASCII header of a QAT-PLY buffer and return metadata + body offset.
 * The buffer can be longer than the header; we only read the header bytes.
 */
export function parseQatPlyHeader(buf: Uint8Array): QatPlyHeader {
  const headerEnd = findHeaderEnd(buf);
  const td = new TextDecoder("ascii");
  // Split header into lines using \n; tolerate \r\n.
  const headerText = td.decode(buf.subarray(0, headerEnd));
  const lines = headerText.split("\n").map((l) => (l.endsWith("\r") ? l.slice(0, -1) : l));

  if (lines.length === 0 || lines[0] !== "ply") {
    throw new Error("Not a PLY file (missing 'ply' magic on line 1)");
  }
  // We require little-endian binary; ASCII PLY is rejected. We do NOT need to
  // explicitly handle big-endian since the encoder always writes LE.
  let sawFormat = false;
  let vertexCount = -1;
  let inVertexElement = false;
  const props: PlyProperty[] = [];
  const comments: string[] = [];
  const constants = new Map<string, number>();
  const quantized = new Map<string, QuantizedField>();

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === "end_header" || line === "") continue;
    const tok = line.split(/\s+/);
    const kw = tok[0];

    if (kw === "format") {
      if (tok[1] !== "binary_little_endian") {
        throw new Error(`Unsupported PLY format: ${tok[1] ?? "?"} (need binary_little_endian)`);
      }
      sawFormat = true;
    } else if (kw === "comment") {
      // Keep raw comment text (everything after the leading "comment ").
      const raw = line.slice("comment ".length);
      comments.push(raw);
      const cparts = raw.trim().split(/\s+/);
      if (cparts.length >= 3 && cparts[0] === "constant_field") {
        const name = cparts[1];
        const value = parseFloatHex(cparts[2]);
        constants.set(name, value);
      } else if (
        cparts.length >= 3
        && cparts[0] === "quantized_field"
        && cparts[2] !== "int8"
        && cparts[2] !== "int4"
      ) {
        // v1 of the spec reserves int2/int16/etc. for future versions; we
        // MUST reject any unknown dtype rather than silently dropping it.
        throw new Error(
          `QAT-PLY v1 decoder rejects quantized_field with dtype '${cparts[2]}': ${raw}`,
        );
      } else if (cparts.length >= 5 && cparts[0] === "quantized_field" && cparts[2] === "int8") {
        const name = cparts[1];
        const kv = parseKv(cparts.slice(3));
        const channels = parseInt(kv.channels ?? "0", 10);
        const scaleB64 = kv.scale_b64;
        if (channels <= 0 || !scaleB64) {
          throw new Error(`Bad quantized_field int8 ${name}: ${raw}`);
        }
        const bytes = decodeBase64(scaleB64);
        if (bytes.byteLength !== channels * 4) {
          throw new Error(
            `quantized_field ${name}: scale_b64 has ${bytes.byteLength} bytes, expected ${channels * 4}`,
          );
        }
        const scales = new Float32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        );
        quantized.set(name, { kind: "int8", name, channels, scales });
      } else if (cparts.length >= 5 && cparts[0] === "quantized_field" && cparts[2] === "int4") {
        const name = cparts[1];
        const kv = parseKv(cparts.slice(3));
        const channels = parseInt(kv.channels ?? "0", 10);
        const packed = parseInt(kv.packed_per_byte ?? "2", 10);
        const scaleKind = kv.scale_kind ?? "per_anchor";
        if (channels <= 0 || packed !== 2 || scaleKind !== "per_anchor") {
          throw new Error(`Bad quantized_field int4 ${name}: ${raw}`);
        }
        quantized.set(name, {
          kind: "int4",
          name,
          channels,
          packedPerByte: 2,
          scaleKind: "per_anchor",
        });
      }
    } else if (kw === "element") {
      if (tok[1] === "vertex") {
        vertexCount = parseInt(tok[2], 10);
        inVertexElement = true;
      } else {
        inVertexElement = false;
      }
    } else if (kw === "property") {
      // We only model the single "vertex" element (the one QAT-PLY uses).
      // Lists are not used by the QAT format; reject defensively.
      if (tok[1] === "list") {
        throw new Error("QAT-PLY does not support 'property list' columns");
      }
      const typeRaw = tok[1];
      const propName = tok[2];
      const ty = TYPE_ALIASES[typeRaw];
      if (!ty) throw new Error(`Unknown PLY property type: ${typeRaw}`);
      if (inVertexElement) {
        props.push({ name: propName, type: ty });
      }
    }
  }

  if (!sawFormat) throw new Error("PLY header missing 'format' line");
  if (vertexCount < 0) throw new Error("PLY header missing 'element vertex N' line");

  let rowStride = 0;
  for (const p of props) rowStride += TYPE_SIZES[p.type];

  return {
    vertexCount,
    properties: props,
    rowStride,
    headerByteLength: headerEnd,
    constants,
    quantized,
    comments,
  };
}

function parseKv(toks: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const t of toks) {
    const eq = t.indexOf("=");
    if (eq < 0) continue;
    out[t.slice(0, eq)] = t.slice(eq + 1);
  }
  return out;
}
