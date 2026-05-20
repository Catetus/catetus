/**
 * Minimal binary-PLY decoder for Inria-format 3DGS chunk files emitted by
 * `catetus lodge build`. Produces canonical SoA bytes + layout in the
 * same shape `ComputeDecodePipeline.uploadChunk` accepts (matches
 * `packages/viewer/bench/scripts/ply-to-soa.py` for byte-level parity).
 *
 * Required properties (all `float`):
 *   x, y, z, opacity,
 *   scale_0, scale_1, scale_2,
 *   rot_0, rot_1, rot_2, rot_3,
 *   f_dc_0, f_dc_1, f_dc_2.
 *
 * Extra properties (e.g. `nx`, `f_rest_*`) are tolerated and skipped — the
 * decoder only consumes the canonical render-shell set today (Phase A.2
 * does not need SH > 0).
 *
 * Conversions match the bench / training-side conventions:
 *   - `scale_*` (stored log-scale) → `exp()` (linear scale)
 *   - `opacity` (stored logit)      → `sigmoid()` (0..1)
 *   - `rot_*`                       → normalized to unit quaternion
 *   - `f_dc_*` (Inria SH-DC0)       → linear color via
 *                                     `0.5 + 0.28209479177 * f_dc`
 */

import type { Bbox, SoaAttributeLayout } from '../manifest.js';

/** SH-DC0 → linear RGB normalisation constant (Inria convention). */
const SH_C0 = 0.28209479177387814;

const REQUIRED = [
  'x',
  'y',
  'z',
  'opacity',
  'scale_0',
  'scale_1',
  'scale_2',
  'rot_0',
  'rot_1',
  'rot_2',
  'rot_3',
  'f_dc_0',
  'f_dc_1',
  'f_dc_2',
] as const;

/** Result of {@link decodePlyToSoa}. */
export interface DecodedPlyChunk {
  /** Splat count (= number of vertices in the PLY). */
  splatCount: number;
  /** Tight SoA bytes:  positions | rotations | scales | opacities | colorDC. */
  bytes: Uint8Array;
  /** SoA layout — offsets are relative to `bytes`. */
  layout: SoaAttributeLayout;
  /** Tight position-only AABB. */
  bbox: Bbox;
}

/**
 * Decode a binary Inria PLY into canonical SoA splat bytes.
 *
 * @throws Error whose message starts with `ply_invalid:` on a malformed
 *   header or short body.
 */
export function decodePlyToSoa(plyBytes: Uint8Array): DecodedPlyChunk {
  const header = parseHeader(plyBytes);
  const n = header.vertexCount;
  const stride = header.props.length * 4; // all float, 4 bytes
  const bodyBytes = n * stride;
  if (header.bodyOffset + bodyBytes > plyBytes.byteLength) {
    throw new Error(
      `ply_invalid: body truncated (need ${bodyBytes} from offset ${header.bodyOffset}, have ${
        plyBytes.byteLength - header.bodyOffset
      })`,
    );
  }
  const body = new DataView(
    plyBytes.buffer,
    plyBytes.byteOffset + header.bodyOffset,
    bodyBytes,
  );

  // Property indices.
  const ix = (name: string): number => {
    const i = header.props.indexOf(name);
    if (i < 0) throw new Error(`ply_invalid: missing required property '${name}'`);
    return i;
  };
  const idx: Record<(typeof REQUIRED)[number], number> = {
    x: ix('x'),
    y: ix('y'),
    z: ix('z'),
    opacity: ix('opacity'),
    scale_0: ix('scale_0'),
    scale_1: ix('scale_1'),
    scale_2: ix('scale_2'),
    rot_0: ix('rot_0'),
    rot_1: ix('rot_1'),
    rot_2: ix('rot_2'),
    rot_3: ix('rot_3'),
    f_dc_0: ix('f_dc_0'),
    f_dc_1: ix('f_dc_1'),
    f_dc_2: ix('f_dc_2'),
  };

  // Output SoA layout — same order as ply-to-soa.py.
  const posBytes = n * 12;
  const rotBytes = n * 16;
  const scaleBytes = n * 12;
  const opBytes = n * 4;
  const dcBytes = n * 12;
  const total = posBytes + rotBytes + scaleBytes + opBytes + dcBytes;
  const out = new Uint8Array(total);
  const outDv = new DataView(out.buffer);
  const posOff = 0;
  const rotOff = posOff + posBytes;
  const scaleOff = rotOff + rotBytes;
  const opOff = scaleOff + scaleBytes;
  const dcOff = opOff + opBytes;

  let bbMin: [number, number, number] = [Infinity, Infinity, Infinity];
  let bbMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  // Row pull helpers (offsets within row in bytes).
  const off = (i: number): number => i * 4;
  for (let v = 0; v < n; v++) {
    const rowStart = v * stride;

    const x = body.getFloat32(rowStart + off(idx.x), true);
    const y = body.getFloat32(rowStart + off(idx.y), true);
    const z = body.getFloat32(rowStart + off(idx.z), true);
    outDv.setFloat32(posOff + v * 12 + 0, x, true);
    outDv.setFloat32(posOff + v * 12 + 4, y, true);
    outDv.setFloat32(posOff + v * 12 + 8, z, true);
    if (x < bbMin[0]) bbMin[0] = x;
    if (y < bbMin[1]) bbMin[1] = y;
    if (z < bbMin[2]) bbMin[2] = z;
    if (x > bbMax[0]) bbMax[0] = x;
    if (y > bbMax[1]) bbMax[1] = y;
    if (z > bbMax[2]) bbMax[2] = z;

    let r0 = body.getFloat32(rowStart + off(idx.rot_0), true);
    let r1 = body.getFloat32(rowStart + off(idx.rot_1), true);
    let r2 = body.getFloat32(rowStart + off(idx.rot_2), true);
    let r3 = body.getFloat32(rowStart + off(idx.rot_3), true);
    const rn = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3) || 1;
    r0 /= rn; r1 /= rn; r2 /= rn; r3 /= rn;
    outDv.setFloat32(rotOff + v * 16 + 0, r0, true);
    outDv.setFloat32(rotOff + v * 16 + 4, r1, true);
    outDv.setFloat32(rotOff + v * 16 + 8, r2, true);
    outDv.setFloat32(rotOff + v * 16 + 12, r3, true);

    const s0 = Math.exp(body.getFloat32(rowStart + off(idx.scale_0), true));
    const s1 = Math.exp(body.getFloat32(rowStart + off(idx.scale_1), true));
    const s2 = Math.exp(body.getFloat32(rowStart + off(idx.scale_2), true));
    outDv.setFloat32(scaleOff + v * 12 + 0, s0, true);
    outDv.setFloat32(scaleOff + v * 12 + 4, s1, true);
    outDv.setFloat32(scaleOff + v * 12 + 8, s2, true);

    const logitOp = body.getFloat32(rowStart + off(idx.opacity), true);
    const op = 1.0 / (1.0 + Math.exp(-logitOp));
    outDv.setFloat32(opOff + v * 4, op, true);

    const dc0 = 0.5 + SH_C0 * body.getFloat32(rowStart + off(idx.f_dc_0), true);
    const dc1 = 0.5 + SH_C0 * body.getFloat32(rowStart + off(idx.f_dc_1), true);
    const dc2 = 0.5 + SH_C0 * body.getFloat32(rowStart + off(idx.f_dc_2), true);
    outDv.setFloat32(dcOff + v * 12 + 0, dc0, true);
    outDv.setFloat32(dcOff + v * 12 + 4, dc1, true);
    outDv.setFloat32(dcOff + v * 12 + 8, dc2, true);
  }

  const layout: SoaAttributeLayout = {
    positions: { byteOffset: posOff, byteLength: posBytes, componentType: 5126 },
    rotations: { byteOffset: rotOff, byteLength: rotBytes, componentType: 5126 },
    scales: { byteOffset: scaleOff, byteLength: scaleBytes, componentType: 5126 },
    opacities: { byteOffset: opOff, byteLength: opBytes, componentType: 5126 },
    colorDC: { byteOffset: dcOff, byteLength: dcBytes, componentType: 5126 },
  };

  if (!Number.isFinite(bbMin[0])) {
    bbMin = [0, 0, 0];
    bbMax = [0, 0, 0];
  }

  return {
    splatCount: n,
    bytes: out,
    layout,
    bbox: { min: bbMin, max: bbMax },
  };
}

interface PlyHeader {
  vertexCount: number;
  /** Names of `property float` columns, in declaration order. */
  props: string[];
  /** Byte offset of the first vertex row (immediately after `end_header\n`). */
  bodyOffset: number;
}

function parseHeader(bytes: Uint8Array): PlyHeader {
  // Header is ASCII-only; scan up to `end_header\n` (max ~256 properties).
  const MAX_HEADER = 1 << 16; // 64 KiB safety cap
  const scanEnd = Math.min(bytes.byteLength, MAX_HEADER);
  let cursor = 0;
  const lines: string[] = [];
  while (cursor < scanEnd) {
    // Locate next '\n'.
    let nl = cursor;
    while (nl < scanEnd && bytes[nl] !== 0x0a) nl++;
    if (nl >= scanEnd) throw new Error('ply_invalid: header overflowed scan window');
    // Strip trailing '\r' if present.
    const lineEnd = nl > cursor && bytes[nl - 1] === 0x0d ? nl - 1 : nl;
    const line = decodeAscii(bytes, cursor, lineEnd);
    lines.push(line);
    cursor = nl + 1;
    if (line === 'end_header') break;
    if (lines.length > MAX_HEADER) throw new Error('ply_invalid: header too long');
  }

  if (lines.length === 0 || lines[0] !== 'ply') {
    throw new Error('ply_invalid: missing magic "ply"');
  }
  const fmtLine = lines.find((l) => l.startsWith('format'));
  if (!fmtLine || !fmtLine.startsWith('format binary_little_endian')) {
    throw new Error(`ply_invalid: unsupported format (${fmtLine ?? 'none'})`);
  }

  let vertexCount = -1;
  const props: string[] = [];
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith('element vertex ')) {
      vertexCount = parseInt(line.split(' ')[2] ?? '-1', 10);
      inVertex = true;
      continue;
    }
    if (line.startsWith('element ')) {
      inVertex = false;
      continue;
    }
    if (inVertex && line.startsWith('property float ')) {
      const name = line.split(' ')[2];
      if (typeof name === 'string') props.push(name);
    }
  }
  if (vertexCount < 0) throw new Error('ply_invalid: no vertex count');

  for (const r of REQUIRED) {
    if (!props.includes(r)) throw new Error(`ply_invalid: missing property '${r}'`);
  }

  return {
    vertexCount,
    props,
    bodyOffset: cursor,
  };
}

function decodeAscii(b: Uint8Array, from: number, to: number): string {
  // ASCII-only inside PLY headers; build the string manually to dodge
  // TextDecoder allocation overhead inside the hot per-line loop.
  let s = '';
  for (let i = from; i < to; i++) s += String.fromCharCode(b[i] ?? 0);
  return s;
}
