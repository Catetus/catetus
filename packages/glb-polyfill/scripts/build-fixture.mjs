#!/usr/bin/env node
/**
 * Synthesize a tiny GLB that exercises all three SF_* extensions:
 *   - CT_zstd_split_buffer  (wraps the whole BIN with zstd frames + byte-plane split)
 *   - CT_gaussian_splatting_palette (45-D VQ codebook .shpal sidecar for SH-rest)
 *   - CT_quat_smallest3 (10-bit packed quaternions in a SCALAR/UINT accessor)
 *
 * Also writes `fixture-reference.json` containing the *exact* expected
 * decoded values (DC color, rotation, scale, etc.) so the vitest suite can
 * assert byte-for-byte fidelity against ground truth — not against itself.
 *
 * Run with:   node scripts/build-fixture.mjs
 * Output:     tests/fixtures/tiny-sf.glb + tiny-sf.glb.shpal + fixture-reference.json
 *
 * Wire format mirrors `crates/catetus-gltf/src/lib.rs` byte-for-byte; see
 * the per-section comments in this file for the producer-side reference.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { zstdCompressSync } from 'node:zlib';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, '..', 'tests', 'fixtures');
mkdirSync(OUT_DIR, { recursive: true });

const N = 8;          // splat count — small but >0 and > a single VQ centroid
const K = 4;          // palette size (must be a small power for the test)
const VQ_DIM = 45;    // per the spec
const SH_DEGREE = 3;  // palette covers full degree 3
const COMP_BITS = 10; // smallest3 component bits

/* ---------------- deterministic reference values ---------------- */

// DC color (raw SH coefficients, no SH_C0 bake) — these are what the polyfill
// returns. We pick small distinctive numbers so a fail jumps out in the diff.
const refDC = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  refDC[i * 3 + 0] = 0.10 + i * 0.05; // R
  refDC[i * 3 + 1] = 0.20 + i * 0.04; // G
  refDC[i * 3 + 2] = 0.30 + i * 0.03; // B
}
const refPositions = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  refPositions[i * 3 + 0] = i * 0.5;
  refPositions[i * 3 + 1] = -i * 0.25;
  refPositions[i * 3 + 2] = 0.1 * (i + 1);
}
const refScales = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  refScales[i * 3 + 0] = -2.0 + i * 0.1;
  refScales[i * 3 + 1] = -2.0 + i * 0.1;
  refScales[i * 3 + 2] = -2.0 + i * 0.1;
}
const refOpacity = new Float32Array(N);
for (let i = 0; i < N; i++) refOpacity[i] = 0.5 + i * 0.05;
// Quaternions — distinct rotations around X. We pick angles small enough
// that the dominant component is always W (so smallest3 tag=3 for all),
// which gives a deterministic packed layout.
const refQuat = new Float32Array(N * 4);
for (let i = 0; i < N; i++) {
  const a = (i + 1) * 0.05; // small angle
  const s = Math.sin(a * 0.5);
  refQuat[i * 4 + 0] = s; // x
  refQuat[i * 4 + 1] = 0; // y
  refQuat[i * 4 + 2] = 0; // z
  refQuat[i * 4 + 3] = Math.cos(a * 0.5); // w
}

/* ---------------- pack quaternions with smallest-3 ---------------- */

function packSmallest3(quat, bits = 10) {
  const levels = (1 << bits) - 1;
  const mask = (1 << bits) - 1;
  const sqrt2 = Math.SQRT2;
  // Normalize.
  let n2 = 0;
  for (let c = 0; c < 4; c++) n2 += quat[c] * quat[c];
  const n = Math.sqrt(Math.max(n2, 1e-24));
  const q = [quat[0] / n, quat[1] / n, quat[2] / n, quat[3] / n];
  let tag = 0;
  let best = Math.abs(q[0]);
  for (let i = 1; i < 4; i++) {
    const v = Math.abs(q[i]);
    if (v > best) { best = v; tag = i; }
  }
  const sgn = q[tag] < 0 ? -1 : 1;
  const comps = [0, 0, 0];
  let k = 0;
  for (let i = 0; i < 4; i++) {
    if (i === tag) continue;
    const v = q[i] * sgn;
    const t = Math.min(1, Math.max(0, v / sqrt2 + 0.5));
    comps[k++] = Math.round(t * levels) & mask;
  }
  // Use unsigned shifts so we don't sign-extend the tag bits.
  const packed = (
    comps[0] >>> 0
    | ((comps[1] >>> 0) << bits)
    | ((comps[2] >>> 0) << (2 * bits))
    | (((tag & 3) >>> 0) << 30)
  ) >>> 0;
  return packed;
}

const packedRot = new Uint32Array(N);
for (let i = 0; i < N; i++) {
  packedRot[i] = packSmallest3(refQuat.subarray(i * 4, i * 4 + 4), COMP_BITS);
}

/* ---------------- DC color as UBYTE-normalized per-channel ---------------- */
// Mirrors `QuantizeDCPacked` (bits<=8): per-channel min/max from the chunk,
// store as UBYTE, dequant `lo + (q/255)*(hi-lo)`.
const dcMin = [Infinity, Infinity, Infinity];
const dcMax = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < N; i++) {
  for (let c = 0; c < 3; c++) {
    const v = refDC[i * 3 + c];
    if (v < dcMin[c]) dcMin[c] = v;
    if (v > dcMax[c]) dcMax[c] = v;
  }
}
const dcBytes = new Uint8Array(N * 3);
for (let i = 0; i < N; i++) {
  for (let c = 0; c < 3; c++) {
    const span = Math.max(dcMax[c] - dcMin[c], 1e-9);
    const t = (refDC[i * 3 + c] - dcMin[c]) / span;
    dcBytes[i * 3 + c] = Math.round(Math.min(1, Math.max(0, t)) * 255);
  }
}
// Round-trip the reference so test asserts the *quantized* value, not the
// pre-quant input — that's what the polyfill is going to return.
const refDCQuant = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  for (let c = 0; c < 3; c++) {
    const t = dcBytes[i * 3 + c] / 255;
    refDCQuant[i * 3 + c] = dcMin[c] + t * (dcMax[c] - dcMin[c]);
  }
}

/* ---------------- opacity UBYTE-normalized ---------------- */
const opacityBytes = new Uint8Array(N);
for (let i = 0; i < N; i++) opacityBytes[i] = Math.round(refOpacity[i] * 255);
const refOpacityQuant = new Float32Array(N);
for (let i = 0; i < N; i++) refOpacityQuant[i] = opacityBytes[i] / 255;

/* ---------------- SCALE as raw FLOAT ---------------- */
const scaleBytes = new Uint8Array(N * 12);
{
  const dv = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
  for (let i = 0; i < N * 3; i++) dv.setFloat32(i * 4, refScales[i], true);
}

/* ---------------- POSITION as raw FLOAT ---------------- */
const posBytes = new Uint8Array(N * 12);
{
  const dv = new DataView(posBytes.buffer, posBytes.byteOffset, posBytes.byteLength);
  for (let i = 0; i < N * 3; i++) dv.setFloat32(i * 4, refPositions[i], true);
}

/* ---------------- ROTATION packed u32 ---------------- */
const rotBytes = new Uint8Array(N * 4);
{
  const dv = new DataView(rotBytes.buffer, rotBytes.byteOffset, rotBytes.byteLength);
  for (let i = 0; i < N; i++) dv.setUint32(i * 4, packedRot[i] >>> 0, true);
}

/* ---------------- assemble BIN (uncompressed) ---------------- */
// Layout: [POSITION | ROTATION | SCALE | DC | OPACITY] — each 4-byte aligned.
function pad4(len) { return (len + 3) & ~3; }
const layout = [
  { name: 'POSITION', bytes: posBytes, stride: 12 },
  { name: 'ROTATION', bytes: rotBytes, stride: 4 },
  { name: 'SCALE',    bytes: scaleBytes, stride: 12 },
  { name: 'SH0',      bytes: dcBytes, stride: 3 }, // UBYTE x3 normalized
  { name: 'OPACITY',  bytes: opacityBytes, stride: 1 },
];
const offsets = [];
let cursor = 0;
for (const v of layout) {
  cursor = pad4(cursor);
  offsets.push(cursor);
  cursor += v.bytes.byteLength;
}
const uncompressedTotal = pad4(cursor);
const uncompressed = new Uint8Array(uncompressedTotal);
for (let i = 0; i < layout.length; i++) {
  uncompressed.set(layout[i].bytes, offsets[i]);
}

/* ---------------- zstd-split: per-view frame, byte-plane split where stride>1 ---------------- */
function bytePlaneSplit(src, stride) {
  // src[i*stride + b] -> dst[b*count + i]
  const out = new Uint8Array(src.length);
  const count = src.length / stride;
  for (let i = 0; i < count; i++) {
    for (let b = 0; b < stride; b++) {
      out[b * count + i] = src[i * stride + b];
    }
  }
  return out;
}

const compChunks = [];
const views = [];
let compCursor = 0;
for (let i = 0; i < layout.length; i++) {
  const v = layout[i];
  const origLen = v.bytes.byteLength;
  const splitApplied = v.stride > 1;
  const planar = splitApplied ? bytePlaneSplit(v.bytes, v.stride) : v.bytes;
  const compressed = zstdCompressSync(planar);
  compChunks.push(compressed);
  views.push({
    compOffset: compCursor,
    compLength: compressed.byteLength,
    origOffset: offsets[i],
    origLength: origLen,
    stride: v.stride,
    splitApplied,
  });
  compCursor += compressed.byteLength;
}
const compBin = new Uint8Array(compCursor);
{
  let o = 0;
  for (const c of compChunks) { compBin.set(c, o); o += c.byteLength; }
}

/* ---------------- .shpal sidecar (palette) ---------------- */
// Per-coefficient ranges (length 45). Pick non-trivial values so the
// dequantized codebook isn't all zeros.
const ranges = new Float32Array(VQ_DIM);
for (let d = 0; d < VQ_DIM; d++) ranges[d] = 0.1 + (d % 7) * 0.02;

// Codebook: K x 45 int8 (codebookBits=8). Pattern: row c, coef d -> sin-ish.
const codebookI8 = new Int8Array(K * VQ_DIM);
for (let c = 0; c < K; c++) {
  for (let d = 0; d < VQ_DIM; d++) {
    codebookI8[c * VQ_DIM + d] = Math.round(Math.sin((c + 1) * 0.3 + d * 0.1) * 100);
  }
}
const refCodebook = new Float32Array(K * VQ_DIM);
for (let c = 0; c < K; c++) {
  for (let d = 0; d < VQ_DIM; d++) {
    refCodebook[c * VQ_DIM + d] = (codebookI8[c * VQ_DIM + d] / 127) * ranges[d];
  }
}

// Indices: per-splat palette assignment.
const indices = new Uint16Array(N);
for (let i = 0; i < N; i++) indices[i] = i % K;

// Header (matches Rust writer exactly):
//   u32 magic "SHPA" (0x53485041 LE) | u32 version=1 | u32 K | u32 N
//   u8 codebookBits | 3 bytes pad | f32[45] ranges | i8[K*45] codebook | u16[N] indices
const headerLen = 4 + 4 + 4 + 4 + 1 + 3 + ranges.byteLength;
const palLen = headerLen + codebookI8.byteLength + indices.byteLength;
const palRaw = new Uint8Array(palLen);
{
  const dv = new DataView(palRaw.buffer, palRaw.byteOffset, palRaw.byteLength);
  dv.setUint32(0, 0x53485041, true);
  dv.setUint32(4, 1, true);
  dv.setUint32(8, K, true);
  dv.setUint32(12, N, true);
  dv.setUint8(16, 8);
  // bytes 17..19 stay zero (alignment pad)
  for (let d = 0; d < VQ_DIM; d++) dv.setFloat32(20 + d * 4, ranges[d], true);
  let off = 20 + VQ_DIM * 4;
  for (let i = 0; i < codebookI8.length; i++) dv.setInt8(off + i, codebookI8[i]);
  off += codebookI8.byteLength;
  for (let i = 0; i < indices.length; i++) dv.setUint16(off + i * 2, indices[i], true);
}
const palCompressed = zstdCompressSync(palRaw);

/* ---------------- assemble glTF JSON ---------------- */

const ACC_FLOAT = 5126;
const ACC_UBYTE = 5121;
const ACC_UINT = 5125;

const bufferViewIdx = layout.map((_, i) => i);
const bufferViews = layout.map((v, i) => ({
  buffer: 0,
  byteOffset: offsets[i],
  byteLength: v.bytes.byteLength,
}));

const accessors = [
  // 0: POSITION
  {
    bufferView: bufferViewIdx[0], componentType: ACC_FLOAT, count: N, type: 'VEC3',
    min: [refPositions[0], refPositions[N * 3 - 2], 0], // dummy — overwritten below
  },
  // 1: ROTATION (SCALAR uint)
  { bufferView: bufferViewIdx[1], componentType: ACC_UINT, count: N, type: 'SCALAR' },
  // 2: SCALE
  { bufferView: bufferViewIdx[2], componentType: ACC_FLOAT, count: N, type: 'VEC3' },
  // 3: SH0 (DC) — UBYTE normalized VEC3 with per-channel min/max
  {
    bufferView: bufferViewIdx[3], componentType: ACC_UBYTE, count: N, type: 'VEC3',
    normalized: true, min: dcMin, max: dcMax,
  },
  // 4: OPACITY — UBYTE normalized SCALAR
  { bufferView: bufferViewIdx[4], componentType: ACC_UBYTE, count: N, type: 'SCALAR', normalized: true },
];
// Real POSITION min/max for the bbox.
let pxMin = Infinity, pyMin = Infinity, pzMin = Infinity;
let pxMax = -Infinity, pyMax = -Infinity, pzMax = -Infinity;
for (let i = 0; i < N; i++) {
  if (refPositions[i * 3 + 0] < pxMin) pxMin = refPositions[i * 3 + 0];
  if (refPositions[i * 3 + 1] < pyMin) pyMin = refPositions[i * 3 + 1];
  if (refPositions[i * 3 + 2] < pzMin) pzMin = refPositions[i * 3 + 2];
  if (refPositions[i * 3 + 0] > pxMax) pxMax = refPositions[i * 3 + 0];
  if (refPositions[i * 3 + 1] > pyMax) pyMax = refPositions[i * 3 + 1];
  if (refPositions[i * 3 + 2] > pzMax) pzMax = refPositions[i * 3 + 2];
}
accessors[0].min = [pxMin, pyMin, pzMin];
accessors[0].max = [pxMax, pyMax, pzMax];

const sidecarUri = 'tiny-sf.glb.shpal';
const gltf = {
  asset: { version: '2.0', generator: 'glb-polyfill/build-fixture' },
  extensionsUsed: [
    'KHR_gaussian_splatting',
    'CT_zstd_split_buffer',
    'CT_gaussian_splatting_palette',
    'CT_quat_smallest3',
  ],
  buffers: [{ byteLength: compBin.byteLength }],
  bufferViews,
  accessors,
  meshes: [{
    primitives: [{
      mode: 0, // POINTS
      attributes: {
        'POSITION': 0,
        'KHR_gaussian_splatting:ROTATION': 1,
        'KHR_gaussian_splatting:SCALE': 2,
        'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 3,
        'KHR_gaussian_splatting:OPACITY': 4,
      },
    }],
  }],
  extensions: {
    KHR_gaussian_splatting: {
      splatCount: N,
      shDegree: SH_DEGREE,
      bbox: { min: [pxMin, pyMin, pzMin], max: [pxMax, pyMax, pzMax] },
    },
    CT_zstd_split_buffer: {
      buffer: 0,
      uncompressedByteLength: uncompressedTotal,
      views,
    },
    CT_gaussian_splatting_palette: {
      uri: sidecarUri,
      shDegree: SH_DEGREE,
      paletteSize: K,
      splatCount: N,
      codebookBits: 8,
    },
    CT_quat_smallest3: {
      componentBits: COMP_BITS,
      componentType: ACC_UINT,
      layout: 'q0|q1|q2|tag',
      tagBits: 2,
    },
  },
};

/* ---------------- assemble GLB container ---------------- */

const enc = new TextEncoder();
let jsonStr = JSON.stringify(gltf);
while ((jsonStr.length + 20) % 4 !== 0) jsonStr += ' '; // pad JSON to 4-byte boundary
const jsonBytes = enc.encode(jsonStr);
// BIN must also be 4-byte aligned.
const binAligned = new Uint8Array(pad4(compBin.byteLength));
binAligned.set(compBin, 0);

const totalLen = 12 + 8 + jsonBytes.byteLength + 8 + binAligned.byteLength;
const glb = new Uint8Array(totalLen);
const dv = new DataView(glb.buffer);
dv.setUint32(0, 0x46546c67, true); // 'glTF'
dv.setUint32(4, 2, true);          // version
dv.setUint32(8, totalLen, true);
// JSON chunk
dv.setUint32(12, jsonBytes.byteLength, true);
dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
glb.set(jsonBytes, 20);
// BIN chunk
const binChunkOff = 20 + jsonBytes.byteLength;
dv.setUint32(binChunkOff, binAligned.byteLength, true);
dv.setUint32(binChunkOff + 4, 0x004e4942, true); // 'BIN\0'
glb.set(binAligned, binChunkOff + 8);

/* ---------------- write outputs ---------------- */

const glbPath = join(OUT_DIR, 'tiny-sf.glb');
const palPath = join(OUT_DIR, sidecarUri);
const refPath = join(OUT_DIR, 'fixture-reference.json');

writeFileSync(glbPath, glb);
writeFileSync(palPath, palCompressed);

// SH-rest from palette: per-splat = codebook[indices[i] * 45 ..].
// The codebook is stored channel-major (R0..R14, G0..G14, B0..B14, with the
// fixed 15-float stride per channel — matching Inria PLY's f_rest_X layout),
// but the polyfill transposes to interleaved [k][rgb] before handing the
// buffer back. Mirror that transpose here so the reference matches what
// `decodeSFExtensions` actually returns. See src/palette.ts (the post-#facf09c
// transpose) for the exact mapping.
const COEF_COUNT = 3 + 5 + 7; // degrees 1+2+3
const CHANNEL_STRIDE = 15;    // VQ_DIM / 3
const refShRest = new Float32Array(N * COEF_COUNT * 3);
for (let i = 0; i < N; i++) {
  const idx = indices[i];
  const cbBase = idx * VQ_DIM;
  for (let k = 0; k < COEF_COUNT; k++) {
    refShRest[i * COEF_COUNT * 3 + k * 3 + 0] = refCodebook[cbBase + 0 * CHANNEL_STRIDE + k];
    refShRest[i * COEF_COUNT * 3 + k * 3 + 1] = refCodebook[cbBase + 1 * CHANNEL_STRIDE + k];
    refShRest[i * COEF_COUNT * 3 + k * 3 + 2] = refCodebook[cbBase + 2 * CHANNEL_STRIDE + k];
  }
}

const ref = {
  schema: 'glb-polyfill.fixture/1',
  splatCount: N,
  shDegree: SH_DEGREE,
  paletteSize: K,
  componentBits: COMP_BITS,
  positions: Array.from(refPositions),
  rotations: Array.from(refQuat),    // unit-norm reference (pre-pack)
  scales: Array.from(refScales),
  opacities_quant: Array.from(refOpacityQuant),
  dc_color_quant: Array.from(refDCQuant),
  sh_rest: Array.from(refShRest),
  bbox: { min: [pxMin, pyMin, pzMin], max: [pxMax, pyMax, pzMax] },
};
writeFileSync(refPath, JSON.stringify(ref, null, 2));

// Sanity-check: not all-zero on the things the test will assert.
const sumAbs = (arr) => arr.reduce((s, v) => s + Math.abs(v), 0);
const dcSum = sumAbs(refDCQuant);
const shSum = sumAbs(refShRest);
const rotSum = sumAbs(refQuat);
if (dcSum === 0) throw new Error('refusing to write all-zero DC color');
if (shSum === 0) throw new Error('refusing to write all-zero SH-rest');
if (rotSum === 0) throw new Error('refusing to write all-zero rotations');

console.log(`wrote ${glbPath} (${glb.byteLength} bytes)`);
console.log(`wrote ${palPath} (${palCompressed.byteLength} bytes)`);
console.log(`wrote ${refPath}`);
console.log(`  DC color |sum|=${dcSum.toFixed(4)}`);
console.log(`  SH-rest  |sum|=${shSum.toFixed(4)}`);
console.log(`  rot      |sum|=${rotSum.toFixed(4)}`);
