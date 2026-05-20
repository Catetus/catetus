/**
 * End-to-end decode test for the SF_* polyfill.
 *
 * Loads a tiny fixture GLB authored by `scripts/build-fixture.mjs` (regenerate
 * with `node scripts/build-fixture.mjs`). The fixture exercises all three
 * extensions; the companion `fixture-reference.json` carries the *expected*
 * decoded values so this suite asserts against ground truth rather than
 * round-tripping itself.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { decodeGlb, decodeSFExtensions } from '../src/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = resolve(__dirname, 'fixtures');

interface FixtureRef {
  splatCount: number;
  shDegree: number;
  paletteSize: number;
  componentBits: number;
  positions: number[];
  rotations: number[];
  scales: number[];
  opacities_quant: number[];
  dc_color_quant: number[];
  sh_rest: number[];
  bbox: { min: [number, number, number]; max: [number, number, number] };
}

function loadGlbChunks(bytes: Uint8Array): { json: unknown; bin: Uint8Array } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  const total = dv.getUint32(8, true);
  let off = 12;
  let json: unknown = null;
  let bin = new Uint8Array(0);
  while (off + 8 <= total) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const slice = bytes.subarray(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(slice));
    } else if (type === 0x004e4942) {
      bin = slice;
    }
    off += 8 + len;
  }
  if (!json) throw new Error('GLB missing JSON chunk');
  return { json, bin };
}

describe('decodeSFExtensions — tiny-sf fixture', () => {
  const glbBytes = new Uint8Array(readFileSync(resolve(FIXTURE_DIR, 'tiny-sf.glb')));
  const palBytes = readFileSync(resolve(FIXTURE_DIR, 'tiny-sf.glb.shpal'));
  const ref = JSON.parse(readFileSync(resolve(FIXTURE_DIR, 'fixture-reference.json'), 'utf8')) as FixtureRef;
  const { json, bin } = loadGlbChunks(glbBytes);

  const decoded = decodeSFExtensions(json, bin, {
    'tiny-sf.glb.shpal': palBytes.buffer.slice(
      palBytes.byteOffset,
      palBytes.byteOffset + palBytes.byteLength,
    ),
  });

  it('applies all three SF extensions', () => {
    expect(decoded.extensionsApplied.zstdSplitBuffer).toBe(true);
    expect(decoded.extensionsApplied.palette).toBe(true);
    expect(decoded.extensionsApplied.smallest3).toBe(true);
  });

  it('reports CT_log_quant_attrs status (absent in tiny fixture)', () => {
    // The build-fixture.mjs script does not emit CT_log_quant_attrs, so this
    // fixture round-trips through linear-space accessor min/max. The
    // provenance flag should therefore be false.
    expect(decoded.extensionsApplied.logQuantAttrs).toBe(false);
  });

  it('returns the correct splat count and SH degree', () => {
    expect(decoded.count).toBe(ref.splatCount);
    expect(decoded.shDegree).toBe(ref.shDegree);
  });

  it('decodes POSITION bit-exactly (FLOAT, no quantization)', () => {
    expect(decoded.positions.length).toBe(ref.positions.length);
    for (let i = 0; i < ref.positions.length; i++) {
      expect(decoded.positions[i]).toBeCloseTo(ref.positions[i], 6);
    }
  });

  it('decodes DC color non-zero and matches quantized reference', () => {
    expect(decoded.dcRaw.length).toBe(ref.dc_color_quant.length);
    // The deprecated `dc_color` alias must point at the same buffer.
    expect(decoded.dc_color).toBe(decoded.dcRaw);
    // Hard assert: not all-zero (the failure mode the user has been burned on).
    let absSum = 0;
    for (let i = 0; i < decoded.dcRaw.length; i++) absSum += Math.abs(decoded.dcRaw[i]);
    expect(absSum).toBeGreaterThan(0);
    // Compare to the quantized round-trip reference within UBYTE quantization
    // tolerance (≈ 1/255 of channel range — fixture range ~0.65 so eps ~0.003).
    for (let i = 0; i < ref.dc_color_quant.length; i++) {
      expect(decoded.dcRaw[i]).toBeCloseTo(ref.dc_color_quant[i], 5);
    }
  });

  it('decodes ROTATION via CT_quat_smallest3 within quantization eps', () => {
    expect(decoded.rotations.length).toBe(ref.rotations.length);
    // 10-bit smallest-3 quantization error is ≈ sqrt(2)/(2^10) ≈ 0.0014 per
    // component; we use 5e-3 as a comfortable bound.
    for (let i = 0; i < ref.rotations.length; i++) {
      expect(decoded.rotations[i]).toBeCloseTo(ref.rotations[i], 2);
    }
    // Each rotation should remain unit-length.
    for (let i = 0; i < decoded.count; i++) {
      const x = decoded.rotations[i * 4 + 0];
      const y = decoded.rotations[i * 4 + 1];
      const z = decoded.rotations[i * 4 + 2];
      const w = decoded.rotations[i * 4 + 3];
      expect(Math.sqrt(x * x + y * y + z * z + w * w)).toBeCloseTo(1, 3);
    }
  });

  it('decodes SCALE bit-exactly (FLOAT)', () => {
    for (let i = 0; i < ref.scales.length; i++) {
      expect(decoded.scales[i]).toBeCloseTo(ref.scales[i], 6);
    }
  });

  it('decodes OPACITY within UBYTE eps', () => {
    for (let i = 0; i < ref.opacities_quant.length; i++) {
      expect(decoded.opacities[i]).toBeCloseTo(ref.opacities_quant[i], 5);
    }
  });

  it('rebuilds SH-rest from CT_gaussian_splatting_palette codebook', () => {
    expect(decoded.sh_rest).not.toBeNull();
    const sh = decoded.sh_rest as Float32Array;
    expect(sh.length).toBe(ref.sh_rest.length);
    let absSum = 0;
    for (let i = 0; i < sh.length; i++) absSum += Math.abs(sh[i]);
    expect(absSum).toBeGreaterThan(0);
    // Int8 codebook quant; per-coef ranges in [0.1, 0.22] → eps ≈ 1.7e-3.
    for (let i = 0; i < ref.sh_rest.length; i++) {
      expect(sh[i]).toBeCloseTo(ref.sh_rest[i], 4);
    }
  });

  it('exposes a bbox', () => {
    expect(decoded.bbox).not.toBeNull();
    const bb = decoded.bbox!;
    expect(bb.min[0]).toBeCloseTo(ref.bbox.min[0], 6);
    expect(bb.max[2]).toBeCloseTo(ref.bbox.max[2], 6);
  });
});

describe('decodeSFExtensions — CT_log_quant_attrs path', () => {
  // Build a tiny in-memory GLB that mirrors what the Rust writer produces
  // when `--log-quant-attrs` is on: SCALE stored as ln(scale) in UBYTE with
  // the accessor min/max in log-space, OPACITY stored as logit(opacity) in
  // UBYTE with min/max in logit-space (clamped to ±OPACITY_LOGIT_RANGE=12
  // by the Rust encoder). Asserts:
  //   1. polyfill detects CT_log_quant_attrs and records provenance via
  //      extensionsApplied.logQuantAttrs
  //   2. opacity dequant uses accessor min/max then eagerly applies sigmoid
  //      → public `opacities` is linear in [0, 1]
  //   3. scale dequant eagerly applies exp → public `scales` is linear
  //   4. round-trip stats land in the same ballpark as the Rust-decoded PLY
  //      from experiments/decoder-conventions-fix/RESULT.md
  function buildLogQuantGlb() {
    const N = 8;
    // Synthetic "bonsai-like" log-space scales spanning the GT range
    // (RESULT.md: scale_log range [-19.3, 1.20], mean ≈ -4.85).
    const logScales = new Float32Array([
      -19.3, -8.1, -4.85, -3.2, -1.7, -0.6, 0.4, 1.2,
      -18.0, -7.5, -4.85, -3.5, -2.0, -0.5, 0.5, 1.0,
      -17.0, -7.0, -4.85, -3.8, -2.5, -0.4, 0.3, 0.8,
    ]); // length N*3 = 24
    // Synthetic logit-space opacities clamped to ±12 (RESULT.md: opacity_logit
    // mean ≈ -0.52, std ≈ 4.1, range clamp [-12, 12]).
    const logitOps = new Float32Array([
      -7.1, -4.0, -1.0, -0.5, 0.0, 0.6, 4.0, 12.0,
    ]);

    // Quantize to UBYTE with affine dequant: q = round((v - lo) / (hi - lo) * 255)
    function quantU8(values: Float32Array, lo: number[], hi: number[], stride: number): Uint8Array {
      const out = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) {
        const c = i % stride;
        const range = hi[c] - lo[c];
        const t = range === 0 ? 0 : (values[i] - lo[c]) / range;
        out[i] = Math.max(0, Math.min(255, Math.round(t * 255)));
      }
      return out;
    }

    // Per-axis log-scale min/max derived from the data (matches Rust
    // `chunk_scale_bbox_in(_, log_space=true)`).
    const sMin = [Infinity, Infinity, Infinity];
    const sMax = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        const v = logScales[i * 3 + c];
        if (v < sMin[c]) sMin[c] = v;
        if (v > sMax[c]) sMax[c] = v;
      }
    }
    const sQuant = quantU8(logScales, sMin, sMax, 3);

    let oMin = Infinity, oMax = -Infinity;
    for (let i = 0; i < N; i++) {
      if (logitOps[i] < oMin) oMin = logitOps[i];
      if (logitOps[i] > oMax) oMax = logitOps[i];
    }
    const oQuant = quantU8(logitOps, [oMin], [oMax], 1);

    // Required attributes that aren't under test: positions, rotations,
    // SH0 DC. Encoded as plain FLOAT to keep the fixture small.
    const positions = new Float32Array([
      0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 1, 1, 1,
    ]);
    const rotations = new Float32Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    ]);
    const dc = new Float32Array(N * 3).fill(0.1);

    function pad(b: Uint8Array): Uint8Array {
      const p = (4 - (b.byteLength & 3)) & 3;
      if (p === 0) return b;
      const out = new Uint8Array(b.byteLength + p);
      out.set(b);
      return out;
    }
    const parts = [
      pad(new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength)),
      pad(new Uint8Array(rotations.buffer, rotations.byteOffset, rotations.byteLength)),
      pad(sQuant),
      pad(oQuant),
      pad(new Uint8Array(dc.buffer, dc.byteOffset, dc.byteLength)),
    ];
    const offsets: number[] = [];
    let cur = 0;
    for (const p of parts) {
      offsets.push(cur);
      cur += p.byteLength;
    }
    const bin = new Uint8Array(cur);
    for (let i = 0; i < parts.length; i++) bin.set(parts[i], offsets[i]);
    const bv = (i: number, len: number) => ({ buffer: 0, byteOffset: offsets[i], byteLength: len });
    const gltf = {
      asset: { version: '2.0' },
      buffers: [{ byteLength: bin.byteLength }],
      bufferViews: [
        bv(0, N * 12),
        bv(1, N * 16),
        bv(2, N * 3),
        bv(3, N),
        bv(4, N * 12),
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: N, type: 'VEC3', min: [-1, -1, -1], max: [1, 1, 1] },
        { bufferView: 1, componentType: 5126, count: N, type: 'VEC4' },
        // Log-space SCALE accessor — min/max are in log-space.
        { bufferView: 2, componentType: 5121, normalized: true, count: N, type: 'VEC3', min: sMin, max: sMax },
        // Logit-space OPACITY accessor — min/max are in logit-space.
        { bufferView: 3, componentType: 5121, normalized: true, count: N, type: 'SCALAR', min: [oMin], max: [oMax] },
        { bufferView: 4, componentType: 5126, count: N, type: 'VEC3' },
      ],
      meshes: [{
        primitives: [{
          mode: 0,
          attributes: {
            'POSITION': 0,
            'KHR_gaussian_splatting:ROTATION': 1,
            'KHR_gaussian_splatting:SCALE': 2,
            'KHR_gaussian_splatting:OPACITY': 3,
            'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 4,
          },
        }],
      }],
      extensionsUsed: ['KHR_gaussian_splatting', 'CT_log_quant_attrs'],
      extensions: {
        KHR_gaussian_splatting: { splatCount: N, shDegree: 0 },
        CT_log_quant_attrs: { scale: 'ln', opacity: 'logit' },
      },
    };
    return { gltf, bin, N, logScales, logitOps, sMin, sMax, oMin, oMax };
  }

  it('detects CT_log_quant_attrs and records provenance', () => {
    const { gltf, bin } = buildLogQuantGlb();
    const decoded = decodeSFExtensions(gltf, bin);
    expect(decoded.extensionsApplied.logQuantAttrs).toBe(true);
  });

  it('dequantizes opacity in logit-space using accessor min/max, then eagerly applies sigmoid → linear [0,1]', () => {
    const { gltf, bin, N, logitOps, oMin, oMax } = buildLogQuantGlb();
    const decoded = decodeSFExtensions(gltf, bin);
    // The polyfill now eagerly applies sigmoid, so the public `opacities`
    // are LINEAR in [0, 1]. Compare to the expected linear values.
    const step = (oMax - oMin) / 255;
    expect(decoded.opacities.length).toBe(N);
    for (let i = 0; i < N; i++) {
      const linearExpected = 1 / (1 + Math.exp(-logitOps[i]));
      // Sigmoid is Lipschitz with slope ≤ 1/4 globally, so the sigmoid of a
      // logit-space value with UBYTE quant step `step` has error ≤ step/4
      // plus a small numerical pad. We assert on that bound directly; a
      // toBeCloseTo(_, 2) here would be tighter than the math allows.
      expect(Math.abs(decoded.opacities[i] - linearExpected)).toBeLessThanOrEqual(step / 4 + 1e-3);
      // And: always inside [0, 1] — closes the foot-gun that produced the
      // bonsai blob (task #113) when callers forgot the flag.
      expect(decoded.opacities[i]).toBeGreaterThanOrEqual(0);
      expect(decoded.opacities[i]).toBeLessThanOrEqual(1);
    }
  });

  it('eagerly applies exp() to SCALE so public `scales` is always linear', () => {
    const { gltf, bin, N, logScales, sMin, sMax } = buildLogQuantGlb();
    const decoded = decodeSFExtensions(gltf, bin);
    expect(decoded.scales.length).toBe(N * 3);
    for (let i = 0; i < N * 3; i++) {
      const c = i % 3;
      const step = (sMax[c] - sMin[c]) / 255;
      const linearExpected = Math.exp(logScales[i]);
      // exp on a quantized log value has multiplicative error up to e^step.
      // For our range that's <= ~e^0.08 ≈ 1.083 → relative error 8 %.
      const tol = Math.max(linearExpected * (Math.exp(step) - 1), 1e-9);
      expect(Math.abs(decoded.scales[i] - linearExpected)).toBeLessThanOrEqual(tol + 1e-9);
      // All scales linear → non-negative.
      expect(decoded.scales[i]).toBeGreaterThanOrEqual(0);
    }
    // Deep-tail scale -19.3 → exp(-19.3) ≈ 4.1e-9. Must NOT be clamped to a
    // log-space floor (task #86 bug family).
    expect(decoded.scales[0]).toBeCloseTo(Math.exp(-19.3), 8);
    expect(decoded.scales[0]).toBeGreaterThan(0);
    expect(decoded.scales[0]).toBeLessThan(1e-7);
  });

  it('matches the Rust-decoded bonsai stats from experiments/decoder-conventions-fix/RESULT.md', () => {
    // RESULT.md (SF AFTER column):
    //   opacity_logit mean -0.528, std 4.106
    //   scale_log     mean -4.849, std 1.848
    // The polyfill now eagerly de-logs / de-logits, so check that the means
    // of `ln(scale)` / `logit(opacity)` recovered from the linear public
    // outputs land in the Rust ballpark.
    const { gltf, bin, N, logScales, logitOps } = buildLogQuantGlb();
    const decoded = decodeSFExtensions(gltf, bin);

    function mean(a: ArrayLike<number>): number {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i];
      return s / a.length;
    }
    const logOf = (arr: Float32Array) => Float32Array.from(arr, Math.log);
    const logitOf = (arr: Float32Array) =>
      // Clamp to keep finite when opacity is exactly 0 or 1 after quant.
      Float32Array.from(arr, (x) => {
        const c = Math.min(Math.max(x, 1e-6), 1 - 1e-6);
        return Math.log(c / (1 - c));
      });

    const polyfillScaleMean = mean(logOf(decoded.scales));
    const polyfillOpMean = mean(logitOf(decoded.opacities));
    const inputScaleMean = mean(logScales);
    const inputOpMean = mean(logitOps);

    expect(Math.abs(polyfillScaleMean - inputScaleMean)).toBeLessThan(0.1);
    expect(Math.abs(polyfillOpMean - inputOpMean)).toBeLessThan(0.1);

    expect(polyfillScaleMean).toBeGreaterThan(-10);
    expect(polyfillScaleMean).toBeLessThan(0);
    expect(polyfillOpMean).toBeGreaterThan(-3);
    expect(polyfillOpMean).toBeLessThan(3);
    void N;
  });

  it('provenance flag is false when CT_log_quant_attrs extension is absent', () => {
    const { gltf, bin } = buildLogQuantGlb();
    // Drop the extension to simulate a legacy GLB.
    const stripped = JSON.parse(JSON.stringify(gltf));
    delete stripped.extensions.CT_log_quant_attrs;
    stripped.extensionsUsed = stripped.extensionsUsed.filter(
      (e: string) => e !== 'CT_log_quant_attrs',
    );
    const decoded = decodeSFExtensions(stripped, bin);
    expect(decoded.extensionsApplied.logQuantAttrs).toBe(false);
  });
});

describe('decodeSFExtensions — no sidecar required when palette absent', () => {
  it('works on plain (uncompressed) GLBs too', () => {
    // Build a minimal in-memory GLB with NO SF_* extensions to make sure the
    // happy path doesn't accidentally require a sidecar.
    const N = 2;
    const pos = new Float32Array([0, 0, 0, 1, 1, 1]);
    const rot = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1]);
    const scl = new Float32Array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]);
    const op = new Float32Array([1, 0.5]);
    const dc = new Float32Array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    function pad(b: Uint8Array): Uint8Array {
      const p = (4 - (b.byteLength & 3)) & 3;
      if (p === 0) return b;
      const out = new Uint8Array(b.byteLength + p);
      out.set(b);
      return out;
    }
    const parts = [pos, rot, scl, op, dc].map((a) => pad(new Uint8Array(a.buffer)));
    const offsets: number[] = [];
    let cur = 0;
    for (const p of parts) {
      offsets.push(cur);
      cur += p.byteLength;
    }
    const bin = new Uint8Array(cur);
    for (let i = 0; i < parts.length; i++) bin.set(parts[i], offsets[i]);
    const bv = (i: number, len: number) => ({ buffer: 0, byteOffset: offsets[i], byteLength: len });
    const gltf = {
      asset: { version: '2.0' },
      buffers: [{ byteLength: bin.byteLength }],
      bufferViews: [
        bv(0, N * 12), bv(1, N * 16), bv(2, N * 12), bv(3, N * 4), bv(4, N * 12),
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: N, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 1] },
        { bufferView: 1, componentType: 5126, count: N, type: 'VEC4' },
        { bufferView: 2, componentType: 5126, count: N, type: 'VEC3' },
        { bufferView: 3, componentType: 5126, count: N, type: 'SCALAR' },
        { bufferView: 4, componentType: 5126, count: N, type: 'VEC3' },
      ],
      meshes: [{
        primitives: [{
          mode: 0,
          attributes: {
            'POSITION': 0,
            'KHR_gaussian_splatting:ROTATION': 1,
            'KHR_gaussian_splatting:SCALE': 2,
            'KHR_gaussian_splatting:OPACITY': 3,
            'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 4,
          },
        }],
      }],
      extensionsUsed: ['KHR_gaussian_splatting'],
      extensions: { KHR_gaussian_splatting: { splatCount: N, shDegree: 0 } },
    };
    const decoded = decodeSFExtensions(gltf, bin);
    expect(decoded.count).toBe(N);
    expect(decoded.extensionsApplied.zstdSplitBuffer).toBe(false);
    expect(decoded.extensionsApplied.smallest3).toBe(false);
    expect(decoded.extensionsApplied.palette).toBe(false);
    expect(decoded.dcRaw[0]).toBeCloseTo(0.3, 6);
    expect(decoded.dcRaw[5]).toBeCloseTo(0.8, 6);
    expect(decoded.sh_rest).toBeNull();
  });
});

describe('decodeGlb — one-shot wrapper (no sidecars)', () => {
  it('decodes the tiny no-palette fixture from raw GLB bytes', () => {
    // Build the same plain GLB inline so we don't depend on a fixture being
    // regenerated against the latest decoder.
    const N = 2;
    const pos = new Float32Array([0, 0, 0, 1, 1, 1]);
    const rot = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1]);
    const scl = new Float32Array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]);
    const op = new Float32Array([1, 0.5]);
    const dc = new Float32Array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    function pad4(b: Uint8Array): Uint8Array {
      const p = (4 - (b.byteLength & 3)) & 3;
      if (p === 0) return b;
      const out = new Uint8Array(b.byteLength + p);
      out.set(b);
      return out;
    }
    const parts = [pos, rot, scl, op, dc].map((a) => pad4(new Uint8Array(a.buffer)));
    const offs: number[] = [];
    let cur = 0;
    for (const p of parts) { offs.push(cur); cur += p.byteLength; }
    const binChunk = new Uint8Array(cur);
    for (let i = 0; i < parts.length; i++) binChunk.set(parts[i], offs[i]);
    const bv = (i: number, len: number) => ({ buffer: 0, byteOffset: offs[i], byteLength: len });
    const gltf = {
      asset: { version: '2.0' },
      buffers: [{ byteLength: binChunk.byteLength }],
      bufferViews: [bv(0, N*12), bv(1, N*16), bv(2, N*12), bv(3, N*4), bv(4, N*12)],
      accessors: [
        { bufferView: 0, componentType: 5126, count: N, type: 'VEC3', min: [0,0,0], max: [1,1,1] },
        { bufferView: 1, componentType: 5126, count: N, type: 'VEC4' },
        { bufferView: 2, componentType: 5126, count: N, type: 'VEC3' },
        { bufferView: 3, componentType: 5126, count: N, type: 'SCALAR' },
        { bufferView: 4, componentType: 5126, count: N, type: 'VEC3' },
      ],
      meshes: [{ primitives: [{ mode: 0, attributes: {
        'POSITION': 0,
        'KHR_gaussian_splatting:ROTATION': 1,
        'KHR_gaussian_splatting:SCALE': 2,
        'KHR_gaussian_splatting:OPACITY': 3,
        'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 4,
      } }] }],
      extensionsUsed: ['KHR_gaussian_splatting'],
      extensions: { KHR_gaussian_splatting: { splatCount: N, shDegree: 0 } },
    };

    // Wrap the JSON + BIN as a real GLB byte stream.
    const jsonStr = JSON.stringify(gltf);
    // pad json to 4
    const jsonPadLen = (4 - (jsonStr.length & 3)) & 3;
    const jsonBytes = new Uint8Array(jsonStr.length + jsonPadLen);
    new TextEncoder().encodeInto(jsonStr, jsonBytes);
    for (let i = jsonStr.length; i < jsonBytes.length; i++) jsonBytes[i] = 0x20; // space
    const binPadLen = (4 - (binChunk.byteLength & 3)) & 3;
    const binPadded = new Uint8Array(binChunk.byteLength + binPadLen);
    binPadded.set(binChunk);
    const totalLen = 12 + 8 + jsonBytes.length + 8 + binPadded.length;
    const glb = new Uint8Array(totalLen);
    const dv = new DataView(glb.buffer);
    dv.setUint32(0, 0x46546c67, true);  // 'glTF'
    dv.setUint32(4, 2, true);
    dv.setUint32(8, totalLen, true);
    dv.setUint32(12, jsonBytes.length, true);
    dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
    glb.set(jsonBytes, 20);
    dv.setUint32(20 + jsonBytes.length, binPadded.length, true);
    dv.setUint32(20 + jsonBytes.length + 4, 0x004e4942, true); // 'BIN\0'
    glb.set(binPadded, 20 + jsonBytes.length + 8);

    const decoded = decodeGlb(glb);
    expect(decoded.count).toBe(N);
    expect(decoded.dcRaw[0]).toBeCloseTo(0.3, 6);
    expect(decoded.dcRaw[5]).toBeCloseTo(0.8, 6);
    expect(decoded.opacities[0]).toBeCloseTo(1, 6);
    expect(decoded.opacities[1]).toBeCloseTo(0.5, 6);
    expect(decoded.scales[0]).toBeCloseTo(0.1, 6);
    expect(decoded.extensionsApplied.logQuantAttrs).toBe(false);
  });

  it('throws a clear error on non-GLB bytes', () => {
    expect(() => decodeGlb(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))
      .toThrow(/bad GLB magic/);
  });
});
