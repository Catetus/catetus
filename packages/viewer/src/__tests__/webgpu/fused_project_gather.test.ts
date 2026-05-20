// SPDX-License-Identifier: Apache-2.0
/**
 * Parity test for the fused vs separate project + gather pipelines.
 *
 * The non-fused (legacy) path is:
 *     cs_project(splats, cam)         → instUnsorted[i], keys[i]=depth, indices[i]=i
 *     radix_sort(keys, indices)       → indices[]: sorted order
 *     cs_gather(instUnsorted, indices) → instanceBuffer[i] = instUnsorted[indices[i]]
 *
 * The fused path is:
 *     cs_keygen(splats, cam)          → keys[i]=depth, indices[i]=i  (no instance write)
 *     radix_sort(keys, indices)       → indices[]: sorted order
 *     cs_project_gather(splats, indices, cam)
 *                                     → instanceBuffer[i] = project(splats[indices[i]], cam)
 *
 * The two final instanceBuffer outputs MUST be bit-identical when the
 * projection math is byte-for-byte the same in both code paths. Byte-stable
 * rendering across paths is a core invariant of the viewer.
 *
 * Strategy:
 *   1. Lock the WGSL projection-math regions to byte-equality (cs_project in
 *      decode.wgsl vs cs_project_gather in cs_project_gather.wgsl). This is a
 *      static guarantee — no GPU required.
 *   2. Run a TS reference implementation of both pipelines over the same
 *      synthetic scene + camera and assert byte-identical final buffers.
 *
 * This catches drift between the two WGSL files at unit-test time, without
 * requiring a WebGPU device (which isn't available in node/vitest).
 */
import { describe, expect, it } from 'vitest';
import {
  DECODE_WGSL,
  PROJECT_GATHER_WGSL,
} from '../../webgpu/index.js';
import { buildSyntheticScene } from '../../../bench/compute-decode.bench.js';
import { decodeSplatsSoa } from '../../renderer/base.js';

/* ------------------------------------------------------------------ */
/* (1) Static check: projection math is bit-identical in both shaders. */
/* ------------------------------------------------------------------ */

/** Extract a substring between two anchor lines (anchors excluded). */
function slice(src: string, fromAnchor: string, toAnchor: string): string {
  const a = src.indexOf(fromAnchor);
  const b = src.indexOf(toAnchor, a);
  if (a < 0 || b < 0) throw new Error(`anchor not found: ${fromAnchor} or ${toAnchor}`);
  return src.slice(a + fromAnchor.length, b);
}

/**
 * Normalise a WGSL chunk for comparison: strip comments, blank lines, leading
 * whitespace, and the differing parameter / global-name prefixes. We're
 * comparing the *algorithm*, not the naming.
 */
function normalize(s: string): string {
  return s
    // strip line comments
    .replace(/\/\/[^\n]*/g, '')
    // collapse whitespace
    .replace(/\s+/g, ' ')
    // unify the two binding-namespace prefixes used by the kernels
    //   decode.wgsl uses `pu.` for uniforms, `p_splats` for splats.
    //   cs_project_gather.wgsl uses `gu.` and `g_splats`.
    .replace(/\bpu\./g, 'U.')
    .replace(/\bgu\./g, 'U.')
    .replace(/\bp_splats\b/g, 'SPLATS')
    .replace(/\bg_splats\b/g, 'SPLATS')
    // covariance fn is named differently
    .replace(/\bcov3d_fg\b/g, 'cov3d')
    // The non-fused path reads splats by global index `i`; the fused path
    // reads them by the sort-order index `splat_idx`. Normalise so the math
    // body compares as identical.
    .replace(/let s = SPLATS\[i\];/g, 'let s = SPLATS[IDX];')
    .replace(/let s = SPLATS\[splat_idx\];/g, 'let s = SPLATS[IDX];')
    .trim();
}

describe('cs_project vs cs_project_gather: WGSL math parity', () => {
  it('project body is byte-equal across the two shader files', () => {
    // Slice the projection math from the first `let pos = s.pos.xyz;` (just
    // *after* the splat read, which differs between the two paths) through
    // `inst.color = vec4<f32>(s.color.rgb, opacity);` (the last common line
    // before each writes to its own output buffer).
    // Anchor on `let opacity = s.pos.w;` — only present in cs_project_gather
    // (not in cs_keygen) and in cs_project (not in cs_decode). This pins the
    // slice to the projection body in both files.
    const projBody = slice(
      DECODE_WGSL,
      'let opacity = s.pos.w;',
      'inst.color = vec4<f32>(s.color.rgb, opacity);',
    );
    const fusedBody = slice(
      PROJECT_GATHER_WGSL,
      'let opacity = s.pos.w;',
      'inst.color = vec4<f32>(s.color.rgb, opacity);',
    );

    const a = normalize(projBody);
    const b = normalize(fusedBody);
    expect(a).toBe(b);
    // Sanity: body is non-trivial so we know the slice grabbed real content.
    expect(a.length).toBeGreaterThan(400);
  });

  it('cs_keygen depth math matches cs_project depth math', () => {
    // Both compute `vz = view[0][2]*x + view[1][2]*y + view[2][2]*z + view[3][2]`,
    // `depth = -vz`, `dpos = max(depth, 0)`, key = 0xffffffff - bitcast(dpos).
    const keygenSlice = slice(
      PROJECT_GATHER_WGSL,
      'fn cs_keygen(@builtin(global_invocation_id) gid : vec3<u32>) {',
      'fn cs_project_gather',
    );
    expect(keygenSlice).toMatch(/view\[0\]\[2\]\s*\*\s*pos\.x/);
    expect(keygenSlice).toMatch(/view\[1\]\[2\]\s*\*\s*pos\.y/);
    expect(keygenSlice).toMatch(/view\[2\]\[2\]\s*\*\s*pos\.z/);
    expect(keygenSlice).toMatch(/0xffffffffu\s*-\s*kd/);

    const projSlice = slice(
      DECODE_WGSL,
      'fn cs_project(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '// Sort key:',
    );
    expect(projSlice).toMatch(/view\[0\]\[2\]\s*\*\s*pos\.x/);
    expect(projSlice).toMatch(/view\[1\]\[2\]\s*\*\s*pos\.y/);
    expect(projSlice).toMatch(/view\[2\]\[2\]\s*\*\s*pos\.z/);
  });

  it('cs_project_gather binding layout matches the orchestrator', () => {
    // The orchestrator binds (0 splats RO, 1 indices RO, 2 inst_out RW, 3 uniform).
    expect(PROJECT_GATHER_WGSL).toMatch(
      /@group\(0\) @binding\(0\) var<storage, read>\s+g_splats/,
    );
    expect(PROJECT_GATHER_WGSL).toMatch(
      /@group\(0\) @binding\(1\) var<storage, read>\s+g_indices/,
    );
    expect(PROJECT_GATHER_WGSL).toMatch(
      /@group\(0\) @binding\(2\) var<storage, read_write> g_inst_out/,
    );
    expect(PROJECT_GATHER_WGSL).toMatch(
      /@group\(0\) @binding\(3\) var<uniform>\s+gu/,
    );
  });
});

/* ------------------------------------------------------------------ */
/* (2) Behavioural check: TS reference of both pipelines is identical. */
/* ------------------------------------------------------------------ */

interface Splat {
  pos: [number, number, number];
  opacity: number;
  scale: [number, number, number];
  rot: [number, number, number, number];
  color: [number, number, number];
}

interface Cam {
  view: Float32Array;  // 16, column-major
  viewProj: Float32Array;  // 16
  focal: [number, number];
  viewport: [number, number];
}

/** Pure TS port of cov3d() from decode.wgsl. */
function cov3d(s: [number, number, number], q: [number, number, number, number]): number[] {
  const ql = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-8);
  const x = q[0] / ql, y = q[1] / ql, z = q[2] / ql, w = q[3] / ql;
  const xx = x*x, yy = y*y, zz = z*z;
  const xy = x*y, xz = x*z, yz = y*z;
  const wx = w*x, wy = w*y, wz = w*z;
  const r00 = 1 - 2*(yy + zz);
  const r10 = 2*(xy + wz);
  const r20 = 2*(xz - wy);
  const r01 = 2*(xy - wz);
  const r11 = 1 - 2*(xx + zz);
  const r21 = 2*(yz + wx);
  const r02 = 2*(xz + wy);
  const r12 = 2*(yz - wx);
  const r22 = 1 - 2*(xx + yy);
  const sx = s[0], sy = s[1], sz = s[2];
  const m00 = r00*sx, m10 = r10*sx, m20 = r20*sx;
  const m01 = r01*sy, m11 = r11*sy, m21 = r21*sy;
  const m02 = r02*sz, m12 = r12*sz, m22 = r22*sz;
  return [
    m00*m00 + m01*m01 + m02*m02,
    m00*m10 + m01*m11 + m02*m12,
    m00*m20 + m01*m21 + m02*m22,
    m10*m10 + m11*m11 + m12*m12,
    m10*m20 + m11*m21 + m12*m22,
    m20*m20 + m21*m21 + m22*m22,
  ];
}

/** Pure TS port of the cs_project per-splat math; returns 12 floats. */
function projectOne(s: Splat, cam: Cam): Float32Array {
  const V = cam.view;
  const VP = cam.viewProj;
  const [px, py, pz] = s.pos;
  // clip = VP * (pos, 1)   — column-major mat4: clip.k = VP[0][k]*x + VP[1][k]*y + VP[2][k]*z + VP[3][k].
  const cx = VP[0]*px + VP[4]*py + VP[8]*pz  + VP[12];
  const cy = VP[1]*px + VP[5]*py + VP[9]*pz  + VP[13];
  const cz = VP[2]*px + VP[6]*py + VP[10]*pz + VP[14];
  const cw = VP[3]*px + VP[7]*py + VP[11]*pz + VP[15];
  const invW = Math.abs(cw) > 1e-12 ? 1 / cw : 1;
  const ndcx = cx * invW, ndcy = cy * invW, ndcz = cz * invW;

  // view-space depth: row 2 of view dotted with (pos, 1) → -z.
  // Column-major: view[col][row]; row 2 of column c is V[c*4 + 2].
  const vz = V[2]*px + V[6]*py + V[10]*pz + V[14];
  const depth = -vz;
  const behind = depth <= 0;

  const Vc = cov3d(s.scale, s.rot);
  // w0/w1 = first/second rows of view's 3×3 rotation (cols 0..2, row 0..1).
  const w0x = V[0],  w0y = V[4],  w0z = V[8];
  const w1x = V[1],  w1y = V[5],  w1z = V[9];
  const a0x = w0x*Vc[0] + w0y*Vc[1] + w0z*Vc[2];
  const a0y = w0x*Vc[1] + w0y*Vc[3] + w0z*Vc[4];
  const a0z = w0x*Vc[2] + w0y*Vc[4] + w0z*Vc[5];
  const a1x = w1x*Vc[0] + w1y*Vc[1] + w1z*Vc[2];
  const a1y = w1x*Vc[1] + w1y*Vc[3] + w1z*Vc[4];
  const a1z = w1x*Vc[2] + w1y*Vc[4] + w1z*Vc[5];
  const vxx = a0x*w0x + a0y*w0y + a0z*w0z;
  const vxy = a0x*w1x + a0y*w1y + a0z*w1z;
  const vyy = a1x*w1x + a1y*w1y + a1z*w1z;

  const zMax = Math.max(Math.abs(depth), 1e-4);
  const jx = cam.focal[0] / zMax;
  const jy = cam.focal[1] / zMax;
  const reg = 0.3;
  const c00 = jx*jx*vxx + reg;
  const c01 = jx*jy*vxy;
  const c11 = jy*jy*vyy + reg;

  const halfTrace = 0.5 * (c00 + c11);
  const det = Math.max(c00*c11 - c01*c01, 0);
  const term = Math.sqrt(Math.max(halfTrace*halfTrace - det, 0));
  const lambdaMax = halfTrace + term;
  let radius = 3 * Math.sqrt(Math.max(lambdaMax, 0));
  if (behind) radius = 0;

  const out = new Float32Array(12);
  const zc = Math.min(1, Math.max(0, ndcz));
  if (behind) {
    out[0] = 2; out[1] = 2; out[2] = 1; out[3] = 1;       // clip_pos
    out[4] = 1; out[5] = 0; out[6] = 1; out[7] = 0;       // cov (radius=0)
  } else {
    out[0] = ndcx; out[1] = ndcy; out[2] = zc; out[3] = cw;
    out[4] = c00;  out[5] = c01;  out[6] = c11; out[7] = radius;
  }
  out[8]  = s.color[0];
  out[9]  = s.color[1];
  out[10] = s.color[2];
  out[11] = s.opacity;
  return out;
}

/** Compute the back-to-front depth sort key the WGSL uses. */
function depthKey(s: Splat, cam: Cam): number {
  const V = cam.view;
  const [px, py, pz] = s.pos;
  const vz = V[2]*px + V[6]*py + V[10]*pz + V[14];
  const depth = -vz;
  const dpos = Math.max(depth, 0);
  // 0xffffffff - bitcast<u32>(dpos)
  const ab = new ArrayBuffer(4);
  new Float32Array(ab)[0] = dpos;
  const kd = new Uint32Array(ab)[0]!;
  return (0xffffffff - kd) >>> 0;
}

/** Stable LSD ascending sort on u32 keys carrying u32 values — matches WGSL. */
function radixSortPaired(keys: Uint32Array, values: Uint32Array): { keys: Uint32Array; values: Uint32Array } {
  const n = keys.length;
  let kIn = new Uint32Array(keys);
  let vIn = new Uint32Array(values);
  let kOut = new Uint32Array(n);
  let vOut = new Uint32Array(n);
  for (let pass = 0; pass < 8; pass++) {
    const shift = pass * 4;
    const hist = new Uint32Array(16);
    for (let i = 0; i < n; i++) hist[(kIn[i]! >>> shift) & 0xf]!++;
    const off = new Uint32Array(16);
    for (let i = 1; i < 16; i++) off[i] = off[i - 1]! + hist[i - 1]!;
    const cur = new Uint32Array(off);
    for (let i = 0; i < n; i++) {
      const b = (kIn[i]! >>> shift) & 0xf;
      const d = cur[b]!++;
      kOut[d] = kIn[i]!;
      vOut[d] = vIn[i]!;
    }
    [kIn, kOut] = [kOut, kIn];
    [vIn, vOut] = [vOut, vIn];
  }
  return { keys: kIn, values: vIn };
}

/* Simulate the non-fused path: project all → sort → gather. */
function runSeparate(splats: Splat[], cam: Cam): Float32Array {
  const n = splats.length;
  const unsorted = new Float32Array(n * 12);
  const keys = new Uint32Array(n);
  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const inst = projectOne(splats[i]!, cam);
    unsorted.set(inst, i * 12);
    keys[i] = depthKey(splats[i]!, cam);
    indices[i] = i;
  }
  const sorted = radixSortPaired(keys, indices);
  const out = new Float32Array(n * 12);
  for (let i = 0; i < n; i++) {
    const src = sorted.values[i]! * 12;
    out.set(unsorted.subarray(src, src + 12), i * 12);
  }
  return out;
}

/* Simulate the fused path: keygen → sort → project_gather direct to out. */
function runFused(splats: Splat[], cam: Cam): Float32Array {
  const n = splats.length;
  const keys = new Uint32Array(n);
  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    keys[i] = depthKey(splats[i]!, cam);
    indices[i] = i;
  }
  const sorted = radixSortPaired(keys, indices);
  const out = new Float32Array(n * 12);
  for (let i = 0; i < n; i++) {
    const splatIdx = sorted.values[i]!;
    out.set(projectOne(splats[splatIdx]!, cam), i * 12);
  }
  return out;
}

/** Build a deterministic camera somewhat-looking-at the unit cube. */
function makeCamera(): Cam {
  // Identity-ish view that translates +5 along Z so splats in [-2,2] are in front.
  // Column-major mat4: V = T(0,0,-5)  →  V[0..15] with V[14] = -5.
  const view = new Float32Array(16);
  view[0] = 1; view[5] = 1; view[10] = 1; view[15] = 1;
  view[14] = -5;
  // viewProj = perspective(60deg, 1, 0.1, 100) * view, but we only need a
  // deterministic, invertible matrix. Compose the perspective + view inline.
  const fov = Math.PI / 3;
  const aspect = 1;
  const near = 0.1;
  const far = 100;
  const f = 1 / Math.tan(fov / 2);
  const p = new Float32Array(16);
  p[0] = f / aspect;
  p[5] = f;
  p[10] = (far + near) / (near - far);
  p[11] = -1;
  p[14] = (2 * far * near) / (near - far);
  // viewProj = p * view (column-major multiply)
  const vp = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let acc = 0;
      for (let k = 0; k < 4; k++) acc += p[k * 4 + r]! * view[c * 4 + k]!;
      vp[c * 4 + r] = acc;
    }
  }
  return {
    view,
    viewProj: vp,
    focal: [800, 800],
    viewport: [1024, 1024],
  };
}

describe('fused vs separate pipeline: byte-identical instance output', () => {
  it('produces bit-identical Float32Array buffers (256 splats)', () => {
    const { bytes, descriptor } = buildSyntheticScene(256, 0xfa11ed);
    const decoded = decodeSplatsSoa(bytes, descriptor.attributeLayout!, 256);
    const splats: Splat[] = decoded.map((s) => ({
      pos: s.position,
      opacity: s.opacity,
      scale: s.scale,
      rot: s.rotation,
      color: s.colorDC,
    }));
    const cam = makeCamera();

    const separate = runSeparate(splats, cam);
    const fused = runFused(splats, cam);

    expect(separate.length).toBe(fused.length);
    // Compare as raw bytes so we catch any IEEE-754 drift.
    const sb = new Uint8Array(separate.buffer);
    const fb = new Uint8Array(fused.buffer);
    expect(fb).toEqual(sb);
  });

  it('produces bit-identical buffers at larger scale (4096 splats)', () => {
    const { bytes, descriptor } = buildSyntheticScene(4096, 0xc0ffee);
    const decoded = decodeSplatsSoa(bytes, descriptor.attributeLayout!, 4096);
    const splats: Splat[] = decoded.map((s) => ({
      pos: s.position,
      opacity: s.opacity,
      scale: s.scale,
      rot: s.rotation,
      color: s.colorDC,
    }));
    const cam = makeCamera();

    const separate = runSeparate(splats, cam);
    const fused = runFused(splats, cam);

    const sb = new Uint8Array(separate.buffer);
    const fb = new Uint8Array(fused.buffer);
    expect(fb).toEqual(sb);
  });
});
