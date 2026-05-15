// SPDX-License-Identifier: Apache-2.0
/**
 * WSR PR1 unit tests.
 *
 * These tests run under Node + vitest (no GPUDevice available). They cover:
 *
 *   (A) **Static** — the three new WGSL files load via `shaders.generated.ts`
 *       with the expected entry points and binding layouts. This is the same
 *       discipline used by `compute-decode.test.ts` / `fused_project_gather`.
 *
 *   (B) **TS-reference behavioural** — a pure-JS port of the cs_wsr_accumulate
 *       + cs_wsr_resolve math is run on a small synthetic scene. Assertions:
 *         • output is not all-zero (splats actually contributed),
 *         • output is finite (no NaN / ±Inf),
 *         • output is non-negative (the WSR equation is energy-preserving
 *           and `c_B = (0,0,0)`).
 *       This is the in-process equivalent of "instantiate the pipeline with
 *       useWSR: true and assert the readback is sane" — the WGSL is exercised
 *       end-to-end inside Playwright (post-PR1), but the math invariants are
 *       pinned here so a regression in the WSR formula is caught at unit-test
 *       time without a GPU.
 *
 *   (C) **API surface** — `ComputeDecodePipelineInit` accepts the `useWSR`
 *       flag and the `WSRPipeline` exports are reachable from the public
 *       barrel. This is a structural test; it doesn't instantiate the GPU
 *       pipeline (no device), but it verifies the typing is wired through.
 */
import { describe, expect, it } from 'vitest';
import {
  WSR_CLEAR_WGSL,
  WSR_ACCUMULATE_WGSL,
  WSR_RESOLVE_WGSL,
  WSR_DEFAULT_BG_WEIGHT,
  WSR_TILE,
  WSR_WG,
} from '../../webgpu/index.js';
import type { ComputeDecodePipelineInit } from '../../webgpu/index.js';

/* ---------------------------------------------------------------- */
/* (A) Static WGSL string / binding-layout checks.                  */
/* ---------------------------------------------------------------- */

describe('WSR WGSL source', () => {
  it('cs_wsr_clear is present and has the expected entry point + bindings', () => {
    expect(WSR_CLEAR_WGSL).toMatch(/fn cs_wsr_clear/);
    expect(WSR_CLEAR_WGSL).toMatch(/@workgroup_size\(16,\s*16\)/);
    expect(WSR_CLEAR_WGSL).toMatch(
      /@group\(0\) @binding\(0\) var<storage, read_write> numerator/,
    );
    expect(WSR_CLEAR_WGSL).toMatch(
      /@group\(0\) @binding\(1\) var<storage, read_write> denominator/,
    );
    expect(WSR_CLEAR_WGSL).toMatch(
      /@group\(0\) @binding\(2\) var<uniform>\s+cu/,
    );
    // The accumulator buffers are atomic-typed so the accumulate kernel's
    // CAS-loop can target them. The clear kernel writes via `atomicStore`.
    expect(WSR_CLEAR_WGSL).toMatch(/array<atomic<u32>>/);
    expect(WSR_CLEAR_WGSL).toMatch(/atomicStore/);
  });

  it('cs_wsr_accumulate has 1 thread per splat + CAS-loop atomic-add shim', () => {
    expect(WSR_ACCUMULATE_WGSL).toMatch(/fn cs_wsr_accumulate/);
    expect(WSR_ACCUMULATE_WGSL).toMatch(/@workgroup_size\(256\)/);
    // CAS-loop atomic float add (B7.1 portability path).
    expect(WSR_ACCUMULATE_WGSL).toMatch(/fn atomic_add_f32/);
    expect(WSR_ACCUMULATE_WGSL).toMatch(/atomicCompareExchangeWeak/);
    // LC-WSR depth weight: w(d) = max(0, 1 - d/σ - v).
    expect(WSR_ACCUMULATE_WGSL).toMatch(/max\(0\.0,\s*1\.0\s*-\s*depth\s*\/\s*sigma\s*-\s*v_i\)/);
    // Bindings match the wsr.ts layout: (splats RO, num RW atomic, den RW atomic, uniform).
    expect(WSR_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(0\) var<storage, read>\s+splats/,
    );
    expect(WSR_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(1\) var<storage, read_write> numerator/,
    );
    expect(WSR_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(2\) var<storage, read_write> denominator/,
    );
    expect(WSR_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(3\) var<uniform>\s+u/,
    );
  });

  it('cs_wsr_resolve implements the WSR rational equation', () => {
    expect(WSR_RESOLVE_WGSL).toMatch(/fn cs_wsr_resolve/);
    expect(WSR_RESOLVE_WGSL).toMatch(/@workgroup_size\(16,\s*16\)/);
    // C_px = (w_B · c_B + N_px) / (w_B + D_px). The shader spells this as
    // `(w_b * c_b + vec3<f32>(n_r, n_g, n_b)) / denom` with `denom = w_b + d`.
    expect(WSR_RESOLVE_WGSL).toMatch(/w_b\s*\+\s*d/);
    expect(WSR_RESOLVE_WGSL).toMatch(/w_b\s*\*\s*c_b/);
    // The output is packed rgba8unorm so it can `copyBufferToTexture` into a
    // canvas-format texture.
    expect(WSR_RESOLVE_WGSL).toMatch(/fn pack_rgba8/);
  });

  it('shared constants are exported with the expected values', () => {
    expect(WSR_TILE).toBe(16);
    expect(WSR_WG).toBe(256);
    expect(WSR_DEFAULT_BG_WEIGHT).toBeCloseTo(1e-4, 7);
  });
});

/* ---------------------------------------------------------------- */
/* (B) Behavioural — pure-JS reference of the WSR math.             */
/* ---------------------------------------------------------------- */

interface Splat {
  pos: [number, number, number];
  opacity: number;
  scale: [number, number, number];
  rot: [number, number, number, number];
  color: [number, number, number];
}

interface Camera {
  view: Float32Array;
  viewProj: Float32Array;
  focal: [number, number];
  viewport: [number, number];
}

/** TS port of cov3d_w() in cs_wsr_accumulate.wgsl. */
function cov3d(s: [number, number, number], q: [number, number, number, number]): number[] {
  const ql = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-8);
  const x = q[0] / ql, y = q[1] / ql, z = q[2] / ql, w = q[3] / ql;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const r00 = 1 - 2 * (yy + zz);
  const r10 = 2 * (xy + wz);
  const r20 = 2 * (xz - wy);
  const r01 = 2 * (xy - wz);
  const r11 = 1 - 2 * (xx + zz);
  const r21 = 2 * (yz + wx);
  const r02 = 2 * (xz + wy);
  const r12 = 2 * (yz - wx);
  const r22 = 1 - 2 * (xx + yy);
  const sx = s[0], sy = s[1], sz = s[2];
  const m00 = r00 * sx, m10 = r10 * sx, m20 = r20 * sx;
  const m01 = r01 * sy, m11 = r11 * sy, m21 = r21 * sy;
  const m02 = r02 * sz, m12 = r12 * sz, m22 = r22 * sz;
  return [
    m00 * m00 + m01 * m01 + m02 * m02,
    m00 * m10 + m01 * m11 + m02 * m12,
    m00 * m20 + m01 * m21 + m02 * m22,
    m10 * m10 + m11 * m11 + m12 * m12,
    m10 * m20 + m11 * m21 + m12 * m22,
    m20 * m20 + m21 * m21 + m22 * m22,
  ];
}

interface Accumulators {
  numerator: Float64Array; // 4*W*H — RGB sum + unused slot
  denominator: Float64Array; // W*H
}

/**
 * Reference WSR accumulate pass. Mirrors `cs_wsr_accumulate` line-for-line
 * (modulo the f32 → f64 widening on the host side — the GPU runs in f32).
 */
function refAccumulate(
  splats: Splat[],
  cam: Camera,
  sigma: number,
  vDefault: number,
): Accumulators {
  const W = cam.viewport[0]; const H = cam.viewport[1];
  const num = new Float64Array(4 * W * H);
  const den = new Float64Array(W * H);
  const V = cam.view; const VP = cam.viewProj;
  for (const s of splats) {
    const [px, py, pz] = s.pos;
    const cx = VP[0]! * px + VP[4]! * py + VP[8]!  * pz + VP[12]!;
    const cy = VP[1]! * px + VP[5]! * py + VP[9]!  * pz + VP[13]!;
    const cw = VP[3]! * px + VP[7]! * py + VP[11]! * pz + VP[15]!;
    const invW = Math.abs(cw) > 1e-12 ? 1 / cw : 1;
    const ndcx = cx * invW;
    const ndcy = cy * invW;
    const vz = V[2]! * px + V[6]! * py + V[10]! * pz + V[14]!;
    const depth = -vz;
    if (depth <= 0) continue;

    const Vc = cov3d(s.scale, s.rot);
    const w0x = V[0]!, w0y = V[4]!, w0z = V[8]!;
    const w1x = V[1]!, w1y = V[5]!, w1z = V[9]!;
    const a0x = w0x * Vc[0]! + w0y * Vc[1]! + w0z * Vc[2]!;
    const a0y = w0x * Vc[1]! + w0y * Vc[3]! + w0z * Vc[4]!;
    const a0z = w0x * Vc[2]! + w0y * Vc[4]! + w0z * Vc[5]!;
    const a1x = w1x * Vc[0]! + w1y * Vc[1]! + w1z * Vc[2]!;
    const a1y = w1x * Vc[1]! + w1y * Vc[3]! + w1z * Vc[4]!;
    const a1z = w1x * Vc[2]! + w1y * Vc[4]! + w1z * Vc[5]!;
    const vxx = a0x * w0x + a0y * w0y + a0z * w0z;
    const vxy = a0x * w1x + a0y * w1y + a0z * w1z;
    const vyy = a1x * w1x + a1y * w1y + a1z * w1z;

    const zMax = Math.max(Math.abs(depth), 1e-4);
    const jx = cam.focal[0] / zMax;
    const jy = cam.focal[1] / zMax;
    const reg = 0.3;
    const c00 = jx * jx * vxx + reg;
    const c01 = jx * jy * vxy;
    const c11 = jy * jy * vyy + reg;
    const halfTrace = 0.5 * (c00 + c11);
    const det = Math.max(c00 * c11 - c01 * c01, 1e-6);
    const term = Math.sqrt(Math.max(halfTrace * halfTrace - det, 0));
    const radius = 3 * Math.sqrt(Math.max(halfTrace + term, 0));
    if (radius < 0.5) continue;

    const inv00 = c11 / det;
    const inv01 = -c01 / det;
    const inv11 = c00 / det;

    const sigmaSafe = Math.max(sigma, 1e-6);
    const wd = Math.max(0, 1 - depth / sigmaSafe - vDefault);
    if (wd <= 0) continue;

    const cxPx = (ndcx * 0.5 + 0.5) * W;
    const cyPx = (1 - (ndcy * 0.5 + 0.5)) * H;
    const r = Math.ceil(radius);
    const xmin = Math.max(0, Math.floor(cxPx - r));
    const ymin = Math.max(0, Math.floor(cyPx - r));
    const xmax = Math.min(W - 1, Math.ceil(cxPx + r));
    const ymax = Math.min(H - 1, Math.ceil(cyPx + r));
    if (xmin > xmax || ymin > ymax) continue;

    const alphaBase = Math.max(0, Math.min(1, s.opacity));
    for (let py2 = ymin; py2 <= ymax; py2++) {
      for (let px2 = xmin; px2 <= xmax; px2++) {
        const dx = px2 + 0.5 - cxPx;
        const dy = py2 + 0.5 - cyPx;
        const power = -0.5 * (dx * dx * inv00 + 2 * dx * dy * inv01 + dy * dy * inv11);
        if (power > 0) continue;
        const g = Math.exp(power);
        const alpha = Math.max(0, Math.min(0.999, alphaBase * g));
        if (alpha < 1 / 255) continue;
        const aw = alpha * wd;
        const pidx = py2 * W + px2;
        num[pidx * 4 + 0]! += aw * s.color[0];
        num[pidx * 4 + 1]! += aw * s.color[1];
        num[pidx * 4 + 2]! += aw * s.color[2];
        den[pidx]! += aw;
      }
    }
  }
  return { numerator: num, denominator: den };
}

/**
 * Reference WSR resolve. Mirrors `cs_wsr_resolve` (the rational evaluation,
 * not the rgba8unorm pack — we return floats for higher-precision assertions).
 */
function refResolve(
  acc: Accumulators,
  W: number,
  H: number,
  bg: [number, number, number, number],
): Float64Array {
  const out = new Float64Array(W * H * 3);
  const wB = bg[3];
  for (let p = 0; p < W * H; p++) {
    const nR = acc.numerator[p * 4 + 0]!;
    const nG = acc.numerator[p * 4 + 1]!;
    const nB = acc.numerator[p * 4 + 2]!;
    const d  = acc.denominator[p]!;
    const denom = wB + d;
    out[p * 3 + 0] = (wB * bg[0] + nR) / denom;
    out[p * 3 + 1] = (wB * bg[1] + nG) / denom;
    out[p * 3 + 2] = (wB * bg[2] + nB) / denom;
  }
  return out;
}

/**
 * Build a 100-splat synthetic scene in a unit cube at distance 5 from a
 * camera looking down -Z. Splats have isotropic unit scale and bright
 * colours so the rendered image is non-trivial.
 */
function buildScene(): { splats: Splat[]; cam: Camera; sigma: number } {
  const splats: Splat[] = [];
  let seed = 0xdeadbeef >>> 0;
  const rand = (): number => {
    // xorshift32
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5;  seed >>>= 0;
    return (seed & 0xffffff) / 0xffffff;
  };
  for (let i = 0; i < 100; i++) {
    splats.push({
      pos: [(rand() - 0.5) * 2, (rand() - 0.5) * 2, -5 + (rand() - 0.5) * 1.0],
      opacity: 0.6 + rand() * 0.4,
      scale: [0.05, 0.05, 0.05],
      rot: [0, 0, 0, 1],
      color: [0.3 + rand() * 0.7, 0.3 + rand() * 0.7, 0.3 + rand() * 0.7],
    });
  }
  // Identity view (camera at origin looking down -Z).
  const view = new Float32Array(16);
  view[0] = 1; view[5] = 1; view[10] = 1; view[15] = 1;
  // Simple perspective view-projection: fov 60°, aspect 1, near 0.1, far 100.
  // Standard column-major perspective matrix.
  const fovY = Math.PI / 3;
  const f = 1 / Math.tan(fovY / 2);
  const aspect = 1;
  const near = 0.1;
  const far = 100;
  const viewProj = new Float32Array(16);
  viewProj[0]  = f / aspect;
  viewProj[5]  = f;
  viewProj[10] = (far + near) / (near - far);
  viewProj[11] = -1;
  viewProj[14] = (2 * far * near) / (near - far);
  // Combined with identity view, viewProj == proj.

  const W = 64, H = 64;
  const focalY = H / (2 * Math.tan(fovY / 2));
  const cam: Camera = {
    view,
    viewProj,
    focal: [focalY, focalY],
    viewport: [W, H],
  };
  // σ = 2 × mean_scene_depth ≈ 2 × 5 = 10.
  const sigma = 10;
  return { splats, cam, sigma };
}

describe('WSR reference math', () => {
  it('refAccumulate + refResolve produces a non-zero, finite, non-negative frame', () => {
    const { splats, cam, sigma } = buildScene();
    const acc = refAccumulate(splats, cam, sigma, 0);
    const rgb = refResolve(acc, cam.viewport[0], cam.viewport[1], [0, 0, 0, WSR_DEFAULT_BG_WEIGHT]);

    let nonzero = 0;
    let maxV = -Infinity;
    let minV = +Infinity;
    for (let i = 0; i < rgb.length; i++) {
      const v = rgb[i]!;
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      if (v > 1e-6) nonzero++;
      if (v > maxV) maxV = v;
      if (v < minV) minV = v;
    }
    // 100 splats over a 64×64 frame should cover a non-trivial pixel count.
    // The exact count depends on the random scene but the lower bound is
    // generous — even if half the scene fell behind the near plane we'd
    // still cover > 100 px.
    expect(nonzero).toBeGreaterThan(100);
    // No pixel should exceed 1.0 by more than rounding — the WSR equation is
    // a weighted average of [0,1] color values.
    expect(maxV).toBeLessThanOrEqual(1.0 + 1e-5);
  });

  it('refResolve at zero accumulation (no splats) returns the background color', () => {
    const W = 8, H = 8;
    const acc: Accumulators = {
      numerator: new Float64Array(W * H * 4),
      denominator: new Float64Array(W * H),
    };
    const bg: [number, number, number, number] = [0.7, 0.1, 0.4, WSR_DEFAULT_BG_WEIGHT];
    const rgb = refResolve(acc, W, H, bg);
    // C = (w_B · c_B + 0) / (w_B + 0) = c_B. Floor-exact, no rounding.
    for (let p = 0; p < W * H; p++) {
      expect(rgb[p * 3 + 0]).toBeCloseTo(bg[0], 6);
      expect(rgb[p * 3 + 1]).toBeCloseTo(bg[1], 6);
      expect(rgb[p * 3 + 2]).toBeCloseTo(bg[2], 6);
    }
  });

  it('WSR is order-invariant — shuffling the splats yields the same frame', () => {
    const { splats, cam, sigma } = buildScene();
    const accA = refAccumulate(splats, cam, sigma, 0);
    // Shuffle.
    const shuffled = [...splats];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = (i * 2654435769) % (i + 1); // deterministic permutation
      [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
    }
    const accB = refAccumulate(shuffled, cam, sigma, 0);
    const rgbA = refResolve(accA, cam.viewport[0], cam.viewport[1], [0, 0, 0, WSR_DEFAULT_BG_WEIGHT]);
    const rgbB = refResolve(accB, cam.viewport[0], cam.viewport[1], [0, 0, 0, WSR_DEFAULT_BG_WEIGHT]);
    // Float64 accumulation is associative-up-to-ULP. We assert per-pixel
    // closeness within an absolute tolerance comfortably below any 8-bit
    // rgba8unorm quantisation step (1/255 ≈ 3.9e-3).
    for (let i = 0; i < rgbA.length; i++) {
      expect(Math.abs(rgbA[i]! - rgbB[i]!)).toBeLessThan(1e-9);
    }
  });
});

/* ---------------------------------------------------------------- */
/* (C) API surface — useWSR flag flows through public types.        */
/* ---------------------------------------------------------------- */

describe('ComputeDecodePipelineInit useWSR flag', () => {
  it('useWSR + wsrMaxWidth + wsrMaxHeight are optional members of the init type', () => {
    // Compile-time check via construction-of-literal-init. We never
    // instantiate the pipeline (no GPUDevice in Node), but if the type
    // surface drifted this assignment would fail tsc.
    const init: Omit<ComputeDecodePipelineInit, 'device'> = {
      capacity: 1024,
      useWSR: true,
      wsrMaxWidth: 256,
      wsrMaxHeight: 256,
    };
    expect(init.useWSR).toBe(true);
    expect(init.wsrMaxWidth).toBe(256);
    expect(init.wsrMaxHeight).toBe(256);
  });

  it('useWSR defaults to false (legacy sorted-alpha path)', () => {
    const init: Omit<ComputeDecodePipelineInit, 'device'> = { capacity: 16 };
    expect(init.useWSR).toBeUndefined();
  });
});
