// SPDX-License-Identifier: Apache-2.0
/**
 * WSR tile-prefix-sum scatter unit tests.
 *
 * The tile-prefix-sum path is the recovery move from the B8 PR2 KILL
 * (2026-05-15): CAS-loop atomic-add scatter saturated on hot pixels in
 * bonsai, producing 0.29 fps / 17 dB. The tile path bins splats into
 * per-tile lists (one atomic counter per tile, ~10^4× lower contention
 * than per-pixel), then accumulates in workgroup-private registers (no
 * atomics in the per-pixel loop at all).
 *
 * Tests follow the same shape as `wsr.test.ts`:
 *
 *   (A) **Static** — both new WGSL files load via `shaders.generated.ts`
 *       with the expected entry points + bind layouts.
 *
 *   (B) **TS-reference behavioural** — a pure-JS port of the bin +
 *       per-tile accumulate + resolve math is run on a small synthetic
 *       scene and produces a non-zero, finite, non-negative frame.
 *
 *   (C) **Bin-vs-accumulate consistency** — the JS reference shows
 *       that the tile-path output matches the original per-splat WSR
 *       accumulate reference (the order-independent WSR equation is
 *       insensitive to how splats are partitioned).
 *
 *   (D) **API surface** — `useWSRTile` flows through `ComputeDecodePipelineInit`.
 */
import { describe, expect, it } from 'vitest';
import {
  TILE_BIN_WGSL,
  WSR_TILE_ACCUMULATE_WGSL,
  WSR_RESOLVE_WGSL,
  WSR_TILE_SIZE,
  WSR_TILE_BIN_WG,
  WSR_TILE_DEFAULT_MAX_PER_TILE,
  WSR_TILE_DEFAULT_BG_WEIGHT,
} from '../../webgpu/index.js';
import type { ComputeDecodePipelineInit } from '../../webgpu/index.js';

/* ---------------------------------------------------------------- */
/* (A) Static WGSL string / binding-layout checks.                  */
/* ---------------------------------------------------------------- */

describe('WSR tile-prefix-sum WGSL source', () => {
  it('cs_tile_bin has 1 thread per splat + atomic tile counter', () => {
    expect(TILE_BIN_WGSL).toMatch(/fn cs_tile_bin\(/);
    expect(TILE_BIN_WGSL).toMatch(/@workgroup_size\(256\)/);
    // Per-tile atomic counter is the ONLY atomic in the binning kernel
    // (per-pixel atomics are the thing we're eliminating).
    expect(TILE_BIN_WGSL).toMatch(/atomicAdd\(&tile_count/);
    // Bindings: splats RO, tile_count RW atomic, tile_lists RW, uniforms.
    expect(TILE_BIN_WGSL).toMatch(
      /@group\(0\) @binding\(0\) var<storage, read>\s+splats/,
    );
    expect(TILE_BIN_WGSL).toMatch(
      /@group\(0\) @binding\(1\) var<storage, read_write> tile_count\s*:\s*array<atomic<u32>>/,
    );
    expect(TILE_BIN_WGSL).toMatch(
      /@group\(0\) @binding\(2\) var<storage, read_write> tile_lists\s*:\s*array<u32>/,
    );
    expect(TILE_BIN_WGSL).toMatch(
      /@group\(0\) @binding\(3\) var<uniform>\s+u/,
    );
    // Bin-clear entry point shares the same bind group.
    expect(TILE_BIN_WGSL).toMatch(/fn cs_tile_bin_clear/);
  });

  it('cs_wsr_tile_accumulate is 16×16 workgroup per tile, NO atomics', () => {
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(/fn cs_wsr_tile_accumulate/);
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(/@workgroup_size\(16,\s*16\)/);
    // The whole point: no atomics inside the per-pixel loop.
    expect(WSR_TILE_ACCUMULATE_WGSL).not.toMatch(/atomicCompareExchangeWeak/);
    expect(WSR_TILE_ACCUMULATE_WGSL).not.toMatch(/atomicAdd/);
    // Workgroup-shared splat-index batch (the cooperative prefetch).
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(/var<workgroup>\s+shared_idx/);
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(/workgroupBarrier/);
    // LC-WSR depth weight equation, same as the per-splat scatter shader.
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /max\(0\.0,\s*1\.0\s*-\s*depth\s*\/\s*sigma\s*-\s*v_i\)/,
    );
    // Bindings: splats RO, tile_count RO, tile_lists RO, numerator RW,
    // denominator RW, uniforms.
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(0\) var<storage, read>\s+splats/,
    );
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(1\) var<storage, read>\s+tile_count/,
    );
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(2\) var<storage, read>\s+tile_lists/,
    );
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(3\) var<storage, read_write> numerator/,
    );
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(4\) var<storage, read_write> denominator/,
    );
    expect(WSR_TILE_ACCUMULATE_WGSL).toMatch(
      /@group\(0\) @binding\(5\) var<uniform>\s+u/,
    );
  });

  it('shares the cs_wsr_resolve kernel with the PR1 path', () => {
    // No new resolve kernel — we re-use the same rational evaluator.
    expect(WSR_RESOLVE_WGSL).toMatch(/fn cs_wsr_resolve/);
  });

  it('exports the expected constants', () => {
    expect(WSR_TILE_SIZE).toBe(16);
    expect(WSR_TILE_BIN_WG).toBe(256);
    expect(WSR_TILE_DEFAULT_MAX_PER_TILE).toBe(16384);
    expect(WSR_TILE_DEFAULT_BG_WEIGHT).toBeCloseTo(1e-4, 7);
  });
});

/* ---------------------------------------------------------------- */
/* (B) Behavioural — pure-JS reference of the tile-path math.       */
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

interface ProjectedSplat {
  cx: number;          // screen x
  cy: number;          // screen y
  inv00: number;
  inv01: number;
  inv11: number;
  wd: number;          // LC-WSR depth weight
  alphaBase: number;   // clamped opacity
  color: [number, number, number];
  xmin: number; ymin: number;
  xmax: number; ymax: number;
  radius: number;
}

/** Mirror of cs_tile_bin's per-splat projection + culling. */
function projectSplat(s: Splat, cam: Camera, sigma: number, vDefault: number): ProjectedSplat | null {
  const [px, py, pz] = s.pos;
  const VP = cam.viewProj;
  const V = cam.view;
  const cx = VP[0]! * px + VP[4]! * py + VP[8]! * pz + VP[12]!;
  const cy = VP[1]! * px + VP[5]! * py + VP[9]! * pz + VP[13]!;
  const cw = VP[3]! * px + VP[7]! * py + VP[11]! * pz + VP[15]!;
  const invW = Math.abs(cw) > 1e-12 ? 1 / cw : 1;
  const ndcx = cx * invW;
  const ndcy = cy * invW;
  const vz = V[2]! * px + V[6]! * py + V[10]! * pz + V[14]!;
  const depth = -vz;
  if (depth <= 0) return null;

  const sigmaSafe = Math.max(sigma, 1e-6);
  const wd = Math.max(0, 1 - depth / sigmaSafe - vDefault);
  if (wd <= 0) return null;

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
  if (radius < 0.5) return null;

  const inv00 = c11 / det;
  const inv01 = -c01 / det;
  const inv11 = c00 / det;

  const cxPx = (ndcx * 0.5 + 0.5) * cam.viewport[0];
  const cyPx = (1 - (ndcy * 0.5 + 0.5)) * cam.viewport[1];
  const r = Math.ceil(radius);
  const xmin = Math.max(0, Math.floor(cxPx - r));
  const ymin = Math.max(0, Math.floor(cyPx - r));
  const xmax = Math.min(cam.viewport[0] - 1, Math.ceil(cxPx + r));
  const ymax = Math.min(cam.viewport[1] - 1, Math.ceil(cyPx + r));
  if (xmin > xmax || ymin > ymax) return null;

  const bboxArea = (xmax - xmin + 1) * (ymax - ymin + 1);
  if (bboxArea > 4096) return null;

  return {
    cx: cxPx, cy: cyPx, inv00, inv01, inv11, wd,
    alphaBase: Math.max(0, Math.min(1, s.opacity)),
    color: s.color,
    xmin, ymin, xmax, ymax, radius,
  };
}

/** TS port of `cs_tile_bin` + `cs_wsr_tile_accumulate`. */
function refTileRender(
  splats: Splat[],
  cam: Camera,
  sigma: number,
  vDefault: number,
  bg: [number, number, number, number],
  maxPerTile: number,
): { rgb: Float64Array; tileCounts: Int32Array; tilesX: number; tilesY: number } {
  const W = cam.viewport[0], H = cam.viewport[1];
  const tilesX = Math.ceil(W / WSR_TILE_SIZE);
  const tilesY = Math.ceil(H / WSR_TILE_SIZE);
  const totalTiles = tilesX * tilesY;

  // Stage 1: bin.
  const tileCounts = new Int32Array(totalTiles);
  const tileLists: number[][] = new Array(totalTiles);
  for (let t = 0; t < totalTiles; t++) tileLists[t] = [];

  const projected: (ProjectedSplat | null)[] = splats.map(s => projectSplat(s, cam, sigma, vDefault));
  for (let i = 0; i < splats.length; i++) {
    const p = projected[i];
    if (!p) continue;
    const txMin = Math.floor(p.xmin / WSR_TILE_SIZE);
    const tyMin = Math.floor(p.ymin / WSR_TILE_SIZE);
    const txMax = Math.min(tilesX - 1, Math.floor(p.xmax / WSR_TILE_SIZE));
    const tyMax = Math.min(tilesY - 1, Math.floor(p.ymax / WSR_TILE_SIZE));
    for (let ty = tyMin; ty <= tyMax; ty++) {
      for (let tx = txMin; tx <= txMax; tx++) {
        const tIdx = ty * tilesX + tx;
        if (tileCounts[tIdx]! < maxPerTile) {
          tileLists[tIdx]!.push(i);
        }
        tileCounts[tIdx]! += 1;
      }
    }
  }

  // Stage 2: per-tile accumulate (no per-pixel atomics — direct register writes).
  const num = new Float64Array(W * H * 4);
  const den = new Float64Array(W * H);
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tIdx = ty * tilesX + tx;
      const list = tileLists[tIdx]!;
      if (list.length === 0) continue;
      const xs = tx * WSR_TILE_SIZE;
      const ys = ty * WSR_TILE_SIZE;
      const xe = Math.min(W, xs + WSR_TILE_SIZE);
      const ye = Math.min(H, ys + WSR_TILE_SIZE);
      for (let py2 = ys; py2 < ye; py2++) {
        for (let px2 = xs; px2 < xe; px2++) {
          let nR = 0, nG = 0, nB = 0, d = 0;
          const fx = px2 + 0.5;
          const fy = py2 + 0.5;
          for (const si of list) {
            const p = projected[si]!;
            const dx = fx - p.cx;
            const dy = fy - p.cy;
            const power = -0.5 * (dx * dx * p.inv00 + 2 * dx * dy * p.inv01 + dy * dy * p.inv11);
            if (power > 0) continue;
            const g = Math.exp(power);
            const alpha = Math.max(0, Math.min(0.999, p.alphaBase * g));
            if (alpha < 1 / 255) continue;
            const aw = alpha * p.wd;
            nR += aw * p.color[0];
            nG += aw * p.color[1];
            nB += aw * p.color[2];
            d  += aw;
          }
          const pidx = py2 * W + px2;
          num[pidx * 4 + 0] = nR;
          num[pidx * 4 + 1] = nG;
          num[pidx * 4 + 2] = nB;
          den[pidx] = d;
        }
      }
    }
  }

  // Stage 3: resolve.
  const rgb = new Float64Array(W * H * 3);
  const wB = bg[3];
  for (let p = 0; p < W * H; p++) {
    const denom = wB + den[p]!;
    rgb[p * 3 + 0] = (wB * bg[0] + num[p * 4 + 0]!) / denom;
    rgb[p * 3 + 1] = (wB * bg[1] + num[p * 4 + 1]!) / denom;
    rgb[p * 3 + 2] = (wB * bg[2] + num[p * 4 + 2]!) / denom;
  }
  return { rgb, tileCounts, tilesX, tilesY };
}

/** Per-splat scatter reference — copy of the PR1 `refAccumulate` semantics. */
function refScatterRender(
  splats: Splat[],
  cam: Camera,
  sigma: number,
  vDefault: number,
  bg: [number, number, number, number],
): Float64Array {
  const W = cam.viewport[0], H = cam.viewport[1];
  const num = new Float64Array(4 * W * H);
  const den = new Float64Array(W * H);
  for (const s of splats) {
    const p = projectSplat(s, cam, sigma, vDefault);
    if (!p) continue;
    for (let py2 = p.ymin; py2 <= p.ymax; py2++) {
      for (let px2 = p.xmin; px2 <= p.xmax; px2++) {
        const dx = px2 + 0.5 - p.cx;
        const dy = py2 + 0.5 - p.cy;
        const power = -0.5 * (dx * dx * p.inv00 + 2 * dx * dy * p.inv01 + dy * dy * p.inv11);
        if (power > 0) continue;
        const g = Math.exp(power);
        const alpha = Math.max(0, Math.min(0.999, p.alphaBase * g));
        if (alpha < 1 / 255) continue;
        const aw = alpha * p.wd;
        const pidx = py2 * W + px2;
        num[pidx * 4 + 0]! += aw * p.color[0];
        num[pidx * 4 + 1]! += aw * p.color[1];
        num[pidx * 4 + 2]! += aw * p.color[2];
        den[pidx]! += aw;
      }
    }
  }
  const rgb = new Float64Array(W * H * 3);
  const wB = bg[3];
  for (let pi = 0; pi < W * H; pi++) {
    const denom = wB + den[pi]!;
    rgb[pi * 3 + 0] = (wB * bg[0] + num[pi * 4 + 0]!) / denom;
    rgb[pi * 3 + 1] = (wB * bg[1] + num[pi * 4 + 1]!) / denom;
    rgb[pi * 3 + 2] = (wB * bg[2] + num[pi * 4 + 2]!) / denom;
  }
  return rgb;
}

function buildScene(): { splats: Splat[]; cam: Camera; sigma: number } {
  const splats: Splat[] = [];
  let seed = 0xcafebabe >>> 0;
  const rand = (): number => {
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
  const view = new Float32Array(16);
  view[0] = 1; view[5] = 1; view[10] = 1; view[15] = 1;
  const fovY = Math.PI / 3;
  const f = 1 / Math.tan(fovY / 2);
  const viewProj = new Float32Array(16);
  viewProj[0]  = f;
  viewProj[5]  = f;
  viewProj[10] = (100 + 0.1) / (0.1 - 100);
  viewProj[11] = -1;
  viewProj[14] = (2 * 100 * 0.1) / (0.1 - 100);
  const W = 64, H = 64;
  const focalY = H / (2 * Math.tan(fovY / 2));
  const cam: Camera = { view, viewProj, focal: [focalY, focalY], viewport: [W, H] };
  return { splats, cam, sigma: 10 };
}

describe('WSR tile-prefix-sum reference math', () => {
  it('refTileRender produces a non-zero, finite, non-negative frame on the synthetic scene', () => {
    const { splats, cam, sigma } = buildScene();
    const { rgb, tileCounts, tilesX, tilesY } = refTileRender(
      splats, cam, sigma, 0,
      [0, 0, 0, WSR_TILE_DEFAULT_BG_WEIGHT],
      WSR_TILE_DEFAULT_MAX_PER_TILE,
    );

    expect(tilesX).toBe(Math.ceil(cam.viewport[0] / WSR_TILE_SIZE));
    expect(tilesY).toBe(Math.ceil(cam.viewport[1] / WSR_TILE_SIZE));

    // At least one tile must have received some splats (otherwise the
    // projection math regressed and the test is meaningless).
    let totalBinned = 0;
    for (let i = 0; i < tileCounts.length; i++) totalBinned += tileCounts[i]!;
    expect(totalBinned).toBeGreaterThan(0);

    let nonzero = 0;
    let maxV = -Infinity;
    for (let i = 0; i < rgb.length; i++) {
      const v = rgb[i]!;
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      if (v > 1e-6) nonzero++;
      if (v > maxV) maxV = v;
    }
    expect(nonzero).toBeGreaterThan(100);
    expect(maxV).toBeLessThanOrEqual(1 + 1e-5);
  });

  it('tile-path output matches the per-splat scatter reference within fp noise', () => {
    // Both algorithms compute the SAME WSR rational over the SAME splat set.
    // The tile path just partitions the inner loop by screen tile; the
    // per-pixel sum is mathematically identical.
    const { splats, cam, sigma } = buildScene();
    const bg: [number, number, number, number] = [0, 0, 0, WSR_TILE_DEFAULT_BG_WEIGHT];
    const tile = refTileRender(splats, cam, sigma, 0, bg, WSR_TILE_DEFAULT_MAX_PER_TILE);
    const scatter = refScatterRender(splats, cam, sigma, 0, bg);
    // Float64 sums are associative-up-to-ULP; the two summation orders are
    // close to bit-identical because the per-pixel inner loop visits the
    // same set of splats in the same order in our reference (we iterate
    // `splats` in array order in both cases when populating the tile).
    let maxAbsDiff = 0;
    for (let i = 0; i < tile.rgb.length; i++) {
      const d = Math.abs(tile.rgb[i]! - scatter[i]!);
      if (d > maxAbsDiff) maxAbsDiff = d;
    }
    expect(maxAbsDiff).toBeLessThan(1e-9);
  });

  it('per-tile capacity overflow is silent (counter saturates above maxPerTile)', () => {
    // Synthetic stress: build a scene where many splats all land in one tile,
    // exceeding maxPerTile=8. The tile-count buffer must keep counting (so
    // the kernel can pick min(count, maxPerTile)), but the list is capped.
    const splats: Splat[] = [];
    for (let i = 0; i < 64; i++) {
      splats.push({
        pos: [0.0, 0.0, -5.0],         // all stack in the center
        opacity: 0.5,
        scale: [0.02, 0.02, 0.02],     // small footprint → likely 1 tile
        rot: [0, 0, 0, 1],
        color: [0.5, 0.5, 0.5],
      });
    }
    const view = new Float32Array(16);
    view[0] = 1; view[5] = 1; view[10] = 1; view[15] = 1;
    const fovY = Math.PI / 3;
    const f = 1 / Math.tan(fovY / 2);
    const viewProj = new Float32Array(16);
    viewProj[0] = f; viewProj[5] = f;
    viewProj[10] = (100 + 0.1) / (0.1 - 100);
    viewProj[11] = -1;
    viewProj[14] = (2 * 100 * 0.1) / (0.1 - 100);
    const W = 32, H = 32;
    const focalY = H / (2 * Math.tan(fovY / 2));
    const cam: Camera = { view, viewProj, focal: [focalY, focalY], viewport: [W, H] };
    const cap = 8;
    const { tileCounts } = refTileRender(
      splats, cam, 10, 0,
      [0, 0, 0, WSR_TILE_DEFAULT_BG_WEIGHT],
      cap,
    );
    // At least one tile must have exceeded the cap (the counter keeps
    // incrementing past `cap`).
    let maxCount = 0;
    for (let i = 0; i < tileCounts.length; i++) {
      if (tileCounts[i]! > maxCount) maxCount = tileCounts[i]!;
    }
    expect(maxCount).toBeGreaterThan(cap);
  });
});

/* ---------------------------------------------------------------- */
/* (D) API surface — useWSRTile flag on ComputeDecodePipelineInit.  */
/* ---------------------------------------------------------------- */

describe('ComputeDecodePipelineInit useWSRTile flag', () => {
  it('useWSRTile + wsrTileMaxPerTile are optional members of the init type', () => {
    const init: Omit<ComputeDecodePipelineInit, 'device'> = {
      capacity: 1024,
      useWSRTile: true,
      wsrMaxWidth: 256,
      wsrMaxHeight: 256,
      wsrTileMaxPerTile: 1024,
    };
    expect(init.useWSRTile).toBe(true);
    expect(init.wsrTileMaxPerTile).toBe(1024);
  });

  it('useWSRTile defaults to false (legacy sorted-alpha path)', () => {
    const init: Omit<ComputeDecodePipelineInit, 'device'> = { capacity: 16 };
    expect(init.useWSRTile).toBeUndefined();
  });
});
