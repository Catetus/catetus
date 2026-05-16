// SPDX-License-Identifier: Apache-2.0
/**
 * Unit tests for the BufferPager + WGSL splats-access templating helper.
 *
 * Real GPU execution lives behind the bench harness; here we only test the
 * pure-CPU helpers (page layout math + WGSL string transformation).
 */
import { describe, expect, it } from 'vitest';
import { computePageLayout, BufferPager, templateSplatsAccess } from '../../webgpu/buffer-pager.js';

describe('computePageLayout', () => {
  it('returns 1 page when totalSplats fits in a single buffer', () => {
    // 1 M splats × 64 B = 64 MB. 1 GiB cap → 1 page.
    const lay = computePageLayout(1_000_000, 1024 * 1024 * 1024);
    expect(lay.numPages).toBe(1);
    expect(lay.pages[0].splatStart).toBe(0);
    expect(lay.pages[0].splatCount).toBe(1_000_000);
  });

  it('rounds splatsPerPage down to a multiple of 256', () => {
    // 100 MB cap → 100 * 1024 * 1024 / 64 = 1638400 splats max → /256 = 6400 → *256 = 1638400.
    // Try a less-clean cap: 100 MB + 13 bytes.
    const lay = computePageLayout(10_000_000, 100 * 1024 * 1024 + 13);
    expect(lay.splatsPerPage % 256).toBe(0);
  });

  it('partitions L0-scale (119M) into 4 pages at 2 GiB cap', () => {
    const lay = computePageLayout(119_000_000, 2 * 1024 * 1024 * 1024);
    expect(lay.numPages).toBeGreaterThanOrEqual(4);
    // Sum of per-page splat counts must equal total.
    const total = lay.pages.reduce((acc, p) => acc + p.splatCount, 0);
    expect(total).toBe(119_000_000);
    // First N-1 pages have splatsPerPage; last has the remainder.
    for (let i = 0; i < lay.numPages - 1; i++) {
      expect(lay.pages[i].splatCount).toBe(lay.splatsPerPage);
    }
    expect(lay.pages[lay.numPages - 1].splatCount).toBeLessThanOrEqual(lay.splatsPerPage);
  });

  it('partitions L1-scale (54M) into >= 2 pages at 2 GiB cap', () => {
    const lay = computePageLayout(54_000_000, 2 * 1024 * 1024 * 1024);
    expect(lay.numPages).toBeGreaterThanOrEqual(2);
    const total = lay.pages.reduce((acc, p) => acc + p.splatCount, 0);
    expect(total).toBe(54_000_000);
  });
});

describe('templateSplatsAccess', () => {
  const SAMPLE_WGSL = `
struct DecodedSplat { pos: vec4<f32>, scale: vec4<f32>, rot: vec4<f32>, color: vec4<f32> };
@group(0) @binding(0) var<storage, read>       splats : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;
@group(0) @binding(2) var<uniform>             u : SomeUniforms;
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = splats[gid.x];
  output[gid.x] = bitcast<u32>(s.pos.x);
}
`;

  it('passes through with N=1 (single page)', () => {
    const r = templateSplatsAccess(SAMPLE_WGSL, 'splats', 1, 65536);
    expect(r.splatsBindings).toEqual([0]);
    expect(r.rebasedBindings.get(1)).toBe(1);
    expect(r.rebasedBindings.get(2)).toBe(2);
    // Body should call read_splats_splats(gid.x).
    expect(r.wgsl).toContain('read_splats_splats(gid.x)');
    // Should declare splats_p0.
    expect(r.wgsl).toContain('splats_p0 : array<DecodedSplat>');
  });

  it('emits N pages and rebases later bindings (N=4)', () => {
    const r = templateSplatsAccess(SAMPLE_WGSL, 'splats', 4, 1_048_576);
    expect(r.splatsBindings).toEqual([0, 1, 2, 3]);
    expect(r.rebasedBindings.get(1)).toBe(4);
    expect(r.rebasedBindings.get(2)).toBe(5);
    // 4 page declarations.
    for (let p = 0; p < 4; p++) {
      expect(r.wgsl).toContain(`splats_p${p} : array<DecodedSplat>`);
    }
    // Helper should switch on page.
    expect(r.wgsl).toContain('SPLATS_PER_PAGE_splats : u32 = 1048576u');
    expect(r.wgsl).toContain('case 0u: { return splats_p0[off]; }');
    expect(r.wgsl).toContain('case 3u: { return splats_p3[off]; }');
    // Body call replaced.
    expect(r.wgsl).toContain('read_splats_splats(gid.x)');
    // Other bindings rebased: original binding 1 (output) → 4, binding 2 (u) → 5.
    expect(r.wgsl).toMatch(/@binding\(4\)\s+var<storage, read_write>\s+output/);
    expect(r.wgsl).toMatch(/@binding\(5\)\s+var<uniform>\s+u/);
  });

  it('handles read_write splats (lod_blend pattern)', () => {
    const RW_SAMPLE = `
struct DecodedSplat { pos: vec4<f32>, scale: vec4<f32>, rot: vec4<f32>, color: vec4<f32> };
@group(0) @binding(0) var<storage, read_write> lb_splats : array<DecodedSplat>;
@group(0) @binding(1) var<storage, read>       chunk_id : array<u32>;
fn dummy() { lb_splats[0u].pos.w = 0.0; }
`;
    // For read_write we still expose the read helper; the host is responsible
    // for never calling templateSplatsAccess for kernels that need random
    // WRITE access. lod_blend uses pattern 1 (per-dispatch slice) instead.
    const r = templateSplatsAccess(RW_SAMPLE, 'lb_splats', 2, 256);
    expect(r.splatsBindings).toEqual([0, 1]);
    // The body's `lb_splats[0u].pos.w = 0.0` will be transformed to
    // `read_splats_lb_splats(0u).pos.w = 0.0` which is invalid WGSL — this
    // is by design: callers should only pass templateSplatsAccess to read-
    // only random-access kernels, not to read-write kernels. We still
    // verify the read helper is emitted.
    expect(r.wgsl).toContain('fn read_splats_lb_splats(');
  });
});
