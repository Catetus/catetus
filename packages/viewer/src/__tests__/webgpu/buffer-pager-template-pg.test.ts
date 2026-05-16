// SPDX-License-Identifier: Apache-2.0
/**
 * Validates that templateSplatsAccess applied to the real PROJECT_GATHER_WGSL
 * source produces well-formed WGSL for both entry points (cs_keygen on
 * k_splats and cs_project_gather on g_splats) without breaking the kernel
 * bodies. Pure CPU — verifies bindings, helper emission, and that the bodies
 * call the `read_splats_*` helper instead of indexing the original array.
 *
 * Stage 6 / sf-154.
 */
import { describe, expect, it } from 'vitest';
import { PROJECT_GATHER_WGSL } from '../../webgpu/shaders.generated.js';
import { templateSplatsAccess } from '../../webgpu/buffer-pager.js';

describe('templateSplatsAccess on PROJECT_GATHER_WGSL', () => {
  it('templates k_splats with N=2 pages and rebases downstream bindings', () => {
    const r = templateSplatsAccess(PROJECT_GATHER_WGSL, 'k_splats', 2, 1024 * 1024);
    expect(r.splatsBindings).toEqual([0, 1]);
    // Original keys/indices/uniforms at bindings 1/2/3 must be rebased to 2/3/4.
    expect(r.rebasedBindings.get(1)).toBe(2);
    expect(r.rebasedBindings.get(2)).toBe(3);
    expect(r.rebasedBindings.get(3)).toBe(4);
    // The two splats pages must be declared.
    expect(r.wgsl).toContain('k_splats_p0 : array<DecodedSplat>');
    expect(r.wgsl).toContain('k_splats_p1 : array<DecodedSplat>');
    // The body's `let s = k_splats[i];` must be transformed into the helper call.
    expect(r.wgsl).toContain('read_splats_k_splats(i)');
    expect(r.wgsl).not.toMatch(/[^_]k_splats\s*\[/);
  });

  it('templates g_splats with N=4 pages (L0-scale)', () => {
    const r = templateSplatsAccess(PROJECT_GATHER_WGSL, 'g_splats', 4, 33_554_432);
    expect(r.splatsBindings).toEqual([0, 1, 2, 3]);
    // Original indices/inst_out/uniforms at bindings 1/2/3 → 4/5/6.
    expect(r.rebasedBindings.get(1)).toBe(4);
    expect(r.rebasedBindings.get(2)).toBe(5);
    expect(r.rebasedBindings.get(3)).toBe(6);
    // Helper switches across all 4 pages.
    for (let p = 0; p < 4; p++) {
      expect(r.wgsl).toContain(`case ${p}u: { return g_splats_p${p}[off]; }`);
    }
    // Body's `let s = g_splats[splat_idx];` → helper call.
    expect(r.wgsl).toContain('read_splats_g_splats(splat_idx)');
  });

  it('only rewrites the named splats binding, not the other one', () => {
    // When templating just `k_splats`, the `g_splats` binding (a different
    // array<DecodedSplat>) must NOT be touched — it stays as a single
    // binding unchanged. (Each kernel gets its own templating pass.)
    const r = templateSplatsAccess(PROJECT_GATHER_WGSL, 'k_splats', 2, 1024 * 1024);
    // g_splats body still uses g_splats[splat_idx] (one binding).
    expect(r.wgsl).toMatch(/g_splats\s*\[\s*splat_idx\s*\]/);
    // And only one g_splats declaration line.
    const gDecls = (r.wgsl.match(/var<storage,\s*read>\s+g_splats\s*:/g) ?? []).length;
    expect(gDecls).toBe(1);
  });
});
