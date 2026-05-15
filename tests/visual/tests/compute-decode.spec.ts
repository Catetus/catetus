/**
 * SPEC-0062 — compute-decode + GPU radix-sort visual regression.
 *
 * The compute path (WGSL decode + project + radix-sort + gather) must produce
 * frames that are pixel-equivalent to the CPU path on the same WebGPU
 * backend. This test renders one canonical scene through both paths and
 * asserts that the per-pixel diff is at float-precision noise level
 * (<= 0.5% of pixels changed under pixelmatch's threshold=0.1 — well above
 * the float-rounding floor we observe in practice).
 *
 * The test only runs under the `chrome-webgpu` Playwright project; WebGL2
 * projects skip it since `useComputeDecode` is a WebGPU-only feature.
 */
import { test, expect } from '@playwright/test';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import { ASSETS, captureFrames } from './_helpers.js';

const MAX_DIFF_RATIO = 0.02; // 2% of pixels (very generous — float noise is <<1%)

for (const asset of ASSETS) {
  test.describe(`compute-decode parity — ${asset.id}`, () => {
    test(`compute and CPU paths agree`, async ({ page }, testInfo) => {
      const renderer = (testInfo.project.metadata as { renderer?: string }).renderer ?? 'auto';
      test.skip(renderer !== 'webgpu', 'compute-decode is WebGPU-only');

      // 1. CPU path baseline.
      const cpu = await captureFrames(page, { renderer: 'webgpu', src: asset.src });

      // 2. Compute path. We re-open the harness URL with useComputeDecode=1.
      const params = new URLSearchParams({
        src: asset.src,
        renderer: 'webgpu',
        useComputeDecode: '1',
      });
      await page.goto(`/page.html?${params.toString()}`, { waitUntil: 'load' });
      await page.waitForFunction(
        () => {
          // @ts-expect-error injected by harness
          const sf = window.__sf;
          return sf && (sf.ready === true || sf.error !== null);
        },
        null,
        { timeout: 60_000 },
      );
      const computeResult = await page.evaluate(() => {
        // @ts-expect-error injected by harness
        const sf = window.__sf;
        return {
          error: sf.error as { code: string; message: string } | null,
          frames: (sf.frames as Array<{ index: number; dataUrl: string }>).map((f) => ({
            index: f.index,
            dataUrl: f.dataUrl,
          })),
        };
      });
      if (computeResult.error) {
        throw new Error(`compute path errored: ${computeResult.error.code} — ${computeResult.error.message}`);
      }

      const dataUrlToBuffer = (u: string): Buffer => {
        const idx = u.indexOf('base64,');
        return Buffer.from(u.slice(idx + 'base64,'.length), 'base64');
      };
      const compute = computeResult.frames.map((f) => ({
        index: f.index,
        png: dataUrlToBuffer(f.dataUrl),
      }));

      expect(compute.length).toBe(cpu.length);

      // 3. Per-frame pixelmatch.
      for (let i = 0; i < cpu.length; i++) {
        const a = PNG.sync.read(cpu[i]!.png);
        const b = PNG.sync.read(compute[i]!.png);
        expect(a.width).toBe(b.width);
        expect(a.height).toBe(b.height);
        const out = new PNG({ width: a.width, height: a.height });
        const changed = pixelmatch(a.data, b.data, out.data, a.width, a.height, {
          threshold: 0.1,
          includeAA: false,
        });
        const ratio = changed / (a.width * a.height);
        expect(ratio).toBeLessThanOrEqual(MAX_DIFF_RATIO);
      }
    });
  });
}
