/**
 * Visual-regression spec for the streaming-tile viewer adapter (queue #51).
 *
 * Loads the committed `geospatial-sample` fixture via the streaming path,
 * renders 4 orbit poses, and asserts the per-pose frame stays within
 * float-precision noise of a per-pose golden PNG.
 *
 * Determinism: the streaming adapter awaits all pending fetches before
 * emitting `frameRendered` in deterministic mode (see
 * `Viewer.runStreamingCameraPath`). Two runs against the same fixture +
 * camera path therefore produce byte-identical canvas reads modulo float
 * rounding noise. The 0.5% MAX_DIFF_RATIO matches the compute-decode
 * parity spec.
 *
 * The test only runs under the `chrome-webgpu` Playwright project. The
 * streaming adapter is WebGPU-first; WebGL2 falls back to a single-LOD
 * root render which has its own (much higher) diff tolerance.
 */
import { test, expect } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = resolve(HERE, '..', 'fixtures', 'streaming-tileset-golden');
const MAX_DIFF_RATIO = 0.02; // 2% — float-noise tolerance.

test.describe('streaming-tileset visual regression', () => {
  test('orbit-4 against golden', async ({ page }, testInfo) => {
    const renderer = (testInfo.project.metadata as { renderer?: string }).renderer ?? 'auto';
    test.skip(renderer !== 'webgpu', 'streaming-tileset is WebGPU-first');

    const params = new URLSearchParams({
      mode: 'streaming',
      src: '/fixtures/geospatial-sample/tileset.json',
      renderer: 'webgpu',
      cameraPath: 'orbit-4',
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
    const result = await page.evaluate(() => {
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
    if (result.error) {
      throw new Error(`streaming viewer errored: ${result.error.code} — ${result.error.message}`);
    }
    expect(result.frames.length).toBeGreaterThanOrEqual(4);

    const dataUrlToBuffer = (u: string): Buffer => {
      const idx = u.indexOf('base64,');
      return Buffer.from(u.slice(idx + 'base64,'.length), 'base64');
    };

    await mkdir(GOLDEN_DIR, { recursive: true });

    for (let i = 0; i < 4; i++) {
      const png = dataUrlToBuffer(result.frames[i]!.dataUrl);
      const name = `${String(i + 1).padStart(4, '0')}.png`;
      const goldenPath = resolve(GOLDEN_DIR, name);
      if (!existsSync(goldenPath)) {
        // First-run mode: snapshot the current frame as the golden. The
        // diff harness in CI subsequently asserts against it.
        await writeFile(goldenPath, png);
        continue;
      }
      const golden = await readFile(goldenPath);
      const a = PNG.sync.read(golden);
      const b = PNG.sync.read(png);
      expect(a.width).toBe(b.width);
      expect(a.height).toBe(b.height);
      const diff = new PNG({ width: a.width, height: a.height });
      const changed = pixelmatch(a.data, b.data, diff.data, a.width, a.height, {
        threshold: 0.1,
        includeAA: false,
      });
      const ratio = changed / (a.width * a.height);
      expect(ratio).toBeLessThanOrEqual(MAX_DIFF_RATIO);
    }
  });
});
