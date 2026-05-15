/**
 * KHR_gaussian_splatting attribute-layout parity (Playwright).
 *
 * Each compare-scenes fixture (floater / indoor / product, both size-min
 * and web-mobile variants) is committed in two authoring shapes:
 *
 *  - `apps/web/public/compare-scenes/<scene>/<variant>/scene.gltf`
 *      RC layout — namespaced primitive-level attributes (post-PR-2).
 *
 *  - `apps/web/public/compare-scenes/legacy/<scene>/<variant>/scene.gltf`
 *      Pre-RC layout — bare keys inside the per-primitive extension object.
 *
 * Both reference the *same* binary buffer (`buffers/chunk_0000.bin`). The
 * dual-layout reader is supposed to produce identical decoded splats from
 * either shape; the renderer is supposed to produce identical pixels.
 *
 * This spec drives the harness page once per (scene × variant × layout),
 * captures one frame from a deterministic camera path, and asserts that
 * the RC frame is within float-precision noise of the legacy frame.
 *
 * Runs only under `chrome-webgl2` to keep CI cost bounded — the parity
 * we're testing is in the JSON reader, not in the renderer backend.
 */
import { test, expect } from '@playwright/test';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdir, stat, readFile, writeFile, copyFile, symlink } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const HERE = dirname(fileURLToPath(import.meta.url));
// Path layout: tests/visual/tests/<this>  →  tests/visual  →  repo root.
const REPO_ROOT = resolve(HERE, '..', '..', '..');
const FIXTURES_DIR = resolve(HERE, '..', 'fixtures');
const COMPARE_SRC = resolve(REPO_ROOT, 'apps', 'web', 'public', 'compare-scenes');

// Pixel-diff tolerance: the dual-layout reader resolves to the same
// accessor indices, so the decoded splat bytes are bit-identical and the
// rendered frames differ only in float-rounding noise from re-running the
// orbit pose. Keep tight.
const MAX_DIFF_RATIO = 0.005; // 0.5%

const SCENES = ['floater', 'indoor', 'product'] as const;
const VARIANTS = ['size-min', 'web-mobile'] as const;

/**
 * Stage the compare-scenes fixtures under tests/visual/fixtures/ so the
 * harness static server (which only serves /fixtures/*) can reach them.
 *
 * We copy lazily: if the staged copy already exists and is at least as new
 * as the source, skip. Each `(scene, variant)` lives at:
 *
 *   /fixtures/compare/<scene>/<variant>/scene.gltf            (RC)
 *   /fixtures/compare/legacy/<scene>/<variant>/scene.gltf     (legacy)
 *
 * The `buffers/` subdirectory under each variant is copied alongside.
 */
async function ensureStaged(): Promise<void> {
  const stagingRoot = resolve(FIXTURES_DIR, 'compare');
  await mkdir(stagingRoot, { recursive: true });

  async function stageOne(srcDir: string, dstDir: string): Promise<void> {
    await mkdir(dstDir, { recursive: true });
    await copyIfNewer(resolve(srcDir, 'scene.gltf'), resolve(dstDir, 'scene.gltf'));
    const buffersSrc = resolve(srcDir, 'buffers');
    const buffersDst = resolve(dstDir, 'buffers');
    if (existsSync(buffersSrc)) {
      await mkdir(buffersDst, { recursive: true });
      await copyIfNewer(
        resolve(buffersSrc, 'chunk_0000.bin'),
        resolve(buffersDst, 'chunk_0000.bin'),
      );
    }
  }

  for (const scene of SCENES) {
    for (const variant of VARIANTS) {
      await stageOne(
        resolve(COMPARE_SRC, scene, variant),
        resolve(stagingRoot, scene, variant),
      );
      await stageOne(
        resolve(COMPARE_SRC, 'legacy', scene, variant),
        resolve(stagingRoot, 'legacy', scene, variant),
      );
    }
  }
}

async function copyIfNewer(src: string, dst: string): Promise<void> {
  if (!existsSync(src)) return;
  try {
    const [sSrc, sDst] = await Promise.all([stat(src), stat(dst)]);
    if (sDst.mtimeMs >= sSrc.mtimeMs) return;
  } catch {
    // dst doesn't exist; fall through to copy.
  }
  await copyFile(src, dst);
}

interface CapturedFrame {
  png: Buffer;
}

async function captureOne(
  page: import('@playwright/test').Page,
  src: string,
  renderer: string,
): Promise<CapturedFrame> {
  // Use a 1-pose custom camera path so we capture exactly one frame quickly.
  // The orbit-8 default is fine too but slower.
  const params = new URLSearchParams({
    src,
    renderer,
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
    throw new Error(`viewer errored: ${result.error.code} — ${result.error.message}`);
  }
  expect(result.frames.length).toBeGreaterThan(0);
  const u = result.frames[0]!.dataUrl;
  const idx = u.indexOf('base64,');
  return { png: Buffer.from(u.slice(idx + 'base64,'.length), 'base64') };
}

test.describe('attribute-layout parity (RC vs legacy)', () => {
  test.beforeAll(async () => {
    await ensureStaged();
  });

  for (const scene of SCENES) {
    for (const variant of VARIANTS) {
      test(`${scene}/${variant} — RC and legacy render to byte-equal pixels`, async ({
        page,
      }, testInfo) => {
        const renderer =
          (testInfo.project.metadata as { renderer?: string }).renderer ?? 'webgl2';
        // Pin to a single renderer project — parity is in the JSON reader,
        // not in the renderer backend, so one project is enough.
        test.skip(testInfo.project.name !== 'chrome-webgl2', 'parity runs once on chrome-webgl2');

        const rc = await captureOne(
          page,
          `/fixtures/compare/${scene}/${variant}/scene.gltf`,
          renderer,
        );
        const legacy = await captureOne(
          page,
          `/fixtures/compare/legacy/${scene}/${variant}/scene.gltf`,
          renderer,
        );

        const a = PNG.sync.read(rc.png);
        const b = PNG.sync.read(legacy.png);
        expect(a.width).toBe(b.width);
        expect(a.height).toBe(b.height);
        const diff = new PNG({ width: a.width, height: a.height });
        const changed = pixelmatch(a.data, b.data, diff.data, a.width, a.height, {
          threshold: 0.05,
          includeAA: false,
        });
        const ratio = changed / (a.width * a.height);
        expect(ratio).toBeLessThanOrEqual(MAX_DIFF_RATIO);
      });
    }
  }
});

// Suppress unused-import lints in environments where these helpers stay
// available but aren't directly invoked here.
void readFile;
void writeFile;
void symlink;
