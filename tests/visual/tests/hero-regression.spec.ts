/**
 * Visual regression harness for the SplatForge homepage hero.
 *
 * Prevents shipping a "pink blob" hero again by, for each
 * (scene, preset) ∈ {bonsai, bicycle, garden, stump}
 *                ×  {web-mobile, web-desktop, quest-browser, hero, quality-max}:
 *
 *   1. Running the CLI (`splatforge optimize --chunked`) on the source PLY,
 *   2. Loading `/preview-hero.html?src=/tmp-scenes/<scene>/<preset>/scene.gltf`,
 *   3. Waiting for `[data-hero-status]` text to reach `live · auto-orbit`,
 *   4. Asserting no console errors (other than expected warnings),
 *   5. Screenshotting `[data-hero-canvas]`,
 *   6. Computing mean luminance + RGB variance over the canvas pixels,
 *   7. Asserting `0.05 < luminance < 0.95` AND `variance > 0.005`.
 *
 * A second test loads `apps/web/public/hero-scene/scene.gltf` directly and
 * applies the same thresholds (warning-only if the production hero fails —
 * it is known-renderable-but-ugly per CLAUDE.md).
 *
 * Optional bonus: when `OPENAI_API_KEY` is set, each screenshot is passed to
 * GPT-4o-mini-vision with a 1-shot classifier prompt. Categories b/c
 * (all-black / sparse fragments) fail the case.
 */
import { test, expect, type Page } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const REPO_ROOT = resolve(__dirname, '../../..');
const CLI_BIN = process.env.SPLATFORGE_BIN
  ?? resolve(REPO_ROOT, 'target/release/splatforge');
const TMP_REGRESSION = resolve(REPO_ROOT, 'tmp/regression');
const SCREENSHOT_DIR = resolve(__dirname, '../__screenshots__');
const PRODUCTION_HERO_GLTF = '/hero-scene/scene.gltf';

const SCENES = ['bonsai', 'bicycle', 'garden', 'stump'] as const;
const PRESETS = ['web-mobile', 'web-desktop', 'quest-browser', 'hero', 'quality-max'] as const;

/**
 * Candidate PLY locations checked in order. The first match wins.
 * Override via `SPLATFORGE_PLY_<SCENE>` env var (uppercased).
 *
 * These paths are agent-machine conventions (`/private/tmp/...`) and ad-hoc
 * 4090-sync mirrors; on CI the workflow stages the PLYs into
 * `tests/visual/fixtures/scenes/<scene>.ply` before invoking the test.
 */
const PLY_CANDIDATES: Record<(typeof SCENES)[number], string[]> = {
  bonsai: [
    '/private/tmp/bonsai-hero.ply',
    '/private/tmp/4090-scenes-sync/bonsai.ply',
    '/private/tmp/sbench/scenes/bonsai.ply',
    resolve(REPO_ROOT, 'tests/visual/fixtures/scenes/bonsai.ply'),
  ],
  bicycle: [
    '/private/tmp/bicycle-hero.ply',
    '/private/tmp/4090-scenes-sync/bicycle.ply',
    '/private/tmp/sbench/scenes/bicycle.ply',
    resolve(REPO_ROOT, 'tests/visual/fixtures/scenes/bicycle.ply'),
  ],
  garden: [
    '/private/tmp/garden-hero.ply',
    '/private/tmp/4090-scenes-sync/garden.ply',
    '/private/tmp/sbench/scenes/garden.ply',
    resolve(REPO_ROOT, 'tests/visual/fixtures/scenes/garden.ply'),
  ],
  stump: [
    '/private/tmp/stump-hero.ply',
    '/private/tmp/4090-scenes-sync/stump.ply',
    '/private/tmp/sbench/scenes/stump.ply',
    resolve(REPO_ROOT, 'tests/visual/fixtures/scenes/stump.ply'),
  ],
};

function findPly(scene: (typeof SCENES)[number]): string | null {
  const override = process.env[`SPLATFORGE_PLY_${scene.toUpperCase()}`];
  if (override && existsSync(override)) return override;
  for (const p of PLY_CANDIDATES[scene]) {
    if (existsSync(p)) return p;
  }
  return null;
}

/** Run the CLI to produce a chunked gltf at the requested out path. */
function runOptimize(plyPath: string, preset: string, outGltf: string): {
  ok: boolean;
  stdout: string;
  stderr: string;
  durationMs: number;
} {
  mkdirSync(dirname(outGltf), { recursive: true });
  const started = Date.now();
  const res = spawnSync(
    CLI_BIN,
    ['optimize', plyPath, '--preset', preset, '--chunked', '--out', outGltf],
    { encoding: 'utf8', timeout: 240_000 },
  );
  return {
    ok: res.status === 0 && existsSync(outGltf),
    stdout: res.stdout ?? '',
    stderr: res.stderr ?? '',
    durationMs: Date.now() - started,
  };
}

/**
 * Compute mean luminance and RGB-variance for the pixels in `png`.
 *
 * Luminance uses Rec. 709 weights, normalised to 0..1.
 * Variance is the mean over channels of `E[(c - E[c])^2]`, normalised so a
 * pure-black or pure-uniform image yields 0 and a 50/50 black/white image
 * yields ~0.25.
 */
function computeStats(png: PNG): { meanLuma: number; variance: number; nPixels: number } {
  const { data, width, height } = png;
  const n = width * height;
  let sumR = 0, sumG = 0, sumB = 0, sumY = 0;
  let sumR2 = 0, sumG2 = 0, sumB2 = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    sumR += r; sumG += g; sumB += b;
    sumR2 += r * r; sumG2 += g * g; sumB2 += b * b;
    sumY += 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  const meanR = sumR / n;
  const meanG = sumG / n;
  const meanB = sumB / n;
  const varR = Math.max(0, sumR2 / n - meanR * meanR);
  const varG = Math.max(0, sumG2 / n - meanG * meanG);
  const varB = Math.max(0, sumB2 / n - meanB * meanB);
  return {
    meanLuma: sumY / n,
    variance: (varR + varG + varB) / 3,
    nPixels: n,
  };
}

/**
 * Capture the live framebuffer of `[data-hero-canvas]`.
 *
 * Element-level `Locator.screenshot()` on an auto-rotating WebGL canvas hangs
 * because playwright's "wait for stability" rule never passes. A full-page
 * screenshot with the canvas's bounding box as a clip is reliable, and it
 * comes from the compositor so `preserveDrawingBuffer: false` does not matter.
 */
async function captureCanvas(page: Page): Promise<Buffer> {
  const box = await page.evaluate(() => {
    const el = document.querySelector<HTMLCanvasElement>('[data-hero-canvas]');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
  if (!box || box.width < 4 || box.height < 4) {
    throw new Error(`[data-hero-canvas] missing or zero-sized: ${JSON.stringify(box)}`);
  }
  return page.screenshot({
    type: 'png',
    animations: 'allow',
    clip: {
      x: Math.max(0, Math.floor(box.x)),
      y: Math.max(0, Math.floor(box.y)),
      width: Math.max(4, Math.floor(box.width)),
      height: Math.max(4, Math.floor(box.height)),
    },
  });
}

/** Filter console errors that are known-noisy and not failure conditions. */
const EXPECTED_CONSOLE_PATTERNS: RegExp[] = [
  /favicon/i,
  /WebGL warning/i,
  /Download the .* error/i,
  /\[viewer\] no manifest/i, // some chunked outputs only emit summary
  /AudioContext was prevented/i,
];

function isExpectedConsole(text: string): boolean {
  return EXPECTED_CONSOLE_PATTERNS.some((re) => re.test(text));
}

/**
 * Optional vision-LLM "antimatter15-eye" check. Disabled unless
 * `OPENAI_API_KEY` is set in the environment.
 *
 * Returns one of `'a' | 'b' | 'c' | null` where:
 *   a -> recognizable 3D scene (PASS),
 *   b -> all black (FAIL),
 *   c -> sparse fragments (FAIL),
 *   null -> classifier unavailable or unsure (treated as PASS).
 */
async function visionCheck(pngBuf: Buffer, label: string): Promise<string | null> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;
  const b64 = pngBuf.toString('base64');
  const body = {
    model: process.env.SPLATFORGE_VISION_MODEL ?? 'gpt-4o-mini',
    messages: [{
      role: 'user',
      content: [
        {
          type: 'text',
          text:
            `You are inspecting a screenshot of a 3D Gaussian splat scene named "${label}". ` +
            `Reply with EXACTLY one letter, no other text:\n` +
            `  a — a recognizable 3D scene (any quality)\n` +
            `  b — entirely black, blank, or pink-blob saturated\n` +
            `  c — only sparse / scattered fragments, no coherent object`,
        },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${b64}` } },
      ],
    }],
    max_tokens: 4,
    temperature: 0,
  };
  try {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(body),
      // 20s ceiling so a hung API call never blocks the suite.
      signal: AbortSignal.timeout(20_000),
    });
    if (!r.ok) return null;
    const j: any = await r.json();
    const txt: string = j?.choices?.[0]?.message?.content ?? '';
    const m = txt.trim().toLowerCase().match(/^[abc]/);
    return m ? m[0] : null;
  } catch {
    return null;
  }
}

mkdirSync(SCREENSHOT_DIR, { recursive: true });
mkdirSync(TMP_REGRESSION, { recursive: true });

// ---------- per-(scene,preset) suite ----------

test.describe('hero-regression: preset × scene grid', () => {
  for (const scene of SCENES) {
    const ply = findPly(scene);
    for (const preset of PRESETS) {
      const id = `${scene}-${preset}`;
      test(id, async ({ page }, testInfo) => {
        // CLI optimize runs inline. On a quiet box bonsai-web-mobile is ~90s;
        // contention from other agents has pushed it past 3 min.
        testInfo.setTimeout(360_000);
        if (!ply) {
          testInfo.skip(true, `no source PLY found for scene "${scene}" — set SPLATFORGE_PLY_${scene.toUpperCase()}`);
          return;
        }
        if (!existsSync(CLI_BIN)) {
          throw new Error(`splatforge CLI not built: expected ${CLI_BIN}. Run \`cargo build --release -p splatforge-cli\`.`);
        }

        const outDir = resolve(TMP_REGRESSION, scene, preset);
        const outGltf = resolve(outDir, 'scene.gltf');
        // Reuse a previously-generated gltf if present; the CLI is deterministic.
        const opt = existsSync(outGltf)
          ? { ok: true, stdout: '', stderr: '', durationMs: 0 }
          : runOptimize(ply, preset, outGltf);
        testInfo.attach(`${id}-cli-stderr.txt`, { body: opt.stderr, contentType: 'text/plain' }).catch(() => {});
        if (!opt.ok) {
          throw new Error(
            `splatforge optimize failed for ${id}\n` +
            `  cmd: ${CLI_BIN} optimize ${ply} --preset ${preset} --chunked --out ${outGltf}\n` +
            `  duration: ${opt.durationMs} ms\n` +
            `  stderr: ${opt.stderr.slice(-2000)}`,
          );
        }

        const consoleErrors: string[] = [];
        page.on('console', (msg) => {
          if (msg.type() === 'error') {
            const txt = msg.text();
            if (!isExpectedConsole(txt)) consoleErrors.push(txt);
          }
        });
        page.on('pageerror', (e) => consoleErrors.push(String(e?.message ?? e)));

        const url = `/preview-hero.html?src=/tmp-scenes/${scene}/${preset}/scene.gltf`;
        await page.goto(url, { waitUntil: 'load' });

        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-hero-status]');
            return !!el && /live · auto-orbit/.test(el.textContent ?? '');
          },
          null,
          { timeout: 30_000 },
        );

        // Settle a couple of frames so the splat sort + first complete redraw lands.
        await page.waitForTimeout(500);

        expect(
          consoleErrors,
          `unexpected console errors for ${id}:\n${consoleErrors.join('\n')}`,
        ).toEqual([]);

        const canvas = page.locator('[data-hero-canvas]');
        await expect(canvas).toBeVisible();
        const buf = await captureCanvas(page);

        const screenshotPath = resolve(SCREENSHOT_DIR, `${id}.png`);
        writeFileSync(screenshotPath, buf);
        testInfo.attach(`${id}.png`, { path: screenshotPath, contentType: 'image/png' }).catch(() => {});

        const png = PNG.sync.read(buf);
        const { meanLuma, variance, nPixels } = computeStats(png);
        const stats = { id, meanLuma, variance, nPixels, gltfBytes: statSync(outGltf).size };
        testInfo.attach(`${id}-stats.json`, {
          body: JSON.stringify(stats, null, 2),
          contentType: 'application/json',
        }).catch(() => {});

        expect(meanLuma, `mean luminance for ${id} (got ${meanLuma.toFixed(4)})`).toBeGreaterThan(0.05);
        expect(meanLuma, `mean luminance for ${id} (got ${meanLuma.toFixed(4)})`).toBeLessThan(0.95);
        expect(variance, `RGB variance for ${id} (got ${variance.toFixed(5)})`).toBeGreaterThan(0.005);

        const verdict = await visionCheck(buf, id);
        if (verdict === 'b' || verdict === 'c') {
          throw new Error(`vision-LLM flagged ${id} as category ${verdict} (b=blank/pink, c=sparse)`);
        }
      });
    }
  }
});

// ---------- production hero (warning-only) ----------

test('hero-regression: production hero asset', async ({ page }, testInfo) => {
  const consoleErrors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const txt = msg.text();
      if (!isExpectedConsole(txt)) consoleErrors.push(txt);
    }
  });
  page.on('pageerror', (e) => consoleErrors.push(String(e?.message ?? e)));

  await page.goto(`/preview-hero.html?src=${PRODUCTION_HERO_GLTF}`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-hero-status]');
      return !!el && /live · auto-orbit/.test(el.textContent ?? '');
    },
    null,
    { timeout: 30_000 },
  );
  await page.waitForTimeout(500);

  const canvas = page.locator('[data-hero-canvas]');
  await expect(canvas).toBeVisible();
  const buf = await captureCanvas(page);

  const baselinePath = resolve(SCREENSHOT_DIR, 'production-hero.png');
  writeFileSync(baselinePath, buf);
  testInfo.attach('production-hero.png', { path: baselinePath, contentType: 'image/png' }).catch(() => {});

  const png = PNG.sync.read(buf);
  const { meanLuma, variance } = computeStats(png);
  testInfo.attach('production-hero-stats.json', {
    body: JSON.stringify({ meanLuma, variance, consoleErrors }, null, 2),
    contentType: 'application/json',
  }).catch(() => {});

  const inRange = meanLuma > 0.05 && meanLuma < 0.95 && variance > 0.005;
  if (!inRange) {
    // CLAUDE.md: production hero is "renderable but ugly" — warn, don't block.
    // eslint-disable-next-line no-console
    console.warn(
      `[hero-regression] PRODUCTION HERO outside thresholds (luma=${meanLuma.toFixed(4)} variance=${variance.toFixed(5)}). ` +
      `This is the known reverted-to 91k/2.5MB asset; flagging for follow-up but not failing the suite.`,
    );
  } else {
    expect(meanLuma).toBeGreaterThan(0.05);
    expect(meanLuma).toBeLessThan(0.95);
    expect(variance).toBeGreaterThan(0.005);
  }
});
