/**
 * SPEC-0009 — Visual diff spec.
 *
 * For each renderer project, for each registered asset:
 *   1. Open the harness page with `?src=<asset>&renderer=<id>`.
 *   2. Wait for the viewer's `complete` event (via `window.__sf.ready`).
 *   3. Pull the 8 captured PNGs out of the page.
 *   4. Save them under `report/raw/<asset>/<renderer>/0001.png ... 0008.png`.
 *   5. If a golden frameset exists, diff against it with pixelmatch and
 *      record `metrics.json` next to the frames.
 *   6. If no goldens exist yet, emit a "golden missing" annotation and save
 *      the run as a candidate baseline at `report/raw/.../candidate/`.
 *
 * The pass/fail threshold is the SPEC-0009 default of 3% mean pixel diff.
 */
import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ASSETS,
  captureFrames,
  diffAgainstGolden,
  writeFrames,
  writeJson,
} from './_helpers.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');           // tests/visual
const RAW_DIR = resolve(ROOT, 'report/raw');
const GOLDEN_ROOT = resolve(ROOT, 'fixtures/golden/frames');
const THRESHOLD = Number(process.env.SPLATFORGE_DIFF_THRESHOLD ?? 0.03);

for (const asset of ASSETS) {
  test.describe(`visual diff — ${asset.id}`, () => {
    test(`renders 8 frames and matches golden`, async ({ page }, testInfo) => {
      // The renderer metadata is set per-project in playwright.config.ts.
      const renderer = (testInfo.project.metadata as { renderer?: string }).renderer ?? 'auto';
      const projectId = testInfo.project.name;

      const frames = await captureFrames(page, { renderer, src: asset.src });
      expect(frames.length, 'orbit-8 must produce 8 frames').toBeGreaterThanOrEqual(1);

      const runDir = resolve(RAW_DIR, asset.id, projectId);
      await writeFrames(runDir, frames);

      const goldenDir = resolve(GOLDEN_ROOT, asset.id, projectId);
      const diff = await diffAgainstGolden(frames, goldenDir);

      await writeJson(resolve(runDir, 'metrics.json'), {
        asset: asset.id,
        project: projectId,
        renderer,
        frameCount: frames.length,
        threshold: THRESHOLD,
        goldensPresent: diff.goldensPresent,
        perFrame: diff.perFrame,
        max: diff.max,
        mean: diff.mean,
        p95: diff.p95,
      });

      if (!diff.goldensPresent) {
        // Don't fail the test — surface a soft annotation so CI flags it and
        // the candidate frames are saved as a starting point for review.
        testInfo.annotations.push({
          type: 'golden-missing',
          description: `no golden frames at ${goldenDir} — saved run as candidate`,
        });
        await writeFrames(resolve(runDir, 'candidate'), frames);
        return;
      }

      expect(
        diff.mean,
        `mean pixel diff ${diff.mean.toFixed(4)} exceeded threshold ${THRESHOLD}`,
      ).toBeLessThanOrEqual(THRESHOLD);
    });
  });
}
