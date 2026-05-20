/**
 * SPEC-0010 — Viewer parity spec.
 *
 * Each project (renderer) records its own metrics.json under
 * `report/raw/<asset>/<project>/`. After all projects have run, the
 * `parity.json` aggregator collates them into the SPEC-0010 schema.
 *
 * Because Playwright runs projects in sequence (or in parallel with shared
 * outputs), we don't try to write parity.json from inside a single test.
 * Instead, the very-last test in this file scans `report/raw/` and writes
 * the aggregate. Running `playwright test` once across all projects is
 * enough to produce a complete matrix; running just one project produces a
 * partial matrix, which is also fine.
 */
import { test } from '@playwright/test';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import {
  ASSETS,
  captureFrames,
  diffAgainstGolden,
  writeJson,
  ratioToScore,
} from './_helpers.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');
const RAW_DIR = resolve(ROOT, 'report/raw');
const GOLDEN_ROOT = resolve(ROOT, 'fixtures/golden/frames');
const PARITY_PATH = resolve(ROOT, 'report/parity.json');

for (const asset of ASSETS) {
  test.describe(`viewer parity — ${asset.id}`, () => {
    test(`record this project's metrics`, async ({ page }, testInfo) => {
      const renderer = (testInfo.project.metadata as { renderer?: string }).renderer ?? 'auto';
      const projectId = testInfo.project.name;

      // The diff spec already wrote frames + metrics; if it ran in this same
      // playwright invocation we could short-circuit, but re-capturing keeps
      // parity.spec runnable in isolation. The deterministic camera path
      // makes the run cost cheap and the bytes identical.
      const frames = await captureFrames(page, { renderer, src: asset.src });
      const goldenDir = resolve(GOLDEN_ROOT, asset.id, projectId);
      const diff = await diffAgainstGolden(frames, goldenDir);

      const score = diff.goldensPresent ? ratioToScore(diff.mean) : NaN;
      await writeJson(resolve(RAW_DIR, asset.id, projectId, 'parity-cell.json'), {
        asset: asset.id,
        project: projectId,
        renderer,
        visualScore: score,
        diffMean: diff.mean,
        diffMax: diff.max,
        diffP95: diff.p95,
        goldensPresent: diff.goldensPresent,
      });
    });
  });
}

/**
 * Aggregator. Runs as its own test so Playwright will only execute it after
 * every project's parity-cell.json is on disk (Playwright runs projects
 * sequentially by default in this config: workers=1).
 *
 * If a cell is missing we still write the parity.json — missing cells are
 * simply absent from the matrix. The reporter handles that gracefully.
 */
test('aggregate parity.json', async () => {
  if (!existsSync(RAW_DIR)) return;
  const assets = await readdir(RAW_DIR, { withFileTypes: true });

  const result: Record<string, { asset: string; matrix: Record<string, unknown> }> = {};
  for (const assetEntry of assets) {
    if (!assetEntry.isDirectory()) continue;
    if (assetEntry.name.startsWith('_')) continue;
    const assetId = assetEntry.name;
    const projectsDir = resolve(RAW_DIR, assetId);
    const projects = await readdir(projectsDir, { withFileTypes: true });
    const matrix: Record<string, unknown> = {};
    for (const p of projects) {
      if (!p.isDirectory()) continue;
      const cellPath = resolve(projectsDir, p.name, 'parity-cell.json');
      if (!existsSync(cellPath)) continue;
      const cell = JSON.parse(await readFile(cellPath, 'utf8')) as {
        visualScore: number;
        diffMean: number;
        goldensPresent: boolean;
      };
      matrix[p.name] = {
        visualScore: Number.isFinite(cell.visualScore) ? cell.visualScore : null,
        diffMean: cell.diffMean,
        goldensPresent: cell.goldensPresent,
      };
    }
    result[assetId] = { asset: assetId, matrix };
  }

  // SPEC-0010 schema is `{ asset, matrix }` per asset. We emit `assets: [...]`
  // here so multiple assets can share a single file; the report builder
  // un-wraps per-asset pages.
  await writeJson(PARITY_PATH, {
    schema: 'catetus.parity/1',
    assets: Object.values(result),
  });
});
