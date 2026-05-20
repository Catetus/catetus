#!/usr/bin/env node
/**
 * Build the aggregate HTML report from raw Playwright outputs.
 *
 * Inputs:
 *   report/raw/<asset>/<project>/metrics.json    (per-renderer diffs)
 *   report/raw/<asset>/<project>/0001.png ...    (captured frames)
 *   report/parity.json                           (matrix aggregate)
 *
 * Outputs:
 *   report/index.html      (parity matrix landing page + diff-report links)
 *   report/diff/<asset>/<project>.html  (per-renderer diff page)
 *
 * The HTML is rendered via `@catetus/report-ui` so the diff CLI and the
 * harness share the same templates.
 */
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderDiffReport, renderParityReport } from '@catetus/report-ui';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');
const RAW_DIR = resolve(ROOT, 'report/raw');
const OUT_INDEX = resolve(ROOT, 'report/index.html');
const PARITY_PATH = resolve(ROOT, 'report/parity.json');
const GOLDEN_ROOT = resolve(ROOT, 'fixtures/golden/frames');

/** Read a PNG file and return a `data:image/png;base64,...` URL or empty. */
async function pngDataUrl(path) {
  if (!existsSync(path)) return '';
  const buf = await readFile(path);
  return `data:image/png;base64,${buf.toString('base64')}`;
}

/** Write a file, mkdir -p its parent. */
async function writeOut(path, contents) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, contents);
}

/** Scan `report/raw/<asset>/<project>/` and build a DiffReportData. */
async function buildDiffData(assetId, projectId) {
  const dir = resolve(RAW_DIR, assetId, projectId);
  const metricsPath = resolve(dir, 'metrics.json');
  if (!existsSync(metricsPath)) return null;
  const metrics = JSON.parse(await readFile(metricsPath, 'utf8'));
  const frames = [];
  for (let i = 1; i <= (metrics.frameCount ?? 8); i++) {
    const name = `${String(i).padStart(4, '0')}.png`;
    const afterPath = resolve(dir, name);
    const beforePath = resolve(GOLDEN_ROOT, assetId, projectId, name);
    const [afterPng, beforePng] = await Promise.all([
      pngDataUrl(afterPath),
      pngDataUrl(beforePath),
    ]);
    if (!afterPng) continue;
    frames.push({
      index: i,
      beforePng: beforePng || afterPng,   // fall back to after if no golden
      afterPng,
      diffPng: afterPng,                  // overlay not emitted by harness — reuse after
      diffRatio: metrics.perFrame?.[i - 1] ?? 0,
    });
  }
  return {
    asset: `${assetId} / ${projectId}`,
    threshold: metrics.threshold ?? 0.03,
    metrics: {
      max: Number.isFinite(metrics.max) ? metrics.max : 0,
      mean: Number.isFinite(metrics.mean) ? metrics.mean : 0,
      p95: Number.isFinite(metrics.p95) ? metrics.p95 : 0,
    },
    frames,
    cameraPath: 'orbit-8',
    frameSize: '512x512',
  };
}

async function buildParityHtml() {
  if (!existsSync(PARITY_PATH)) return '';
  const parity = JSON.parse(await readFile(PARITY_PATH, 'utf8'));
  return parity.assets
    .map((entry) => {
      const matrix = {};
      for (const [proj, cell] of Object.entries(entry.matrix)) {
        matrix[proj] = {
          visualScore: cell.visualScore ?? 0,
          warnings: cell.goldensPresent ? [] : ['golden_missing'],
        };
      }
      return renderParityReport({ asset: entry.asset, matrix });
    })
    .join('\n<hr />\n');
}

async function main() {
  // Per-renderer diff pages.
  const links = [];
  if (existsSync(RAW_DIR)) {
    const assets = (await readdir(RAW_DIR, { withFileTypes: true })).filter(
      (e) => e.isDirectory() && !e.name.startsWith('_'),
    );
    for (const a of assets) {
      const projects = (await readdir(resolve(RAW_DIR, a.name), { withFileTypes: true })).filter(
        (e) => e.isDirectory(),
      );
      for (const p of projects) {
        const data = await buildDiffData(a.name, p.name);
        if (!data) continue;
        const html = renderDiffReport(data);
        const outPath = resolve(ROOT, 'report/diff', a.name, `${p.name}.html`);
        await writeOut(outPath, html);
        links.push({ asset: a.name, project: p.name, href: `diff/${a.name}/${p.name}.html` });
      }
    }
  }

  const parityHtml = await buildParityHtml();
  const linksHtml = links
    .map((l) => `<li><a href="${l.href}">${l.asset} / ${l.project}</a></li>`)
    .join('');

  const index = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Catetus report</title>
<style>
  body { background:#0b0d10; color:#d7dde4; font-family: ui-monospace, Menlo, monospace; padding: 24px; }
  a { color:#5ac8fa; }
  h1 { font-size: 18px; }
  h2 { font-size: 14px; color:#7a8591; text-transform:uppercase; letter-spacing:.08em; }
</style></head><body>
<h1>Catetus report</h1>
<h2>Per-renderer diffs</h2>
<ul>${linksHtml || '<li><em>no runs found</em></li>'}</ul>
<h2>Parity matrix</h2>
${parityHtml || '<p><em>no parity.json yet</em></p>'}
</body></html>`;

  await writeOut(OUT_INDEX, index);
  process.stderr.write(`[report] wrote ${OUT_INDEX}\n`);
}

main().catch((err) => { console.error(err); process.exit(1); });
