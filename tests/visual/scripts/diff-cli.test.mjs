#!/usr/bin/env node
/**
 * Self-contained unit test for diff-cli.mjs's diff math.
 *
 * Run: `node tests/visual/scripts/diff-cli.test.mjs`
 *
 * Covers:
 *   - aggregateMetrics on empty / typical / identical inputs
 *   - pixelDiffPair with two identical 4x4 PNGs -> ratio 0
 *   - pixelDiffPair with two different 4x4 PNGs -> ratio > 0
 *
 * Exits non-zero on any assertion failure. No test framework — keeps the
 * dep surface small so this works in any sandbox with Node 20+ and pngjs.
 */
import { aggregateMetrics, pixelDiffPair } from './diff-cli.mjs';

let failures = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    failures++;
  } else {
    console.log(`  ok: ${msg}`);
  }
}
function approxEq(a, b, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

/* ------------------------- aggregateMetrics ----------------------- */
{
  const m = aggregateMetrics([]);
  assert(m.max === 0 && m.mean === 0 && m.p95 === 0 && m.count === 0, 'empty input -> zeros');
}
{
  const m = aggregateMetrics([0, 0, 0, 0]);
  assert(m.max === 0 && m.mean === 0 && m.p95 === 0 && m.count === 4, 'all-zero -> zero metrics');
}
{
  const m = aggregateMetrics([0.1, 0.2, 0.3, 0.4]);
  assert(approxEq(m.max, 0.4), `max=0.4 got ${m.max}`);
  assert(approxEq(m.mean, 0.25), `mean=0.25 got ${m.mean}`);
  assert(m.p95 > 0.35 && m.p95 <= 0.4, `p95 in (0.35, 0.4] got ${m.p95}`);
  assert(m.count === 4, 'count=4');
}

/* ------------------------- pixelDiffPair -------------------------- */
let PNG;
try {
  ({ PNG } = await import('pngjs'));
} catch {
  console.warn('skipping pixelDiffPair: pngjs not installed');
  process.exit(failures === 0 ? 0 : 1);
}

function makePng(w, h, fill) {
  const png = new PNG({ width: w, height: h });
  for (let i = 0; i < png.data.length; i += 4) {
    png.data[i] = fill[0];
    png.data[i + 1] = fill[1];
    png.data[i + 2] = fill[2];
    png.data[i + 3] = fill[3];
  }
  return PNG.sync.write(png);
}

{
  const a = makePng(4, 4, [128, 128, 128, 255]);
  const b = makePng(4, 4, [128, 128, 128, 255]);
  const { ratio } = await pixelDiffPair(a, b);
  assert(ratio === 0, `identical 4x4 -> ratio 0, got ${ratio}`);
}
{
  const a = makePng(4, 4, [0, 0, 0, 255]);
  const b = makePng(4, 4, [255, 255, 255, 255]);
  const { ratio } = await pixelDiffPair(a, b);
  assert(ratio > 0, `differing 4x4 -> ratio > 0, got ${ratio}`);
  assert(ratio <= 1, `ratio <= 1, got ${ratio}`);
}
{
  // Different sizes -> ratio 1.
  const a = makePng(4, 4, [0, 0, 0, 255]);
  const b = makePng(8, 8, [0, 0, 0, 255]);
  const { ratio } = await pixelDiffPair(a, b);
  assert(ratio === 1, `size mismatch -> ratio 1, got ${ratio}`);
}

if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}
console.log('all diff-cli.test.mjs assertions passed');
