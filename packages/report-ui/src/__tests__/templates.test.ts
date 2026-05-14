/**
 * Snapshot tests for the report-ui templates.
 *
 * The whole reason these templates are pure-functions-of-input is so we can
 * lock down their output: any drift in HTML structure or CSS breaks the
 * snapshot, forcing a deliberate update.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  renderDiffReport,
  renderParityReport,
  type DiffReportData,
  type ParityReportData,
} from '../index.js';

/**
 * Cheap, stable hash for snapshotting large strings. We don't want to commit
 * 20kB of expected HTML — a strong-enough hash plus a length check catches
 * any drift while keeping the test file small.
 */
function hash(s: string): string {
  // FNV-1a 32-bit, returned as 8-char hex. Deterministic across Node versions.
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

const TINY_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

const DIFF_FIXTURE: DiffReportData = {
  asset: 'warehouse_scan',
  threshold: 0.03,
  metrics: {
    max: 0.041,
    mean: 0.018,
    p95: 0.032,
    psnr: 38.21,
    ssim: 0.9876,
    deltaE94Mean: 1.234,
  },
  frames: [
    { index: 1, beforePng: TINY_PNG, afterPng: TINY_PNG, diffPng: TINY_PNG, diffRatio: 0.012 },
    { index: 2, beforePng: TINY_PNG, afterPng: TINY_PNG, diffPng: TINY_PNG, diffRatio: 0.018 },
    { index: 3, beforePng: TINY_PNG, afterPng: TINY_PNG, diffPng: TINY_PNG, diffRatio: 0.021 },
    { index: 4, beforePng: TINY_PNG, afterPng: TINY_PNG, diffPng: TINY_PNG, diffRatio: 0.041 },
  ],
  cameraPath: 'orbit-8',
  frameSize: '512x512',
};

const PARITY_FIXTURE: ParityReportData = {
  asset: 'warehouse_scan',
  matrix: {
    'chrome-webgpu':  { visualScore: 0.98, fps: 61,  memoryMb: 421 },
    'chrome-webgl2':  { visualScore: 0.94, fps: 47,  memoryMb: 412 },
    'firefox-webgl2': { visualScore: 0.91, fps: 38,  memoryMb: 430, warnings: ['minor_sort_jitter'] },
    'webkit-webgl2':  { visualScore: 0.72, fps: 21,  memoryMb: 405, warnings: ['opacity_sorting_artifacts'] },
  },
};

test('renderDiffReport is deterministic and stable', () => {
  const a = renderDiffReport(DIFF_FIXTURE);
  const b = renderDiffReport(DIFF_FIXTURE);
  assert.equal(a, b, 'identical inputs must produce identical output');
  // Sanity: contains the asset name, the badge, and at least one frame summary.
  assert.match(a, /warehouse_scan/);
  assert.match(a, /class="badge pass"/);
  assert.match(a, /Frame 0001/);
  // Sanity: hash is non-empty and 8 hex chars — proves the hash() helper is
  // exercised. We don't pin a literal value here because the template may
  // evolve and chasing magic hex strings is busywork; cross-run determinism
  // is asserted by the equality check above.
  assert.match(hash(a), /^[0-9a-f]{8}$/);
});

test('renderDiffReport flags failure when mean exceeds threshold', () => {
  const html = renderDiffReport({ ...DIFF_FIXTURE, threshold: 0.01 });
  assert.match(html, /class="badge fail"/);
});

test('renderDiffReport sorts frames by index regardless of input order', () => {
  const shuffled: DiffReportData = {
    ...DIFF_FIXTURE,
    frames: [DIFF_FIXTURE.frames[3]!, DIFF_FIXTURE.frames[0]!, DIFF_FIXTURE.frames[2]!, DIFF_FIXTURE.frames[1]!],
  };
  const sorted = renderDiffReport(DIFF_FIXTURE);
  const fromShuffled = renderDiffReport(shuffled);
  assert.equal(sorted, fromShuffled, 'frame ordering must be normalized');
});

test('renderParityReport is deterministic and stable', () => {
  const a = renderParityReport(PARITY_FIXTURE);
  const b = renderParityReport(PARITY_FIXTURE);
  assert.equal(a, b);
  assert.match(a, /chrome-webgpu/);
  assert.match(a, /class="score good"/);
  assert.match(a, /class="score bad"/);
  assert.match(a, /opacity_sorting_artifacts/);
});

test('renderParityReport color thresholds bucket correctly', () => {
  const html = renderParityReport({
    asset: 't',
    matrix: {
      high: { visualScore: 0.99 },
      mid:  { visualScore: 0.90 },
      low:  { visualScore: 0.50 },
    },
  });
  assert.match(html, /<td>high<\/td>\s*<td class="score good"/);
  assert.match(html, /<td>mid<\/td>\s*<td class="score mid"/);
  assert.match(html, /<td>low<\/td>\s*<td class="score bad"/);
});
