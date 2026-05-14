/**
 * Shared helpers for the visual-diff and viewer-parity Playwright specs.
 *
 * Kept separate from the spec files so the diff-cli script (which does not
 * use the Playwright test runner) can reuse the same primitives via plain
 * dynamic import.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import type { Page } from '@playwright/test';

/** Reasonable upper bound on how long the harness may take to finish loading. */
export const HARNESS_TIMEOUT_MS = 60_000;

/** Stable per-test asset list. Extend as fixtures land. */
export const ASSETS: Array<{ id: string; src: string }> = [
  { id: 'tiny_cube', src: '/fixtures/tiny/cube.gltf' },
];

/** One frame captured from `window.__sf.frames` in the harness page. */
export interface CapturedFrame {
  /** 1-indexed frame number. */
  index: number;
  /** PNG bytes. */
  png: Buffer;
}

/** Aggregate result of comparing a renderer's run to its golden frames. */
export interface DiffResult {
  /** Per-frame ratios in 0..1 (changed pixels / total pixels). */
  perFrame: number[];
  /** Aggregate metrics matching SPEC-0009. */
  max: number;
  mean: number;
  p95: number;
  /** True if every frame had a matching golden file. */
  goldensPresent: boolean;
}

/**
 * Drive the harness page to the asset under the given renderer, then read
 * back the 8 captured frames.
 *
 * The harness page exposes `window.__sf` — see `harness/page.html`.
 */
export async function captureFrames(
  page: Page,
  opts: { renderer: string; src: string; seed?: number },
): Promise<CapturedFrame[]> {
  const params = new URLSearchParams({
    src: opts.src,
    renderer: opts.renderer,
    ...(opts.seed !== undefined ? { seed: String(opts.seed) } : {}),
  });
  await page.goto(`/page.html?${params.toString()}`, { waitUntil: 'load' });

  // Wait until the harness flips __sf.ready = true (or reports an error).
  await page.waitForFunction(
    () => {
      // @ts-expect-error injected by harness
      const sf = window.__sf;
      return sf && (sf.ready === true || sf.error !== null);
    },
    null,
    { timeout: HARNESS_TIMEOUT_MS },
  );

  const result = await page.evaluate(() => {
    // @ts-expect-error injected by harness
    const sf = window.__sf;
    return {
      error: sf.error,
      warnings: sf.warnings,
      frames: sf.frames.map((f: { index: number; dataUrl: string }) => ({
        index: f.index,
        dataUrl: f.dataUrl,
      })),
    } as {
      error: { code: string; message: string } | null;
      warnings: Array<{ code: string; message: string }>;
      frames: Array<{ index: number; dataUrl: string }>;
    };
  });

  if (result.error) {
    throw new Error(`viewer reported error: ${result.error.code} — ${result.error.message}`);
  }
  if (result.frames.length === 0) {
    throw new Error('viewer produced 0 frames');
  }

  return result.frames.map((f) => ({
    index: f.index,
    png: dataUrlToBuffer(f.dataUrl),
  }));
}

/** Decode a `data:image/png;base64,...` URL to its raw bytes. */
export function dataUrlToBuffer(dataUrl: string): Buffer {
  const idx = dataUrl.indexOf('base64,');
  if (idx < 0) throw new Error('expected base64 data URL');
  return Buffer.from(dataUrl.slice(idx + 'base64,'.length), 'base64');
}

/** Write a frame's PNG to `dir/0001.png` etc. */
export async function writeFrames(dir: string, frames: CapturedFrame[]): Promise<void> {
  await mkdir(dir, { recursive: true });
  for (const f of frames) {
    const name = `${String(f.index).padStart(4, '0')}.png`;
    await writeFile(resolve(dir, name), f.png);
  }
}

/**
 * Compare a freshly-captured run against a golden directory.
 *
 * If the golden directory or any individual golden file is missing, that
 * frame's ratio is recorded as `NaN` and `goldensPresent` is `false`. We
 * still return a partial result so callers can save the run as a new
 * candidate golden.
 */
export async function diffAgainstGolden(
  frames: CapturedFrame[],
  goldenDir: string,
): Promise<DiffResult> {
  const ratios: number[] = [];
  let goldensPresent = true;

  for (const f of frames) {
    const goldenPath = resolve(goldenDir, `${String(f.index).padStart(4, '0')}.png`);
    if (!existsSync(goldenPath)) {
      goldensPresent = false;
      ratios.push(NaN);
      continue;
    }
    const goldenBuf = await readFile(goldenPath);
    const a = PNG.sync.read(goldenBuf);
    const b = PNG.sync.read(f.png);
    if (a.width !== b.width || a.height !== b.height) {
      // Size mismatch is a hard failure — treat as 100% diff.
      ratios.push(1);
      continue;
    }
    const diff = new PNG({ width: a.width, height: a.height });
    const changed = pixelmatch(a.data, b.data, diff.data, a.width, a.height, {
      threshold: 0.1,
      includeAA: false,
    });
    ratios.push(changed / (a.width * a.height));
  }

  const valid = ratios.filter((r) => !Number.isNaN(r));
  return {
    perFrame: ratios,
    max: valid.length ? Math.max(...valid) : NaN,
    mean: valid.length ? valid.reduce((s, x) => s + x, 0) / valid.length : NaN,
    p95: percentile(valid, 0.95),
    goldensPresent,
  };
}

/** Linear-interp percentile over a sorted copy of `xs`. */
export function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return NaN;
  const sorted = [...xs].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! * (hi - idx) + sorted[hi]! * (idx - lo);
}

/** Convert a 0..1 diff-ratio into the parity-matrix visual score (1 - ratio). */
export function ratioToScore(ratio: number): number {
  return Math.max(0, Math.min(1, 1 - ratio));
}

/** Write a JSON file atomically-enough for CI (mkdir -p + write). */
export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2) + '\n');
}
