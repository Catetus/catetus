/**
 * Playwright configuration for the hero-regression visual harness.
 *
 * Separate from `playwright.config.ts` so the hero-regression run uses its
 * own static server (rooted at `apps/web/public`, with `tmp/regression`
 * mounted under `/tmp-scenes/`) and never collides with the SPEC-0009 /
 * SPEC-0010 harness ports.
 */
import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.HERO_REGRESSION_PORT ?? 4319);
const BASE = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './tests',
  testMatch: /hero-regression\.spec\.ts$/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  // Each (scene, preset) optimize + load + screenshot easily takes 30s+,
  // and the suite covers up to 5 presets × 4 scenes = 20 cases.
  // Each test runs splatforge optimize + viewer load + screenshot inline.
  // Per-test override via testInfo.setTimeout still applies.
  timeout: 360_000,
  globalTimeout: 3_600_000,
  expect: { timeout: 30_000 },
  reporter: [['list'], ['json', { outputFile: 'report/hero/_playwright.json' }]],
  outputDir: 'report/hero/_artifacts',

  use: {
    baseURL: BASE,
    headless: true,
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
    timezoneId: 'UTC',
    locale: 'en-US',
  },

  projects: [
    {
      name: 'chrome-webgl2',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    command: `node harness/hero-server.mjs --port ${PORT}`,
    url: `${BASE}/preview-hero.html`,
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'pipe',
    timeout: 30_000,
    env: { HERO_REGRESSION_PORT: String(PORT) },
  },
});
