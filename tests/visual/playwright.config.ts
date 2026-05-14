/**
 * Playwright configuration for the SplatForge visual-diff (SPEC-0009) and
 * viewer-parity (SPEC-0010) harness.
 *
 * Each renderer is a separate Playwright "project" so the report tree has
 * stable per-renderer subdirectories. The webServer block starts the tiny
 * static server in `harness/server.mjs` which serves:
 *   - `harness/page.html`            (mounted at `/page.html`)
 *   - the built viewer dist          (mounted at `/viewer/`)
 *   - the fixtures directory         (mounted at `/fixtures/`)
 *
 * Tests pass `?renderer=<id>&src=<asset-path>` to `page.html`. The harness
 * page reads `renderer` and constructs a `SplatForgeViewer` with the matching
 * `renderer` option, deterministic=true and cameraPath='orbit-8'.
 */
import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.SPLATFORGE_HARNESS_PORT ?? 4317);
const BASE = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './tests',
  // Deterministic runs are the whole point: no parallel sharding within a
  // file, no retries (failures should be reproducible).
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: [['list'], ['json', { outputFile: 'report/raw/_playwright.json' }]],
  outputDir: 'report/raw/_artifacts',

  use: {
    baseURL: BASE,
    headless: true,
    viewport: { width: 512, height: 512 },
    deviceScaleFactor: 1,
    // Disable animations and force a fixed timezone for determinism.
    timezoneId: 'UTC',
    locale: 'en-US',
  },

  // One project per renderer. The `renderer` metadata is read by tests to
  // build the URL query string; we also pass `forceRenderer` so the harness
  // page can ignore the auto-probe and pin the backend.
  projects: [
    {
      name: 'chrome-webgpu',
      metadata: { renderer: 'webgpu' },
      use: {
        ...devices['Desktop Chrome'],
        // WebGPU requires the unsafe flag headfully or the headless-shell
        // build with `--enable-unsafe-webgpu`.
        launchOptions: {
          args: ['--enable-unsafe-webgpu', '--use-vulkan=swiftshader', '--enable-features=Vulkan'],
        },
      },
    },
    {
      name: 'chrome-webgl2',
      metadata: { renderer: 'webgl2' },
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox-webgl2',
      metadata: { renderer: 'webgl2' },
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit-webgl2',
      metadata: { renderer: 'webgl2' },
      use: { ...devices['Desktop Safari'] },
    },
  ],

  webServer: {
    command: `node harness/server.mjs --port ${PORT}`,
    url: `${BASE}/page.html`,
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'pipe',
    timeout: 30_000,
  },
});
