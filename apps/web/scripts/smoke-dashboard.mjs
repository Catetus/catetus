#!/usr/bin/env node
/**
 * Playwright smoke test for /dashboard.
 *
 * Verifies:
 *   1. Unauthenticated visit shows the "Sign in" region (unauth state).
 *   2. After binding an API key + mocking /v1/me/usage, the page
 *      renders all three sections (usage / recent jobs / get help)
 *      and lands in the `ready` data-state.
 *
 * Usage:
 *   pnpm --filter @catetus/web build      # produces dist/
 *   pnpm --filter @catetus/web preview &  # serves on :4321
 *   SF_PREVIEW_URL=http://127.0.0.1:4321 node apps/web/scripts/smoke-dashboard.mjs
 *
 * Exits 0 on success, non-zero on failure. Designed for CI; can be
 * folded into a Playwright config in a follow-up session.
 */
import { chromium } from "playwright";

const BASE = process.env.SF_PREVIEW_URL ?? "http://127.0.0.1:4321";

function fail(msg) {
  console.error(`[smoke-dashboard] FAIL: ${msg}`);
  process.exit(1);
}
function pass(msg) {
  console.error(`[smoke-dashboard] ok: ${msg}`);
}

const browser = await chromium.launch({ headless: true });
try {
  // ---- Test 1: unauthenticated visit
  {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 900 },
    });
    const page = await ctx.newPage();
    await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
    // Wait for the page script to finish its initial setState() call.
    await page.waitForFunction(
      () => {
        const el = document.querySelector("main.dashboard");
        return el && el.dataset.state && el.dataset.state !== "loading";
      },
      { timeout: 8000 },
    );
    const state = await page.getAttribute("main.dashboard", "data-state");
    if (state !== "unauth") {
      await page.screenshot({ path: "apps/web/screenshots/dashboard-unauth-FAIL.png" });
      fail(`expected unauth state, got "${state}"`);
    }
    const unauthVisible = await page.isVisible(
      'main.dashboard [data-region="unauth"]',
    );
    if (!unauthVisible) fail("unauth region not visible");
    pass("unauthenticated visit shows unauth region");
    await ctx.close();
  }

  // ---- Test 2: with mocked API → renders all 3 sections
  {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 900 },
    });
    const page = await ctx.newPage();

    // Intercept /v1/me/usage and return a fixture payload.
    await page.route("**/v1/me/usage*", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          plan: "paid",
          key_masked: "sk_test_",
          email: "test@catetus.com",
          usage: {
            repack_runs: 7,
            repack_seconds: 145,
            period_start: "2026-05-01T00:00:00Z",
          },
          recent_jobs: [
            {
              timestamp: "2026-05-16T12:34:56Z",
              route: "/v1/jobs/:id/repack",
              method: "POST",
              status: 200,
              duration_ms: 18000,
            },
            {
              timestamp: "2026-05-16T12:00:00Z",
              route: "/v1/jobs",
              method: "POST",
              status: 201,
              duration_ms: 42,
            },
          ],
        }),
      }),
    );

    // Seed the localStorage with a fake API key before navigation by
    // injecting an init script that runs before the page scripts.
    await ctx.addInitScript(() => {
      try {
        localStorage.setItem(
          "catetus.apiKey",
          "sk_test_fake_key_for_smoke",
        );
        // Point the client at the same origin so route() catches it.
        localStorage.setItem("catetus.apiBase", window.location.origin);
      } catch (_e) {}
    });
    await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });

    await page.waitForFunction(
      () => {
        const el = document.querySelector("main.dashboard");
        return el && el.dataset.state === "ready";
      },
      { timeout: 8000 },
    );
    pass("page reached ready state after mocked /v1/me/usage");

    // Header bindings
    const planText = await page.textContent('[data-bind="plan"]');
    if (planText?.trim() !== "Paid") fail(`plan binding: "${planText}"`);
    const emailText = await page.textContent('[data-bind="email"]');
    if (!emailText?.includes("test@catetus.com"))
      fail(`email binding: "${emailText}"`);

    // Section 1: Usage
    const repackRuns = await page.textContent('[data-bind="repack-runs"]');
    if (repackRuns?.trim() !== "7")
      fail(`repack-runs binding: "${repackRuns}"`);
    const repackSeconds = await page.textContent(
      '[data-bind="repack-seconds"]',
    );
    if (!repackSeconds?.includes("2m"))
      fail(`repack-seconds binding: "${repackSeconds}" (expected 2m 25s)`);

    // Section 2: Recent jobs — should have 2 rows
    const rowCount = await page.locator(
      '[data-bind="jobs-tbody"] tr',
    ).count();
    if (rowCount !== 2) fail(`recent-jobs row count: ${rowCount}`);
    pass(`recent-jobs table rendered ${rowCount} rows`);

    // Section 3: Get help — must contain at least one help link
    const helpLink = await page
      .locator('.help-card a[href^="mailto:"]')
      .count();
    if (helpLink < 1) fail("help section missing mailto link");
    pass("all 3 dashboard sections rendered");

    await ctx.close();
  }
  console.error("[smoke-dashboard] ALL CHECKS PASSED");
} finally {
  await browser.close();
}
