#!/usr/bin/env node
import { chromium } from "playwright";
import { resolve, dirname } from "node:path";
import { fileURLToPath, URL as NodeURL } from "node:url";
const __dirname = fileURLToPath(new NodeURL(".", import.meta.url));
const OUT = resolve(__dirname, "..", "screenshots");
const BASE = process.env.SF_PREVIEW_URL ?? "http://127.0.0.1:4327";

const browser = await chromium.launch({ headless: true });
try {
  // Unauth screenshot.
  {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 800 },
    });
    const page = await ctx.newPage();
    await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => {
      const el = document.querySelector("main.dashboard");
      return el && el.dataset.state && el.dataset.state !== "loading";
    });
    await page.screenshot({ path: resolve(OUT, "dashboard-unauth.png"), fullPage: true });
    await ctx.close();
  }
  // Ready (mocked) screenshot.
  {
    const ctx = await browser.newContext({
      viewport: { width: 1280, height: 1100 },
    });
    await ctx.addInitScript(() => {
      try {
        localStorage.setItem("splatforge.apiKey", "sk_test_fake_key_for_smoke");
        localStorage.setItem("splatforge.apiBase", window.location.origin);
      } catch (_e) {}
    });
    const page = await ctx.newPage();
    await page.route("**/v1/me/usage*", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          plan: "paid",
          key_masked: "sk_test_",
          email: "monte@splatforge.dev",
          usage: { repack_runs: 7, repack_seconds: 145, period_start: "2026-05-01T00:00:00Z" },
          recent_jobs: [
            { timestamp: "2026-05-16T12:34:56Z", route: "/v1/jobs/:id/repack", method: "POST", status: 200, duration_ms: 18000 },
            { timestamp: "2026-05-16T12:00:00Z", route: "/v1/jobs", method: "POST", status: 201, duration_ms: 42 },
            { timestamp: "2026-05-15T20:00:00Z", route: "/v1/jobs/:id/upload", method: "POST", status: 200, duration_ms: 4500 },
          ],
        }),
      }),
    );
    await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => {
      const el = document.querySelector("main.dashboard");
      return el && el.dataset.state === "ready";
    });
    await page.screenshot({ path: resolve(OUT, "dashboard-ready.png"), fullPage: true });
    await ctx.close();
  }
  console.error("[shoot-dashboard] wrote dashboard-unauth.png + dashboard-ready.png");
} finally {
  await browser.close();
}
