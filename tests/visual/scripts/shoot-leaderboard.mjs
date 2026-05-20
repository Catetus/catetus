#!/usr/bin/env node
/**
 * Visual verification harness for the SplatBench leaderboard page.
 * Mirrors the pattern in `screenshot-pages.mjs` but focuses on the
 * leaderboard component specifically — captures default state, "real
 * scenes" filter, and "size-min" preset state so v2 changes are
 * easy to inspect at a glance.
 *
 * Required by memory rule `feedback_verify_ui_visually_before_handoff`:
 * never report a UI change as done without an actual screenshot.
 *
 * Assumes `astro preview` (or `astro dev`) is already running on
 * http://localhost:4321. If not, start it with:
 *
 *   cd apps/web && pnpm run build && pnpm exec astro preview --port 4321
 */
import { chromium } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, "..", "..", "..", "tasks", "screenshots", "leaderboard");
fs.mkdirSync(outDir, { recursive: true });

const BASE = process.env.PREVIEW_URL ?? "http://localhost:4321";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1600 } });
let failed = 0;

async function shoot(name, fn) {
  const page = await ctx.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errs.push(`console: ${m.text()}`));
  try {
    await fn(page);
    const file = path.join(outDir, `${name}.png`);
    await page.screenshot({ path: file, fullPage: true });
    console.log(`OK   ${name} → ${file}`);
  } catch (e) {
    console.error(`FAIL ${name}: ${e.message}`);
    failed++;
  }
  if (errs.length) {
    console.log(`     errors: ${errs.join(" | ")}`);
    failed++;
  }
  await page.close();
}

await shoot("bench-default", async (page) => {
  await page.goto(`${BASE}/bench`, { waitUntil: "networkidle", timeout: 30_000 });
  await page.waitForSelector("#leaderboard");
});

await shoot("bench-real-only", async (page) => {
  await page.goto(`${BASE}/bench`, { waitUntil: "networkidle", timeout: 30_000 });
  await page.waitForSelector('[data-source-chip="real"]');
  await page.click('[data-source-chip="real"]');
  // small idle wait so the filter count + sort transition can flush
  await page.waitForTimeout(200);
});

await shoot("bench-size-min", async (page) => {
  await page.goto(`${BASE}/bench`, { waitUntil: "networkidle", timeout: 30_000 });
  await page.waitForSelector("#preset-select");
  await page.selectOption("#preset-select", "sizeMin");
  await page.waitForTimeout(200);
});

await browser.close();
process.exit(failed > 0 ? 1 : 0);
