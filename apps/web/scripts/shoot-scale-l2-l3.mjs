// Playwright shooter for /scale after L2 + L3 publish.
// Captures: initial L5 state, L3 loading + mid-stream, L2 loading + mid-stream.
// Usage:  node tasks/scripts/sf-scale-l2-l3/shoot-scale-l2-l3.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, "screenshots");
mkdirSync(OUT_DIR, { recursive: true });

const URL = process.env.SCALE_URL ?? "http://127.0.0.1:4321/scale/";

function readStats(page) {
  return page.evaluate(() => ({
    state: document.querySelector("[data-status-state]")?.textContent,
    detail: document.querySelector("[data-status-detail]")?.textContent,
    splats: document.querySelector("[data-stat-splats]")?.textContent,
    fps: document.querySelector("[data-stat-fps]")?.textContent,
    bytes: document.querySelector("[data-stat-bytes]")?.textContent,
    l2Enabled: !document.querySelector('[data-lod="2"]')?.disabled,
    l3Enabled: !document.querySelector('[data-lod="3"]')?.disabled,
    l4Enabled: !document.querySelector('[data-lod="4"]')?.disabled,
    l5Enabled: !document.querySelector('[data-lod="5"]')?.disabled,
  }));
}

async function waitForMidstream(page, minSplatsM, timeoutMs) {
  return page
    .waitForFunction(
      (min) => {
        const s = document.querySelector("[data-status-state]");
        const t = (s?.textContent || "").trim();
        if (t === "ready" || t === "error") return t;
        const splatsText = document.querySelector("[data-stat-splats]")?.textContent || "";
        const m = splatsText.match(/^([\d.]+)\s*([MkG])?$/);
        if (m) {
          let n = parseFloat(m[1]);
          if (m[2] === "k") n *= 1e3;
          else if (m[2] === "M") n *= 1e6;
          else if (m[2] === "G") n *= 1e9;
          if (n > min * 1e6) return "midstream";
        }
        return null;
      },
      minSplatsM,
      { timeout: timeoutMs, polling: 1000 },
    )
    .then((h) => h.jsonValue())
    .catch(() => "timeout");
}

async function main() {
  const errors = [];
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPU",
      "--use-vulkan=swiftshader",
      "--disable-vulkan-fallback-to-gl-for-testing",
      "--ignore-gpu-blocklist",
      "--enable-gpu",
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    const t = m.type();
    const txt = m.text();
    console.log(`[browser ${t}]`, txt);
    if (t === "error") errors.push(txt);
  });
  page.on("pageerror", (err) => {
    console.error("[browser ERROR]", err.message);
    errors.push(err.message);
  });

  console.log(`Opening ${URL}...`);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

  // 1. Initial state — wait for L5 to at least begin loading.
  await waitForMidstream(page, 0.5, 60_000);
  await page.waitForTimeout(2_000);
  await page.screenshot({ path: resolve(OUT_DIR, "01-initial-l5.png"), fullPage: false });
  const initial = await readStats(page);
  console.log("initial stats:", initial);

  // 2. Click L3.
  console.log("\n=== L3 ===");
  const l3Btn = await page.$('[data-lod="3"]:not([disabled])');
  if (!l3Btn) {
    console.error("L3 button is disabled — published manifest is missing L3!");
    await page.screenshot({ path: resolve(OUT_DIR, "02-l3-DISABLED.png"), fullPage: false });
    errors.push("L3 button disabled");
  } else {
    await l3Btn.click();
    const l3State = await waitForMidstream(page, 3.0, 360_000); // expect >3M streamed
    console.log("L3 state:", l3State);
    await page.waitForTimeout(3_000);
    await page.screenshot({ path: resolve(OUT_DIR, "02-l3-loaded.png"), fullPage: false });
    const l3Stats = await readStats(page);
    console.log("L3 stats:", l3Stats);
  }

  // 3. Click L2.
  console.log("\n=== L2 ===");
  const l2Btn = await page.$('[data-lod="2"]:not([disabled])');
  if (!l2Btn) {
    console.error("L2 button is disabled — either HTML still says `disabled` or runtime disabled it!");
    await page.screenshot({ path: resolve(OUT_DIR, "03-l2-DISABLED.png"), fullPage: false });
    errors.push("L2 button disabled");
  } else {
    await l2Btn.click();
    // L2 is heavy; just confirm it AT LEAST starts loading (1M+ splats) within 5 min.
    const l2State = await waitForMidstream(page, 1.0, 300_000);
    console.log("L2 state:", l2State);
    await page.waitForTimeout(3_000);
    await page.screenshot({ path: resolve(OUT_DIR, "03-l2-loaded.png"), fullPage: false });
    const l2Stats = await readStats(page);
    console.log("L2 stats:", l2Stats);
  }

  const final = await readStats(page);
  const summary = {
    url: URL,
    initial,
    final,
    consoleErrors: errors,
  };
  writeFileSync(resolve(OUT_DIR, "summary.json"), JSON.stringify(summary, null, 2));
  console.log("\n=== SUMMARY ===");
  console.log(JSON.stringify(summary, null, 2));

  await browser.close();
  if (errors.length > 0) {
    console.error(`\nFAIL: ${errors.length} console error(s) observed`);
    process.exit(2);
  }
  console.log("\nOK: no console errors");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
