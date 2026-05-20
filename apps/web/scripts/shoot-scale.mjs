// Playwright shooter for /scale (loading + L5 live + L4 attempt).
// Run with: node apps/web/scripts/shoot-scale.mjs
//
// IMPORTANT: WebGPU on the Mac runs through Metal under Chromium, but
// only headed/headless-new mode exposes navigator.gpu. Pass
// --enable-unsafe-webgpu and --enable-features=Vulkan,WebGPU.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, "..", "screenshots");
mkdirSync(OUT_DIR, { recursive: true });

const URL = process.env.SCALE_URL ?? "http://127.0.0.1:4321/scale/";

async function main() {
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
  page.on("console", (m) => console.log(`[browser ${m.type()}]`, m.text()));
  page.on("pageerror", (err) => console.error("[browser ERROR]", err.message));

  console.log(`Opening ${URL}...`);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

  // 1. Loading state — wait until the status bar shows progress >0%.
  await page.waitForFunction(
    () => {
      const b = document.querySelector("[data-status-bar]");
      const s = document.querySelector("[data-status-state]");
      return s && (s.textContent || "").trim() === "loading"
        && b && parseFloat(b.style.width || "0") > 5;
    },
    { timeout: 30_000 },
  ).catch(() => console.log("loading state never observed (may have skipped); continuing"));

  await page.screenshot({
    path: resolve(OUT_DIR, "scale-page-loading.png"),
    fullPage: false,
  });
  console.log("captured: loading");

  // 2. L5 live — prefer ready, but accept "enough chunks loaded so the
  // canvas is actually rendering something" if the full level is slow.
  const finalState = await page.waitForFunction(
    () => {
      const s = document.querySelector("[data-status-state]");
      const t = (s?.textContent || "").trim();
      if (t === "ready" || t === "error") return t;
      // Mid-stream: accept if at least 8 chunks are uploaded AND
      // pipeline.splatCount > 500k. We expose splatCount via the stats DOM.
      const splatsText = document.querySelector("[data-stat-splats]")?.textContent || "";
      // Match e.g. "1.23M" or "780k".
      const m = splatsText.match(/^([\d.]+)\s*([MkG])?$/);
      if (m) {
        let n = parseFloat(m[1]);
        if (m[2] === "k") n *= 1e3;
        else if (m[2] === "M") n *= 1e6;
        else if (m[2] === "G") n *= 1e9;
        if (n > 500_000) return "midstream";
      }
      return null;
    },
    { timeout: 360_000, polling: 1_000 },
  ).then((h) => h.jsonValue()).catch(() => "timeout");

  console.log(`final state: ${finalState}`);

  // Render a few frames to make sure the canvas has content.
  await page.waitForTimeout(2_000);
  await page.screenshot({
    path: resolve(OUT_DIR, "scale-page-l5-live.png"),
    fullPage: false,
  });
  console.log("captured: l5-live");

  // Diagnostic: read stats from the DOM.
  const stats = await page.evaluate(() => ({
    state: document.querySelector("[data-status-state]")?.textContent,
    detail: document.querySelector("[data-status-detail]")?.textContent,
    splats: document.querySelector("[data-stat-splats]")?.textContent,
    fps: document.querySelector("[data-stat-fps]")?.textContent,
    bytes: document.querySelector("[data-stat-bytes]")?.textContent,
    canvasW: document.querySelector("[data-canvas]")?.width,
    canvasH: document.querySelector("[data-canvas]")?.height,
  }));
  console.log("L5 stats:", stats);

  // 3. Switch to L4. The page supports interrupting an in-flight load —
  // we don't need to wait for L5 ready. Click L4 and capture mid-stream.

  const l4Btn = await page.$("[data-lod=\"4\"]:not([disabled])");
  if (l4Btn) {
    await l4Btn.click();
    await page.waitForFunction(
      () => {
        const s = document.querySelector("[data-status-state]");
        const t = (s?.textContent || "").trim();
        if (t === "ready" || t === "error") return t;
        const splatsText = document.querySelector("[data-stat-splats]")?.textContent || "";
        const m = splatsText.match(/^([\d.]+)\s*([MkG])?$/);
        if (m) {
          let n = parseFloat(m[1]);
          if (m[2] === "k") n *= 1e3;
          else if (m[2] === "M") n *= 1e6;
          return n > 800_000 ? "midstream" : null;
        }
        return null;
      },
      { timeout: 360_000, polling: 1_000 },
    ).catch(() => console.log("L4 mid-state never reached"));
    await page.waitForTimeout(2_000);
  } else {
    // Fallback: button stayed disabled — hover for visual state.
    await page.hover("[data-lod=\"4\"]").catch(() => {});
    await page.waitForTimeout(1_000);
  }
  await page.screenshot({
    path: resolve(OUT_DIR, "scale-page-l4-attempt.png"),
    fullPage: false,
  });
  console.log("captured: l4-attempt");

  const l4Stats = await page.evaluate(() => ({
    state: document.querySelector("[data-status-state]")?.textContent,
    detail: document.querySelector("[data-status-detail]")?.textContent,
    splats: document.querySelector("[data-stat-splats]")?.textContent,
    fps: document.querySelector("[data-stat-fps]")?.textContent,
    bytes: document.querySelector("[data-stat-bytes]")?.textContent,
  }));
  console.log("L4 stats:", l4Stats);

  // Write a small summary alongside the PNGs.
  writeFileSync(
    resolve(OUT_DIR, "scale-page-summary.json"),
    JSON.stringify({ url: URL, l5: stats, l4: l4Stats }, null, 2),
  );

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
