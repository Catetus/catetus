import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(root, "screenshots");
await import("node:fs").then(fs => fs.mkdirSync(outDir, { recursive: true }));

const PAGES = [
  { url: "http://localhost:4321/bench",              file: path.join(outDir, "bench.png") },
  { url: "http://localhost:4321/vs-splat-transform", file: path.join(outDir, "vs-splat-transform.png") },
  { url: "http://localhost:4321/",                   file: path.join(outDir, "index.png") },
];

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
let failed = 0;
for (const p of PAGES) {
  const page = await ctx.newPage();
  const errs = [];
  page.on("pageerror", e => errs.push(`pageerror: ${e.message}`));
  page.on("console", m => m.type() === "error" && errs.push(`console: ${m.text()}`));
  try {
    await page.goto(p.url, { waitUntil: "networkidle", timeout: 30000 });
    await page.screenshot({ path: p.file, fullPage: true });
    console.log(`OK   ${p.url} → ${p.file}`);
  } catch (e) {
    console.error(`FAIL ${p.url}: ${e.message}`);
    failed++;
  }
  if (errs.length) {
    console.log(`     errors: ${errs.join(" | ")}`);
    failed++;
  }
  await page.close();
}
await browser.close();
process.exit(failed > 0 ? 1 : 0);
