#!/usr/bin/env node
/**
 * Merge the fidelity runner's output (`benches/reports/fidelity-v0.json`) into
 * the published leaderboard artifacts:
 *
 *   - `benches/reports/splatbench-v0.json`  ← gains a `fidelity` block per scene
 *   - `benches/reports/splatbench-v0.md`    ← gains a fidelity leaderboard
 *   - `benches/reports/splatbench-v0.html`  ← gains a fidelity column
 *
 * Reads on-disk inputs; writes the same files back. Idempotent.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const REPORTS = resolve(__dirname, 'reports');
const SPLATBENCH_JSON = resolve(REPORTS, 'splatbench-v0.json');
const SPLATBENCH_MD = resolve(REPORTS, 'splatbench-v0.md');
const SPLATBENCH_HTML = resolve(REPORTS, 'splatbench-v0.html');
const FIDELITY_JSON = resolve(REPORTS, 'fidelity-v0.json');
const VERSION = '0.1.1';

const fidelity = JSON.parse(readFileSync(FIDELITY_JSON, 'utf8'));
const splatbench = JSON.parse(readFileSync(SPLATBENCH_JSON, 'utf8'));

/* ------------- helper formatters --------------------------------------- */

const pct = (v) => `${(v * 100).toFixed(2)}%`;
const fmtBytes = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1073741824) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(2)} GB`;
};

/**
 * Map ΔE94 (0..1, where 1 = ΔE94 of 100) onto a coarse band:
 *   pass        : mean < 2% AND max < 5%
 *   borderline  : mean < 3% AND max < 8%
 *   fail        : everything else
 */
function band(mean, max) {
  if (mean < 0.02 && max < 0.05) return 'pass';
  if (mean < 0.03 && max < 0.08) return 'borderline';
  return 'fail';
}

/* ------------- 1. JSON update ------------------------------------------ */

const fidelityById = new Map(fidelity.scenes.map((s) => [s.id, s.presets]));

splatbench.splatforgeVersion = VERSION;
splatbench.runDate = fidelity.runDate || splatbench.runDate;
let scenesWithFidelity = 0;
let scenesMissing = [];
for (const scene of splatbench.scenes) {
  const presets = fidelityById.get(scene.id);
  if (!presets) {
    scenesMissing.push(scene.id);
    continue;
  }
  scenesWithFidelity++;
  const presetSummary = (presetName) => {
    const m = presets[presetName];
    if (!m) return null;
    return {
      meanDeltaE94: round4(m.deltaE94.mean),
      maxDeltaE94: round4(m.deltaE94.max),
      p95DeltaE94: round4(m.deltaE94.p95),
      meanPixelMatch: round4(m.pixelMatch.mean),
      meanSsimLoss: round4(m.ssimLoss.mean),
      status: m.status,
      passed: m.passed,
    };
  };
  scene.fidelity = {
    baseline: 'lossless-repack',
    renderer: fidelity.renderer,
    cameraPath: fidelity.cameraPath,
    frameSize: fidelity.frameSize,
    losslessRepack: presetSummary('lossless-repack'),
    webMobile: presetSummary('web-mobile'),
    sizeMin: presetSummary('size-min'),
  };
}
writeFileSync(SPLATBENCH_JSON, JSON.stringify(splatbench, null, 2) + '\n');

function round4(x) {
  return Math.round(x * 1e6) / 1e6;
}

/* ------------- 2. HTML update ------------------------------------------ */

let html = readFileSync(SPLATBENCH_HTML, 'utf8');

// Replace the DATA array. The original is a one-line-per-scene block between
// `const DATA = [` and `];`. We rewrite it adding a `fidelity` field.
const dataStart = html.indexOf('const DATA = [');
const dataEnd = html.indexOf('];', dataStart);
if (dataStart === -1 || dataEnd === -1) {
  throw new Error('splatbench-v0.html: cannot locate DATA array');
}
const existingDataBlock = html.slice(dataStart, dataEnd + 2);

// Parse the existing data lines so we can rewrite while preserving every
// original field (we only add `fidelity`).
const dataLines = existingDataBlock
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l.startsWith('{ id:'));
const newDataLines = dataLines.map((line) => {
  const m = line.match(/id:"([^"]+)"/);
  if (!m) return line;
  const id = m[1];
  const presets = fidelityById.get(id);
  if (!presets) return line;
  const f = presets['web-mobile'];
  const g = presets['size-min'];
  const fidelityField = `, fidelity:{ webMobile:{ mean:${f.deltaE94.mean.toFixed(4)}, max:${f.deltaE94.max.toFixed(4)}, status:"${f.status}" }, sizeMin:{ mean:${g.deltaE94.mean.toFixed(4)}, max:${g.deltaE94.max.toFixed(4)}, status:"${g.status}" } }`;
  const updated = line.replace(/(\s*}\s*,?\s*)$/, (_, tail) => `${fidelityField} ${tail.trimStart()}`);
  return updated;
});

const newDataBlock =
  'const DATA = [\n  ' +
  newDataLines
    .map((l, i) => (i === newDataLines.length - 1 ? l.replace(/,$/, '') : l))
    .join('\n  ') +
  '\n];';
html = html.slice(0, dataStart) + newDataBlock + html.slice(dataEnd + 2);

// Add a Fidelity column to the leaderboard table and a fidelity cell to each
// rendered row.
html = html.replace(
  /<th>Ratio<\/th>\s*<th>analyze<\/th>/,
  '<th>Ratio</th>\n        <th>Fidelity (ΔE94)</th>\n        <th>analyze</th>',
);
html = html.replace(
  /(<td><span class="ratio \$\{cls\}">[^]*?<\/td>)\s*\n\s*<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
  (whole) => {
    return whole.replace(
      /<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
      `<td>\${fmtFidelity(r, preset)}</td>\n      <td>\${fmtMs(r.analyze)}</td>`,
    );
  },
);

// Inject helper functions ahead of `function renderTable()`. We insert a small
// block once, keyed by a marker comment for idempotency.
if (!html.includes('// fidelity-helper-injected')) {
  const helpers = `// fidelity-helper-injected\nfunction fidelityKey(preset) {\n  return preset === "webMobile" ? "webMobile" : "sizeMin";\n}\nfunction fidelityClass(status) {\n  if (status === "pass") return "fid-pass";\n  if (status === "borderline") return "fid-borderline";\n  if (status === "fail") return "fid-fail";\n  return "fid-na";\n}\nfunction fmtFidelity(r, preset) {\n  if (!r.fidelity) return "<span class=\\"fid-na\\">—</span>";\n  const f = r.fidelity[fidelityKey(preset)];\n  if (!f) return "<span class=\\"fid-na\\">—</span>";\n  const cls = fidelityClass(f.status);\n  const pct = (v) => (v * 100).toFixed(2) + "%";\n  return '<span class="fid ' + cls + '">' + pct(f.mean) + ' / ' + pct(f.max) + '</span>';\n}\n`;
  html = html.replace(/function renderTable\(\)/, `${helpers}function renderTable()`);
}

// Add styles for the fidelity pills if not already present.
if (!html.includes('.fid-pass')) {
  const fidStyles = `\n  .fid { font-variant-numeric: tabular-nums; padding: 2px 6px; border-radius: 4px; }\n  .fid-pass { background: rgba(52, 211, 153, 0.18); color: var(--good); }\n  .fid-borderline { background: rgba(251, 191, 36, 0.18); color: var(--warn); }\n  .fid-fail { background: rgba(248, 113, 113, 0.18); color: var(--bad); }\n  .fid-na { color: var(--fg-dim); }\n`;
  html = html.replace(/(\.ratio-low \{[^}]*\}\s*)/m, `$1${fidStyles}`);
  // Fallback if the .ratio-low rule wasn't found — append before </style>.
  if (!html.includes('.fid-pass')) {
    html = html.replace('</style>', `${fidStyles}</style>`);
  }
}

// Cross out the "Visual fidelity" pending row in the "What v0 doesn't yet measure" table.
html = html.replace(
  /<tr><td>Visual fidelity \(ΔE94, SSIM, PSNR\)<\/td><td>pending<\/td><td>needs <code class="inline">playwright-core \+ chromium<\/code><\/td><\/tr>/,
  '<tr><td>Visual fidelity (ΔE94, SSIM, PSNR)</td><td><strong>shipped v0.1.1</strong></td><td>—</td></tr>',
);

// Update sub-text noting the new measurements.
html = html.replace(
  /class="sub">[^<]*</,
  'class="sub">Initial benchmark — 7 scenes, 3 presets, 3DGS-format-only. v0.1.1 adds visual fidelity (ΔE94 / pixelmatch / SSIM) via @splatforge/viewer headless renders.<',
);

writeFileSync(SPLATBENCH_HTML, html);

/* ------------- 3. Markdown update -------------------------------------- */

let md = readFileSync(SPLATBENCH_MD, 'utf8');

// Build the fidelity leaderboard table once.
const fidelityRows = splatbench.scenes
  .map((s) => {
    const f = s.fidelity?.webMobile;
    const g = s.fidelity?.sizeMin;
    if (!f || !g) return null;
    return { id: s.id, f, g };
  })
  .filter(Boolean)
  .sort((a, b) => a.f.meanDeltaE94 - b.f.meanDeltaE94);

const fidelitySection = `## Leaderboard — visual fidelity (v0.1.1)

Frames captured via \`@splatforge/viewer\` in headless Chromium (SwiftShader software-rendered WebGL2), 8 deterministic orbit poses at 512×512. \`lossless-repack\` is the per-scene baseline. ΔE94 is normalized to 0..1 (i.e. \`3%\` = 3 absolute ΔE94 units, the perceptibility threshold of an attentive observer).

| Rank | Scene | web-mobile ΔE94 mean / max | status | size-min ΔE94 mean / max | status |
| ---: | ----- | ---: | :---: | ---: | :---: |
${fidelityRows
  .map(
    (r, i) =>
      `| ${i + 1} | \`${r.id}\` | ${pct(r.f.meanDeltaE94)} / ${pct(r.f.maxDeltaE94)} | **${r.f.status}** | ${pct(r.g.meanDeltaE94)} / ${pct(r.g.maxDeltaE94)} | **${r.g.status}** |`,
  )
  .join('\n')}

**Pass criterion:** mean ΔE94 < 3% AND max ΔE94 < 8%. **Borderline:** mean 2–3% or max 5–8%. **Pass:** mean < 2% AND max < 5%.

Software-rendered numbers may differ slightly from hardware-accelerated chromium; see \`fidelity-v0.json\` for per-frame raw metrics and \`benches/reports/frames/<scene>/<preset>/0001.png\` etc. for the actual frames.

`;

// Drop the entire "What's intentionally missing from v0" section's visual-fidelity bullet.
md = md.replace(
  /\* \*\*Visual-fidelity scores\*\*[^\n]*\n[^\n]*\n/,
  '',
);

// Insert the fidelity section before "## Corpus composition" if not already present.
if (!md.includes('Leaderboard — visual fidelity')) {
  md = md.replace(/## Corpus composition/, `${fidelitySection}## Corpus composition`);
}

// Update headline KPIs to mention fidelity.
const presetWMpass = splatbench.scenes.filter(
  (s) => s.fidelity?.webMobile?.status !== 'fail',
).length;
const presetSMpass = splatbench.scenes.filter(
  (s) => s.fidelity?.sizeMin?.status !== 'fail',
).length;
const fidHeadline = `| \`web-mobile\` fidelity passing | **${presetWMpass} / ${splatbench.scenes.length}** scenes within PRD threshold |
| \`size-min\` fidelity passing | **${presetSMpass} / ${splatbench.scenes.length}** scenes within PRD threshold |
`;
md = md.replace(/(\| \`size-min\` ratio[^\n]+\n)/, `$1${fidHeadline}`);

// Bump the version line and run date.
md = md.replace(/\*\*SplatForge version:\*\* `0\.1\.0`/, `**SplatForge version:** \`${VERSION}\``);

writeFileSync(SPLATBENCH_MD, md);

console.error(`updated ${SPLATBENCH_JSON}`);
console.error(`updated ${SPLATBENCH_MD}`);
console.error(`updated ${SPLATBENCH_HTML}`);
console.error(`scenes with fidelity: ${scenesWithFidelity}/${splatbench.scenes.length}`);
if (scenesMissing.length > 0) {
  console.error(`scenes still missing fidelity: ${scenesMissing.join(', ')}`);
}
