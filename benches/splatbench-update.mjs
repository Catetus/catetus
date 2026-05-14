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
const DIFF_REPACK_JSON = resolve(REPORTS, 'diff-repack-v0.json');
const VERSION = '0.1.1';

const fidelity = JSON.parse(readFileSync(FIDELITY_JSON, 'utf8'));
const splatbench = JSON.parse(readFileSync(SPLATBENCH_JSON, 'utf8'));
// Optional — only present once `splatforge-pro` runs differentiable repack on
// a scene. Joined by `scene_id`.
let diffRepackById = new Map();
try {
  const dr = JSON.parse(readFileSync(DIFF_REPACK_JSON, 'utf8'));
  diffRepackById = new Map((dr.results || []).map((r) => [r.scene_id, r]));
} catch (_) {
  // missing file is fine — the column just stays empty
}

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
    const out = {
      meanDeltaE94: round4(m.deltaE94.mean),
      maxDeltaE94: round4(m.deltaE94.max),
      p95DeltaE94: round4(m.deltaE94.p95),
      meanPixelMatch: round4(m.pixelMatch.mean),
      meanSsimLoss: round4(m.ssimLoss.mean),
      status: m.status,
      passed: m.passed,
    };
    if (m.mlScore) {
      // splatforge-pro splat-aware perceptual metric (proprietary; private
      // companion build). The metric values land in the public benchmark so
      // visitors can read them, but only `splatforge-pro` reproduces them.
      out.mlScore = round4(m.mlScore.mean);
      out.mlScoreMax = round4(m.mlScore.max);
      out.mlScoreVersion = m.mlScoreVersion;
    }
    return out;
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

  // splatforge-pro DifferentiableRepack — proprietary premium-tier result.
  // Joined by `scene_id` from diff-repack-v0.json. Only scenes that have
  // been run through the gsplat repack have an entry; others stay empty.
  const dr = diffRepackById.get(scene.id);
  if (dr) {
    scene.repack = {
      targetRatio: 0.5,
      splatsIn: dr.splats_in,
      splatsOut: dr.splats_out,
      bytesIn: dr.bytes_in,
      bytesOut: dr.bytes_out,
      psnrRepackDb: dr.psnr_repack_db,
      psnrOpacityPruneDb: dr.psnr_opacity_prune_db,
      psnrDeltaDb: dr.psnr_delta_db,
    };
    // Optional multi-seed annotations (v0.1.1+). Present whenever the row
    // was produced from an N>1 sweep; absent means "single-run number,
    // treat as noisy" — the HTML displays a `—` instead of `± stdev` for
    // those.
    if (typeof dr.seed_count === 'number') scene.repack.seedCount = dr.seed_count;
    if (typeof dr.psnr_delta_db_mean === 'number') scene.repack.psnrDeltaDbMean = dr.psnr_delta_db_mean;
    if (typeof dr.psnr_delta_db_stdev === 'number') scene.repack.psnrDeltaDbStdev = dr.psnr_delta_db_stdev;
    if (typeof dr.psnr_delta_db_min === 'number') scene.repack.psnrDeltaDbMin = dr.psnr_delta_db_min;
    if (typeof dr.psnr_delta_db_max === 'number') scene.repack.psnrDeltaDbMax = dr.psnr_delta_db_max;
  }
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
  const mlField = (m) => (m?.mlScore ? `, ml:${m.mlScore.mean.toFixed(4)}` : '');
  const fidelityField = `, fidelity:{ webMobile:{ mean:${f.deltaE94.mean.toFixed(4)}, max:${f.deltaE94.max.toFixed(4)}, status:"${f.status}"${mlField(f)} }, sizeMin:{ mean:${g.deltaE94.mean.toFixed(4)}, max:${g.deltaE94.max.toFixed(4)}, status:"${g.status}"${mlField(g)} } }`;
  const dr = diffRepackById.get(id);
  let repackField = '';
  if (dr) {
    const extras = [];
    if (typeof dr.psnr_delta_db_stdev === 'number') {
      extras.push(`stdev:${dr.psnr_delta_db_stdev.toFixed(2)}`);
    }
    if (typeof dr.seed_count === 'number') {
      extras.push(`n:${dr.seed_count}`);
    }
    const extrasField = extras.length ? `, ${extras.join(', ')}` : '';
    repackField = `, repack:{ deltaDb:${dr.psnr_delta_db.toFixed(2)}, repackDb:${dr.psnr_repack_db.toFixed(2)}, pruneDb:${dr.psnr_opacity_prune_db.toFixed(2)}${extrasField} }`;
  }
  // Strip any prior `, fidelity:{...}` AND `, repack:{...}` blocks so
  // re-runs don't duplicate fields.
  const stripped = line
    .replace(
      /,\s*fidelity:\{\s*webMobile:\{[^}]*\},\s*sizeMin:\{[^}]*\}\s*\}/g,
      '',
    )
    .replace(/,\s*repack:\{[^}]*\}/g, '');
  const updated = stripped.replace(
    /(\s*}\s*,?\s*)$/,
    (_, tail) => `${fidelityField}${repackField} ${tail.trimStart()}`,
  );
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
// Match either the original `Ratio → analyze` or the v0.1.1 `Ratio → Fidelity
// → analyze` and upgrade to `Ratio → Fidelity → ML Score → analyze`.
if (!html.includes('<th>ML Score')) {
  if (html.match(/<th>Fidelity \(ΔE94\)<\/th>\s*<th>analyze<\/th>/)) {
    html = html.replace(
      /<th>Fidelity \(ΔE94\)<\/th>\s*<th>analyze<\/th>/,
      '<th>Fidelity (ΔE94)</th>\n        <th>ML Score <span class="ml-tag">pro</span></th>\n        <th>analyze</th>',
    );
  } else {
    html = html.replace(
      /<th>Ratio<\/th>\s*<th>analyze<\/th>/,
      '<th>Ratio</th>\n        <th>Fidelity (ΔE94)</th>\n        <th>ML Score <span class="ml-tag">pro</span></th>\n        <th>analyze</th>',
    );
  }
}
if (!html.includes('<th>Repack ΔPSNR')) {
  html = html.replace(
    /<th>ML Score <span class="ml-tag">pro<\/span><\/th>\s*<th>analyze<\/th>/,
    '<th>ML Score <span class="ml-tag">pro</span></th>\n        <th>Repack ΔPSNR <span class="ml-tag">premium</span></th>\n        <th>analyze</th>',
  );
}
if (!html.includes('fmtMlScore(r, preset)')) {
  if (html.includes('${fmtFidelity(r, preset)}')) {
    // Fidelity cell already present — insert ML cell after it.
    html = html.replace(
      /<td>\$\{fmtFidelity\(r, preset\)\}<\/td>\s*\n\s*<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
      `<td>\${fmtFidelity(r, preset)}</td>\n      <td>\${fmtMlScore(r, preset)}</td>\n      <td>\${fmtMs(r.analyze)}</td>`,
    );
  } else {
    html = html.replace(
      /(<td><span class="ratio \$\{cls\}">[^]*?<\/td>)\s*\n\s*<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
      (whole) =>
        whole.replace(
          /<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
          `<td>\${fmtFidelity(r, preset)}</td>\n      <td>\${fmtMlScore(r, preset)}</td>\n      <td>\${fmtMs(r.analyze)}</td>`,
        ),
    );
  }
}
if (!html.includes('fmtRepack(r)')) {
  html = html.replace(
    /<td>\$\{fmtMlScore\(r, preset\)\}<\/td>\s*\n\s*<td>\$\{fmtMs\(r\.analyze\)\}<\/td>/,
    `<td>\${fmtMlScore(r, preset)}</td>\n      <td>\${fmtRepack(r)}</td>\n      <td>\${fmtMs(r.analyze)}</td>`,
  );
}

// Inject helper functions ahead of `function renderTable()`. We insert a small
// block once, keyed by a marker comment for idempotency.
// Helper injection is idempotent: each block is delimited by a
// `// <name>-helper-injected` opening marker and a `// <name>-helper-end`
// closing marker so we can rewrite the body unconditionally on every run.
// (The previous "only inject if marker missing" pattern silently froze old
// helper bodies — e.g. an updated fmtRepack that styles negative deltas in
// red never took effect because the marker was already there.)
const fidelityHelper = `// fidelity-helper-injected
function fidelityKey(preset) {
  return preset === "webMobile" ? "webMobile" : "sizeMin";
}
function fidelityClass(status) {
  if (status === "pass") return "fid-pass";
  if (status === "borderline") return "fid-borderline";
  if (status === "fail") return "fid-fail";
  return "fid-na";
}
function fmtFidelity(r, preset) {
  if (!r.fidelity) return "<span class=\\"fid-na\\">—</span>";
  const f = r.fidelity[fidelityKey(preset)];
  if (!f) return "<span class=\\"fid-na\\">—</span>";
  const cls = fidelityClass(f.status);
  const pct = (v) => (v * 100).toFixed(2) + "%";
  return '<span class="fid ' + cls + '">' + pct(f.mean) + ' / ' + pct(f.max) + '</span>';
}
// fidelity-helper-end
`;
const mlHelper = `// ml-helper-injected
function fmtMlScore(r, preset) {
  if (!r.fidelity) return "<span class=\\"ml-na\\">—</span>";
  const f = r.fidelity[fidelityKey(preset)];
  if (!f || typeof f.ml !== "number") return "<span class=\\"ml-na\\">—</span>";
  return '<span class="ml">' + (f.ml * 100).toFixed(2) + '%</span>';
}
// ml-helper-end
`;
const repackHelper = `// repack-helper-injected
function fmtRepack(r) {
  if (!r.repack || typeof r.repack.deltaDb !== "number") return "<span class=\\"ml-na\\">—</span>";
  const d = r.repack.deltaDb;
  const sign = d >= 0 ? "+" : "";
  const cls = d >= 0 ? "repack-win" : "repack-loss";
  const hasStdev = typeof r.repack.stdev === "number";
  const hasN = typeof r.repack.n === "number";
  const stdevSuffix = hasStdev ? " ± " + r.repack.stdev.toFixed(2) : "";
  const nSuffix = hasN ? " (n=" + r.repack.n + ")" : "";
  const baseTitle = d < 0
    ? "Repack loses to opacity-prune on this scene. Translucent volumes are a structural failure mode for saliency-based hard-pruning."
    : "Median improvement of differentiable-repack over opacity-prune at the same byte budget.";
  const seedNote = hasN ? " " + r.repack.n + "-seed median ± stdev." : " Single-seed number — treat as noisy.";
  const title = ' title="' + baseTitle + seedNote + '"';
  return '<span class="' + cls + '"' + title + '>' + sign + d.toFixed(2) + " dB" + stdevSuffix + nSuffix + '</span>';
}
// repack-helper-end
`;
function upsertHelper(name, body) {
  const re = new RegExp(`// ${name}-helper-injected[\\s\\S]*?// ${name}-helper-end\\n`);
  if (re.test(html)) {
    html = html.replace(re, body);
  } else {
    html = html.replace(/function renderTable\(\)/, `${body}function renderTable()`);
  }
}
upsertHelper('fidelity', fidelityHelper);
upsertHelper('ml', mlHelper);
upsertHelper('repack', repackHelper);

// Idempotent style upsert — same delimiter pattern as the helpers.
const fidStyles = `/* fid-styles-injected */
  .fid { font-variant-numeric: tabular-nums; padding: 2px 6px; border-radius: 4px; }
  .fid-pass { background: rgba(52, 211, 153, 0.18); color: var(--good); }
  .fid-borderline { background: rgba(251, 191, 36, 0.18); color: var(--warn); }
  .fid-fail { background: rgba(248, 113, 113, 0.18); color: var(--bad); }
  .fid-na { color: var(--fg-dim); }
  .ml { font-variant-numeric: tabular-nums; color: var(--fg); }
  .ml-na { color: var(--fg-dim); }
  .ml-tag { font-size: 0.7em; padding: 1px 4px; border-radius: 3px; background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); color: white; margin-left: 4px; vertical-align: middle; }
  .repack-win { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--good); padding: 2px 6px; border-radius: 4px; background: rgba(52, 211, 153, 0.12); }
  .repack-loss { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--bad); padding: 2px 6px; border-radius: 4px; background: rgba(248, 113, 113, 0.12); border-bottom: 1px dotted rgba(248,113,113,0.55); cursor: help; }
  /* fid-styles-end */
`;
const fidStylesRe = /\/\* fid-styles-injected \*\/[\s\S]*?\/\* fid-styles-end \*\/\n/;
if (fidStylesRe.test(html)) {
  html = html.replace(fidStylesRe, fidStyles);
} else if (html.match(/\.ratio-low \{[^}]*\}\s*/m)) {
  html = html.replace(/(\.ratio-low \{[^}]*\}\s*)/m, `$1${fidStyles}`);
} else {
  html = html.replace('</style>', `${fidStyles}</style>`);
}
// Older runs may have left an undelimited copy of these style rules in
// the file. They use the same class names as the delimited copy above,
// so the cascade keeps the (later) delimited copy in effect; we leave
// the dupes alone rather than risk an over-eager regex.

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

const hasMl = fidelityRows.some((r) => typeof r.f.mlScore === 'number');
const mlHeader = hasMl ? ' web-mobile ML | size-min ML |' : '';
const mlAlign = hasMl ? ' ---: | ---: |' : '';
const mlVersion = splatbench.scenes.find((s) => s.fidelity?.webMobile?.mlScoreVersion)?.fidelity
  ?.webMobile?.mlScoreVersion;

const fidelitySection = `## Leaderboard — visual fidelity (v0.1.1)

Frames captured via \`@splatforge/viewer\` in headless Chromium (SwiftShader software-rendered WebGL2), 8 deterministic orbit poses at 512×512. \`lossless-repack\` is the per-scene baseline. ΔE94 is normalized to 0..1 (i.e. \`3%\` = 3 absolute ΔE94 units, the perceptibility threshold of an attentive observer).${
    hasMl
      ? `\n\n**ML Score** is the splat-aware perceptual metric from \`splatforge-pro\` (version \`${mlVersion}\`), a proprietary build that scores rendered vs baseline frames with a model tuned for Gaussian-splat failure modes. Higher is better; 100% means visually identical. ML Score values are published; reproducing them requires the \`splatforge-pro\` binary.`
      : ''
  }

| Rank | Scene | web-mobile ΔE94 mean / max | status | size-min ΔE94 mean / max | status |${mlHeader}
| ---: | ----- | ---: | :---: | ---: | :---: |${mlAlign}
${fidelityRows
  .map((r, i) => {
    const ml = hasMl
      ? ` ${pct(r.f.mlScore ?? 0)} | ${pct(r.g.mlScore ?? 0)} |`
      : '';
    return `| ${i + 1} | \`${r.id}\` | ${pct(r.f.meanDeltaE94)} / ${pct(r.f.maxDeltaE94)} | **${r.f.status}** | ${pct(r.g.meanDeltaE94)} / ${pct(r.g.maxDeltaE94)} | **${r.g.status}** |${ml}`;
  })
  .join('\n')}

**Pass criterion:** mean ΔE94 < 3% AND max ΔE94 < 8%. **Borderline:** mean 2–3% or max 5–8%. **Pass:** mean < 2% AND max < 5%.

Software-rendered numbers may differ slightly from hardware-accelerated chromium; see \`fidelity-v0.json\` for per-frame raw metrics and \`benches/reports/frames/<scene>/<preset>/0001.png\` etc. for the actual frames.

`;

// Drop the entire "What's intentionally missing from v0" section's visual-fidelity bullet.
md = md.replace(
  /\* \*\*Visual-fidelity scores\*\*[^\n]*\n[^\n]*\n/,
  '',
);

// Always rewrite the fidelity section so updates (e.g., new ML Score column)
// land idempotently. We delimit it by the next "## " header.
if (md.includes('Leaderboard — visual fidelity')) {
  md = md.replace(
    /## Leaderboard — visual fidelity[^]*?(?=\n## )/,
    fidelitySection.trimEnd() + '\n\n',
  );
} else {
  md = md.replace(/## Corpus composition/, `${fidelitySection}## Corpus composition`);
}

// Recompute corpus aggregates from the scenes array so the JSON's `aggregates`
// block and the markdown headline never drift apart from the actual data.
const wmRatios = splatbench.scenes.map((s) => s.webMobileRatio).sort((a, b) => a - b);
const smRatios = splatbench.scenes.map((s) => s.sizeMinRatio).sort((a, b) => a - b);
const median = (arr) => {
  if (arr.length === 0) return 0;
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
};
const presetWMpass = splatbench.scenes.filter(
  (s) => s.fidelity?.webMobile?.status !== 'fail',
).length;
const presetSMpass = splatbench.scenes.filter(
  (s) => s.fidelity?.sizeMin?.status !== 'fail',
).length;
const scenesReal = splatbench.scenes.filter((s) => s.source === 'real').length;
const scenesSynthetic = splatbench.scenes.filter((s) => s.source === 'synthetic').length;
const splatTotal = splatbench.scenes.reduce((a, s) => a + (s.splatCount || 0), 0);
const bytesInTotal = splatbench.scenes.reduce((a, s) => a + (s.bytesIn || 0), 0);
const webMobileSpzTotal = splatbench.scenes.reduce((a, s) => a + (s.webMobileSpzBytes || 0), 0);
const sizeMinSpzTotal = splatbench.scenes.reduce((a, s) => a + (s.sizeMinSpzBytes || 0), 0);

splatbench.aggregates = {
  scenesTotal: splatbench.scenes.length,
  scenesReal,
  scenesSynthetic,
  splatCountTotal: splatTotal,
  bytesInTotal,
  webMobileSpzTotal,
  sizeMinSpzTotal,
  webMobileRatioOverall: +(bytesInTotal / Math.max(1, webMobileSpzTotal)).toFixed(2),
  sizeMinRatioOverall: +(bytesInTotal / Math.max(1, sizeMinSpzTotal)).toFixed(2),
  webMobileRatioMin: +wmRatios[0]?.toFixed(2) || 0,
  webMobileRatioMedian: +median(wmRatios).toFixed(2),
  webMobileRatioMax: +wmRatios[wmRatios.length - 1]?.toFixed(2) || 0,
  sizeMinRatioMin: +smRatios[0]?.toFixed(2) || 0,
  sizeMinRatioMedian: +median(smRatios).toFixed(2),
  sizeMinRatioMax: +smRatios[smRatios.length - 1]?.toFixed(2) || 0,
  fidelityWebMobilePass: presetWMpass,
  fidelitySizeMinPass: presetSMpass,
};
writeFileSync(SPLATBENCH_JSON, JSON.stringify(splatbench, null, 2) + '\n');

const fmtSplatCount = (n) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
};

// Rebuild the Headline table from scratch so re-runs can't accumulate
// duplicate "fidelity passing" rows.
const a = splatbench.aggregates;
const headline = `## Headline

| Metric | Value |
| ---: | ---: |
| Scenes total | **${a.scenesTotal}** (${a.scenesReal} real + ${a.scenesSynthetic} synthetic) |
| Splats total | **${fmtSplatCount(a.splatCountTotal)}** across the corpus |
| Input total | **${fmtBytes(a.bytesInTotal)}** raw PLY |
| Corpus total \`web-mobile\` | **${fmtBytes(a.bytesInTotal)} → ${fmtBytes(a.webMobileSpzTotal)}** (${a.webMobileRatioOverall}× overall) |
| Corpus total \`size-min\` | **${fmtBytes(a.bytesInTotal)} → ${fmtBytes(a.sizeMinSpzTotal)}** (${a.sizeMinRatioOverall}× overall) |
| \`web-mobile\` ratio (min / median / max) | **${a.webMobileRatioMin}× / ${a.webMobileRatioMedian}× / ${a.webMobileRatioMax}×** |
| \`size-min\` ratio (min / median / max) | **${a.sizeMinRatioMin}× / ${a.sizeMinRatioMedian}× / ${a.sizeMinRatioMax}×** |
| \`web-mobile\` fidelity passing | **${a.fidelityWebMobilePass} / ${a.scenesTotal}** scenes within PRD threshold |
| \`size-min\` fidelity passing | **${a.fidelitySizeMinPass} / ${a.scenesTotal}** scenes within PRD threshold |

`;
md = md.replace(/## Headline\n[^]*?(?=\n## )/, headline.trimEnd() + '\n\n');

// Rebuild the per-preset leaderboards from the JSON so new corpus scenes show
// up here too. Sorted by ratio descending.
const fmtMs = (n) => `${n.toLocaleString('en-US')} ms`;
const fmtSplats = (n) => n.toLocaleString('en-US');
const fmtBytesShort = (n) => {
  if (n >= 1073741824) return `${(n / 1073741824).toFixed(2)} GB`;
  if (n >= 1048576) return `${(n / 1048576).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
};
const wmRows = [...splatbench.scenes].sort((x, y) => y.webMobileRatio - x.webMobileRatio);
const smRows = [...splatbench.scenes].sort((x, y) => y.sizeMinRatio - x.sizeMinRatio);
const wmTable =
  `## Leaderboard — \`web-mobile\` preset\n\n` +
  `| Rank | Scene | Class | Source | Splats | Input | SPZ out | **Ratio** | analyze |\n` +
  `| ---: | ----- | ----- | ------ | ---: | ---: | ---: | ---: | ---: |\n` +
  wmRows
    .map(
      (s, i) =>
        `| ${i + 1} | \`${s.id}\` | ${s.class} | ${s.source === 'real' ? '**real**' : 'synthetic'} | ${fmtSplats(s.splatCount)} | ${fmtBytesShort(s.bytesIn)} | ${fmtBytesShort(s.webMobileSpzBytes)} | **${s.webMobileRatio.toFixed(2)}×** | ${fmtMs(s.analyzeMs)} |`,
    )
    .join('\n') +
  '\n\n';
const smTable =
  `## Leaderboard — \`size-min\` preset\n\n` +
  `| Rank | Scene | SPZ out | **Ratio** |\n` +
  `| ---: | ----- | ---: | ---: |\n` +
  smRows
    .map(
      (s, i) =>
        `| ${i + 1} | \`${s.id}\` | ${fmtBytesShort(s.sizeMinSpzBytes)} | **${s.sizeMinRatio.toFixed(2)}×** |`,
    )
    .join('\n') +
  '\n\n';
md = md.replace(/## Leaderboard — `web-mobile` preset\n[^]*?(?=\n## )/, wmTable.trimEnd() + '\n\n');
md = md.replace(/## Leaderboard — `size-min` preset\n[^]*?(?=\n## )/, smTable.trimEnd() + '\n\n');

// Bump the version line and run date.
md = md.replace(/\*\*SplatForge version:\*\* `0\.1\.0`/, `**SplatForge version:** \`${VERSION}\``);
md = md.replace(/\*\*Run date:\*\* `?[^\n`]+`?/, `**Run date:** ${splatbench.runDate}`);

writeFileSync(SPLATBENCH_MD, md);

console.error(`updated ${SPLATBENCH_JSON}`);
console.error(`updated ${SPLATBENCH_MD}`);
console.error(`updated ${SPLATBENCH_HTML}`);
console.error(`scenes with fidelity: ${scenesWithFidelity}/${splatbench.scenes.length}`);
if (scenesMissing.length > 0) {
  console.error(`scenes still missing fidelity: ${scenesMissing.join(', ')}`);
}
