/**
 * Visual-diff report template (SPEC-0009).
 *
 * Renders a self-contained HTML page showing before/after/diff frames
 * side-by-side, aggregate metrics at the top, and a pass/fail badge driven
 * by the configured threshold (defaults to 0.03 mean pixel diff).
 */

/** Aggregate metrics over the orbit-8 camera path. */
export interface DiffMetrics {
  /** Max per-frame pixel L1 difference, in 0..1. */
  max: number;
  /** Mean per-frame pixel L1 difference, in 0..1. */
  mean: number;
  /** 95th percentile per-frame pixel L1 difference, in 0..1. */
  p95: number;
  /** Optional PSNR (higher is better). */
  psnr?: number;
  /** Optional SSIM (1.0 = identical). */
  ssim?: number;
  /** Optional perceptual ΔE94 mean over OKLab. */
  deltaE94Mean?: number;
}

/** Per-frame data — base64 data URLs for self-contained reports. */
export interface DiffFrame {
  /** 1-indexed frame number, matches `0001..0008.png` on disk. */
  index: number;
  /** `data:image/png;base64,...` for the "before" frame. */
  beforePng: string;
  /** `data:image/png;base64,...` for the "after" frame. */
  afterPng: string;
  /** `data:image/png;base64,...` for the diff overlay. */
  diffPng: string;
  /** Per-frame pixel L1 in 0..1. */
  diffRatio: number;
}

/** Full input to {@link renderDiffReport}. */
export interface DiffReportData {
  /** Logical asset identifier, e.g. `warehouse_scan`. */
  asset: string;
  /** Pass/fail threshold on `metrics.mean`. Default 0.03. */
  threshold: number;
  /** Aggregate metrics. */
  metrics: DiffMetrics;
  /** Per-frame frames. Sorted by `index` ascending for determinism. */
  frames: DiffFrame[];
  /** Optional camera path label, e.g. `orbit-8`. */
  cameraPath?: string;
  /** Optional frame size label, e.g. `512x512`. */
  frameSize?: string;
  /**
   * If set, the report embeds this ISO-8601 timestamp. Omit (the default)
   * for deterministic, byte-stable output — snapshot tests rely on this.
   */
  generatedAt?: string;
}

/** HTML-escape a string. Safe for text content and attribute values. */
function esc(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Format a 0..1 ratio as a fixed 4-decimal percent. */
function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

/** Format an optional numeric metric to 3 decimals, or `n/a`. */
function fmt(value: number | undefined, digits = 3): string {
  return value === undefined ? 'n/a' : value.toFixed(digits);
}

const STYLE = `
  :root {
    color-scheme: dark;
    --bg: #0b0d10;
    --panel: #14181d;
    --border: #232a32;
    --text: #d7dde4;
    --muted: #7a8591;
    --accent: #5ac8fa;
    --pass: #4ade80;
    --fail: #f87171;
    --warn: #facc15;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px;
    line-height: 1.5;
  }
  h1 { font-size: 18px; margin: 0 0 8px; }
  h2 { font-size: 14px; margin: 24px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge.pass { background: rgba(74, 222, 128, 0.15); color: var(--pass); }
  .badge.fail { background: rgba(248, 113, 113, 0.15); color: var(--fail); }
  .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 12px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
  .card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  .card .value { font-size: 18px; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  details { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 8px; }
  details summary { cursor: pointer; padding: 10px 12px; font-weight: 600; list-style: none; }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: '\\25B6'; display: inline-block; width: 14px; color: var(--muted); transition: transform 0.15s; }
  details[open] summary::before { transform: rotate(90deg); }
  details .body { padding: 12px; border-top: 1px solid var(--border); display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  details img { width: 100%; height: auto; border-radius: 4px; image-rendering: pixelated; background: #000; }
  details .col { display: flex; flex-direction: column; gap: 4px; }
  details .col .cap { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .meta { color: var(--muted); font-size: 11px; margin-top: 4px; }
`;

/**
 * Render a SPEC-0009 visual-diff report to a single self-contained HTML
 * document.
 *
 * The output is deterministic given the same input: no timestamps are
 * embedded unless {@link DiffReportData.generatedAt} is supplied.
 */
export function renderDiffReport(data: DiffReportData): string {
  const passed = data.metrics.mean <= data.threshold;
  const sortedFrames = [...data.frames].sort((a, b) => a.index - b.index);
  const generated = data.generatedAt
    ? `<div class="meta">Generated: ${esc(data.generatedAt)}</div>`
    : '';
  const cameraLabel = data.cameraPath ?? 'orbit-8';
  const frameSizeLabel = data.frameSize ?? '512x512';

  const framesHtml = sortedFrames
    .map((f) => {
      const idx = String(f.index).padStart(4, '0');
      return `<details>
  <summary>Frame ${esc(idx)} — diff ${esc(pct(f.diffRatio))}</summary>
  <div class="body">
    <div class="col"><span class="cap">before</span><img alt="before frame ${esc(idx)}" src="${esc(f.beforePng)}" /></div>
    <div class="col"><span class="cap">after</span><img alt="after frame ${esc(idx)}" src="${esc(f.afterPng)}" /></div>
    <div class="col"><span class="cap">diff</span><img alt="diff frame ${esc(idx)}" src="${esc(f.diffPng)}" /></div>
  </div>
</details>`;
    })
    .join('\n');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SplatForge diff — ${esc(data.asset)}</title>
<style>${STYLE}</style>
</head>
<body>
<header>
  <h1>SplatForge visual diff — ${esc(data.asset)}
    <span class="badge ${passed ? 'pass' : 'fail'}">${passed ? 'PASS' : 'FAIL'}</span>
  </h1>
  <div class="meta">camera ${esc(cameraLabel)} · frame ${esc(frameSizeLabel)} · threshold ${esc(pct(data.threshold))}</div>
  ${generated}
</header>

<h2>Aggregate</h2>
<div class="summary">
  <div class="card"><div class="label">mean</div><div class="value">${esc(pct(data.metrics.mean))}</div></div>
  <div class="card"><div class="label">max</div><div class="value">${esc(pct(data.metrics.max))}</div></div>
  <div class="card"><div class="label">p95</div><div class="value">${esc(pct(data.metrics.p95))}</div></div>
  <div class="card"><div class="label">psnr</div><div class="value">${esc(fmt(data.metrics.psnr, 2))}</div></div>
  <div class="card"><div class="label">ssim</div><div class="value">${esc(fmt(data.metrics.ssim, 4))}</div></div>
  <div class="card"><div class="label">ΔE94</div><div class="value">${esc(fmt(data.metrics.deltaE94Mean, 3))}</div></div>
</div>

<h2>Frames</h2>
${framesHtml}
</body>
</html>
`;
}
