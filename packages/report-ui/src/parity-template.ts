/**
 * Viewer-parity report template (SPEC-0010).
 *
 * Renders the renderer × asset matrix as an HTML table. Visual-score cells are
 * color-coded:
 *
 *   - green  : visualScore >= 0.95
 *   - yellow : 0.85 <= visualScore < 0.95
 *   - red    : visualScore < 0.85
 */

/** One cell of the parity matrix, keyed by renderer name. */
export interface ParityCell {
  /** Visual score in 0..1, where 1.0 == pixel-identical to the reference. */
  visualScore: number;
  /** Average frames-per-second over the camera path. */
  fps?: number;
  /** Peak GPU memory in megabytes. */
  memoryMb?: number;
  /** Stable warning codes, e.g. `opacity_sorting_artifacts`. */
  warnings?: string[];
  /** True if the renderer ran without crashing. */
  ok?: boolean;
}

/** Full input to {@link renderParityReport}. */
export interface ParityReportData {
  /** Logical asset identifier. */
  asset: string;
  /** Renderer-name -> cell. Order is preserved in the rendered table. */
  matrix: Record<string, ParityCell>;
  /** Score thresholds used to color cells. Defaults: pass=0.95, warn=0.85. */
  thresholds?: { pass: number; warn: number };
  /**
   * If set, the report embeds this ISO-8601 timestamp. Omit (the default) for
   * deterministic, byte-stable output.
   */
  generatedAt?: string;
}

/** HTML-escape. */
function esc(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** 3-decimal score format. */
function fmtScore(value: number): string {
  return value.toFixed(3);
}

/** `value` or `n/a`. */
function fmtOpt(value: number | undefined, digits = 0): string {
  return value === undefined ? 'n/a' : value.toFixed(digits);
}

/** Cell color class given thresholds. */
function cellClass(score: number, pass: number, warn: number): string {
  if (score >= pass) return 'good';
  if (score >= warn) return 'mid';
  return 'bad';
}

const STYLE = `
  :root {
    color-scheme: dark;
    --bg: #0b0d10;
    --panel: #14181d;
    --border: #232a32;
    --text: #d7dde4;
    --muted: #7a8591;
    --good-bg: rgba(74, 222, 128, 0.18);
    --good-fg: #4ade80;
    --mid-bg: rgba(250, 204, 21, 0.18);
    --mid-fg: #facc15;
    --bad-bg: rgba(248, 113, 113, 0.18);
    --bad-fg: #f87171;
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
  .meta { color: var(--muted); font-size: 11px; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; background: #10141a; }
  tr:last-child td { border-bottom: 0; }
  td.score { font-weight: 600; }
  td.score.good { background: var(--good-bg); color: var(--good-fg); }
  td.score.mid  { background: var(--mid-bg);  color: var(--mid-fg); }
  td.score.bad  { background: var(--bad-bg);  color: var(--bad-fg); }
  td.warnings { color: var(--mid-fg); font-size: 11px; }
  ul.warns { margin: 0; padding-left: 18px; }
`;

/**
 * Render a SPEC-0010 viewer-parity report to a single self-contained HTML
 * document.
 *
 * Deterministic given the same input: no timestamp embedded unless
 * {@link ParityReportData.generatedAt} is set.
 */
export function renderParityReport(data: ParityReportData): string {
  const thresholds = data.thresholds ?? { pass: 0.95, warn: 0.85 };
  const generated = data.generatedAt
    ? `<div class="meta">Generated: ${esc(data.generatedAt)}</div>`
    : '';

  const rows = Object.entries(data.matrix)
    .map(([renderer, cell]) => {
      const klass = cellClass(cell.visualScore, thresholds.pass, thresholds.warn);
      const warnings = (cell.warnings ?? []).slice().sort();
      const warnHtml =
        warnings.length === 0
          ? '<span class="meta">none</span>'
          : `<ul class="warns">${warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>`;
      return `<tr>
  <td>${esc(renderer)}</td>
  <td class="score ${klass}">${esc(fmtScore(cell.visualScore))}</td>
  <td>${esc(fmtOpt(cell.fps, 1))}</td>
  <td>${esc(fmtOpt(cell.memoryMb, 0))}</td>
  <td class="warnings">${warnHtml}</td>
</tr>`;
    })
    .join('\n');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Catetus parity — ${esc(data.asset)}</title>
<style>${STYLE}</style>
</head>
<body>
<header>
  <h1>Catetus viewer parity — ${esc(data.asset)}</h1>
  <div class="meta">pass &ge; ${esc(fmtScore(thresholds.pass))} · warn &ge; ${esc(fmtScore(thresholds.warn))}</div>
  ${generated}
</header>

<h2>Matrix</h2>
<table>
  <thead>
    <tr>
      <th>renderer</th>
      <th>visual score</th>
      <th>fps</th>
      <th>memory MB</th>
      <th>warnings</th>
    </tr>
  </thead>
  <tbody>
${rows}
  </tbody>
</table>
</body>
</html>
`;
}
