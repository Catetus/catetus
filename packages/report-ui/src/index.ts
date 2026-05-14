/**
 * `@splatforge/report-ui`
 *
 * Pure, deterministic HTML rendering helpers used by:
 *   - `splatforge diff` CLI         (SPEC-0009)
 *   - `tests/visual` parity harness (SPEC-0010)
 *
 * No external CDN dependencies, no runtime deps. All output is a single
 * self-contained HTML string with inlined CSS. The functions are pure of any
 * timestamp / random source so the rendered HTML is byte-stable given the
 * same input — snapshot tests rely on this.
 */
export {
  renderDiffReport,
  type DiffReportData,
  type DiffFrame,
  type DiffMetrics,
} from './diff-template.js';

export {
  renderParityReport,
  type ParityReportData,
  type ParityCell,
} from './parity-template.js';
