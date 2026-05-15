#!/usr/bin/env node
/**
 * Copy benchmark report JSON into `src/data/` and rendered orbit frames
 * into `public/rate-frames/` so the Astro project is self-contained at
 * build time. Vercel uploads only the project root (`apps/web/`) during
 * deploys; this script materializes any cross-package dependencies
 * inside that root before the build runs.
 *
 * Source of truth lives at `benches/reports/` in the repo root.
 *
 *   - JSON / markdown    → `apps/web/src/data/`
 *   - Frame PNGs         → `apps/web/public/rate-frames/<scene>/<preset>/`
 *   - Frame manifest     → `apps/web/src/data/rate-frames.json`
 *
 * The frame manifest is what `/rate` reads at build time to know which
 * (scene, preset, frame_index) tuples are available. Only scenes that
 * have a `lossless-repack` reference AND at least one non-reference
 * candidate are listed — those are the only pairs that make sense to
 * show. Scenes with <2 candidates are skipped because the page needs
 * two distinct candidate presets per pair.
 *
 * Missing frame source is tolerated — the script writes an empty
 * manifest and the page renders an "awaiting bench renders" empty
 * state. This keeps `npm run build` green in environments where the
 * bench harness hasn't shipped frames yet.
 */
import {
  copyFileSync,
  mkdirSync,
  existsSync,
  readdirSync,
  writeFileSync,
  statSync,
} from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const APP_ROOT = resolve(__dirname, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');
const SOURCES = [
  'splatbench-v0.json',
  'splatbench-v0.encoders.json',
  'bonsai-real-demo.md',
];
const DEST_DIR = resolve(APP_ROOT, 'src', 'data');

mkdirSync(DEST_DIR, { recursive: true });

let copied = 0;
let skipped = 0;
for (const name of SOURCES) {
  const src = resolve(REPO_ROOT, 'benches', 'reports', name);
  if (!existsSync(src)) {
    console.warn(`[sync-data] missing source: ${src} (skipping)`);
    skipped++;
    continue;
  }
  const dst = resolve(DEST_DIR, name);
  copyFileSync(src, dst);
  copied++;
}
console.error(
  `[sync-data] copied ${copied} files into ${DEST_DIR} (${skipped} missing)`,
);

/* -------- frame sync for the /rate page (fidelity-ml v0.4 collection) -------- */

// Candidate roots: try the worktree's benches/reports/frames first, then
// fall back to a sibling main-repo checkout if this is a git worktree.
// The latter is uncommitted (frames are gitignored) so the dev shell
// keeps a single copy on disk under the primary worktree.
const FRAME_CANDIDATES = [
  resolve(REPO_ROOT, 'benches', 'reports', 'frames'),
];
// If the repo lives inside `.claude/worktrees/<id>/` walk back up two
// levels and try the main checkout.
const m = REPO_ROOT.match(/^(.*?)\/\.claude\/worktrees\//);
if (m) {
  FRAME_CANDIDATES.push(resolve(m[1], 'benches', 'reports', 'frames'));
}

const FRAMES_SRC = FRAME_CANDIDATES.find((p) => existsSync(p));
const FRAMES_DST = resolve(APP_ROOT, 'public', 'rate-frames');
const REFERENCE_PRESET = 'lossless-repack';
// Bound the frame count so we don't ship hundreds of MB of PNG into the
// static site. orbit-8 means at most 8 frames per (scene, preset).
const MAX_FRAMES_PER_PRESET = 8;

const manifest = { reference_preset: REFERENCE_PRESET, scenes: [] };
let frames_copied = 0;

if (FRAMES_SRC) {
  mkdirSync(FRAMES_DST, { recursive: true });
  for (const sceneDir of readdirSync(FRAMES_SRC)) {
    const sceneSrc = join(FRAMES_SRC, sceneDir);
    if (!statSync(sceneSrc).isDirectory()) continue;

    const presets = readdirSync(sceneSrc).filter((p) => {
      const dir = join(sceneSrc, p);
      if (!statSync(dir).isDirectory()) return false;
      return readdirSync(dir).some((f) => f.endsWith('.png'));
    });
    if (!presets.includes(REFERENCE_PRESET)) continue;
    const otherPresets = presets.filter((p) => p !== REFERENCE_PRESET);
    if (otherPresets.length < 2) continue;

    const sceneFrameCounts = {};
    for (const preset of presets) {
      const presetSrc = join(sceneSrc, preset);
      const presetDst = join(FRAMES_DST, sceneDir, preset);
      mkdirSync(presetDst, { recursive: true });
      const pngs = readdirSync(presetSrc)
        .filter((f) => f.endsWith('.png'))
        .sort()
        .slice(0, MAX_FRAMES_PER_PRESET);
      for (const png of pngs) {
        copyFileSync(join(presetSrc, png), join(presetDst, png));
        frames_copied++;
      }
      sceneFrameCounts[preset] = pngs.length;
    }
    manifest.scenes.push({
      scene_id: sceneDir,
      reference_preset: REFERENCE_PRESET,
      candidate_presets: otherPresets,
      frame_indices: Array.from(
        { length: Math.min(MAX_FRAMES_PER_PRESET, sceneFrameCounts[REFERENCE_PRESET] || 0) },
        (_, i) => i + 1,
      ),
    });
  }
  writeFileSync(
    resolve(DEST_DIR, 'rate-frames.json'),
    JSON.stringify(manifest, null, 2) + '\n',
  );
  console.error(
    `[sync-data] copied ${frames_copied} frame PNGs into ${FRAMES_DST}; ${manifest.scenes.length} scenes listed`,
  );
} else {
  console.warn(
    `[sync-data] no frame sources at any of: ${FRAME_CANDIDATES.join(', ')} — writing empty rate-frames manifest`,
  );
  writeFileSync(
    resolve(DEST_DIR, 'rate-frames.json'),
    JSON.stringify({ reference_preset: REFERENCE_PRESET, scenes: [] }, null, 2) + '\n',
  );
}
