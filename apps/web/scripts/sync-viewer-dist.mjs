#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
/**
 * sync-viewer-dist.mjs — copy `packages/viewer/dist/` into
 * `apps/web/public/viewer/` so Astro pages can `import("/viewer/index.js")`
 * (and `/viewer/lodge/index.js`) without bundling. This replaces the
 * older hand-written re-export shim that lived in `public/viewer/` and
 * was missing the `lodge/`, `streaming/`, `progressive/` subtrees.
 *
 * Run this before `astro build` (the package.json build script chains
 * sync-data.mjs -> build-explore-index.mjs -> sync-viewer-dist.mjs ->
 * astro check -> astro build).
 *
 * The viewer package is ESM-only with `.js` import paths, which Astro
 * serves verbatim from `public/`. No bundler needed.
 */

import { cpSync, existsSync, mkdirSync, readdirSync, statSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = resolve(HERE, '..');
const DIST = resolve(APP_ROOT, '..', '..', 'packages', 'viewer', 'dist');
const PUBLIC_VIEWER = resolve(APP_ROOT, 'public', 'viewer');

if (!existsSync(DIST)) {
  // On Vercel the workspace `packages/viewer` is checked out but not
  // built (Vercel only runs the `apps/web` build script with
  // rootDirectory=apps/web). The committed `apps/web/public/viewer/`
  // tree IS the canonical browser-side surface in that environment.
  // Skip the sync rather than fail the build.
  console.warn(
    `sync-viewer-dist: ${DIST} does not exist; assuming pre-synced ` +
      `public/viewer/ (Vercel-style build). Skipping sync.`,
  );
  process.exit(0);
}

// Wipe + rebuild so renames in the source don't leave stale files behind.
// Preserve /viewer/vendor/ — it holds importmap targets (fflate, fzstd) that
// aren't part of the packages/viewer build output but are referenced by the
// viewer's bare specifiers via the importmap in apps/web/src/layouts/Base.astro.
if (existsSync(PUBLIC_VIEWER)) {
  for (const name of readdirSync(PUBLIC_VIEWER)) {
    if (name === 'vendor') continue;
    rmSync(join(PUBLIC_VIEWER, name), { recursive: true, force: true });
  }
}
mkdirSync(PUBLIC_VIEWER, { recursive: true });

// Copy everything except .map files (they bloat deploy by ~3x and the
// browser doesn't need them for the demo).
const KEEP_EXT = new Set(['.js']);

const SKIP_DIRS = new Set(['__tests__']);

function walk(dir, dest) {
  for (const name of readdirSync(dir)) {
    if (SKIP_DIRS.has(name)) continue;
    const src = join(dir, name);
    const dst = join(dest, name);
    const st = statSync(src);
    if (st.isDirectory()) {
      mkdirSync(dst, { recursive: true });
      walk(src, dst);
    } else {
      const ext = name.slice(name.lastIndexOf('.'));
      if (KEEP_EXT.has(ext)) {
        cpSync(src, dst);
      }
    }
  }
}

walk(DIST, PUBLIC_VIEWER);
console.log(`sync-viewer-dist: copied ${DIST} -> ${PUBLIC_VIEWER}`);

// Phase 3: also drop in the WebGPU preview shell (drag-drop UI + controls
// + format loaders) as /viewer/index.html so visitors hitting /viewer/
// directly get a working interactive viewer, not a directory listing.
//
// `packages/viewer/src/index.html` is the canonical preview shell. Its
// importmap and module imports use relative paths assuming `dist/` is a
// sibling — we rewrite them on the fly so they resolve under /viewer/ on
// the deployed site.
const SRC_HTML = resolve(APP_ROOT, '..', '..', 'packages', 'viewer', 'src', 'index.html');
if (existsSync(SRC_HTML)) {
  const { readFileSync, writeFileSync } = await import('node:fs');
  const html = readFileSync(SRC_HTML, 'utf8')
    .replaceAll('../dist/index.js', './index.js')
    .replaceAll('../dist/renderer/webgpu.js', './renderer/webgpu.js')
    .replaceAll('../dist/camera-state.js', './camera-state.js')
    .replaceAll('../dist/controls.js', './controls.js')
    .replaceAll('../../glb-polyfill/dist/index.js', './glb-polyfill/index.js');
  writeFileSync(join(PUBLIC_VIEWER, 'index.html'), html);
  console.log(`sync-viewer-dist: wrote preview shell at ${PUBLIC_VIEWER}/index.html`);
}

// Phase 3: also stage the glb-polyfill dist under /viewer/glb-polyfill/
// so the preview shell's importmap entry resolves on the deployed site.
const POLYFILL_DIST = resolve(APP_ROOT, '..', '..', 'packages', 'glb-polyfill', 'dist');
if (existsSync(POLYFILL_DIST)) {
  const POLYFILL_DEST = join(PUBLIC_VIEWER, 'glb-polyfill');
  mkdirSync(POLYFILL_DEST, { recursive: true });
  walk(POLYFILL_DIST, POLYFILL_DEST);
  console.log(`sync-viewer-dist: copied ${POLYFILL_DIST} -> ${POLYFILL_DEST}`);
}
