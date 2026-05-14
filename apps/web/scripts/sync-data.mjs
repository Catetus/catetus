#!/usr/bin/env node
/**
 * Copy benchmark report JSON into `src/data/` so the Astro project is
 * self-contained at build time. Vercel uploads only the project root
 * (`apps/web/`) during deploys; this script materializes any cross-package
 * dependencies inside that root before the build runs.
 *
 * Source of truth lives at `benches/reports/*.json` in the repo root.
 */
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const APP_ROOT = resolve(__dirname, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');
const SOURCES = ['splatbench-v0.json', 'bonsai-real-demo.md'];
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
console.error(`[sync-data] copied ${copied} files into ${DEST_DIR} (${skipped} missing)`);
