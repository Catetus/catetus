#!/usr/bin/env node
/**
 * Rewrite `apps/web/public/compare-scenes/<scene>/<variant>/scene.gltf` from
 * the legacy KHR_gaussian_splatting attribute layout (bare keys
 * `_ROTATION`/`_SCALE`/`_OPACITY`/`_COLOR_DC` nested inside the
 * per-primitive extension object) to the RC layout (namespaced keys
 * `KHR_gaussian_splatting:ROTATION` etc. on `primitive.attributes` next to
 * `mode`). Pre-existing legacy fixtures are preserved under
 * `compare-scenes/legacy/` for back-compat regression coverage.
 *
 * Idempotent: re-running on already-RC files is a no-op.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, '../..');
const BASE = join(ROOT, 'apps/web/public/compare-scenes');

const SCENES = ['floater', 'indoor', 'product'];
const VARIANTS = ['size-min', 'web-mobile'];

const GS = 'KHR_gaussian_splatting';
const LEGACY_TO_RC = {
  POSITION: `${GS}:POSITION`,
  _ROTATION: `${GS}:ROTATION`,
  _SCALE: `${GS}:SCALE`,
  _OPACITY: `${GS}:OPACITY`,
  _COLOR_DC: `${GS}:COLOR_DC`,
  _COLOR_SH: `${GS}:COLOR_SH`,
};

function migrate(doc) {
  let changed = false;
  for (const mesh of doc.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const ext = prim.extensions?.[GS];
      const legacyAttrs = ext?.attributes;
      const primAttrs = prim.attributes ?? {};
      const alreadyRc = Object.keys(primAttrs).some((k) => k.startsWith(`${GS}:`));
      if (alreadyRc) continue;
      if (!legacyAttrs || typeof legacyAttrs !== 'object') continue;

      const newAttrs = { ...primAttrs };
      for (const [oldKey, accessor] of Object.entries(legacyAttrs)) {
        const rcKey = LEGACY_TO_RC[oldKey];
        if (!rcKey) continue;
        newAttrs[rcKey] = accessor;
      }
      prim.attributes = newAttrs;
      if (prim.mode === undefined) prim.mode = 0;
      // Strip the legacy `.attributes` from the primitive extension; keep the
      // extension object present (empty) so conformance checkers still see the
      // primitive as a splat primitive.
      delete ext.attributes;
      changed = true;
    }
  }
  return changed;
}

let migrated = 0;
for (const scene of SCENES) {
  for (const variant of VARIANTS) {
    const p = join(BASE, scene, variant, 'scene.gltf');
    const text = readFileSync(p, 'utf8');
    const doc = JSON.parse(text);
    if (migrate(doc)) {
      writeFileSync(p, JSON.stringify(doc, null, 2));
      console.log(`migrated: ${scene}/${variant}/scene.gltf`);
      migrated += 1;
    } else {
      console.log(`unchanged: ${scene}/${variant}/scene.gltf`);
    }
  }
}
console.log(`done — ${migrated} file(s) rewritten`);
