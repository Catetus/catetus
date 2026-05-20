#!/usr/bin/env node
/**
 * Build the Explore registry index.
 *
 * Reads:
 *   - benches/reports/splatbench-v0.json  (canonical scenes we own + bench)
 *   - benches/seed-corpus/external-manifest.json (third-party public splats)
 *
 * Emits:
 *   - apps/web/src/data/explore-index.json — single normalized list, sorted,
 *     ready for Astro to import at build time and for static-side filter JS
 *     to consume in the browser.
 *
 * Each entry has the same shape, regardless of source:
 *   {
 *     id, source ("splatbench"|"external"), displayName, license, attribution,
 *     profile, splatCount, bytesIn, format, originUrl,
 *     selfHosted (bool — can we serve it through our viewer today?),
 *     khrConformance ("pass"|"fail-expected"|"unknown"),
 *     fidelity (optional, only for splatbench entries with measured fidelity),
 *     thumbHint
 *   }
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const APP_ROOT = resolve(__dirname, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');

const SPLATBENCH_PATH = resolve(REPO_ROOT, 'benches', 'reports', 'splatbench-v0.json');
const EXTERNAL_PATH = resolve(REPO_ROOT, 'benches', 'seed-corpus', 'external-manifest.json');
const OUT_PATH = resolve(APP_ROOT, 'src', 'data', 'explore-index.json');

/** Map raw splatbench class tag to a coarse profile bucket for the filter UI. */
function classToProfile(cls) {
  if (!cls) return 'other';
  if (cls.includes('indoor')) return 'indoor';
  if (cls.includes('outdoor')) return 'outdoor';
  if (cls.includes('foliage') || cls.includes('translucency')) return 'foliage';
  if (cls.includes('portrait')) return 'portrait';
  if (cls.includes('product')) return 'product-scan';
  if (cls.includes('specular') || cls.includes('lowlight') || cls.includes('texture') ||
      cls.includes('depth') || cls.includes('motion') || cls.includes('banding') ||
      cls.includes('transparency') || cls.includes('noisy') || cls.includes('dense')) {
    return 'synthetic-probe';
  }
  return 'other';
}

/** Coarse license bucket for the filter UI. */
function licenseBucket(raw) {
  if (!raw) return 'unknown';
  const r = raw.toLowerCase();
  if (r.includes('cc0')) return 'CC0';
  if (r.includes('cc-by-sa') || r.includes('cc by-sa')) return 'CC-BY-SA';
  if (r.includes('cc-by') || r.includes('cc by')) return 'CC-BY';
  if (r.includes('apache')) return 'Apache-2.0';
  if (r.includes('mit')) return 'MIT';
  if (r.includes('mip-nerf') || r.includes('deep blending') || r.includes('research')) {
    return 'research-open';
  }
  return 'other';
}

function attributionForSplatbench(scene) {
  if (scene.source === 'real') {
    if (scene.id.includes('mipnerf360')) {
      return 'Barron et al. (Mip-NeRF 360, CVPR 2022). 3DGS by dylanebert (HuggingFace).';
    }
    return 'Public research dataset.';
  }
  return 'Catetus synthetic probe (Apache-2.0, deterministic).';
}

function buildSplatbenchEntries(splatbench) {
  return splatbench.scenes.map((s) => {
    const fid = s.fidelity?.webMobile;
    const passed = fid?.passed === true;
    return {
      id: s.id,
      source: 'splatbench',
      displayName: s.id.replace(/_/g, ' '),
      license: s.license,
      licenseBucket: licenseBucket(s.license),
      attribution: attributionForSplatbench(s),
      profile: classToProfile(s.class),
      profileRaw: s.class,
      splatCount: s.splatCount,
      bytesIn: s.bytesIn,
      format: 'ply',
      originUrl: s.origin,
      selfHosted: true,
      // splatbench scenes were processed through our own writer and pass the
      // Catetus KHR-spz subset by construction (we wrote them).
      khrConformance: passed ? 'pass' : 'unknown',
      fidelity: fid ? {
        meanDeltaE94: fid.meanDeltaE94,
        passed: fid.passed,
        status: fid.status,
      } : null,
      thumbHint: classToProfile(s.class),
      // For self-hosted scenes the viewer can use the bench buffer URLs — but
      // we don't ship the per-scene gltf payloads on the public site for the
      // alpha. Instead the per-scene page renders metadata and a CTA to the
      // optimizer. The hero scene (`bonsai`) is the only one with a public
      // viewer-ready gltf under /hero-scene/.
      viewerSrc: s.id === 'bonsai_mipnerf360_iter7k' ? '/hero-scene/scene.gltf' : null,
    };
  });
}

function buildExternalEntries(manifest) {
  return manifest.scenes.map((s) => ({
    id: s.id,
    source: 'external',
    displayName: s.displayName,
    license: s.license,
    licenseBucket: licenseBucket(s.license),
    attribution: s.attribution,
    profile: classToProfile(s.profile) === 'other' ? s.profile : classToProfile(s.profile),
    profileRaw: s.profile,
    splatCount: s.splatCountApprox,
    bytesIn: s.bytesApprox,
    format: s.format,
    originUrl: s.url,
    selfHosted: false,
    khrConformance: s.khrConformance,
    fidelity: null,
    thumbHint: s.thumbHint,
    notes: s.notes,
    viewerSrc: null,
  }));
}

function main() {
  const splatbench = JSON.parse(readFileSync(SPLATBENCH_PATH, 'utf8'));
  const external = JSON.parse(readFileSync(EXTERNAL_PATH, 'utf8'));

  const splatbenchEntries = buildSplatbenchEntries(splatbench);
  const externalEntries = buildExternalEntries(external);

  // Sort: self-hosted first (best UX — viewer works inline), then by license
  // openness (CC0 > CC-BY > research-open > other), then by splat count desc.
  const licenseOrder = { 'CC0': 0, 'CC-BY': 1, 'CC-BY-SA': 2, 'Apache-2.0': 3, 'MIT': 4, 'research-open': 5, 'other': 6, 'unknown': 7 };
  const all = [...splatbenchEntries, ...externalEntries].sort((a, b) => {
    if (a.selfHosted !== b.selfHosted) return a.selfHosted ? -1 : 1;
    const la = licenseOrder[a.licenseBucket] ?? 99;
    const lb = licenseOrder[b.licenseBucket] ?? 99;
    if (la !== lb) return la - lb;
    return (b.splatCount ?? 0) - (a.splatCount ?? 0);
  });

  // Distinct facets, for the filter UI.
  const facets = {
    licenses: [...new Set(all.map((s) => s.licenseBucket))].sort(),
    profiles: [...new Set(all.map((s) => s.profile))].sort(),
    sources: ['splatbench', 'external'],
    khrConformance: [...new Set(all.map((s) => s.khrConformance))].sort(),
  };

  const index = {
    schema: 'catetus.explore-index/0.1',
    generatedAt: new Date().toISOString().slice(0, 10),
    splatbenchVersion: splatbench.catetusVersion,
    splatbenchRunDate: splatbench.runDate,
    countsBySource: {
      splatbench: splatbenchEntries.length,
      external: externalEntries.length,
      total: all.length,
    },
    facets,
    scenes: all,
  };

  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, JSON.stringify(index, null, 2) + '\n', 'utf8');
  console.error(
    `[build-explore-index] wrote ${all.length} entries ` +
    `(splatbench=${splatbenchEntries.length}, external=${externalEntries.length}) ` +
    `to ${OUT_PATH}`,
  );
}

main();
