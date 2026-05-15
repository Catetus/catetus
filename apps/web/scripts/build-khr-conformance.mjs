#!/usr/bin/env node
/**
 * Build apps/web/src/data/khr-conformance.json by:
 *   1. Asking cargo to build the splatforge-khr-conformance fixture
 *      generator + validator (no-op if already built).
 *   2. Regenerating the fixture corpus into a tempdir.
 *   3. Running the validator in --json mode against every fixture.
 *   4. Writing a combined report JSON consumed by /khr-conformance.astro.
 *
 * If cargo is unavailable (Vercel's build environment usually is)
 * we fall through to whatever khr-conformance.json was committed —
 * the page is still SSR-rendered from disk, just with the last-known
 * good report instead of a freshly minted one. The committed JSON is
 * regenerated locally before pushing.
 */
import { execFileSync } from 'node:child_process';
import {
  mkdtempSync,
  existsSync,
  readdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const APP_ROOT = resolve(__dirname, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');
const OUT_DIR = resolve(APP_ROOT, 'src', 'data');
const OUT_PATH = resolve(OUT_DIR, 'khr-conformance.json');

// The committed fallback — never deleted by this script, always kept fresh
// by the local pre-commit invocation in scripts/build-khr-conformance.mjs.

function haveCargo() {
  try {
    execFileSync('cargo', ['--version'], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

function specMetadata() {
  // Hard-coded to the RC commit the validator was authored against.
  // Bumped together with crates/splatforge-khr-conformance/Cargo.toml.
  return {
    repo: 'KhronosGroup/glTF',
    path: 'extensions/2.0/Khronos/KHR_gaussian_splatting',
    commitSha: '63770cc70a3709cf101a42cece0bdf602b37e2e7',
    commitDate: '2026-04-15',
    commitMessage: 'Editorial review (#2567)',
    url: 'https://github.com/KhronosGroup/glTF/tree/63770cc70a3709cf101a42cece0bdf602b37e2e7/extensions/2.0/Khronos/KHR_gaussian_splatting',
    status: 'Release Candidate',
  };
}

function clauseDescriptions() {
  // Mirrors Clause::description in the Rust crate. Kept here so the SSR
  // page never has to spawn cargo for a build that lacks Rust toolchains.
  return {
    EXT_USED: 'Root extensionsUsed array MUST list "KHR_gaussian_splatting".',
    ASSET_VERSION: 'asset.version MUST be "2.0" per glTF 2.0.',
    PRIM_EXT: "At least one primitive's extensions block MUST declare KHR_gaussian_splatting.",
    PRIM_MODE_POINTS: 'Primitive carrying KHR_gaussian_splatting MUST set mode to POINTS (0).',
    EXT_KERNEL: 'The extension object MUST declare a string `kernel`.',
    EXT_COLOR_SPACE: 'The extension object MUST declare a string `colorSpace`.',
    EXT_PROJECTION: 'If `projection` is present it MUST be a string (default "perspective").',
    EXT_SORTING: 'If `sortingMethod` is present it MUST be a string (default "cameraDistance").',
    ATTR_POSITION: 'Primitive attributes MUST declare a POSITION accessor.',
    ATTR_ROTATION: 'Attributes MUST declare KHR_gaussian_splatting:ROTATION.',
    ATTR_SCALE: 'Attributes MUST declare KHR_gaussian_splatting:SCALE.',
    ATTR_OPACITY: 'Attributes MUST declare KHR_gaussian_splatting:OPACITY.',
    ATTR_SH_DC: 'Attributes MUST declare KHR_gaussian_splatting:SH_DEGREE_0_COEF_0.',
    ACC_POSITION: 'POSITION accessor MUST be VEC3 (FLOAT or normalized integer).',
    ACC_ROTATION: 'ROTATION accessor MUST be VEC4 (FLOAT or normalized signed byte/short).',
    ACC_SCALE: 'SCALE accessor MUST be VEC3 (FLOAT or unsigned-integer normalized variants).',
    ACC_OPACITY: 'OPACITY accessor MUST be SCALAR (FLOAT or normalized UByte/UShort).',
    ACC_SH_COEF: 'Every SH_DEGREE_l_COEF_n accessor MUST be VEC3 FLOAT.',
    ACC_POSITION_MINMAX: 'POSITION accessor MUST provide both min and max arrays.',
    SH_DEGREES_FULL: 'SH degrees MUST be fully defined; using degree l requires degrees 0..l-1.',
    ACC_COUNTS_AGREE: 'All per-splat accessors MUST share the same count.',
    BUFFERVIEW_BOUNDS: "Every accessor's bufferView byte footprint MUST fit inside its parent buffer.",
    ATTRS_KNOWN_ONLY: 'All KHR_gaussian_splatting:* attribute keys MUST be defined by the spec.',
  };
}

function runBuild() {
  if (!haveCargo()) {
    if (existsSync(OUT_PATH)) {
      console.error('[khr-conformance] cargo not found; reusing committed', OUT_PATH);
      return;
    }
    throw new Error('cargo not found and no committed khr-conformance.json to fall back to');
  }

  console.error('[khr-conformance] building validator + fixture binaries');
  execFileSync('cargo', ['build', '-p', 'splatforge-khr-conformance', '--quiet'], {
    cwd: REPO_ROOT,
    stdio: 'inherit',
  });

  // Regenerate fixtures into a tempdir so we never mutate the committed corpus.
  const tmp = mkdtempSync(join(tmpdir(), 'khr-fixtures-'));
  try {
    const fixturesBin = resolve(REPO_ROOT, 'target', 'debug', 'splatforge-khr-fixtures');
    execFileSync(fixturesBin, [tmp], { stdio: 'inherit' });

    const validateBin = resolve(REPO_ROOT, 'target', 'debug', 'splatforge-khr-validate');
    const fixtures = readdirSync(tmp)
      .filter((f) => f.endsWith('.glb') || f.endsWith('.gltf'))
      .sort();

    const reports = [];
    let sampleClauses = [];
    for (const f of fixtures) {
      const p = join(tmp, f);
      let out;
      try {
        out = execFileSync(validateBin, [p, '--json'], { encoding: 'utf8' });
      } catch (e) {
        // Validator exits 1 on failing clauses — that's expected for negative
        // fixtures. The JSON is still on stdout in that case.
        if (e.stdout) {
          out = e.stdout.toString('utf8');
        } else {
          throw e;
        }
      }
      const report = JSON.parse(out);
      report.fixture = f;
      // Rewrite source to be repo-relative so the public report doesn't
      // leak the tempdir path.
      report.source = `crates/splatforge-khr-conformance/fixtures/${f}`;
      report.expected_pass = f.startsWith('01_') || f.startsWith('02_') ||
        f.startsWith('03_') || f.startsWith('04_') || f.startsWith('05_');
      if (sampleClauses.length === 0) {
        sampleClauses = report.clauses.map((c) => c.id);
      }
      reports.push(report);
    }

    const descs = clauseDescriptions();
    const payload = {
      generatedAt: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      spec: specMetadata(),
      crate: { name: 'splatforge-khr-conformance', version: '0.2.0' },
      clauses: sampleClauses.map((id) => ({ id, description: descs[id] || '' })),
      fixtures: reports,
    };

    writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2) + '\n');
    console.error(`[khr-conformance] wrote ${OUT_PATH}`);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

runBuild();
