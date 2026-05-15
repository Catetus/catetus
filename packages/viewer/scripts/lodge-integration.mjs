#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
/**
 * Lodge Phase-A.2 integration sanity script — node-side.
 *
 * Walks a `.lodge` directory built by `splatforge lodge build`:
 *   - parses the manifest JSON,
 *   - fetches + decodes every per-(level, chunk) PLY through the same
 *     `decodePlyToSoa` the browser path uses,
 *   - reports per-level splat counts, PSNR-equivalent count parity, and
 *     wall-clock decode latency.
 *
 * Usage:  node packages/viewer/scripts/lodge-integration.mjs <lodge-dir>
 *
 * Reports a non-zero exit code on any mismatch — used as the
 * artifact-verification gate per `feedback_artifact_pipelines_need_
 * artifact_verification.md`.
 */
import { readFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { performance } from 'node:perf_hooks';

import { parseLodgeManifest, selectLodLevel } from '../dist/lodge/manifest.js';
import { decodePlyToSoa } from '../dist/lodge/ply.js';

async function main() {
  const dir = resolve(process.argv[2] ?? '');
  if (!dir) {
    console.error('usage: lodge-integration.mjs <lodge-dir>');
    process.exit(2);
  }
  const manifestPath = join(dir, 'manifest.json');
  const json = await readFile(manifestPath, 'utf-8');
  const m = parseLodgeManifest(json);
  console.error(
    `[lodge-it] manifest v${m.version} from ${m.source}: original=${m.originalSplatCount} levels=${m.levels.length}`,
  );

  let grandSplats = 0;
  let grandBytes = 0;
  let grandDecodeMs = 0;

  for (const level of m.levels) {
    const t0 = performance.now();
    let levelSplats = 0;
    let levelBytes = 0;
    for (const chunk of level.chunks) {
      const p = join(dir, chunk.path);
      const ply = await readFile(p);
      const decoded = decodePlyToSoa(new Uint8Array(ply.buffer, ply.byteOffset, ply.byteLength));
      if (decoded.splatCount !== chunk.splatCount) {
        console.error(
          `[lodge-it] MISMATCH ${chunk.path}: manifest=${chunk.splatCount} decoded=${decoded.splatCount}`,
        );
        process.exit(1);
      }
      levelSplats += decoded.splatCount;
      levelBytes += decoded.bytes.byteLength;
    }
    const elapsed = performance.now() - t0;
    grandSplats += levelSplats;
    grandBytes += levelBytes;
    grandDecodeMs += elapsed;
    console.error(
      `[lodge-it] level ${level.level}: ${levelSplats} splats / ${level.chunks.length} chunks / ${(levelBytes / 1_048_576).toFixed(1)} MiB SoA / ${elapsed.toFixed(0)} ms decode`,
    );
    if (levelSplats !== level.splatCount) {
      console.error(`[lodge-it] level ${level.level} splat-count mismatch`);
      process.exit(1);
    }
  }

  // Sanity: a camera at the scene centroid must pick level 0; a camera
  // outside the scene radius * 1.5 must pick the coarsest level.
  const c = [
    (m.bbox[0][0] + m.bbox[1][0]) * 0.5,
    (m.bbox[0][1] + m.bbox[1][1]) * 0.5,
    (m.bbox[0][2] + m.bbox[1][2]) * 0.5,
  ];
  const near = selectLodLevel(m, c);
  const far = selectLodLevel(m, [c[0] + 1000, c[1], c[2]]);
  console.error(`[lodge-it] LOD selector: near=L${near} far=L${far}`);
  if (near !== 0) {
    console.error('[lodge-it] expected near=L0');
    process.exit(1);
  }
  if (far !== m.levels.length - 1) {
    console.error(`[lodge-it] expected far=L${m.levels.length - 1}`);
    process.exit(1);
  }

  console.error(
    `[lodge-it] OK: ${grandSplats} splats across ${m.levels.length} levels in ${grandDecodeMs.toFixed(0)} ms`,
  );
}

main().catch((err) => {
  console.error(`[lodge-it] FAILED: ${err.stack ?? err.message ?? err}`);
  process.exit(1);
});
