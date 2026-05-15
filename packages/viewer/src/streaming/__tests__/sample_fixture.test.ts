/**
 * Smoke test: parse the committed `geospatial-sample` tileset fixture and
 * round-trip each tile's GLB through the streaming loader. This validates
 * (a) `parseTileset` accepts the producer's exact byte format, (b)
 * `decodeGlb` extracts JSON+BIN cleanly, and (c) the existing
 * `parseManifest` reads the resulting per-tile glTF JSON.
 *
 * Runs purely on the filesystem — no GPU, no fetch — so it executes in CI
 * (Node) without WebGPU.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseTileset } from '../tileset_loader.js';
import { decodeGlb, manifestFromGlb } from '../glb.js';

const FIXTURE_DIR = resolve(
  __dirname,
  '../../../../../crates/splatforge-optimize/tests/fixtures/geospatial-sample',
);

describe('geospatial-sample fixture round-trip', () => {
  it('parses tileset.json into a 4-tile chain', () => {
    const json = readFileSync(resolve(FIXTURE_DIR, 'tileset.json'), 'utf-8');
    const t = parseTileset(json, `file://${FIXTURE_DIR}/tileset.json`);
    expect(t.tiles).toHaveLength(4);
    expect(t.extensionsRequired).toContain('KHR_gaussian_splatting');
    // Root is the coarsest (lod3.glb), leaf is the finest (lod0.glb).
    expect(t.root.contentUri).toBe('lod3.glb');
    let leaf = t.root;
    while (leaf.children.length) leaf = leaf.children[0]!;
    expect(leaf.contentUri).toBe('lod0.glb');
    expect(leaf.geometricError).toBe(0);
  });

  it('each tile GLB decodes + manifestFromGlb yields a renderable chunk', () => {
    const json = readFileSync(resolve(FIXTURE_DIR, 'tileset.json'), 'utf-8');
    const t = parseTileset(json, `file://${FIXTURE_DIR}/tileset.json`);
    const expectedCounts: Record<string, number> = {
      'lod0.glb': 450,
      'lod1.glb': 225,
      'lod2.glb': 113,
      'lod3.glb': 57,
    };
    for (const tile of t.tiles) {
      const glbBytes = readFileSync(resolve(FIXTURE_DIR, tile.contentUri));
      const out = decodeGlb(new Uint8Array(glbBytes));
      const { manifest, bin } = manifestFromGlb(out);
      expect(manifest.splatCount).toBe(expectedCounts[tile.contentUri]);
      expect(manifest.chunks.length).toBe(1);
      expect(manifest.chunks[0]!.attributeLayout).toBeDefined();
      expect(bin.byteLength).toBeGreaterThan(0);
    }
  });
});
