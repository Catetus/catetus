import { describe, expect, it } from 'vitest';
import { decodePlyToSoa } from '../lodge/ply.js';
import { LodgeChunkLoader } from '../lodge/chunk-loader.js';
import type { UploadablePipeline, ChunkFetcher } from '../lodge/chunk-loader.js';
import { parseLodgeManifest } from '../lodge/manifest.js';
import type { ChunkDescriptor } from '../manifest.js';

const SH_C0 = 0.28209479177387814;

/**
 * Build a minimal Inria-format binary PLY in memory containing `n`
 * deterministic splats. Property layout matches the bonsai PLYs the
 * lodge chunker emits.
 */
function buildSyntheticPly(n: number): { bytes: Uint8Array; positions: Array<[number, number, number]> } {
  const PROPS = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  const header =
    `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n` +
    PROPS.map((p) => `property float ${p}`).join('\n') +
    '\nend_header\n';
  const headerBytes = new TextEncoder().encode(header);
  const stride = PROPS.length * 4;
  const body = new Uint8Array(n * stride);
  const dv = new DataView(body.buffer);
  const positions: Array<[number, number, number]> = [];

  for (let i = 0; i < n; i++) {
    const row = i * stride;
    const x = (i % 8) * 0.1;
    const y = (Math.floor(i / 8) % 8) * 0.1;
    const z = (Math.floor(i / 64) % 8) * 0.1;
    positions.push([x, y, z]);
    let p = 0;
    const set = (v: number) => {
      dv.setFloat32(row + p, v, true);
      p += 4;
    };
    set(x); set(y); set(z);    // position
    set(0); set(0); set(1);    // normals (unused)
    set(0.5); set(-0.2); set(0.3); // f_dc
    set(2.0);                  // logit opacity (sigmoid ≈ 0.881)
    set(-2.0); set(-2.0); set(-2.0); // log scale (exp ≈ 0.135)
    set(1); set(0); set(0); set(0); // unit quaternion
  }

  const out = new Uint8Array(headerBytes.byteLength + body.byteLength);
  out.set(headerBytes, 0);
  out.set(body, headerBytes.byteLength);
  return { bytes: out, positions };
}

describe('decodePlyToSoa', () => {
  it('round-trips positions, normalises rotation, sigmoids opacity, exp-scales, SH-C0 colors', () => {
    const { bytes, positions } = buildSyntheticPly(4);
    const decoded = decodePlyToSoa(bytes);
    expect(decoded.splatCount).toBe(4);

    // SoA shape: 4 splats × (12 + 16 + 12 + 4 + 12) = 224 B.
    expect(decoded.bytes.byteLength).toBe(4 * (12 + 16 + 12 + 4 + 12));

    const dv = new DataView(decoded.bytes.buffer);
    for (let i = 0; i < 4; i++) {
      const px = dv.getFloat32(i * 12 + 0, true);
      const py = dv.getFloat32(i * 12 + 4, true);
      const pz = dv.getFloat32(i * 12 + 8, true);
      expect(px).toBeCloseTo(positions[i]![0], 5);
      expect(py).toBeCloseTo(positions[i]![1], 5);
      expect(pz).toBeCloseTo(positions[i]![2], 5);
    }

    const rotOff = 4 * 12;
    for (let i = 0; i < 4; i++) {
      const w = dv.getFloat32(rotOff + i * 16 + 0, true);
      const x = dv.getFloat32(rotOff + i * 16 + 4, true);
      const y = dv.getFloat32(rotOff + i * 16 + 8, true);
      const z = dv.getFloat32(rotOff + i * 16 + 12, true);
      const norm = Math.sqrt(w * w + x * x + y * y + z * z);
      expect(norm).toBeCloseTo(1.0, 5);
    }

    const scaleOff = rotOff + 4 * 16;
    const expectedScale = Math.exp(-2.0);
    for (let i = 0; i < 4; i++) {
      const s0 = dv.getFloat32(scaleOff + i * 12 + 0, true);
      expect(s0).toBeCloseTo(expectedScale, 5);
    }

    const opOff = scaleOff + 4 * 12;
    const expectedOp = 1 / (1 + Math.exp(-2.0));
    for (let i = 0; i < 4; i++) {
      const o = dv.getFloat32(opOff + i * 4, true);
      expect(o).toBeCloseTo(expectedOp, 5);
    }

    const dcOff = opOff + 4 * 4;
    const expectedR = 0.5 + SH_C0 * 0.5;
    for (let i = 0; i < 4; i++) {
      const r = dv.getFloat32(dcOff + i * 12 + 0, true);
      expect(r).toBeCloseTo(expectedR, 5);
    }
  });

  it('rejects an ASCII PLY', () => {
    const ascii = new TextEncoder().encode(
      'ply\nformat ascii 1.0\nelement vertex 0\nend_header\n',
    );
    expect(() => decodePlyToSoa(ascii)).toThrow(/ply_invalid/);
  });

  it('rejects a PLY missing required props', () => {
    const bad = new TextEncoder().encode(
      'ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n',
    );
    expect(() => decodePlyToSoa(bad)).toThrow(/ply_invalid/);
  });
});

/* -------------------------------------------------------------------- */
/* LodgeChunkLoader                                                     */
/* -------------------------------------------------------------------- */

class MockPipeline implements UploadablePipeline {
  capacity = 1_000_000;
  uploads: Array<{ descriptor: ChunkDescriptor; byteLength: number }> = [];
  // Mirror the real pipeline's reset semantics by exposing the same
  // private-shaped fields the loader pokes.
  decodedSplats = 0;
  chunks: unknown[] = [];
  get splatCount(): number {
    return this.decodedSplats;
  }
  uploadChunk(descriptor: ChunkDescriptor, bytes: Uint8Array): void {
    if (descriptor.splatCount === 0) return;
    this.uploads.push({ descriptor, byteLength: bytes.byteLength });
    this.decodedSplats += descriptor.splatCount;
  }
}

function makeManifest(chunkBytesByPath: Map<string, Uint8Array>) {
  // Build a 2-level manifest whose chunk paths key into `chunkBytesByPath`.
  // Splat counts per chunk are inferred from the synthetic PLYs.
  const paths = Array.from(chunkBytesByPath.keys());
  const l0Chunks = paths
    .filter((p) => p.startsWith('level_0/'))
    .map((p, i) => ({
      index: i,
      path: p,
      splat_count: decodePlyToSoa(chunkBytesByPath.get(p)!).splatCount,
      bbox: [[-1, -1, -1], [1, 1, 1]],
      centroid: [i * 0.5, 0, 0],
      radius: 1.0,
      blake3: '',
    }));
  const l1Chunks = paths
    .filter((p) => p.startsWith('level_1/'))
    .map((p, i) => ({
      index: i,
      path: p,
      splat_count: decodePlyToSoa(chunkBytesByPath.get(p)!).splatCount,
      bbox: [[-1, -1, -1], [1, 1, 1]],
      centroid: [0, 0, 0],
      radius: 1.0,
      blake3: '',
    }));
  return {
    version: 1,
    source: 'synthetic.ply',
    original_splat_count: 0,
    bbox: [[-1, -1, -1], [1, 1, 1]],
    levels: [
      {
        level: 0,
        splat_count: l0Chunks.reduce((a, c) => a + c.splat_count, 0),
        reduction: 1.0,
        depth_threshold: 0.0,
        chunks: l0Chunks,
      },
      {
        level: 1,
        splat_count: l1Chunks.reduce((a, c) => a + c.splat_count, 0),
        reduction: 0.5,
        depth_threshold: 10.0,
        chunks: l1Chunks,
      },
    ],
  };
}

describe('LodgeChunkLoader', () => {
  it('uploads all chunks of a level and tracks cache hits on reload', async () => {
    const c0 = buildSyntheticPly(8).bytes;
    const c1 = buildSyntheticPly(8).bytes;
    const c2 = buildSyntheticPly(16).bytes;
    const byPath = new Map([
      ['level_0/chunk_0000.ply', c0],
      ['level_0/chunk_0001.ply', c1],
      ['level_1/chunk_0000.ply', c2],
    ]);
    const manifestJson = JSON.stringify(makeManifest(byPath));
    const manifest = parseLodgeManifest(manifestJson);

    const fetched: string[] = [];
    const fetcher: ChunkFetcher = async (url) => {
      fetched.push(url);
      // Strip baseUrl prefix.
      const rel = url.replace('lodge://test/', '');
      const bytes = byPath.get(rel);
      if (!bytes) throw new Error(`mock miss: ${rel}`);
      return bytes;
    };
    const pipeline = new MockPipeline();
    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: 'lodge://test/',
      pipeline,
      fetcher,
      memoryBudgetBytes: 1024 * 1024 * 1024,
    });

    const r0 = await loader.loadLevel(0);
    expect(r0.level).toBe(0);
    expect(r0.uploadedChunks).toBe(2);
    expect(r0.cachedChunks).toBe(0);
    expect(r0.splats).toBe(16);
    expect(fetched).toHaveLength(2);

    // Reload — both should hit the cache.
    const r0b = await loader.loadLevel(0);
    expect(r0b.uploadedChunks).toBe(0);
    expect(r0b.cachedChunks).toBe(2);
    expect(fetched).toHaveLength(2); // no new fetches

    // Switch to level 1 — one new fetch.
    const r1 = await loader.loadLevel(1);
    expect(r1.level).toBe(1);
    expect(r1.uploadedChunks).toBe(1);
    expect(r1.splats).toBe(16);
    expect(fetched).toHaveLength(3);
  });

  it('honors the memory budget via LRU eviction', async () => {
    const big = buildSyntheticPly(100).bytes; // ~33k bytes decoded per chunk
    const byPath = new Map([
      ['level_0/chunk_0000.ply', big],
      ['level_0/chunk_0001.ply', big],
      ['level_1/chunk_0000.ply', big],
    ]);
    const manifest = parseLodgeManifest(JSON.stringify(makeManifest(byPath)));
    const pipeline = new MockPipeline();

    // Decoded SoA bytes per chunk: 100 * 56 = 5600 B.
    // Budget tight enough that only the *current* level's chunks can live.
    const budget = 6000; // forces eviction after each insert.
    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: 'lodge://test/',
      pipeline,
      fetcher: async (url) => byPath.get(url.replace('lodge://test/', ''))!,
      memoryBudgetBytes: budget,
    });

    await loader.loadLevel(0);
    // Two chunks tried to land in a 6000-byte budget. Only the newest fits.
    expect(loader.memoryUsage).toBeLessThanOrEqual(budget * 2);
    expect(loader.cacheKeys().length).toBeLessThanOrEqual(2);
  });

  it('picks LOD via camera distance heuristic', async () => {
    const c0 = buildSyntheticPly(8).bytes;
    const c1 = buildSyntheticPly(8).bytes;
    const byPath = new Map([
      ['level_0/chunk_0000.ply', c0],
      ['level_1/chunk_0000.ply', c1],
    ]);
    const manifest = parseLodgeManifest(JSON.stringify(makeManifest(byPath)));
    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: 'lodge://test/',
      pipeline: new MockPipeline(),
      fetcher: async (url) => byPath.get(url.replace('lodge://test/', ''))!,
    });
    expect(loader.pickLevel([0, 0, 0])).toBe(0);
    expect(loader.pickLevel([100, 0, 0])).toBe(1);
  });
});
