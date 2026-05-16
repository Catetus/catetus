import { describe, expect, it } from 'vitest';
import {
  LodgeLODPipeline,
  LodgeChunkLoader,
  CHUNK_RECORD_BYTES,
  LEVEL_RECORD_BYTES,
  ACTIVATION_BYTES,
  LOD_UNIFORMS_BYTES,
  type FrameDecision,
} from '../lodge/index.js';
import { parseLodgeManifest } from '../lodge/manifest.js';
import type { ChunkDescriptor } from '../manifest.js';

const FIXTURE = {
  version: 1,
  source: 'test.ply',
  original_splat_count: 4000,
  bbox: [
    [-10, -10, -10],
    [10, 10, 10],
  ],
  levels: [
    {
      level: 0,
      splat_count: 2000,
      reduction: 1.0,
      depth_threshold: 0.0,
      chunks: [
        {
          index: 0,
          path: 'level_0/chunk_0000.ply',
          splat_count: 1000,
          bbox: [
            [-10, -10, -10],
            [0, 10, 10],
          ],
          centroid: [-5, 0, 0],
          radius: 5,
          blake3: 'aa',
        },
        {
          index: 1,
          path: 'level_0/chunk_0001.ply',
          splat_count: 1000,
          bbox: [
            [0, -10, -10],
            [10, 10, 10],
          ],
          centroid: [5, 0, 0],
          radius: 5,
          blake3: 'bb',
        },
      ],
    },
    {
      level: 1,
      splat_count: 1000,
      reduction: 0.5,
      depth_threshold: 20.0,
      chunks: [
        {
          index: 0,
          path: 'level_1/chunk_0000.ply',
          splat_count: 1000,
          bbox: [
            [-10, -10, -10],
            [10, 10, 10],
          ],
          centroid: [0, 0, 0],
          radius: 10,
          blake3: 'cc',
        },
      ],
    },
    {
      level: 2,
      splat_count: 200,
      reduction: 0.1,
      depth_threshold: 50.0,
      chunks: [
        {
          index: 0,
          path: 'level_2/chunk_0000.ply',
          splat_count: 200,
          bbox: [
            [-10, -10, -10],
            [10, 10, 10],
          ],
          centroid: [0, 0, 0],
          radius: 10,
          blake3: 'dd',
        },
      ],
    },
  ],
};

class MockPipeline {
  uploadedChunks: Array<{ desc: ChunkDescriptor; bytes: Uint8Array }> = [];
  decodedSplats = 0;
  chunks: ChunkDescriptor[] = [];
  capacity = 100_000_000;
  uploadChunk(desc: ChunkDescriptor, bytes: Uint8Array): void {
    this.uploadedChunks.push({ desc, bytes: bytes.slice(0) });
    this.decodedSplats += desc.splatCount;
    this.chunks.push(desc);
  }
  get splatCount(): number {
    return this.decodedSplats;
  }
  reset(): void {
    this.uploadedChunks.length = 0;
    this.decodedSplats = 0;
    this.chunks.length = 0;
  }
}

function makePipeline(): LodgeLODPipeline {
  const manifest = parseLodgeManifest(JSON.stringify(FIXTURE));
  const loader = new LodgeChunkLoader(manifest, {
    baseUrl: 'http://test/lodge/',
    pipeline: new MockPipeline(),
    fetcher: async () => new Uint8Array(0), // never called in these tests
  });
  return new LodgeLODPipeline(loader, { ssSizeThreshold: 0 });
}

describe('LodgeLODPipeline.prepareFrame', () => {
  it('picks level 0 at scene center, level 2 at scene-far', () => {
    const p = makePipeline();

    const near = p.prepareFrame([0, 0, 0], 1000);
    expect(near.selectedLevel).toBe(0);

    const far = p.prepareFrame([100, 0, 0], 1000);
    expect(far.selectedLevel).toBe(2);
  });

  it('emits one record per (level, chunk) in manifest order', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    // 2 + 1 + 1 = 4 records flattened.
    expect(d.records).toHaveLength(4);
    expect(d.activations).toHaveLength(4);
    expect(d.records[0]?.level).toBe(0);
    expect(d.records[1]?.level).toBe(0);
    expect(d.records[2]?.level).toBe(1);
    expect(d.records[3]?.level).toBe(2);
  });

  it('marks exactly the chunks on the selected level as active', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    expect(d.selectedLevel).toBe(0);
    expect(d.activations[0]?.active).toBe(1);
    expect(d.activations[1]?.active).toBe(1);
    expect(d.activations[2]?.active).toBe(0);
    expect(d.activations[3]?.active).toBe(0);
  });

  it('records active splat count = sum of selected-level chunk counts', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    expect(d.activeSplats).toBe(2000);
    expect(d.selectedLevelSplats).toBe(2000);
  });

  it('picks correct near / far chunks at off-center cameras', () => {
    const p = makePipeline();
    const d = p.prepareFrame([-3, 0, 0], 1000);
    // Level 0 selected; chunk 0 (centroid -5) is the near, chunk 1 (centroid 5) is the far.
    expect(d.selectedLevel).toBe(0);
    expect(d.nearChunkIndex).toBe(0);
    expect(d.farChunkIndex).toBe(1);
  });
});

describe('LodgeLODPipeline byte layouts', () => {
  it('emits CHUNK_RECORD_BYTES = 32, LEVEL_RECORD_BYTES = 16, ACTIVATION_BYTES = 16', () => {
    expect(CHUNK_RECORD_BYTES).toBe(32);
    expect(LEVEL_RECORD_BYTES).toBe(16);
    expect(ACTIVATION_BYTES).toBe(16);
  });

  it('encodeChunkRecords matches WGSL ChunkDesc layout', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    const buf = p.encodeChunkRecords(d.records);
    expect(buf.byteLength).toBe(4 * CHUNK_RECORD_BYTES);

    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    // record 0: chunk0 of level 0, centroid [-5, 0, 0], radius 5.
    expect(f[0]).toBe(-5);
    expect(f[1]).toBe(0);
    expect(f[2]).toBe(0);
    expect(f[3]).toBe(5);
    expect(u[4]).toBe(0); // level
    expect(u[5]).toBe(0); // chunkIndex
    expect(u[6]).toBe(1000); // splatCount
  });

  it('encodeLevelRecords fills 8 slots, padding with +inf', () => {
    const p = makePipeline();
    const buf = p.encodeLevelRecords();
    expect(buf.byteLength).toBe(8 * LEVEL_RECORD_BYTES);

    const f = new Float32Array(buf);
    expect(f[0]).toBe(0); // level 0 depth_threshold
    expect(f[4]).toBe(20); // level 1 depth_threshold
    expect(f[8]).toBe(50); // level 2 depth_threshold
    // Unused slots are +inf.
    expect(f[12]).toBe(Number.POSITIVE_INFINITY);
    expect(f[28]).toBe(Number.POSITIVE_INFINITY);
  });

  it('encodeLodSelectUniforms packs to LOD_UNIFORMS_BYTES', () => {
    const p = makePipeline();
    const d = p.prepareFrame([-3, 0, 0], 1000);
    const buf = p.encodeLodSelectUniforms(d, [-3, 0, 0], 1000);
    expect(buf.byteLength).toBe(LOD_UNIFORMS_BYTES);

    const f = new Float32Array(buf);
    expect(f[0]).toBe(-3); // camera.x
    expect(f[1]).toBe(0); // camera.y
    expect(f[2]).toBe(0); // camera.z
    expect(f[4]).toBe(0); // scene_center.x
    expect(f[5]).toBe(0); // scene_center.y
    expect(f[8]).toBe(0); // depth_threshold[0]
    expect(f[9]).toBe(20); // depth_threshold[1]
    expect(f[10]).toBe(50); // depth_threshold[2]
  });

  it('decodeActivations round-trips the activation buffer', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    // Synthesize a GPU output: pack the JS-side decisions and decode.
    const N = d.activations.length;
    const buf = new ArrayBuffer(N * ACTIVATION_BYTES);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    for (let i = 0; i < N; i++) {
      const a = d.activations[i]!;
      u[i * 4] = a.level;
      u[i * 4 + 1] = a.active;
      u[i * 4 + 2] = a.slot;
      f[i * 4 + 3] = a.tBlend;
    }
    const back = p.decodeActivations(buf);
    expect(back).toHaveLength(N);
    for (let i = 0; i < N; i++) {
      expect(back[i]?.level).toBe(d.activations[i]?.level);
      expect(back[i]?.active).toBe(d.activations[i]?.active);
      expect(back[i]?.slot).toBe(d.activations[i]?.slot);
      expect(back[i]?.tBlend).toBeCloseTo(d.activations[i]!.tBlend, 6);
    }
  });
});

describe('LodgeLODPipeline.lastFrameDecision', () => {
  it('starts as null', () => {
    const p = makePipeline();
    expect(p.lastFrameDecision).toBeNull();
  });

  it('stores the latest call', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    expect(p.lastFrameDecision).toBe(d);
  });
});

describe('streamFrame integration with a mock fetcher', () => {
  it('fetches both near and far chunks on first call, cache hits on second', async () => {
    // Build synthetic SoA PLYs for the two chunks at level 0.
    // We re-use the LodgeChunkLoader's fetcher contract: bytes are Inria
    // binary PLY. The integration test in lodge-chunk-loader.test.ts
    // already covers the decode path; here we want fetch-count
    // semantics. Stub the fetcher to count calls + return a hand-built
    // minimal PLY that decodes to splat_count=1000 / 1.
    const manifest = parseLodgeManifest(JSON.stringify(FIXTURE));
    const mock = new MockPipeline();
    let fetchCount = 0;
    // Build a tiny valid PLY for each chunk. Format: Inria binary,
    // x,y,z,scale_0..2,rot_0..3,opacity,f_dc_0..2 = 14 floats / splat.
    const buildPly = (n: number): Uint8Array => {
      const header =
        `ply\n` +
        `format binary_little_endian 1.0\n` +
        `element vertex ${n}\n` +
        `property float x\nproperty float y\nproperty float z\n` +
        `property float scale_0\nproperty float scale_1\nproperty float scale_2\n` +
        `property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n` +
        `property float opacity\n` +
        `property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n` +
        `end_header\n`;
      const headerBytes = new TextEncoder().encode(header);
      const body = new Float32Array(n * 14);
      for (let i = 0; i < n; i++) {
        const o = i * 14;
        // Fill with valid quaternion + log-scale + opacity-logit values.
        body[o + 6] = 1; // rot_0 = w
        // scale_0..2 = 0 (log) → exp(0) = 1; opacity logit 0 → sigmoid = 0.5.
      }
      const out = new Uint8Array(headerBytes.byteLength + body.byteLength);
      out.set(headerBytes);
      out.set(new Uint8Array(body.buffer), headerBytes.byteLength);
      return out;
    };
    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: 'http://test/lodge/',
      pipeline: mock,
      fetcher: async (url) => {
        fetchCount++;
        const m = url.match(/chunk_(\d+)\.ply/);
        const i = m ? Number(m[1]) : 0;
        // Look up the chunk by path to learn its splat_count.
        for (const lvl of manifest.levels) {
          for (const c of lvl.chunks) {
            if (url.endsWith(c.path)) return buildPly(c.splatCount);
          }
        }
        return buildPly(i);
      },
    });
    const p = new LodgeLODPipeline(loader, { ssSizeThreshold: 0 });

    const r1 = await p.streamFrame([0, 0, 0], 1000);
    expect(r1.selectedLevel).toBe(0);
    expect(r1.uploadedChunks).toBe(2);
    expect(r1.cachedChunks).toBe(0);

    const beforeFetch = fetchCount;
    const r2 = await p.streamFrame([0, 0, 0], 1000);
    expect(r2.uploadedChunks).toBe(0);
    expect(r2.cachedChunks).toBe(2);
    // The level reload still touches the cache to re-upload to the GPU,
    // but it should NOT issue new HTTP fetches for the cached chunks.
    expect(fetchCount).toBe(beforeFetch);
  });
});

describe('frame decision exposes elapsedMs', () => {
  it('elapsedMs is non-negative', () => {
    const p = makePipeline();
    const d: FrameDecision = p.prepareFrame([0, 0, 0], 1000);
    expect(d.elapsedMs).toBeGreaterThanOrEqual(0);
  });
});
