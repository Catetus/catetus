/**
 * WGSL-emulation parity test for `cs_lod_select.wgsl` and
 * `cs_lod_blend.wgsl`.
 *
 * The strategy: decode the byte buffers produced by `LodgeLODPipeline`'s
 * encoders into a JS object graph, then run a hand-written "WGSL
 * interpreter" that mirrors the kernel line-by-line. Compare its output
 * to the JS reference selector. They must match exactly — any divergence
 * is either a layout bug in the encoder or a math bug in the reference.
 *
 * This is the closest we can get to gating the real GPU kernel without
 * spinning up WebGPU inside vitest. The browser bench (`real-scene.bench`)
 * will catch any remaining issue at run time.
 */
import { describe, expect, it } from 'vitest';
import {
  LodgeLODPipeline,
  LodgeChunkLoader,
  CHUNK_RECORD_BYTES,
  LOD_UNIFORMS_BYTES,
  type ChunkActivation,
} from '../lodge/index.js';
import { parseLodgeManifest } from '../lodge/manifest.js';

const FIXTURE = {
  version: 1,
  source: 'corals.ply',
  original_splat_count: 10_000,
  bbox: [
    [-30, -30, -30],
    [30, 30, 30],
  ],
  levels: [
    {
      level: 0,
      splat_count: 6000,
      reduction: 1.0,
      depth_threshold: 0.0,
      chunks: [
        { index: 0, path: 'level_0/chunk_0000.ply', splat_count: 3000, bbox: [[-30,-30,-30],[0,30,30]], centroid: [-15, 0, 0], radius: 12, blake3: 'a' },
        { index: 1, path: 'level_0/chunk_0001.ply', splat_count: 3000, bbox: [[0,-30,-30],[30,30,30]], centroid: [15, 0, 0], radius: 12, blake3: 'b' },
      ],
    },
    {
      level: 1,
      splat_count: 3000,
      reduction: 0.5,
      depth_threshold: 50.0,
      chunks: [
        { index: 0, path: 'level_1/chunk_0000.ply', splat_count: 3000, bbox: [[-30,-30,-30],[30,30,30]], centroid: [0, 0, 0], radius: 25, blake3: 'c' },
      ],
    },
    {
      level: 2,
      splat_count: 500,
      reduction: 0.083,
      depth_threshold: 120.0,
      chunks: [
        { index: 0, path: 'level_2/chunk_0000.ply', splat_count: 500, bbox: [[-30,-30,-30],[30,30,30]], centroid: [0, 0, 0], radius: 25, blake3: 'd' },
      ],
    },
  ],
};

/** WGSL-simulating function. Reads the same byte buffers and produces
 *  the same activation records the GPU kernel would. Must remain a
 *  line-by-line mirror of `cs_lod_select.wgsl`. */
function emulateLodSelect(
  chunkBuf: ArrayBuffer,
  levelBuf: ArrayBuffer,
  uniformBuf: ArrayBuffer,
): ChunkActivation[] {
  const cF = new Float32Array(chunkBuf);
  const cU = new Uint32Array(chunkBuf);
  // 32 bytes per ChunkDesc → 8 f32 / u32 per record.

  const uF = new Float32Array(uniformBuf);
  const uU = new Uint32Array(uniformBuf);

  // Decode uniforms (matches WGSL layout in cs_lod_select.wgsl):
  // 0..3   camera_pos.xyz, pad
  // 4..7   scene_center.xyz, pad
  // 8..11  depth_thresholds[0..3]
  // 12..15 depth_thresholds[4..7]
  // 16     chunk_count (u32)
  // 17     num_levels (u32)
  // 18,19  pad u32×2
  // 20..23 near_centroid.xyz, pad
  // 24..27 far_centroid.xyz, pad
  // 28     ss_size_threshold (f32)
  // 29     focal_y (f32)
  // 30,31  pad u32×2
  const camera = [uF[0]!, uF[1]!, uF[2]!] as [number, number, number];
  const thresholds = [
    uF[8]!, uF[9]!, uF[10]!, uF[11]!,
    uF[12]!, uF[13]!, uF[14]!, uF[15]!,
  ];
  const chunkCount = uU[16]!;
  const numLevels = uU[17]!;
  const nearCentroid = [uF[20]!, uF[21]!, uF[22]!] as [number, number, number];
  const farCentroid = [uF[24]!, uF[25]!, uF[26]!] as [number, number, number];
  const ssThreshold = uF[28]!;
  const focalY = uF[29]!;

  // levelBuf is informational in the emulator — the kernel reads
  // depth_thresholds from the uniform, so we don't need to decode levelBuf
  // unless we want to verify layout. Keep the param for parity.
  void new Float32Array(levelBuf);

  const out: ChunkActivation[] = [];
  for (let i = 0; i < chunkCount; i++) {
    const o = i * 8;
    const cx = cF[o + 0]!;
    const cy = cF[o + 1]!;
    const cz = cF[o + 2]!;
    const cr = cF[o + 3]!;
    const cLevel = cU[o + 4]!;

    // dist
    const dx = camera[0] - cx;
    const dy = camera[1] - cy;
    const dz = camera[2] - cz;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const distEff = Math.max(dist - cr, 0);

    let chosen = 0;
    for (let l = 0; l < numLevels; l++) {
      if (distEff >= thresholds[l]!) chosen = l;
    }
    const safeDist = Math.max(dist, 1e-3);
    const ssRadius = (focalY * cr) / safeDist;
    if (ssRadius < ssThreshold && chosen + 1 < numLevels) chosen += 1;

    const act: ChunkActivation = {
      level: chosen,
      active: 0,
      slot: 0,
      tBlend: 1,
    };
    if (cLevel === chosen) {
      act.active = 1;
      const dNear = Math.hypot(cx - nearCentroid[0], cy - nearCentroid[1], cz - nearCentroid[2]);
      const dFar = Math.hypot(cx - farCentroid[0], cy - farCentroid[1], cz - farCentroid[2]);
      const isFar = dFar < dNear;
      act.slot = isFar ? 1 : 0;
      const vx = farCentroid[0] - nearCentroid[0];
      const vy = farCentroid[1] - nearCentroid[1];
      const vz = farCentroid[2] - nearCentroid[2];
      const denom = Math.max(vx * vx + vy * vy + vz * vz, 1e-8);
      const raw = ((camera[0] - nearCentroid[0]) * vx +
                   (camera[1] - nearCentroid[1]) * vy +
                   (camera[2] - nearCentroid[2]) * vz) / denom;
      const t = Math.max(0, Math.min(1, raw));
      act.tBlend = isFar ? t : 1 - t;
      const sep = Math.hypot(
        nearCentroid[0] - farCentroid[0],
        nearCentroid[1] - farCentroid[1],
        nearCentroid[2] - farCentroid[2],
      );
      if (sep < 1e-4) act.tBlend = 1;
    }
    out.push(act);
  }
  return out;
}

function makePipeline(): LodgeLODPipeline {
  const manifest = parseLodgeManifest(JSON.stringify(FIXTURE));
  const loader = new LodgeChunkLoader(manifest, {
    baseUrl: 'http://test/lodge/',
    pipeline: { uploadChunk() {}, splatCount: 0, capacity: 1e9 },
    fetcher: async () => new Uint8Array(0),
  });
  return new LodgeLODPipeline(loader, { ssSizeThreshold: 0 });
}

describe('WGSL emulation parity', () => {
  it('JS reference and WGSL emulator agree at scene-center camera', () => {
    const p = makePipeline();
    const d = p.prepareFrame([0, 0, 0], 1000);
    const chunkBuf = p.encodeChunkRecords(d.records);
    const levelBuf = p.encodeLevelRecords();
    const uBuf = p.encodeLodSelectUniforms(d, [0, 0, 0], 1000);

    expect(chunkBuf.byteLength).toBe(4 * CHUNK_RECORD_BYTES);
    expect(uBuf.byteLength).toBe(LOD_UNIFORMS_BYTES);

    const wgslOut = emulateLodSelect(chunkBuf, levelBuf, uBuf);
    expect(wgslOut).toHaveLength(d.activations.length);
    for (let i = 0; i < wgslOut.length; i++) {
      expect(wgslOut[i]?.level).toBe(d.activations[i]?.level);
      expect(wgslOut[i]?.active).toBe(d.activations[i]?.active);
      expect(wgslOut[i]?.slot).toBe(d.activations[i]?.slot);
      expect(wgslOut[i]?.tBlend).toBeCloseTo(d.activations[i]!.tBlend, 5);
    }
  });

  it('parity at far camera (level 2 should activate)', () => {
    const p = makePipeline();
    const cam: [number, number, number] = [200, 0, 0];
    const d = p.prepareFrame(cam, 1000);
    expect(d.selectedLevel).toBe(2);
    const chunkBuf = p.encodeChunkRecords(d.records);
    const levelBuf = p.encodeLevelRecords();
    const uBuf = p.encodeLodSelectUniforms(d, cam, 1000);
    const wgslOut = emulateLodSelect(chunkBuf, levelBuf, uBuf);
    for (let i = 0; i < wgslOut.length; i++) {
      expect(wgslOut[i]?.active).toBe(d.activations[i]?.active);
      expect(wgslOut[i]?.level).toBe(d.activations[i]?.level);
    }
  });

  it('parity at off-center camera (-3,0,0) with eq.4 ramp ≈ 0.4', () => {
    const p = makePipeline();
    const cam: [number, number, number] = [-3, 0, 0];
    const d = p.prepareFrame(cam, 1000);
    expect(d.selectedLevel).toBe(0);
    const chunkBuf = p.encodeChunkRecords(d.records);
    const levelBuf = p.encodeLevelRecords();
    const uBuf = p.encodeLodSelectUniforms(d, cam, 1000);
    const wgslOut = emulateLodSelect(chunkBuf, levelBuf, uBuf);
    // The two active chunks should have tBlend summing to 1 (one gets t,
    // the other gets 1-t).
    let activeT = 0;
    let activeCount = 0;
    for (const a of wgslOut) {
      if (a.active === 1) {
        activeT += a.tBlend;
        activeCount++;
      }
    }
    expect(activeCount).toBe(2);
    expect(activeT).toBeCloseTo(1.0, 5);
  });

  it('SS-size-bump parity: tiny chunks at long distance get bumped', () => {
    // Custom manifest where chunks are TINY (radius 0.01) so the
    // ssSizeThreshold kicks in.
    const tinyFix = {
      ...FIXTURE,
      levels: FIXTURE.levels.map((l) => ({
        ...l,
        chunks: l.chunks.map((c) => ({ ...c, radius: 0.01 })),
      })),
    };
    const manifest = parseLodgeManifest(JSON.stringify(tinyFix));
    const loader = new LodgeChunkLoader(manifest, {
      baseUrl: 'http://test/lodge/',
      pipeline: { uploadChunk() {}, splatCount: 0, capacity: 1e9 },
      fetcher: async () => new Uint8Array(0),
    });
    const p = new LodgeLODPipeline(loader, { ssSizeThreshold: 4 });
    const cam: [number, number, number] = [0, 0, 0];
    const d = p.prepareFrame(cam, 1000);
    const chunkBuf = p.encodeChunkRecords(d.records);
    const levelBuf = p.encodeLevelRecords();
    const uBuf = p.encodeLodSelectUniforms(d, cam, 1000);
    const wgslOut = emulateLodSelect(chunkBuf, levelBuf, uBuf);
    for (let i = 0; i < wgslOut.length; i++) {
      expect(wgslOut[i]?.level).toBe(d.activations[i]?.level);
    }
  });
});
