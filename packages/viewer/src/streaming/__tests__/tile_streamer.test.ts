import { describe, expect, it } from 'vitest';
import { TileStreamer } from '../tile_streamer.js';
import type { TileNode } from '../tileset_loader.js';

/** Build a minimal valid GLB blob with an empty BIN chunk + 2-byte JSON. */
function makeGlb(jsonText: string): Uint8Array {
  const jsonBytes = new TextEncoder().encode(jsonText);
  // Pad JSON to a multiple of 4.
  const jsonPad = (4 - (jsonBytes.length & 3)) & 3;
  const jsonLen = jsonBytes.length + jsonPad;
  const total = 12 + 8 + jsonLen + 8 + 0; // header + json chunk + empty BIN
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  // header
  dv.setUint32(0, 0x46546c67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  // JSON chunk
  dv.setUint32(12, jsonLen, true);
  dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  new Uint8Array(buf, 20, jsonBytes.length).set(jsonBytes);
  // Space-pad
  for (let i = 0; i < jsonPad; i++) {
    new DataView(buf).setUint8(20 + jsonBytes.length + i, 0x20);
  }
  // BIN chunk (empty)
  dv.setUint32(20 + jsonLen, 0, true);
  dv.setUint32(24 + jsonLen, 0x004e4942, true); // 'BIN\0'
  return new Uint8Array(buf);
}

function makeTile(id: string, url: string): TileNode {
  return {
    id,
    aabb: { min: [0, 0, 0], max: [1, 1, 1] },
    geometricError: 1,
    refine: 'REPLACE',
    contentUrl: url,
    contentUri: url,
    children: [],
    depth: 0,
  };
}

function mockFetch(map: Record<string, Uint8Array>): typeof fetch {
  return (async (url: string | URL | Request) => {
    const k = url.toString();
    const bytes = map[k];
    if (!bytes) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) } as unknown as Response;
    return { ok: true, status: 200, arrayBuffer: async () => bytes.slice().buffer } as unknown as Response;
  }) as unknown as typeof fetch;
}

describe('TileStreamer', () => {
  it('fetches and caches a tile', async () => {
    const glb = makeGlb('{"asset":{"version":"2.0"}}');
    const streamer = new TileStreamer({ fetch: mockFetch({ 'http://x/a.glb': glb }) });
    const tile = makeTile('a', 'http://x/a.glb');
    const p1 = streamer.fetchTile(tile);
    const p2 = streamer.fetchTile(tile);
    // Same in-flight promise.
    expect(p1).toBe(p2);
    await p1;
    expect(streamer.stateOf(tile)).toBe('loaded');
    expect(streamer.residentTileCount).toBe(1);
  });

  it('marks failed fetches', async () => {
    const streamer = new TileStreamer({ fetch: mockFetch({}) });
    const tile = makeTile('a', 'http://x/missing.glb');
    await expect(streamer.fetchTile(tile)).rejects.toThrow();
    expect(streamer.stateOf(tile)).toBe('failed');
  });

  it('evicts LRU when over budget', async () => {
    const glb = makeGlb('{"asset":{"version":"2.0"}}');
    const streamer = new TileStreamer({
      maxBytes: glb.byteLength + 4, // room for ~1
      fetch: mockFetch({ 'http://x/a.glb': glb, 'http://x/b.glb': glb, 'http://x/c.glb': glb }),
    });
    const a = makeTile('a', 'http://x/a.glb');
    const b = makeTile('b', 'http://x/b.glb');
    const c = makeTile('c', 'http://x/c.glb');
    // Fetch a, advance frame; fetch b, advance frame; fetch c, advance frame.
    // Eviction inside maybeEvict pins tiles touched on the *current* frame;
    // advancing the frame after the fetch lands lets the eviction kick in
    // on the subsequent fetch.
    await streamer.fetchTile(a);
    streamer.touch([]); // bump frame, no touch
    await streamer.fetchTile(b);
    streamer.touch([]);
    await streamer.fetchTile(c);
    // After c lands, eviction has been invoked but tiles touched on the
    // current frame (which is c's load-frame) are pinned. We advance once
    // more and then trigger a small extra touch to drain anything older
    // than the budget cap.
    const stats = streamer.stats();
    // Eviction has fired at least twice (a and b were strictly older than
    // c when c landed).
    expect(stats.evictions).toBeGreaterThanOrEqual(2);
    // Only c is resident (a + b were the LRU victims).
    expect(stats.residentTiles).toBe(1);
    expect(streamer.stateOf(c)).toBe('loaded');
  });
});
