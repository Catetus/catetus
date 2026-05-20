/**
 * Off-main-thread depth sort for the splat renderer.
 *
 * Architecture (mirrors antimatter15/splat):
 *   - On 'init', main thread transfers the splat positions (Float32Array) once.
 *   - On 'sort', main thread sends the current view matrix + a monotonically
 *     increasing requestId. We compute view-space z per splat, run a 16-bit
 *     counting sort (same algorithm as renderer.ts), and post back the sorted
 *     ORIGINAL splat indices as a Float32Array (that's what the renderer's
 *     vertex attribute consumes — `gl.FLOAT`, divisor 1).
 *   - Posting back uses Transferable so it's a pointer swap, not a copy.
 *   - Main thread drops stale results (older than its current requestId).
 *
 * The renderer keeps the previously uploaded sort while we work, so the UI
 * thread never blocks on sort. This eliminates the ~16 ms hitch at scene
 * orbit at 1.24 M splats.
 *
 * Scratch buffers are kept alive between sort calls; we re-allocate the
 * out-going Float32Array each call because it's transferred (zero-copy) to
 * the main thread.
 */

interface InitMsg {
  type: 'init';
  positions: Float32Array;
}
interface SortMsg {
  type: 'sort';
  requestId: number;
  view: Float32Array;            // length 16, column-major
}
type InMsg = InitMsg | SortMsg;

interface SortedMsg {
  type: 'sorted';
  requestId: number;
  indices: Float32Array;         // sorted ORIGINAL splat indices
}

let positions: Float32Array | null = null;
let count = 0;
// Persistent scratch buffers (re-used across sort calls).
let depths: Float32Array | null = null;
let bucketOf: Uint32Array | null = null;
let outIdx: Uint32Array | null = null;

function ensureScratch(N: number): void {
  if (!depths || depths.length !== N) {
    depths = new Float32Array(N);
    bucketOf = new Uint32Array(N);
    outIdx = new Uint32Array(N);
  }
}

function sortAndPost(requestId: number, view: Float32Array): void {
  if (!positions || count === 0) return;
  ensureScratch(count);
  const N = count;
  const d = depths!;
  const buckets = bucketOf!;
  const out = outIdx!;
  const pos = positions;
  // view-space z (camera looks down -Z; smaller = farther).
  // view-z = view[2]*x + view[6]*y + view[10]*z + view[14]
  const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
  let dmin = Infinity, dmax = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = pos[i * 3 + 0];
    const y = pos[i * 3 + 1];
    const z = pos[i * 3 + 2];
    const vz = v2 * x + v6 * y + v10 * z + v14;
    d[i] = vz;
    if (vz < dmin) dmin = vz;
    if (vz > dmax) dmax = vz;
  }
  // Transfer a fresh Float32Array (we hand off the buffer to the main thread).
  const indices = new Float32Array(N);
  const range = dmax - dmin;
  if (range < 1e-9) {
    // Degenerate (all coplanar): identity order.
    for (let i = 0; i < N; i++) indices[i] = i;
    const msg: SortedMsg = { type: 'sorted', requestId, indices };
    (self as unknown as Worker).postMessage(msg, [indices.buffer]);
    return;
  }
  const NBUCKETS = 65536;
  const counts = new Uint32Array(NBUCKETS);
  const inv = (NBUCKETS - 1) / range;
  for (let i = 0; i < N; i++) {
    const b = ((d[i] - dmin) * inv) | 0;
    buckets[i] = b;
    counts[b]++;
  }
  // Exclusive prefix sum.
  let total = 0;
  for (let b = 0; b < NBUCKETS; b++) {
    const c = counts[b];
    counts[b] = total;
    total += c;
  }
  // Stable scatter into out (Uint32), then float-convert into the transferable.
  for (let i = 0; i < N; i++) {
    out[counts[buckets[i]]++] = i;
  }
  for (let i = 0; i < N; i++) indices[i] = out[i];
  const msg: SortedMsg = { type: 'sorted', requestId, indices };
  (self as unknown as Worker).postMessage(msg, [indices.buffer]);
}

(self as unknown as Worker).addEventListener('message', (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  if (msg.type === 'init') {
    positions = msg.positions;
    count = positions.length / 3 | 0;
    // Drop any old scratch sized for a previous scene.
    depths = null;
    bucketOf = null;
    outIdx = null;
  } else if (msg.type === 'sort') {
    sortAndPost(msg.requestId, msg.view);
  }
});

export type SortWorkerInbound = InMsg;
export type SortWorkerOutbound = SortedMsg;
