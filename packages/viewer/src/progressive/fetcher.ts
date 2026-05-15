// SPDX-License-Identifier: Apache-2.0
/**
 * Streaming `.mgs2` fetcher.
 *
 * Pulls bytes from a network URL (or any `Response`) and exposes them as a
 * pull-based async iterator that emits:
 *
 *   1. one `{kind: 'header', ...}` event once the fixed prefix + PLY header
 *      have been fully buffered (typically ~1.5 KB total).
 *   2. zero-or-more `{kind: 'chunk', ...}` events as additional complete
 *      record payload arrives. Each chunk event carries one full batch of
 *      whole records — partial trailing records are kept in an internal
 *      buffer until the next chunk arrives or the stream completes.
 *   3. one terminal `{kind: 'done', ...}` event with the total bytes seen.
 *
 * Header layout (per `crates/splatforge-ply/src/progressive.rs`):
 *
 *   [4]  magic           = b"MGS2"
 *   [4]  version u32     = 1
 *   [4]  flags   u32     = 0
 *   [8]  n_splats u64
 *   [4]  record_size u32
 *   [4]  ply_header_len u32
 *   [H]  ply_header_bytes
 *   [N * record_size] splat records, descending importance order.
 *
 * The fetcher also parses the embedded PLY header to expose per-field byte
 * offsets — the uploader needs these to map AoS records → SoA chunks for
 * the WebGPU compute-decode path (see `uploader.ts`).
 *
 * Designed for both:
 *   - Real `fetch(url)` in a browser (uses `Response.body.getReader()`).
 *   - A test harness that wants to drive batches synchronously — supply a
 *     custom `source` async iterator.
 */

/** `.mgs2` magic bytes (ASCII "MGS2"). */
export const MGS2_MAGIC = new Uint8Array([0x4d, 0x47, 0x53, 0x32]);
/** Bitstream version we understand. */
export const MGS2_VERSION = 1;
/** Bytes in the fixed prefix: magic + version + flags + n_splats + record_size + ply_header_len. */
export const MGS2_PREFIX_LEN = 4 + 4 + 4 + 8 + 4 + 4;

/** Per-field byte offsets parsed out of the embedded PLY header. */
export interface PlyFieldOffsets {
  /** Byte stride of a single PLY vertex record. Must equal `record_size`. */
  recordSize: number;
  x: number;
  y: number;
  z: number;
  scale0: number;
  scale1: number;
  scale2: number;
  /** PLY convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z. */
  rot0: number;
  rot1: number;
  rot2: number;
  rot3: number;
  opacity: number;
  fDc0: number;
  fDc1: number;
  fDc2: number;
}

/** Header event emitted once after the prefix + PLY header have been read. */
export interface ProgressiveHeaderEvent {
  kind: 'header';
  /** Total splats in the bitstream (= source PLY vertex count). */
  nSplats: number;
  /** Bytes per PLY vertex record. */
  recordSize: number;
  /** Verbatim PLY header bytes (ASCII, ending in `end_header\n`). */
  plyHeader: Uint8Array;
  /** Byte offsets of every field we care about. */
  fieldOffsets: PlyFieldOffsets;
  /** Total `.mgs2` byte length (sum of all subsequent chunks + the bytes already consumed). May be `undefined` if the source didn't carry a Content-Length. */
  totalBytes?: number;
}

/** Chunk of fully-decoded PLY records (raw AoS bytes). */
export interface ProgressiveChunkEvent {
  kind: 'chunk';
  /** Byte offset (from start of file) at which this batch's payload begins. */
  offset: number;
  /** Raw payload bytes — always a whole number of records. */
  bytes: Uint8Array;
  /** Number of records in this batch (`bytes.byteLength / recordSize`). */
  splatsAdded: number;
  /** Total records emitted so far across all chunks. */
  splatsSoFar: number;
}

/** End-of-stream event. */
export interface ProgressiveDoneEvent {
  kind: 'done';
  /** Total bytes consumed from the source. */
  totalBytesRead: number;
  /** Total records emitted. */
  totalSplats: number;
}

/** Union of every event a fetcher emits. */
export type ProgressiveEvent =
  | ProgressiveHeaderEvent
  | ProgressiveChunkEvent
  | ProgressiveDoneEvent;

/**
 * Parameters for {@link fetchProgressive}.
 */
export interface FetchProgressiveOptions {
  /**
   * Maximum bytes of *payload* to buffer before emitting a chunk event. The
   * fetcher splits the stream into batches of this size (rounded down to a
   * whole record). Defaults to 1 MB.
   */
  batchBytes?: number;
  /**
   * Override the underlying byte source. When set, `url` is ignored. Used by
   * the test harness to inject a synthetic slow stream.
   */
  source?: AsyncIterable<Uint8Array>;
  /**
   * Optional abort signal. The fetcher cancels the underlying reader when the
   * signal aborts and ends the iterator early.
   */
  signal?: AbortSignal;
}

/* ------------------------------------------------------------------------ */
/* PLY header parse — minimal subset matching `parse_inria_ply_header`.     */
/* ------------------------------------------------------------------------ */

/** Map a PLY scalar property type to its byte width. Unknown types throw. */
function plyTypeBytes(ty: string): number {
  switch (ty) {
    case 'float':
    case 'float32':
    case 'int':
    case 'int32':
    case 'uint':
    case 'uint32':
      return 4;
    case 'double':
    case 'float64':
      return 8;
    case 'short':
    case 'int16':
    case 'ushort':
    case 'uint16':
      return 2;
    case 'char':
    case 'int8':
    case 'uchar':
    case 'uint8':
      return 1;
    default:
      throw new Error(`mgs2: unsupported PLY property type "${ty}"`);
  }
}

/**
 * Parse an Inria 3DGS PLY ASCII header and compute the byte offset of every
 * field we'll need at upload time. Mirrors the Rust `parse_inria_ply_header`.
 *
 * @throws Error when the header is not `binary_little_endian` or is missing
 *         a required field (x, y, z, scale_0..2, rot_0..3, opacity, f_dc_0..2).
 */
export function parsePlyHeader(headerBytes: Uint8Array): PlyFieldOffsets {
  const text = new TextDecoder('ascii').decode(headerBytes);
  const lines = text.split('\n');
  if (lines[0]?.trim() !== 'ply') {
    throw new Error('mgs2: PLY header missing "ply" magic');
  }
  let sawFormat = false;
  let inVertex = false;
  let offset = 0;
  const fields: Partial<Record<string, number>> = {};
  for (let i = 1; i < lines.length; i++) {
    const trimmed = lines[i]!.trim();
    if (trimmed === 'end_header') break;
    if (trimmed.length === 0 || trimmed.startsWith('comment')) continue;
    const parts = trimmed.split(/\s+/);
    if (parts[0] === 'format') {
      if (parts[1] !== 'binary_little_endian') {
        throw new Error(`mgs2: PLY format must be binary_little_endian, got "${parts[1]}"`);
      }
      sawFormat = true;
    } else if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
    } else if (parts[0] === 'property') {
      if (parts[1] === 'list') {
        throw new Error('mgs2: PLY list properties not supported');
      }
      const ty = parts[1]!;
      const name = parts[2]!;
      const size = plyTypeBytes(ty);
      if (inVertex) {
        fields[name] = offset;
        offset += size;
      }
    }
  }
  if (!sawFormat) {
    throw new Error('mgs2: PLY header missing format directive');
  }
  const need = (n: string): number => {
    const v = fields[n];
    if (v === undefined) {
      throw new Error(`mgs2: PLY header missing required field "${n}"`);
    }
    return v;
  };
  return {
    recordSize: offset,
    x: need('x'),
    y: need('y'),
    z: need('z'),
    scale0: need('scale_0'),
    scale1: need('scale_1'),
    scale2: need('scale_2'),
    rot0: need('rot_0'),
    rot1: need('rot_1'),
    rot2: need('rot_2'),
    rot3: need('rot_3'),
    opacity: need('opacity'),
    fDc0: need('f_dc_0'),
    fDc1: need('f_dc_1'),
    fDc2: need('f_dc_2'),
  };
}

/* ------------------------------------------------------------------------ */
/* Internal byte-buffer accumulator.                                        */
/* ------------------------------------------------------------------------ */

class ByteAccumulator {
  private parts: Uint8Array[] = [];
  private total = 0;
  push(part: Uint8Array): void {
    this.parts.push(part);
    this.total += part.byteLength;
  }
  get length(): number {
    return this.total;
  }
  /** Concatenate everything and clear. */
  drain(): Uint8Array {
    if (this.parts.length === 1) {
      const single = this.parts[0]!;
      this.parts = [];
      this.total = 0;
      return single;
    }
    const out = new Uint8Array(this.total);
    let off = 0;
    for (const p of this.parts) {
      out.set(p, off);
      off += p.byteLength;
    }
    this.parts = [];
    this.total = 0;
    return out;
  }
  /** Read `n` bytes from the head, leaving the rest. Throws if `n > length`. */
  take(n: number): Uint8Array {
    if (n > this.total) throw new Error(`ByteAccumulator: take(${n}) > length(${this.total})`);
    const buf = this.drain();
    const head = buf.subarray(0, n);
    const tail = buf.subarray(n);
    if (tail.byteLength > 0) this.push(tail);
    // Detach `head` from the rest with a copy so callers can retain it past
    // future drains without aliasing.
    return new Uint8Array(head);
  }
}

/* ------------------------------------------------------------------------ */
/* Stream → AsyncIterable<Uint8Array> adapter.                              */
/* ------------------------------------------------------------------------ */

async function* readableStreamToAsyncIter(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<Uint8Array> {
  try {
    for (;;) {
      if (signal?.aborted) break;
      const { value, done } = await reader.read();
      if (done) return;
      if (value && value.byteLength > 0) yield value;
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* ignore */
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Public API.                                                              */
/* ------------------------------------------------------------------------ */

/**
 * Stream a `.mgs2` bitstream from `url` (or the supplied `source` iterator)
 * and yield header / chunk / done events as bytes arrive.
 *
 * Usage:
 *
 *   for await (const ev of fetchProgressive('/scenes/bonsai.mgs2')) {
 *     if (ev.kind === 'header') console.log('total splats', ev.nSplats);
 *     else if (ev.kind === 'chunk') uploader.addBatch(ev.bytes, ev.splatsAdded);
 *   }
 *
 * The async iterator is single-shot — you can't restart it.
 *
 * @throws Error when the source ends before the fixed `.mgs2` prefix has been
 *         received, or when the header is malformed.
 */
export async function* fetchProgressive(
  url: string,
  opts: FetchProgressiveOptions = {},
): AsyncGenerator<ProgressiveEvent> {
  const batchBytes = Math.max(opts.batchBytes ?? 1 << 20, 1);
  let totalBytesHint: number | undefined;

  let source: AsyncIterable<Uint8Array>;
  if (opts.source) {
    source = opts.source;
  } else {
    const res = await fetch(url, opts.signal ? { signal: opts.signal } : {});
    if (!res.ok && res.status !== 206) {
      throw new Error(`mgs2_fetch_failed: HTTP ${res.status} for ${url}`);
    }
    const cl = res.headers.get('content-length');
    if (cl) {
      const n = Number(cl);
      if (Number.isFinite(n) && n > 0) totalBytesHint = n;
    }
    const body = res.body;
    if (!body) throw new Error('mgs2_fetch_failed: response has no body');
    source = readableStreamToAsyncIter(body.getReader(), opts.signal);
  }

  const acc = new ByteAccumulator();
  let totalBytesRead = 0;
  let payloadOffset = 0;
  let totalSplats = 0;
  let header: ProgressiveHeaderEvent | undefined;
  let recordSize = 0;
  let nSplats = 0;
  let prefixReady = false;
  let sourceExhausted = false;

  // Single pump iterator. We pull from it every time the state machine
  // below needs more bytes.
  const pump = source[Symbol.asyncIterator]();

  /** Pull one more chunk from the source. Returns false on EOF. */
  const pullMore = async (): Promise<boolean> => {
    if (sourceExhausted) return false;
    const { value, done } = await pump.next();
    if (done) {
      sourceExhausted = true;
      return false;
    }
    if (value && value.byteLength > 0) {
      acc.push(value);
      totalBytesRead += value.byteLength;
    }
    return true;
  };

  // Phase A: collect the fixed prefix.
  while (acc.length < MGS2_PREFIX_LEN) {
    const got = await pullMore();
    if (!got) {
      throw new Error(
        `mgs2_truncated_header: only ${acc.length} of ${MGS2_PREFIX_LEN} prefix bytes received`,
      );
    }
    if (opts.signal?.aborted) return;
  }
  prefixReady = true;
  void prefixReady;
  for (;;) {
    if (acc.length >= MGS2_PREFIX_LEN && !header) {
      // Parse the prefix.
      const prefix = acc.take(MGS2_PREFIX_LEN);
      const dv = new DataView(prefix.buffer, prefix.byteOffset, MGS2_PREFIX_LEN);
      // Magic.
      for (let i = 0; i < 4; i++) {
        if (prefix[i] !== MGS2_MAGIC[i]) {
          throw new Error(
            `mgs2_bad_magic: expected "MGS2", got "${new TextDecoder('ascii').decode(prefix.subarray(0, 4))}"`,
          );
        }
      }
      const version = dv.getUint32(4, true);
      if (version !== MGS2_VERSION) {
        throw new Error(`mgs2_unsupported_version: ${version} (expected ${MGS2_VERSION})`);
      }
      const flags = dv.getUint32(8, true);
      if (flags !== 0) {
        throw new Error(`mgs2_unsupported_flags: 0x${flags.toString(16)} (expected 0)`);
      }
      // n_splats is u64. JS Numbers cap at 2^53, well above any plausible splat count.
      const nSplatsHi = dv.getUint32(16, true);
      const nSplatsLo = dv.getUint32(12, true);
      nSplats = nSplatsHi * 0x1_0000_0000 + nSplatsLo;
      recordSize = dv.getUint32(20, true);
      const plyHeaderLen = dv.getUint32(24, true);
      if (recordSize === 0 || recordSize > 1 << 16) {
        throw new Error(`mgs2_bad_record_size: ${recordSize}`);
      }
      if (plyHeaderLen === 0 || plyHeaderLen > 1 << 16) {
        throw new Error(`mgs2_bad_ply_header_len: ${plyHeaderLen}`);
      }

      // Wait for the PLY header bytes.
      while (acc.length < plyHeaderLen) {
        const got = await pullMore();
        if (!got) {
          throw new Error(
            `mgs2_truncated_ply_header: ${acc.length} of ${plyHeaderLen} bytes received`,
          );
        }
        if (opts.signal?.aborted) return;
      }
      const plyHeader = acc.take(plyHeaderLen);
      const fieldOffsets = parsePlyHeader(plyHeader);
      if (fieldOffsets.recordSize !== recordSize) {
        throw new Error(
          `mgs2_record_size_mismatch: header says ${recordSize}, PLY columns sum to ${fieldOffsets.recordSize}`,
        );
      }
      payloadOffset = MGS2_PREFIX_LEN + plyHeaderLen;
      header = {
        kind: 'header',
        nSplats,
        recordSize,
        plyHeader,
        fieldOffsets,
        totalBytes: totalBytesHint,
      };
      yield header;
    }

    // Phase C: drain payload in batches of whole records.
    if (header && acc.length >= recordSize) {
      // Emit as many full records as we have, up to `batchBytes`.
      const records = Math.floor(acc.length / recordSize);
      if (records > 0) {
        const remaining = nSplats - totalSplats;
        const recsToEmit = Math.min(
          records,
          Math.floor(batchBytes / recordSize) || 1,
          remaining,
        );
        if (recsToEmit > 0) {
          const bytes = acc.take(recsToEmit * recordSize);
          const ev: ProgressiveChunkEvent = {
            kind: 'chunk',
            offset: payloadOffset,
            bytes,
            splatsAdded: recsToEmit,
            splatsSoFar: totalSplats + recsToEmit,
          };
          payloadOffset += bytes.byteLength;
          totalSplats += recsToEmit;
          yield ev;
          if (totalSplats >= nSplats) {
            yield {
              kind: 'done',
              totalBytesRead,
              totalSplats,
            } satisfies ProgressiveDoneEvent;
            return;
          }
          // Loop again to drain whatever's still buffered.
          continue;
        }
      }
    }

    // Phase D: pull more bytes if available.
    const got = await pullMore();
    if (!got) {
      // Source exhausted. Emit any leftover whole records (defensive — should
      // never happen on a well-formed stream, since the producer always sends
      // exactly n_splats * record_size payload bytes).
      if (header && acc.length >= recordSize) continue;
      yield {
        kind: 'done',
        totalBytesRead,
        totalSplats,
      } satisfies ProgressiveDoneEvent;
      return;
    }
    if (opts.signal?.aborted) return;
  }
}
