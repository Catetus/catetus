import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { decodeV5TailBytes } from '../src/v5tail.js';

const GOLDEN = resolve(
  __dirname,
  '..',
  '..',
  '..',
  'experiments',
  'v5-2-composed',
  'data',
  'sidecar_v5_2.bin',
);

describe('decodeV5TailBytes (golden Python sidecar)', () => {
  it('matches the bonsai V5.2 header shape', () => {
    if (!existsSync(GOLDEN)) {
      // The fixture is not always checked in (it's a 800-KB artifact). Skip
      // silently if absent — the Rust round-trip test in
      // `crates/catetus-gltf/src/v5_tail.rs` covers the same path.
      console.warn(`v5tail golden test: ${GOLDEN} missing, skipping`);
      return;
    }
    const bytes = new Uint8Array(readFileSync(GOLDEN));
    const decoded = decodeV5TailBytes(bytes);
    expect(decoded.header.nSplats).toBe(1_244_819);
    expect(decoded.header.kSelected).toBe(12_448);
    expect(decoded.header.shRestCoefs).toBe(15);
    expect(decoded.header.nCells).toBe(64);
    expect(decoded.selIdx.length).toBe(12_448);
    // SF-ascending invariant.
    for (let i = 1; i < decoded.selIdx.length; i++) {
      expect(decoded.selIdx[i]).toBeGreaterThan(decoded.selIdx[i - 1]);
    }
    // Per-group row counts match K × n_chan.
    expect(decoded.pos.length).toBe(12_448 * 3);
    expect(decoded.rot.length).toBe(12_448 * 4);
    expect(decoded.opa.length).toBe(12_448);
    expect(decoded.sca.length).toBe(12_448 * 3);
    expect(decoded.dc.length).toBe(12_448 * 3);
    expect(decoded.shr.length).toBe(12_448 * 15 * 3);
  });

  /**
   * Phase D Path B back-compat: the polyfill decoder must accept both
   * version=1 (legacy 8/10/12/12/8/8 ship) and version=2 (current 8/10/
   * 14/14/8/8 default) headers. The wire format is identical between
   * versions — per-group bit_depth lives in each group's u8 header — so
   * we synthesize a "v=2" stream by flipping only the version byte on
   * the golden v=1 fixture and confirming the decode is byte-equivalent.
   */
  it('accepts both version 1 (legacy) and version 2 (Path B) headers', () => {
    if (!existsSync(GOLDEN)) {
      console.warn(`v5tail v=2 back-compat test: ${GOLDEN} missing, skipping`);
      return;
    }
    const v1Bytes = new Uint8Array(readFileSync(GOLDEN));
    const v2Bytes = new Uint8Array(v1Bytes); // copy
    // u16 LE at offset 8 — flip 1 -> 2.
    expect(v2Bytes[8]).toBe(1);
    expect(v2Bytes[9]).toBe(0);
    v2Bytes[8] = 2;
    const decodedV1 = decodeV5TailBytes(v1Bytes);
    const decodedV2 = decodeV5TailBytes(v2Bytes);
    // Header n_splats / k_selected / shapes must match.
    expect(decodedV2.header.nSplats).toBe(decodedV1.header.nSplats);
    expect(decodedV2.header.kSelected).toBe(decodedV1.header.kSelected);
    expect(decodedV2.header.shRestCoefs).toBe(decodedV1.header.shRestCoefs);
    expect(decodedV2.header.nCells).toBe(decodedV1.header.nCells);
    expect(decodedV2.selIdx).toEqual(decodedV1.selIdx);
    // Float content equality on a couple of groups (full equality would be
    // slow on ~12K * 48 floats; spot-check is sufficient given the byte
    // streams are identical post-version-byte).
    expect(decodedV2.pos[0]).toBe(decodedV1.pos[0]);
    expect(decodedV2.opa[0]).toBe(decodedV1.opa[0]);
    expect(decodedV2.sca[0]).toBe(decodedV1.sca[0]);
  });

  it('rejects unsupported version bytes', () => {
    if (!existsSync(GOLDEN)) {
      console.warn(`v5tail bad-version test: ${GOLDEN} missing, skipping`);
      return;
    }
    const bytes = new Uint8Array(readFileSync(GOLDEN));
    bytes[8] = 99; // unsupported version
    expect(() => decodeV5TailBytes(bytes)).toThrowError(
      /unsupported version 99/,
    );
  });
});
