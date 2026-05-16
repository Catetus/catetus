// wasm.test.ts — Conformance test for the WebAssembly build of the
// QAT-PLY v1 reference decoder. Loads each fixture in
// apps/codec/conformance/conformance.json, parses + dequantizes each
// declared quantized_field via the WASM module, and asserts the fp32
// output matches the JSON expectation byte-for-byte.
//
// SPDX-License-Identifier: MIT

import { describe, it, expect, beforeAll } from "vitest";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WASM_JS = path.resolve(HERE, "../dist/qat_ply_decode.js");
const CONF    = path.resolve(HERE, "../../conformance/conformance.json");
const FIX_DIR = path.resolve(HERE, "../../conformance/fixtures");

function findEndHeader(bytes: Uint8Array): number {
    const needle = new TextEncoder().encode("\nend_header\n");
    outer: for (let i = 0; i + needle.length <= bytes.length; i++) {
        for (let j = 0; j < needle.length; j++) {
            if (bytes[i + j] !== needle[j]) continue outer;
        }
        return i + needle.length;
    }
    return -1;
}

// Lightweight PLY property-table parser. Returns array of
// {name, type, size} in declaration order, plus n_verts.
function parseProps(headerStr: string): { n_verts: number; props: { name: string; type: string; size: number }[] } {
    const TYPES: Record<string, number> = {
        char: 1, uchar: 1, short: 2, ushort: 2,
        int: 4, uint: 4, float: 4, double: 8, float32: 4,
    };
    let n_verts = 0;
    const props: { name: string; type: string; size: number }[] = [];
    let in_vertex = false;
    for (const line of headerStr.split("\n")) {
        const s = line.trim();
        if (s.startsWith("element vertex ")) {
            n_verts = parseInt(s.slice(15), 10);
            in_vertex = true;
        } else if (s.startsWith("element ")) {
            in_vertex = false;
        } else if (in_vertex && s.startsWith("property ")) {
            const parts = s.split(/\s+/);
            const type = parts[1];
            const name = parts[2];
            props.push({ name, type, size: TYPES[type] ?? 0 });
        }
    }
    return { n_verts, props };
}

function b64ToFloat32(b64: string): Float32Array {
    const bin = Buffer.from(b64, "base64");
    return new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4).slice();
}

let mod: any;

beforeAll(async () => {
    const createQatPlyModule = (await import(WASM_JS)).default;
    mod = await createQatPlyModule();
});

interface ConfCase {
    name: string;
    n_anchors: number;
    fields: Record<string, { shape: [number, number]; expected_fp32_b64: string }>;
}

const conformance: { version: number; cases: ConfCase[] } =
    JSON.parse(await readFile(CONF, "utf8"));

describe("QAT-PLY WASM conformance", () => {
    for (const c of conformance.cases) {
        it(`${c.name} byte-exact vs reference JSON`, async () => {
            const buf = await readFile(path.join(FIX_DIR, c.name));
            const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
            const bodyOff = findEndHeader(bytes);
            expect(bodyOff).toBeGreaterThan(0);

            const headerStr = new TextDecoder().decode(bytes.subarray(0, bodyOff));
            const body = bytes.subarray(bodyOff);

            const hdr = mod.parseHeader(headerStr);
            const { n_verts, props } = parseProps(headerStr);
            expect(n_verts).toBe(c.n_anchors);

            const rowSize = props.reduce((s, p) => s + p.size, 0);
            const propOff: Record<string, number> = {};
            { let o = 0; for (const p of props) { propOff[p.name] = o; o += p.size; } }

            for (let fi = 0; fi < hdr.fields.size(); fi++) {
                const f = hdr.fields.get(fi);
                const expected = c.fields[f.name];
                if (!expected) continue;
                const C = f.channels;

                let actual: Float32Array;
                if (f.dtype === 8) {
                    // int8
                    const q = new Uint8Array(n_verts * C);
                    for (let ci = 0; ci < C; ci++) {
                        const col = `${f.name}_q_${ci}`;
                        const off = propOff[col];
                        for (let r = 0; r < n_verts; r++) {
                            q[r * C + ci] = body[r * rowSize + off];
                        }
                    }
                    const scales = mod.base64DecodeFp32(
                        headerStr.substring(f.scaleB64Offset, f.scaleB64Offset + f.scaleB64Len),
                        C
                    );
                    actual = mod.dequantInt8(q, scales, n_verts, C);
                } else {
                    // int4
                    const B = (C + 1) >> 1;
                    const packed = new Uint8Array(n_verts * B);
                    for (let bi = 0; bi < B; bi++) {
                        const col = `${f.name}_q_${bi}`;
                        const off = propOff[col];
                        for (let r = 0; r < n_verts; r++) {
                            packed[r * B + bi] = body[r * rowSize + off];
                        }
                    }
                    const scaleOff = propOff[`${f.name}_scale`];
                    const scales = new Float32Array(n_verts);
                    const dv = new DataView(body.buffer, body.byteOffset, body.byteLength);
                    for (let r = 0; r < n_verts; r++) {
                        scales[r] = dv.getFloat32(r * rowSize + scaleOff, true);
                    }
                    actual = mod.dequantInt4Packed(packed, scales, n_verts, C);
                }

                const expectedF32 = b64ToFloat32(expected.expected_fp32_b64);
                expect(actual.length).toBe(expectedF32.length);
                // Byte-exact: compare the underlying bytes, not just numeric equality.
                const aBytes = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength);
                const eBytes = new Uint8Array(expectedF32.buffer, expectedF32.byteOffset, expectedF32.byteLength);
                expect(aBytes.length).toBe(eBytes.length);
                for (let i = 0; i < aBytes.length; i++) {
                    if (aBytes[i] !== eBytes[i]) {
                        throw new Error(`byte mismatch at ${i} for ${c.name}/${f.name}: got ${aBytes[i]} expected ${eBytes[i]}`);
                    }
                }
            }
        });
    }
});
