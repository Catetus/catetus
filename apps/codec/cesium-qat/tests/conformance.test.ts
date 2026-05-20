// Runs each plugin against the canonical QAT-PLY v1 conformance suite at
// apps/codec/conformance/. A decoder is conformant when its dequant output
// matches every fixture's `expected_fp32_b64` byte-for-byte.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  parseQatPlyHeader,
  type QuantizedField,
} from "../src/qatHeaderParser.js";
import {
  computeColumnLayout,
  decodeQuantizedInt4Field,
  decodeQuantizedInt8Field,
} from "../src/qatDequant.js";

const CONFORMANCE_DIR = resolve(__dirname, "../../conformance");

interface ConformanceCase {
  name: string;
  n_anchors: number;
  fields: Record<
    string,
    { shape: [number, number]; expected_fp32_b64: string }
  >;
}

interface ConformanceManifest {
  version: number;
  cases: ConformanceCase[];
}

const manifest = JSON.parse(
  readFileSync(resolve(CONFORMANCE_DIR, "conformance.json"), "utf-8"),
) as ConformanceManifest;

describe("QAT-PLY v1 conformance suite", () => {
  expect(manifest.version).toBe(1);
  expect(manifest.cases.length).toBe(10);

  for (const c of manifest.cases) {
    it(`decodes ${c.name} byte-for-byte`, () => {
      const fixturePath = resolve(CONFORMANCE_DIR, "fixtures", c.name);
      const bytes = new Uint8Array(readFileSync(fixturePath));
      const header = parseQatPlyHeader(bytes);
      expect(header.vertexCount).toBe(c.n_anchors);
      const body = bytes.subarray(header.headerByteLength);
      const layout = computeColumnLayout(header);
      for (const [fieldName, info] of Object.entries(c.fields)) {
        const meta = header.quantized.get(fieldName) as QuantizedField | undefined;
        expect(meta, `${c.name}: missing quantized_field ${fieldName}`).toBeTruthy();
        let decoded: Float32Array;
        if (meta!.kind === "int8") {
          decoded = decodeQuantizedInt8Field(header, body, layout, meta!);
        } else {
          decoded = decodeQuantizedInt4Field(header, body, layout, meta!);
        }
        const expectedBytes = Buffer.from(info.expected_fp32_b64, "base64");
        const decodedBytes = Buffer.from(decoded.buffer, decoded.byteOffset, decoded.byteLength);
        expect(
          decodedBytes.equals(expectedBytes),
          `${c.name} / ${fieldName}: decoded bytes do not match expected_fp32_b64`,
        ).toBe(true);
      }
    });
  }
});
