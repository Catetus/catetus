import { describe, it, expect } from "vitest";
import {
  parseQatPlyHeader,
  parseFloatHex,
  decodeBase64,
} from "../src/qatHeaderParser.js";
import {
  computeColumnLayout,
  decodeQuantizedInt4Field,
  decodeQuantizedInt8Field,
  readDcColors,
  readFloatColumn,
  readPositions,
} from "../src/qatDequant.js";
import { QATPlyLoader } from "../src/QATPlyLoader.js";
import { buildSyntheticFixture, floatToHex, floatArrayToBase64 } from "../../fixtures/buildFixture.js";

describe("qatHeaderParser primitives", () => {
  it("parseFloatHex round-trips fp32 values bit-exactly", () => {
    const samples = [0, -0, 1, -1, 0.5, 1024, -3.5, 1e-12, 1e12, Math.PI];
    for (const v of samples) {
      const hex = floatToHex(v);
      const back = parseFloatHex(hex);
      expect(Object.is(back, v)).toBe(true);
    }
  });

  it("decodeBase64 round-trips arbitrary fp32 scales", () => {
    const arr = new Float32Array([0.001, -0.5, 7.25, 1e-7, -3.4e10]);
    const b64 = floatArrayToBase64(arr);
    const decoded = decodeBase64(b64);
    const f = new Float32Array(decoded.buffer.slice(decoded.byteOffset, decoded.byteOffset + decoded.byteLength));
    expect(Array.from(f)).toEqual(Array.from(arr));
  });
});

describe("synthetic QAT-PLY round-trip", () => {
  const fx = buildSyntheticFixture(32);
  const header = parseQatPlyHeader(fx.bytes);
  const body = fx.bytes.subarray(header.headerByteLength);
  const layout = computeColumnLayout(header);

  it("header reports correct vertex count + row stride", () => {
    expect(header.vertexCount).toBe(32);
    expect(header.rowStride).toBeGreaterThan(0);
  });

  it("constant_field comments are parsed into the constants map", () => {
    expect(header.constants.get("nx")).toBe(0);
    expect(header.constants.get("opacity")).toBe(-2.5);
    expect(header.constants.get("rot_0")).toBe(1);
  });

  it("readPositions returns the same xyz the fixture wrote", () => {
    const pos = readPositions(header, body, layout);
    expect(Array.from(pos)).toEqual(Array.from(fx.expectedPositions));
  });

  it("readFloatColumn materializes constant_field values for missing columns", () => {
    const opacity = readFloatColumn(header, body, layout, "opacity");
    for (let i = 0; i < fx.N; i++) expect(opacity[i]).toBe(-2.5);
  });

  it("decodeQuantizedInt8Field matches numpy int8 * scales dequant bit-exactly", () => {
    const q = header.quantized.get("f_anchor_feat");
    expect(q?.kind).toBe("int8");
    if (q?.kind !== "int8") throw new Error("unreachable");
    const out = decodeQuantizedInt8Field(header, body, layout, q);
    expect(Array.from(out)).toEqual(Array.from(fx.expectedAnchorFeat));
  });

  it("decodeQuantizedInt4Field unpacks nibbles + applies per-anchor scale", () => {
    const q = header.quantized.get("f_offset");
    expect(q?.kind).toBe("int4");
    if (q?.kind !== "int4") throw new Error("unreachable");
    const out = decodeQuantizedInt4Field(header, body, layout, q);
    expect(out.length).toBe(fx.N * fx.expectedOffsetChannels);
    expect(Array.from(out)).toEqual(Array.from(fx.expectedOffset));
  });

  it("readDcColors clamps 0.5 + SH_C0 * f_dc to [0,1]", () => {
    const colors = readDcColors(header, body, layout)!;
    expect(colors).not.toBeNull();
    expect(Array.from(colors)).toEqual(Array.from(fx.expectedDcColors));
  });
});

describe("QATPlyLoader.parse end-to-end", () => {
  it("builds a Three.js BufferGeometry with position + color attributes", () => {
    const fx = buildSyntheticFixture(16);
    const loader = new QATPlyLoader();
    const result = loader.parse(fx.bytes);
    expect(result.vertexCount).toBe(16);
    const pos = result.geometry.getAttribute("position");
    expect(pos.count).toBe(16);
    expect(pos.itemSize).toBe(3);
    const color = result.geometry.getAttribute("color");
    expect(color).toBeTruthy();
    expect(result.anchorFeat?.channels).toBe(4);
    expect(result.offset?.channels).toBe(6);
    // Points object should be ready to add to a scene.
    expect(result.points.name).toBe("QATPlyPoints");
  });

  it("rejects truncated bodies with a clear error", () => {
    const fx = buildSyntheticFixture(8);
    const truncated = fx.bytes.subarray(0, fx.bytes.length - 4);
    expect(() => new QATPlyLoader().parse(truncated)).toThrow(/truncated/);
  });
});

describe("v1 spec safety", () => {
  it("rejects quantized_field with unknown dtype (forward-compat)", () => {
    const lines =
      "ply\nformat binary_little_endian 1.0\n" +
      "comment quantized_field f_future int16 channels=2 scale_b64=AAAAAAAAAAA=\n" +
      "element vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    const buf = new TextEncoder().encode(lines);
    expect(() => parseQatPlyHeader(buf)).toThrow(/int16/);
  });
});
