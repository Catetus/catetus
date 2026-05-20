import {
  BufferAttribute,
  BufferGeometry,
  FileLoader,
  Loader,
  Points,
  PointsMaterial,
} from "three";
import type { LoadingManager } from "three";

import {
  parseQatPlyHeader,
  type QatPlyHeader,
  type QuantizedField,
} from "./qatHeaderParser.js";
import {
  computeColumnLayout,
  decodeQuantizedInt4Field,
  decodeQuantizedInt8Field,
  readDcColors,
  readFloatColumn,
  readPositions,
} from "./qatDequant.js";

export interface QATPlyParseResult {
  /** Three.js geometry containing `position` and (when available) `color` attributes. */
  geometry: BufferGeometry;
  /** Convenience Points object using a vertex-color material. Add directly to a scene. */
  points: Points;
  /** Parsed PLY header — exposed for advanced consumers. */
  header: QatPlyHeader;
  /** Dense fp32 (N, C) decode of `f_anchor_feat` if present; otherwise undefined. */
  anchorFeat?: { channels: number; data: Float32Array };
  /** Dense fp32 (N, C) decode of `f_offset` if present; otherwise undefined. */
  offset?: { channels: number; data: Float32Array };
  /** Map of fp32 columns that were emitted in the body (excluding quantized fields). */
  vertexCount: number;
}

/**
 * Three.js loader for the Catetus QAT-PLY format.
 *
 *   import { QATPlyLoader } from "@catetus/three-qat";
 *   const loader = new QATPlyLoader();
 *   const result = await loader.loadAsync("/scene.qat.ply");
 *   scene.add(result.points);
 *
 * The loader does NOT (yet) ship a custom Gaussian-splat material — it renders
 * the anchors as colored points using the DC SH coefficients when present.
 * Downstream projects can use `result.geometry` + their own splat material.
 */
export class QATPlyLoader extends Loader {
  constructor(manager?: LoadingManager) {
    super(manager);
  }

  override load(
    url: string,
    onLoad: (result: QATPlyParseResult) => void,
    onProgress?: (event: ProgressEvent) => void,
    onError?: (err: unknown) => void,
  ): void {
    const fl = new FileLoader(this.manager);
    fl.setPath(this.path);
    fl.setResponseType("arraybuffer");
    fl.setRequestHeader(this.requestHeader);
    fl.setWithCredentials(this.withCredentials);
    fl.load(
      url,
      (data) => {
        try {
          const buf = new Uint8Array(data as ArrayBuffer);
          onLoad(this.parse(buf));
        } catch (e) {
          if (onError) onError(e);
          else this.manager.itemError(url);
        }
      },
      onProgress,
      (e) => {
        if (onError) onError(e);
      },
    );
  }

  /** Parse an in-memory buffer. Synchronous; useful for tests + fetch pipelines. */
  parse(buf: Uint8Array): QATPlyParseResult {
    const header = parseQatPlyHeader(buf);
    const body = buf.subarray(header.headerByteLength);
    const expectedBodyBytes = header.vertexCount * header.rowStride;
    if (body.byteLength < expectedBodyBytes) {
      throw new Error(
        `QAT-PLY body truncated: have ${body.byteLength} bytes, need ${expectedBodyBytes}`,
      );
    }
    const layout = computeColumnLayout(header);

    const positions = readPositions(header, body, layout);
    const colors = readDcColors(header, body, layout);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    if (colors) geometry.setAttribute("color", new BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();

    const material = new PointsMaterial({
      size: 0.01,
      vertexColors: !!colors,
      sizeAttenuation: true,
    });
    const points = new Points(geometry, material);
    points.name = "QATPlyPoints";

    const result: QATPlyParseResult = {
      geometry,
      points,
      header,
      vertexCount: header.vertexCount,
    };

    const anchorFeat = header.quantized.get("f_anchor_feat") as QuantizedField | undefined;
    if (anchorFeat && anchorFeat.kind === "int8") {
      const data = decodeQuantizedInt8Field(header, body, layout, anchorFeat);
      result.anchorFeat = { channels: anchorFeat.channels, data };
    }
    const offsetField = header.quantized.get("f_offset") as QuantizedField | undefined;
    if (offsetField && offsetField.kind === "int4") {
      const data = decodeQuantizedInt4Field(header, body, layout, offsetField);
      result.offset = { channels: offsetField.channels, data };
    }
    return result;
  }
}
