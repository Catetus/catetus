import { Color3 } from "@babylonjs/core/Maths/math.color.js";
import { Vector3 } from "@babylonjs/core/Maths/math.vector.js";
import { PointsCloudSystem } from "@babylonjs/core/Particles/pointsCloudSystem.js";
import { Mesh } from "@babylonjs/core/Meshes/mesh.js";
import type { Scene } from "@babylonjs/core/scene.js";

import {
  parseQatPlyHeader,
  type QatPlyHeader,
} from "./qatHeaderParser.js";
import {
  computeColumnLayout,
  decodeQuantizedInt4Field,
  decodeQuantizedInt8Field,
  readDcColors,
  readPositions,
} from "./qatDequant.js";

export interface QATPlyParseResult {
  /** Header metadata for advanced consumers. */
  header: QatPlyHeader;
  /** Total anchor count. */
  vertexCount: number;
  /** Flat fp32 xyz xyz xyz... array. */
  positions: Float32Array;
  /** Flat fp32 rgb rgb rgb... array (length 3N) if DC SH columns are present. */
  colors?: Float32Array;
  /** int8 anchor_feat dequantized to (N, C) row-major fp32 (if present). */
  anchorFeat?: { channels: number; data: Float32Array };
  /** int4 offset dequantized to (N, C) row-major fp32 (if present). */
  offset?: { channels: number; data: Float32Array };
}

/**
 * Babylon.js loader for Catetus QAT-PLY.
 *
 *   import { QATPlyLoader } from "@catetus/babylonjs-qat";
 *
 *   const loader = new QATPlyLoader();
 *   const buf = new Uint8Array(await (await fetch("/scene.qat.ply")).arrayBuffer());
 *   const mesh = await loader.loadIntoScene(scene, buf, "my-splats");
 *
 * Babylon's SceneLoader plugin API requires more boilerplate (extension
 * registration via SceneLoader.RegisterPlugin) and locks us into specific
 * Babylon versions; we instead expose a small, explicit factory that any
 * Babylon app can call from its own loader pipeline. Returns a `Mesh` owned
 * by the scene's PointsCloudSystem.
 */
export class QATPlyLoader {
  /** Parse an in-memory buffer (no Babylon scene required). */
  parse(buf: Uint8Array): QATPlyParseResult {
    const header = parseQatPlyHeader(buf);
    const body = buf.subarray(header.headerByteLength);
    const expected = header.vertexCount * header.rowStride;
    if (body.byteLength < expected) {
      throw new Error(
        `QAT-PLY body truncated: have ${body.byteLength} bytes, need ${expected}`,
      );
    }
    const layout = computeColumnLayout(header);
    const positions = readPositions(header, body, layout);
    const colors = readDcColors(header, body, layout) ?? undefined;

    const result: QATPlyParseResult = {
      header,
      vertexCount: header.vertexCount,
      positions,
      colors,
    };
    const feat = header.quantized.get("f_anchor_feat");
    if (feat?.kind === "int8") {
      result.anchorFeat = {
        channels: feat.channels,
        data: decodeQuantizedInt8Field(header, body, layout, feat),
      };
    }
    const offset = header.quantized.get("f_offset");
    if (offset?.kind === "int4") {
      result.offset = {
        channels: offset.channels,
        data: decodeQuantizedInt4Field(header, body, layout, offset),
      };
    }
    return result;
  }

  /**
   * Decode and add a points cloud to a Babylon scene. Resolves to the
   * underlying Mesh (built by Babylon's PointsCloudSystem).
   */
  async loadIntoScene(scene: Scene, buf: Uint8Array, name = "QATPly"): Promise<Mesh> {
    const result = this.parse(buf);
    const pcs = new PointsCloudSystem(name, 1, scene);
    pcs.addPoints(result.vertexCount, (particle: unknown, i: number) => {
      const p = particle as { position: Vector3; color?: Color3 };
      p.position = new Vector3(
        result.positions[3 * i],
        result.positions[3 * i + 1],
        result.positions[3 * i + 2],
      );
      if (result.colors) {
        p.color = new Color3(
          result.colors[3 * i],
          result.colors[3 * i + 1],
          result.colors[3 * i + 2],
        );
      }
    });
    return await pcs.buildMeshAsync();
  }
}
