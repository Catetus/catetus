import {
  Cartesian3,
  Color,
  CustomDataSource,
  Entity,
  Math as CesiumMath,
  PointPrimitiveCollection,
  PrimitiveCollection,
  Resource,
  Transforms,
  type Scene,
  type Viewer,
} from "cesium";

import { parseQatPlyHeader, type QatPlyHeader } from "./qatHeaderParser.js";
import {
  computeColumnLayout,
  decodeQuantizedInt4Field,
  decodeQuantizedInt8Field,
  readDcColors,
  readPositions,
} from "./qatDequant.js";

export interface QATPlyLoadOptions {
  /** Anchor (longitude, latitude, height in meters) for placing the splat scene on the globe. */
  origin: { longitude: number; latitude: number; height?: number };
  /**
   * Scale applied to the model-space xyz coords before placing on the globe.
   * QAT-PLY positions are arbitrary "scene units"; default 1.0.
   */
  scale?: number;
  /** Optional friendly name. */
  name?: string;
  /** Override the default point pixel size. */
  pixelSize?: number;
}

export interface QATPlyDecodeResult {
  /** Header metadata. */
  header: QatPlyHeader;
  /** Anchor count. */
  vertexCount: number;
  /** Flat fp32 xyz xyz xyz array (model space). */
  positions: Float32Array;
  /** Per-anchor RGB (length 3N) when f_dc_{0,1,2} are present. */
  colors?: Float32Array;
  /** int8 anchor_feat dequantized to (N, C) when present. */
  anchorFeat?: { channels: number; data: Float32Array };
  /** int4 offset dequantized to (N, C) when present. */
  offset?: { channels: number; data: Float32Array };
}

/**
 * Cesium DataSource that decodes a Catetus QAT-PLY into a
 * `PointPrimitiveCollection` anchored at a given (lon, lat, height).
 *
 * Cesium has no native Gaussian splat primitive (yet) — this DataSource is the
 * pragmatic mapping that any geographic Cesium app can drop in to display
 * Catetus-quantized scenes alongside their other globe content. The decoded
 * float buffers are also exposed (`decode()`) so apps with a custom
 * GS-on-globe renderer can use this package as a pure codec.
 *
 *   const ds = new QATPlyDataSource();
 *   await ds.load(viewer.scene, "/scene.qat.ply", {
 *     origin: { longitude: -122.4194, latitude: 37.7749, height: 0 },
 *     scale: 1.0,
 *   });
 *   viewer.dataSources.add(ds);
 */
export class QATPlyDataSource extends CustomDataSource {
  private _pointCollection: PointPrimitiveCollection | null = null;
  private _ownedSceneScratch: Scene | null = null;

  constructor(name = "QATPly") {
    super(name);
  }

  /**
   * Pure decode of an in-memory buffer. No Cesium/scene side-effects.
   * Useful for tests and for app code that wants to drive its own rendering.
   */
  decode(buf: Uint8Array): QATPlyDecodeResult {
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
    const result: QATPlyDecodeResult = {
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
   * Fetch a URL, decode the PLY, and populate the DataSource with a
   * PointPrimitiveCollection at the requested geographic origin.
   */
  async load(
    scene: Scene,
    urlOrBytes: string | Uint8Array,
    options: QATPlyLoadOptions,
  ): Promise<void> {
    this._setLoading(true);
    try {
      const bytes = typeof urlOrBytes === "string"
        ? new Uint8Array(
            await (await Resource.fetchArrayBuffer({ url: urlOrBytes })) as ArrayBuffer,
          )
        : urlOrBytes;
      const result = this.decode(bytes);
      this._placeOnGlobe(scene, result, options);
    } finally {
      this._setLoading(false);
    }
  }

  /**
   * Place an already-decoded scene on the globe. Useful when the app already
   * has the bytes in memory (cached, streamed, etc.).
   */
  placeDecoded(
    scene: Scene,
    decoded: QATPlyDecodeResult,
    options: QATPlyLoadOptions,
  ): void {
    this._placeOnGlobe(scene, decoded, options);
  }

  private _placeOnGlobe(
    scene: Scene,
    decoded: QATPlyDecodeResult,
    options: QATPlyLoadOptions,
  ): void {
    const scale = options.scale ?? 1.0;
    const pixelSize = options.pixelSize ?? 2.0;
    const origin = Cartesian3.fromDegrees(
      options.origin.longitude,
      options.origin.latitude,
      options.origin.height ?? 0,
    );
    const enuToFixed = Transforms.eastNorthUpToFixedFrame(origin);

    // Lazily create + attach the points collection.
    let collection = this._pointCollection;
    if (!collection) {
      collection = new PointPrimitiveCollection();
      this._pointCollection = collection;
      // Add to the scene's primitives — DataSource entities can't host
      // PointPrimitive directly, but the DataSource holds onto the collection
      // for lifetime management. The PrimitiveCollection on the scene takes
      // ownership of disposal.
      this._ownedSceneScratch = scene;
      (scene.primitives as PrimitiveCollection).add(collection);
    }

    const N = decoded.vertexCount;
    const tmp = new Cartesian3();
    const out = new Cartesian3();
    for (let i = 0; i < N; i++) {
      tmp.x = decoded.positions[3 * i] * scale;
      tmp.y = decoded.positions[3 * i + 1] * scale;
      tmp.z = decoded.positions[3 * i + 2] * scale;
      // ENU rotation/translation -> Cesium fixed-frame coords.
      const mat = enuToFixed;
      const x = mat[0] * tmp.x + mat[4] * tmp.y + mat[8] * tmp.z + mat[12];
      const y = mat[1] * tmp.x + mat[5] * tmp.y + mat[9] * tmp.z + mat[13];
      const z = mat[2] * tmp.x + mat[6] * tmp.y + mat[10] * tmp.z + mat[14];
      out.x = x; out.y = y; out.z = z;

      const color = decoded.colors
        ? new Color(
            decoded.colors[3 * i],
            decoded.colors[3 * i + 1],
            decoded.colors[3 * i + 2],
            1.0,
          )
        : Color.WHITE;
      collection.add({ position: Cartesian3.clone(out), color, pixelSize });
    }

    // Also drop a single Entity at the origin so the DataSource's entities
    // collection is non-empty (handy for viewer.flyTo etc.).
    const anchor = new Entity({
      id: this.name + "-anchor",
      name: this.name,
      position: origin,
    });
    this.entities.add(anchor);
  }

  private _setLoading(b: boolean): void {
    // CustomDataSource exposes `_loading` change events via `loadingEvent`;
    // we toggle the public `isLoading` via the documented setter pattern.
    // Cast through unknown to avoid touching private impl details across
    // Cesium versions.
    const self = this as unknown as { isLoading: boolean };
    self.isLoading = b;
  }

  /**
   * Convert (lon, lat, height) to ECEF cartesian — exposed for tests that
   * don't want to instantiate a Cesium Scene. Pure math, no global state.
   */
  static cartesianFromDegrees(
    longitude: number,
    latitude: number,
    height = 0,
  ): { x: number; y: number; z: number } {
    const c = Cartesian3.fromDegrees(longitude, latitude, height);
    return { x: c.x, y: c.y, z: c.z };
  }

  /** Re-exported for downstream apps that don't import cesium directly. */
  static get CESIUM_PI(): number {
    return CesiumMath.PI;
  }
}
