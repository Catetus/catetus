/**
 * `tileset.json` fetcher + tree builder for Catetus streaming tiles.
 *
 * Consumes the 3D Tiles 1.1 hierarchy emitted by
 * `catetus-optimize::tileset::write_tileset` and produces an in-memory
 * tree of {@link TileNode}s. Each node carries the bounding AABB derived
 * from the on-disk 12-float OBB, the `geometricError`, the content URI
 * (resolved against the tileset's base URL), and a parent pointer used by
 * the LOD selector's "resident ancestor" fallback.
 *
 * The loader intentionally does NOT fetch any tile content — only the root
 * `tileset.json`. Content fetches are deferred to {@link TileStreamer}
 * which the runtime invokes from inside the per-frame walk.
 *
 * Determinism: the tree is built in document order (root.children[0],
 * children[1], ...), and each tile's `id` is a stable path string
 * (`"root"`, `"root/0"`, `"root/0/1"`, ...). Two runs against the same
 * `tileset.json` produce structurally identical trees.
 */

import { aabbFromObb12, type Aabb } from './frustum.js';

/** One in-memory tile. */
export interface TileNode {
  /** Stable path identifier. Used as the LRU cache key and for diagnostics. */
  id: string;
  /** Bounding volume in the tileset's local frame. */
  aabb: Aabb;
  /** World-space refinement error from `tileset.json`. */
  geometricError: number;
  /** Refinement mode — `"REPLACE"` is the only mode Catetus emits. */
  refine: 'REPLACE' | 'ADD';
  /** Absolute URL of this tile's content GLB. */
  contentUrl: string;
  /** Original relative URI (kept for diagnostics / error messages). */
  contentUri: string;
  /** Child tiles. Empty for leaves. */
  children: TileNode[];
  /** Parent tile, or `undefined` for the root. */
  parent?: TileNode;
  /** Depth from the root (root = 0). Cached for selector heuristics. */
  depth: number;
}

/** Fully-parsed tileset. */
export interface Tileset {
  /** Absolute URL the tileset.json was loaded from. */
  url: string;
  /** Tileset's overall geometricError (root-replacement bound). */
  geometricError: number;
  /** Required extensions a runtime must understand. */
  extensionsRequired: string[];
  /** Root tile. */
  root: TileNode;
  /** Pre-built flat list of every tile in document (preorder) order. */
  tiles: TileNode[];
}

interface RawTileset {
  asset?: { version?: string };
  geometricError?: number;
  extensionsRequired?: string[];
  extensionsUsed?: string[];
  root?: RawTile;
}

interface RawTile {
  boundingVolume?: { box?: number[]; region?: number[]; sphere?: number[] };
  geometricError?: number;
  refine?: string;
  content?: { uri?: string; url?: string };
  children?: RawTile[];
}

/**
 * Fetch and parse a `tileset.json` at `url`. Throws an `Error` whose message
 * starts with `tileset_not_found:` (404) or `tileset_invalid:` (parse error).
 */
export async function loadTilesetJson(url: string): Promise<Tileset> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new Error(`tileset_not_found: ${(err as Error).message}`);
  }
  if (res.status === 404) {
    throw new Error(`tileset_not_found: ${url}`);
  }
  if (!res.ok) {
    throw new Error(`tileset_not_found: HTTP ${res.status} for ${url}`);
  }
  const text = await res.text();
  return parseTileset(text, url);
}

/**
 * Parse a `tileset.json` body string. Exposed for tests that don't want to
 * spin up a fetch mock.
 */
export function parseTileset(json: string, baseUrl: string): Tileset {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (err) {
    throw new Error(`tileset_invalid: not valid JSON (${(err as Error).message})`);
  }
  if (!raw || typeof raw !== 'object') {
    throw new Error('tileset_invalid: root must be an object');
  }
  const t = raw as RawTileset;
  if (!t.root) {
    throw new Error('tileset_invalid: missing root tile');
  }

  const tiles: TileNode[] = [];
  const root = buildTileNode(t.root, 'root', baseUrl, undefined, 0, tiles);

  return {
    url: baseUrl,
    geometricError: typeof t.geometricError === 'number' ? t.geometricError : 0,
    extensionsRequired: Array.isArray(t.extensionsRequired) ? t.extensionsRequired.slice() : [],
    root,
    tiles,
  };
}

function buildTileNode(
  raw: RawTile,
  id: string,
  baseUrl: string,
  parent: TileNode | undefined,
  depth: number,
  acc: TileNode[],
): TileNode {
  const obb = raw.boundingVolume?.box;
  if (!obb || obb.length !== 12) {
    throw new Error(`tileset_invalid: tile ${id} missing box bounding volume`);
  }
  const aabb = aabbFromObb12(obb);
  const contentUri = raw.content?.uri ?? raw.content?.url ?? '';
  if (!contentUri) {
    throw new Error(`tileset_invalid: tile ${id} missing content.uri`);
  }
  const contentUrl = resolveAgainst(baseUrl, contentUri);
  const refine = raw.refine === 'ADD' ? 'ADD' : 'REPLACE';

  const node: TileNode = {
    id,
    aabb,
    geometricError: typeof raw.geometricError === 'number' ? raw.geometricError : 0,
    refine,
    contentUrl,
    contentUri,
    children: [],
    parent,
    depth,
  };
  acc.push(node);
  const rawChildren = Array.isArray(raw.children) ? raw.children : [];
  for (let i = 0; i < rawChildren.length; i++) {
    node.children.push(
      buildTileNode(rawChildren[i]!, `${id}/${i}`, baseUrl, node, depth + 1, acc),
    );
  }
  return node;
}

/**
 * Resolve a relative URI against a base URL the same way Cesium does:
 * relative paths resolve against the directory of the tileset.json.
 *
 * The function tries the WHATWG URL constructor first; for harness paths
 * like `"/fixtures/foo/tileset.json"` that aren't valid absolute URLs in
 * Node, we anchor to `globalThis.location` when available, falling back to
 * a simple last-segment swap.
 */
export function resolveAgainst(baseUrl: string, uri: string): string {
  try {
    return new URL(uri, baseUrl).toString();
  } catch {
    // baseUrl wasn't absolute. Try anchoring at globalThis.location.
    const pageHref =
      typeof globalThis !== 'undefined' &&
      (globalThis as { location?: { href?: string } }).location?.href;
    if (typeof pageHref === 'string' && pageHref.length > 0) {
      try {
        const absoluteBase = new URL(baseUrl, pageHref).toString();
        return new URL(uri, absoluteBase).toString();
      } catch {
        /* fall through */
      }
    }
    // Last-resort: replace the trailing filename in baseUrl with `uri`.
    const slash = baseUrl.lastIndexOf('/');
    if (slash < 0) return uri;
    return baseUrl.slice(0, slash + 1) + uri;
  }
}
