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
import { aabbFromObb12 } from './frustum.js';
/**
 * Fetch and parse a `tileset.json` at `url`. Throws an `Error` whose message
 * starts with `tileset_not_found:` (404) or `tileset_invalid:` (parse error).
 */
export async function loadTilesetJson(url) {
    let res;
    try {
        res = await fetch(url);
    }
    catch (err) {
        throw new Error(`tileset_not_found: ${err.message}`);
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
export function parseTileset(json, baseUrl) {
    let raw;
    try {
        raw = JSON.parse(json);
    }
    catch (err) {
        throw new Error(`tileset_invalid: not valid JSON (${err.message})`);
    }
    if (!raw || typeof raw !== 'object') {
        throw new Error('tileset_invalid: root must be an object');
    }
    const t = raw;
    if (!t.root) {
        throw new Error('tileset_invalid: missing root tile');
    }
    const tiles = [];
    const root = buildTileNode(t.root, 'root', baseUrl, undefined, 0, tiles);
    return {
        url: baseUrl,
        geometricError: typeof t.geometricError === 'number' ? t.geometricError : 0,
        extensionsRequired: Array.isArray(t.extensionsRequired) ? t.extensionsRequired.slice() : [],
        root,
        tiles,
    };
}
function buildTileNode(raw, id, baseUrl, parent, depth, acc) {
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
    const node = {
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
        node.children.push(buildTileNode(rawChildren[i], `${id}/${i}`, baseUrl, node, depth + 1, acc));
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
export function resolveAgainst(baseUrl, uri) {
    try {
        return new URL(uri, baseUrl).toString();
    }
    catch {
        // baseUrl wasn't absolute. Try anchoring at globalThis.location.
        const pageHref = typeof globalThis !== 'undefined' &&
            globalThis.location?.href;
        if (typeof pageHref === 'string' && pageHref.length > 0) {
            try {
                const absoluteBase = new URL(baseUrl, pageHref).toString();
                return new URL(uri, absoluteBase).toString();
            }
            catch {
                /* fall through */
            }
        }
        // Last-resort: replace the trailing filename in baseUrl with `uri`.
        const slash = baseUrl.lastIndexOf('/');
        if (slash < 0)
            return uri;
        return baseUrl.slice(0, slash + 1) + uri;
    }
}
//# sourceMappingURL=tileset_loader.js.map