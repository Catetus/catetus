/**
 * Cesium-style screen-space-error (SSE) LOD selector.
 *
 * Reference: CesiumJS `Source/Scene/Cesium3DTileset.js` —
 * `getScreenSpaceError(tile, frameState)`. The formula:
 *
 *   sse = geometricError * height / (distance * 2 * tan(fovY / 2))
 *
 * `geometricError` is the world-space error the parent introduces by *not*
 * refining into this child; `distance` is the eye-to-tile distance;
 * `height` is the viewport height in pixels; `fovY` is the vertical FOV in
 * radians. The result is a pixel count — the parent's error projected onto
 * the screen.
 *
 * Refinement rule for `refine: "REPLACE"` (the only mode Catetus emits):
 *   - If `sse > maximumScreenSpaceError`, descend into children (replace
 *     this tile with its children).
 *   - Otherwise, render this tile as-is.
 *
 * Leaves (geometricError = 0) always render, never refine — there's nothing
 * to refine *into*.
 *
 * `maximumScreenSpaceError` defaults to 16, matching Cesium's default. The
 * value can be tuned per asset; the v2 plan calls for 8 on mobile (higher
 * quality, more tiles loaded) and 24 on desktop low-budget mode.
 */
import { aabbIntersectsFrustum, distanceFromCamera, } from './frustum.js';
/**
 * Screen-space error formula from CesiumJS, broken out for unit tests.
 *
 * `geometricError * height / (distance * 2 * tan(fovY / 2))`.
 */
export function screenSpaceError(geometricError, distance, fovY, viewportHeight) {
    const denom = distance * 2 * Math.tan(fovY * 0.5);
    if (denom <= 0)
        return Infinity;
    return (geometricError * viewportHeight) / denom;
}
/**
 * Walk the tile tree top-down and produce the set of tiles to render this
 * frame, along with any tiles that still need to be fetched.
 *
 * Algorithm:
 *   1. If the tile's AABB doesn't intersect the frustum, cull (recurse no
 *      further — Cesium's behavior, since children are bounded by parent).
 *   2. Compute `sse` for this tile.
 *   3. If `sse <= sseMax` OR the tile has no children, render this tile
 *      (or its nearest resident ancestor when `resident` is set).
 *   4. Else, recurse into each child.
 *
 * Determinism: the selector visits children in their declared order, never
 * branches on float comparisons that flip near zero (we use a small epsilon
 * on `sse - sseMax`), and the output ordering is the tree's preorder
 * traversal.
 */
export function selectVisibleTiles(root, opts) {
    const sseMax = opts.maximumScreenSpaceError ?? 16;
    const render = [];
    const fetch = [];
    // Map of tile -> sse for sort. We collect during the walk and sort once.
    const fetchSse = new Map();
    const visit = (tile) => {
        if (!aabbIntersectsFrustum(tile.aabb, opts.frustum))
            return;
        const dist = distanceFromCamera(tile.aabb, opts.eye);
        const sse = screenSpaceError(tile.geometricError, dist, opts.fovY, opts.viewportHeight);
        const hasChildren = tile.children.length > 0;
        // Leaves: render unconditionally (no further refinement available).
        // Internal nodes: refine if `sse > sseMax + epsilon` (epsilon avoids
        // float-tie flicker between two refinement states).
        const epsilon = 1e-6;
        const shouldRefine = hasChildren && sse > sseMax + epsilon;
        if (shouldRefine) {
            // Children carry tile content, but if any of them aren't resident yet
            // we still need to draw *something* this frame — fall back to this
            // tile (resident or not) and enqueue the children for fetch.
            let allChildrenResident = true;
            if (opts.resident) {
                for (const c of tile.children) {
                    if (!opts.resident.has(c.id)) {
                        allChildrenResident = false;
                        fetch.push(c);
                        fetchSse.set(c, sse);
                    }
                }
            }
            if (allChildrenResident || !opts.resident) {
                for (const c of tile.children)
                    visit(c);
            }
            else {
                // Render the resident ancestor (this tile) as a placeholder while
                // children stream in.
                pushRenderable(tile, render, fetch, fetchSse, sse, opts.resident);
            }
        }
        else {
            pushRenderable(tile, render, fetch, fetchSse, sse, opts.resident);
        }
    };
    visit(root);
    // Sort fetch queue by SSE descending — bigger pixel-error tiles win
    // bandwidth first.
    fetch.sort((a, b) => (fetchSse.get(b) ?? 0) - (fetchSse.get(a) ?? 0));
    // Dedup fetch (a child can be queued multiple times if multiple ancestors
    // are missing it; we keep only the first occurrence).
    const seen = new Set();
    const dedup = fetch.filter((t) => {
        if (seen.has(t.id))
            return false;
        seen.add(t.id);
        return true;
    });
    return { render, fetch: dedup };
}
/**
 * Push `tile` (or its nearest resident ancestor) into `render`. If neither
 * the tile nor any ancestor is resident, enqueue this tile for fetch — we'd
 * rather hold off than draw nothing.
 */
function pushRenderable(tile, render, fetch, fetchSse, sse, resident) {
    if (!resident) {
        render.push(tile);
        return;
    }
    // Walk up looking for a resident ancestor.
    let cur = tile;
    while (cur && !resident.has(cur.id)) {
        cur = cur.parent;
    }
    if (cur) {
        render.push(cur);
    }
    if (!resident.has(tile.id)) {
        fetch.push(tile);
        fetchSse.set(tile, sse);
    }
}
//# sourceMappingURL=lod_selector.js.map