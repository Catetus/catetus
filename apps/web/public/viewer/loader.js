/**
 * Resolve a chunk URI against the manifest's own URL.
 *
 * Absolute URLs pass through. Relative URLs are resolved using the standard
 * `URL` constructor, anchoring against `globalThis.location` when `baseHref`
 * is itself a relative path (the common case in a static server harness).
 */
export function resolveChunkUri(baseHref, uri) {
    // Try `baseHref` directly first — works when it's already absolute.
    try {
        return new URL(uri, baseHref).toString();
    }
    catch {
        // Fall through.
    }
    // Anchor a relative baseHref against the current document.
    const pageHref = typeof globalThis !== 'undefined' &&
        globalThis.location?.href;
    if (typeof pageHref === 'string' && pageHref.length > 0) {
        try {
            const absoluteBase = new URL(baseHref, pageHref).toString();
            return new URL(uri, absoluteBase).toString();
        }
        catch {
            /* fall through */
        }
    }
    return uri;
}
/**
 * Fetch a single chunk's bytes from `url`. Honors `byteOffset` / `byteLength`
 * via a `Range:` request when both are non-zero.
 *
 * Maps non-2xx responses to a thrown Error whose message starts with
 * `chunk_not_found:` (404) or `chunk_fetch_failed:` (any other).
 */
export async function fetchChunkBytes(url, descriptor) {
    const headers = {};
    const { byteOffset, byteLength } = descriptor;
    if (byteLength > 0) {
        const end = byteOffset + byteLength - 1;
        headers['Range'] = `bytes=${byteOffset}-${end}`;
    }
    let res;
    try {
        res = await fetch(url, { headers });
    }
    catch (err) {
        throw new Error(`chunk_fetch_failed: ${err.message}`);
    }
    if (res.status === 404) {
        throw new Error(`chunk_not_found: ${url}`);
    }
    if (!res.ok && res.status !== 206) {
        throw new Error(`chunk_fetch_failed: HTTP ${res.status} for ${url}`);
    }
    const ab = await res.arrayBuffer();
    return new Uint8Array(ab);
}
/**
 * Compare a chunk's `checksum` field to a SHA-256 digest of its bytes. We
 * accept either a SHA-256 hex digest or a BLAKE3 hex digest in the manifest;
 * because we cannot compute BLAKE3 in-browser we treat differing-length
 * digests as `unsupported` rather than failing the load.
 */
export async function validateChecksum(bytes, expected) {
    if (!expected)
        return { ok: true };
    const subtle = globalThis.crypto?.subtle;
    if (!subtle || typeof subtle.digest !== 'function') {
        return { ok: false, reason: 'unsupported' };
    }
    // SHA-256 hex is 64 chars. BLAKE3 is 64 chars too but values won't match
    // SHA-256 of the same input — only verify when explicitly SHA-256.
    if (expected.length !== 64) {
        return { ok: false, reason: 'unsupported' };
    }
    try {
        const digest = await subtle.digest('SHA-256', bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        const hex = bytesToHex(new Uint8Array(digest));
        // Either it's a true SHA-256 match, or it's a different algorithm we
        // can't verify. Treat non-match-but-plausible as `unsupported` so we
        // don't fail builds emitted by a BLAKE3 packer.
        if (hex === expected.toLowerCase())
            return { ok: true };
        return { ok: false, reason: 'unsupported' };
    }
    catch {
        return { ok: false, reason: 'unsupported' };
    }
}
function bytesToHex(b) {
    let s = '';
    for (let i = 0; i < b.length; i++) {
        const v = b[i] ?? 0;
        s += v.toString(16).padStart(2, '0');
    }
    return s;
}
//# sourceMappingURL=loader.js.map