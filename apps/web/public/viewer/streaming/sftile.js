/**
 * `.sftile` v1 decoder for the streaming-tile loader.
 *
 * STREAM-1's `catetus-tileset` crate emits each octree tile as an `.sftile`
 * binary (see `crates/catetus-tileset/src/codec.rs`). Until STREAM-2's GLB
 * tile codec lands, this is the only real on-disk tile payload, so the
 * streamer must be able to decode it directly.
 *
 * ## `.sftile` v1 layout (little-endian) — verbatim from `codec.rs`
 *
 * ```text
 *   offset  size  field
 *   0       4     magic "SFT1"
 *   4       4     u32  splat_count (N)
 *   8       1     u8   sh_degree
 *   9       3     pad (zero)
 *   12      N*?   splats, each:
 *                   3*f32 position
 *                   4*f32 rotation (x,y,z,w)   [IR order]
 *                   3*f32 scale (LINEAR)
 *                   1*f32 opacity (linear [0,1])
 *                   C*f32 color coeffs (C = 3*(sh_degree+1)^2; DC-first)
 * ```
 *
 * The Catetus IR (and thus `.sftile`) stores **linear** scale and a
 * **baked-ish RGB** DC color (`Color::Rgb`), whereas the WebGPU SoA chunk
 * (`splatSceneToSoaChunk`) expects **log-space** scale (it re-applies `exp`)
 * and **raw `f_dc`** (the decode shader re-applies `0.5 + SH_C0 * f_dc`). We
 * therefore invert both transforms here so a decoded tile flows through the
 * exact same `splatSceneToSoaChunk → renderer.uploadChunk` path the file
 * loaders use:
 *   - scale_log = ln(max(scale_linear, eps))
 *   - f_dc      = (rgb - 0.5) / SH_C0
 *
 * The returned object is shaped like the {@link SplatScene} that
 * `splatSceneToSoaChunk` consumes (`count`, `positions`, `rotations`,
 * `scales`, `opacity`, `dcRaw`, `bbox`, plus optional `shRest`/`shDegree`).
 */
const MAGIC_SFT1 = 0x31544653; // "SFT1" little-endian (bytes 0x53 0x46 0x54 0x31)
/** Inria SH band-0 constant; matches `loader/splat-scene.js`'s SH_C0. */
const SH_C0 = 0.28209479177387814;
/** Color coefficient count (across RGB) for an SH degree. Mirrors codec.rs. */
function coeffsForDegree(degree) {
    const bands = (degree + 1) * (degree + 1);
    return 3 * bands;
}
/**
 * Detect a `.sftile` blob by its 4-byte magic without decoding it.
 */
export function isSftile(bytes) {
    if (!bytes || bytes.byteLength < 4)
        return false;
    return (bytes[0] === 0x53 && // 'S'
        bytes[1] === 0x46 && // 'F'
        bytes[2] === 0x54 && // 'T'
        bytes[3] === 0x31 // '1'
    );
}
/**
 * Decode an `.sftile` v1 blob into a {@link SplatScene}-shaped object ready
 * for `splatSceneToSoaChunk`. Throws an `Error` whose message starts with
 * `sftile_invalid:` on malformed input.
 */
export function decodeSftile(bytes) {
    if (bytes.byteLength < 12) {
        throw new Error('sftile_invalid: header too short');
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = dv.getUint32(0, true);
    if (magic !== MAGIC_SFT1) {
        throw new Error('sftile_invalid: bad magic (not "SFT1")');
    }
    const N = dv.getUint32(4, true);
    const shDegree = dv.getUint8(8);
    // bytes[9..12] are zero pad.
    const coeffLen = coeffsForDegree(shDegree);
    const perFixed = (3 + 4 + 3 + 1) * 4; // pos + rot + scale + opacity = 44 B
    const perSplat = perFixed + coeffLen * 4;
    const need = 12 + N * perSplat;
    if (bytes.byteLength < need) {
        throw new Error(`sftile_invalid: payload shorter than declared count (have ${bytes.byteLength}, need ${need})`);
    }
    const positions = new Float32Array(N * 3);
    const rotations = new Float32Array(N * 4);
    const scales = new Float32Array(N * 3); // log-space (converted below)
    const opacity = new Float32Array(N);
    const dcRaw = new Float32Array(N * 3); // raw f_dc (converted below)
    // SH-rest (degrees >= 1): the .sftile stores coeffs DC-first as
    // [r_dc,g_dc,b_dc, r_1,g_1,b_1, ...]. The WebGPU SoA expects the SH-rest
    // tail in splat-major / k-major / channel-minor float32 order, i.e. the
    // same [coef][channel] interleave minus the DC triple. We pass it through
    // 1:1 (DC stripped) when present.
    const restCoef = coeffLen / 3 - 1; // number of non-DC coefficient triples
    const shRest = restCoef > 0 ? new Float32Array(N * restCoef * 3) : null;
    let off = 12;
    let xMin = Infinity, yMin = Infinity, zMin = Infinity;
    let xMax = -Infinity, yMax = -Infinity, zMax = -Infinity;
    const EPS = 1e-8;
    for (let i = 0; i < N; i++) {
        const px = dv.getFloat32(off, true);
        const py = dv.getFloat32(off + 4, true);
        const pz = dv.getFloat32(off + 8, true);
        off += 12;
        positions[i * 3 + 0] = px;
        positions[i * 3 + 1] = py;
        positions[i * 3 + 2] = pz;
        if (px < xMin) xMin = px; if (px > xMax) xMax = px;
        if (py < yMin) yMin = py; if (py > yMax) yMax = py;
        if (pz < zMin) zMin = pz; if (pz > zMax) zMax = pz;
        rotations[i * 4 + 0] = dv.getFloat32(off, true);
        rotations[i * 4 + 1] = dv.getFloat32(off + 4, true);
        rotations[i * 4 + 2] = dv.getFloat32(off + 8, true);
        rotations[i * 4 + 3] = dv.getFloat32(off + 12, true);
        off += 16;
        // Linear scale → log-space (splatSceneToSoaChunk re-applies exp).
        scales[i * 3 + 0] = Math.log(Math.max(dv.getFloat32(off, true), EPS));
        scales[i * 3 + 1] = Math.log(Math.max(dv.getFloat32(off + 4, true), EPS));
        scales[i * 3 + 2] = Math.log(Math.max(dv.getFloat32(off + 8, true), EPS));
        off += 12;
        opacity[i] = dv.getFloat32(off, true);
        off += 4;
        // First color triple = DC. Convert baked-RGB → raw f_dc so the decode
        // shader's `0.5 + SH_C0 * f_dc` reproduces the original RGB.
        const r = dv.getFloat32(off, true);
        const g = dv.getFloat32(off + 4, true);
        const b = dv.getFloat32(off + 8, true);
        dcRaw[i * 3 + 0] = (r - 0.5) / SH_C0;
        dcRaw[i * 3 + 1] = (g - 0.5) / SH_C0;
        dcRaw[i * 3 + 2] = (b - 0.5) / SH_C0;
        off += 12;
        // Remaining coefficient triples → SH-rest (passed through verbatim).
        if (shRest) {
            for (let k = 0; k < restCoef * 3; k++) {
                shRest[i * restCoef * 3 + k] = dv.getFloat32(off, true);
                off += 4;
            }
        }
    }
    if (N === 0) {
        xMin = yMin = zMin = 0;
        xMax = yMax = zMax = 0;
    }
    const scene = {
        count: N,
        positions,
        rotations,
        scales,
        opacity,
        dcRaw,
        bbox: { min: [xMin, yMin, zMin], max: [xMax, yMax, zMax] },
        meta: { format: 'sftile', source: 'tile' },
    };
    if (shRest && shDegree > 0) {
        scene.shRest = shRest;
        scene.shDegree = shDegree;
    }
    return scene;
}
