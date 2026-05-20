/**
 * V5.2 joint-tail residual sidecar decoder + applier.
 *
 * Port of `crates/catetus-gltf/src/v5_tail.rs` (`decode_v5tail_bytes` +
 * `apply_v5tail_to_scene`). Same `SFV51TAL` wire format — works on both the
 * GLB sidecar (`.glb.v5tail`) and the SOG sidecar (`.sog.v5tail`); the
 * sidecar format does not know or care which container produced the base
 * encode.
 *
 * On-disk layout (see the Rust module doc for the full spec):
 *   Header (32 B): magic "SFV51TAL" + u16 version + u8 variant + u8 flags +
 *     u32 n_splats + u32 k_selected + u8 n_groups + u8 sh_rest_coefs +
 *     u16 n_cells + 8 B reserved.
 *   Three length-prefixed zstd blobs: mask, morton_idx, cell_offsets.
 *   Six per-group records: u8 n_chan + u8 bit_depth + zstd(meta) + zstd(payload).
 *
 * The apply path adds residuals in raw 3DGS-PLY space:
 *   position += pos_res
 *   rotation += rot_res (caller re-normalises if it wants)
 *   opacity   = sigmoid(logit(opacity) + opa_res)
 *   scale     = exp(ln(scale) + sca_res)            [linear-scale IR]
 *   dc       += dc_res
 *   sh_rest  += shr_res
 *
 * The caller hands us already-decoded SOG (or GLB) splats in IR space; we
 * mutate the typed arrays in place. The Rust reference matches the V5.2
 * Python prototype to within 0.1 dB on the bonsai bench (58.679 dB).
 */
import { decompress as fzstdDecompress } from 'fzstd';
/** Magic bytes "SFV51TAL". */
const MAGIC = [0x53, 0x46, 0x56, 0x35, 0x31, 0x54, 0x41, 0x4c];
/**
 * Supported sidecar versions.
 *   v=1 → V5.2 Phase C ship (8/10/12/12/8/8 bit-depth profile, baked into
 *         `experiments/v5-2-composed/data/sidecar_v5_2.bin` and the early
 *         bonsai shipped GLB).
 *   v=2 → V5.2 Phase D Path B (8/10/14/14/8/8 default profile — opa/sca
 *         widened to recover the `log_quant_attrs` UBYTE damage on bonsai).
 * The wire format is identical between versions — per-group `bit_depth` is
 * stored in each group's u8 header, so a single decoder handles both. The
 * version byte is purely an encoder-identity signal so future format
 * revisions can rev it without ambiguity.
 */
const VERSION_V1 = 1;
const VERSION = 2;
const VARIANT_PER_CELL_AFFINE = 2;
const N_ATTR_GROUPS = 6;
/**
 * Parse a V5.2 sidecar byte slice. Returns the per-group residuals already
 * de-Morton-permuted into ascending-SF order (so `out.pos[k*3..k*3+3]` is
 * the residual for splat `out.selIdx[k]`).
 *
 * Throws on bad magic / version / variant / truncation / malformed groups.
 */
export function decodeV5TailBytes(bytes, zstdDecompress) {
    const decoder = zstdDecompress ?? ((b) => fzstdDecompress(b));
    if (bytes.length < 32) {
        throw new Error(`v5tail: sidecar shorter than 32 B header (${bytes.length})`);
    }
    for (let i = 0; i < MAGIC.length; i++) {
        if (bytes[i] !== MAGIC[i]) {
            throw new Error(`v5tail: bad magic — expected SFV51TAL, got ${Array.from(bytes.subarray(0, 8), (b) => String.fromCharCode(b)).join('')}`);
        }
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = dv.getUint16(8, true);
    if (version !== VERSION_V1 && version !== VERSION) {
        throw new Error(`v5tail: unsupported version ${version} (expected ${VERSION_V1} or ${VERSION})`);
    }
    const variant = dv.getUint8(10);
    if (variant !== VARIANT_PER_CELL_AFFINE) {
        throw new Error(`v5tail: unsupported variant ${variant} (only per-cell affine = 2 implemented)`);
    }
    const flags = dv.getUint8(11);
    const nSplats = dv.getUint32(12, true);
    const kSelected = dv.getUint32(16, true);
    const nGroups = dv.getUint8(20);
    if (nGroups !== N_ATTR_GROUPS) {
        throw new Error(`v5tail: unsupported n_groups ${nGroups}`);
    }
    const shRestCoefs = dv.getUint8(21);
    const nCells = dv.getUint16(22, true);
    // reserved bytes 24..32 ignored
    let pos = 32;
    // --- mask ---
    const readBlob = () => {
        if (pos + 4 > bytes.length) {
            throw new Error(`v5tail: blob length read past end at offset ${pos}`);
        }
        const len = dv.getUint32(pos, true);
        pos += 4;
        if (pos + len > bytes.length) {
            throw new Error(`v5tail: blob length ${len} overruns buffer at offset ${pos}`);
        }
        const slice = bytes.subarray(pos, pos + len);
        pos += len;
        return slice;
    };
    const maskZ = readBlob();
    const maskBytes = decoder(maskZ);
    const selBool = new Uint8Array(nSplats); // 0 / 1
    // LSB-first bit unpack matches `numpy.packbits(bitorder="little")`.
    for (let i = 0; i < nSplats; i++) {
        selBool[i] = (maskBytes[i >> 3] >> (i & 7)) & 1;
    }
    const selIdx = new Uint32Array(kSelected);
    let kk = 0;
    for (let i = 0; i < nSplats; i++) {
        if (selBool[i]) {
            selIdx[kk++] = i;
        }
    }
    if (kk !== kSelected) {
        throw new Error(`v5tail: sel_bool popcount ${kk} != header kSelected ${kSelected}`);
    }
    // --- morton_idx ---
    const mortonZ = readBlob();
    const mortonBytes = decoder(mortonZ);
    if (mortonBytes.length !== kSelected * 4) {
        throw new Error(`v5tail: morton_idx size ${mortonBytes.length} != K*4 = ${kSelected * 4}`);
    }
    const mortonIdx = new Uint32Array(mortonBytes.buffer.slice(mortonBytes.byteOffset, mortonBytes.byteOffset + mortonBytes.byteLength));
    // inv_morton[k_sf] = position in Morton-ordered array where SF-sorted row k_sf lives.
    const invMorton = new Uint32Array(kSelected);
    for (let m = 0; m < kSelected; m++) {
        invMorton[mortonIdx[m]] = m;
    }
    // --- cell_offsets ---
    const cellZ = readBlob();
    const cellBytes = decoder(cellZ);
    if (cellBytes.length % 4 !== 0) {
        throw new Error(`v5tail: cell_offsets size ${cellBytes.length} not divisible by 4`);
    }
    const cellOffsets = new Uint32Array(cellBytes.buffer.slice(cellBytes.byteOffset, cellBytes.byteOffset + cellBytes.byteLength));
    const actualNCells = cellOffsets.length - 1;
    if (actualNCells !== nCells) {
        throw new Error(`v5tail: decoded n_cells ${actualNCells} != header ${nCells}`);
    }
    // Helper: bit-unpack `n_values` u32s at `bd` bits (LSB-first). Mirrors
    // `bit_unpack_fast` in the Rust decoder.
    const bitUnpack = (buf, nValues, bd) => {
        const out = new Uint32Array(nValues);
        if (bd === 8) {
            if (buf.length < nValues) {
                throw new Error(`v5tail: bit_unpack 8-bit truncated: ${buf.length}/${nValues}`);
            }
            for (let i = 0; i < nValues; i++)
                out[i] = buf[i];
            return out;
        }
        if (bd === 16) {
            const needed = nValues * 2;
            if (buf.length < needed) {
                throw new Error(`v5tail: bit_unpack 16-bit truncated`);
            }
            for (let i = 0; i < nValues; i++) {
                out[i] = buf[i * 2] | (buf[i * 2 + 1] << 8);
            }
            return out;
        }
        const totalBits = nValues * bd;
        const needed = (totalBits + 7) >> 3;
        if (buf.length < needed) {
            throw new Error(`v5tail: bit_unpack ${bd}-bit truncated: have ${buf.length} need ${needed}`);
        }
        const mask = bd >= 32 ? 0xffffffff : (1 << bd) - 1;
        for (let i = 0; i < nValues; i++) {
            let bitPos = i * bd;
            let bytePos = bitPos >> 3;
            let bitOff = bitPos & 7;
            let remaining = bd;
            let placed = 0;
            let val = 0;
            while (remaining > 0) {
                const space = 8 - bitOff;
                const take = space < remaining ? space : remaining;
                const tMask = take >= 8 ? 0xff : (1 << take) - 1;
                const chunk = (buf[bytePos] >> bitOff) & tMask;
                val |= chunk << placed;
                placed += take;
                remaining -= take;
                bitPos += take;
                bytePos = bitPos >> 3;
                bitOff = bitPos & 7;
            }
            out[i] = val & mask;
        }
        return out;
    };
    const groupNChan = (gi) => {
        switch (gi) {
            case 0: return 3; // pos
            case 1: return 4; // rot
            case 2: return 1; // opa
            case 3: return 3; // sca
            case 4: return 3; // dc
            case 5: return shRestCoefs * 3; // shr
            default: throw new Error(`unknown group ${gi}`);
        }
    };
    const dequantGroup = (gi) => {
        if (pos + 2 > bytes.length) {
            throw new Error(`v5tail: group ${gi} header truncated`);
        }
        const nChanRead = dv.getUint8(pos);
        pos += 1;
        const bd = dv.getUint8(pos);
        pos += 1;
        const nChan = groupNChan(gi);
        if (nChanRead !== nChan) {
            throw new Error(`v5tail: group ${gi} n_chan ${nChanRead} != expected ${nChan}`);
        }
        const metaRaw = decoder(readBlob());
        const metaFloats = actualNCells * nChan * 2;
        if (metaRaw.length !== metaFloats * 4) {
            throw new Error(`v5tail: group ${gi} meta size ${metaRaw.length} != expected ${metaFloats * 4}`);
        }
        const meta = new Float32Array(metaRaw.buffer.slice(metaRaw.byteOffset, metaRaw.byteOffset + metaRaw.byteLength));
        const packed = decoder(readBlob());
        const q = bitUnpack(packed, kSelected * nChan, bd);
        // Dequant per cell into Morton-order residuals, then de-permute into
        // SF-ascending order.
        const residualMorton = new Float32Array(kSelected * nChan);
        for (let ci = 0; ci < actualNCells; ci++) {
            const a = cellOffsets[ci];
            const b = cellOffsets[ci + 1];
            if (b <= a)
                continue;
            for (let c = 0; c < nChan; c++) {
                const scale = meta[(ci * nChan + c) * 2];
                const offset = meta[(ci * nChan + c) * 2 + 1];
                for (let row = a; row < b; row++) {
                    const idx = row * nChan + c;
                    residualMorton[idx] = q[idx] * scale + offset;
                }
            }
        }
        const out = new Float32Array(kSelected * nChan);
        for (let kSf = 0; kSf < kSelected; kSf++) {
            const m = invMorton[kSf];
            for (let c = 0; c < nChan; c++) {
                out[kSf * nChan + c] = residualMorton[m * nChan + c];
            }
        }
        return out;
    };
    const posR = dequantGroup(0);
    const rotR = dequantGroup(1);
    const opaR = dequantGroup(2);
    const scaR = dequantGroup(3);
    const dcR = dequantGroup(4);
    const shrR = dequantGroup(5);
    return {
        header: {
            variant,
            flags,
            nSplats,
            kSelected,
            shRestCoefs,
            nCells,
        },
        selIdx,
        pos: posR,
        rot: rotR,
        opa: opaR,
        sca: scaR,
        dc: dcR,
        shr: shrR,
    };
}
/**
 * Apply a decoded V5.2 sidecar to a splat scene (in-place mutation of the
 * scene's typed arrays). Returns the number of splats actually modified.
 *
 * Coordinate conventions mirror the Rust apply path:
 *   * `opacity`: residual is logit-space → round-trip through logit + sigmoid.
 *   * `scale`:   residual is log-space → round-trip through ln + exp.
 *   * `position` / `rotation` / `dc` / `sh_rest`: linear additive.
 *
 * The caller is responsible for re-normalizing rotation quats (if it cares)
 * AFTER this returns. We don't normalize here because the V5.2 prototype
 * adds the un-normalized PLY residual, matching the Rust apply path.
 */
export function applyV5TailToScene(scene, decoded) {
    const k = decoded.header.kSelected;
    const n = scene.positions.length / 3;
    if (decoded.header.nSplats !== n) {
        throw new Error(`v5tail apply: sidecar n_splats ${decoded.header.nSplats} != scene count ${n}`);
    }
    const shrChanScene = scene.shRestCoefs * 3;
    const shrChanSidecar = decoded.header.shRestCoefs * 3;
    // We accept either matching or larger sidecar shr — the apply only writes
    // the first `min` channels, matching the Rust path's "truncate to scene
    // capacity" behaviour.
    const shrChan = shrChanScene < shrChanSidecar ? shrChanScene : shrChanSidecar;
    for (let kk = 0; kk < k; kk++) {
        const i = decoded.selIdx[kk];
        // position
        scene.positions[i * 3 + 0] += decoded.pos[kk * 3 + 0];
        scene.positions[i * 3 + 1] += decoded.pos[kk * 3 + 1];
        scene.positions[i * 3 + 2] += decoded.pos[kk * 3 + 2];
        // rotation (raw additive — caller re-normalises if it cares)
        scene.rotations[i * 4 + 0] += decoded.rot[kk * 4 + 0];
        scene.rotations[i * 4 + 1] += decoded.rot[kk * 4 + 1];
        scene.rotations[i * 4 + 2] += decoded.rot[kk * 4 + 2];
        scene.rotations[i * 4 + 3] += decoded.rot[kk * 4 + 3];
        // opacity: logit-space residual.
        scene.opacities[i] = sigmoid(logit(scene.opacities[i]) + decoded.opa[kk]);
        // scale: log-space residual.
        for (let c = 0; c < 3; c++) {
            const raw = Math.log(Math.max(scene.scales[i * 3 + c], 1.175494e-38));
            scene.scales[i * 3 + c] = Math.exp(raw + decoded.sca[kk * 3 + c]);
        }
        // dc: linear additive.
        scene.dcRaw[i * 3 + 0] += decoded.dc[kk * 3 + 0];
        scene.dcRaw[i * 3 + 1] += decoded.dc[kk * 3 + 1];
        scene.dcRaw[i * 3 + 2] += decoded.dc[kk * 3 + 2];
        // sh_rest: linear additive (clipped to scene's coef capacity).
        if (scene.shRest && shrChan > 0) {
            for (let c = 0; c < shrChan; c++) {
                scene.shRest[i * shrChanScene + c] += decoded.shr[kk * shrChanSidecar + c];
            }
        }
    }
    return k;
}
function logit(p) {
    const c = p < 1e-7 ? 1e-7 : p > 1 - 1e-7 ? 1 - 1e-7 : p;
    return Math.log(c / (1 - c));
}
function sigmoid(x) {
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
}
//# sourceMappingURL=v5tail.js.map