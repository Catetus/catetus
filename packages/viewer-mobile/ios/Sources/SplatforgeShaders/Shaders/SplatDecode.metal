// SPDX-License-Identifier: Apache-2.0
//
// SplatDecode.metal — port of `packages/viewer/src/webgpu/decode.wgsl`
// (compute decode kernel only; the project pass lives in ProjectGather.metal).
//
// Input  (device, read):  packed bytes of one SoA chunk —
//                         POSITION u16x3 normalized | min,max from uniform
//                         ROTATION f32x4
//                         SCALE    u8x3  normalized | min,max from uniform
//                         OPACITY  u8    normalized | min=0,max=1
//                         COLOR_DC u8x3  normalized | min=0,max=1
//
// Output (device, write): one `DecodedSplat` per splat, the canonical form fed
//                         to ProjectGather.
//
// One threadgroup-thread = one splat. tg_size 256 picked to match WGSL.
//
// The math mirrors `decodeAttribute` in renderer/base.ts so the GPU and CPU
// paths produce bit-identical results modulo float-precision noise.

#include <metal_stdlib>
using namespace metal;

struct DecodedSplat {
    // vec4 packing keeps storage-buffer alignment at 16B.
    //   pos.xyz   = world position
    //   pos.w     = opacity
    //   scale.xyz = anisotropic scale
    //   scale.w   = reserved (0)
    //   rot.xyzw  = quaternion (x,y,z,w)
    //   color.rgb = DC SH coefficient as linear RGB
    //   color.a   = reserved (1)
    float4 pos;
    float4 scale;
    float4 rot;
    float4 color;
};

struct AttributeSlice {
    uint  byte_offset;
    uint  comp_type;     // 5121 u8, 5123 u16, 5126 f32
    uint  normalized;    // 0/1
    uint  _pad;
    // Per-component dequant bounds. comp 0..3.
    float4 vmin;
    float4 vmax;
};

struct DecodeUniforms {
    uint splat_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    AttributeSlice positions;
    AttributeSlice rotations;
    AttributeSlice scales;
    AttributeSlice opacities;
    AttributeSlice color_dc;
};

// Load an unsigned byte from a u32 storage buffer at absolute byte offset `b`.
static inline uint load_u8(const device uint *src, uint b) {
    uint word = src[b >> 2u];
    uint sh   = (b & 3u) * 8u;
    return (word >> sh) & 0xffu;
}

// Load an unsigned 16-bit little-endian short.
static inline uint load_u16(const device uint *src, uint b) {
    uint lo = load_u8(src, b);
    uint hi = load_u8(src, b + 1u);
    return lo | (hi << 8u);
}

// Load a little-endian f32 by reading 4 contiguous bytes.
static inline float load_f32(const device uint *src, uint b) {
    uint b0 = load_u8(src, b);
    uint b1 = load_u8(src, b + 1u);
    uint b2 = load_u8(src, b + 2u);
    uint b3 = load_u8(src, b + 3u);
    return as_type<float>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}

// Decode one scalar attribute component at byte offset `b` according to
// the slice's comp_type / normalized / min / max.
static inline float decode_component(const device uint *src,
                                     AttributeSlice slice,
                                     uint b,
                                     uint k) {
    if (slice.comp_type == 5126u) {
        return load_f32(src, b);
    }
    if (slice.comp_type == 5123u) {
        float raw = (float)load_u16(src, b);
        if (slice.normalized == 1u) {
            float lo = slice.vmin[k];
            float hi = slice.vmax[k];
            return lo + (raw / 65535.0f) * (hi - lo);
        }
        return raw;
    }
    // default: 5121 u8
    float raw = (float)load_u8(src, b);
    if (slice.normalized == 1u) {
        float lo = slice.vmin[k];
        float hi = slice.vmax[k];
        return lo + (raw / 255.0f) * (hi - lo);
    }
    return raw;
}

static inline uint comp_stride(AttributeSlice slice) {
    if (slice.comp_type == 5126u) { return 4u; }
    if (slice.comp_type == 5123u) { return 2u; }
    return 1u;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void cs_decode(
    const device uint           *src_bytes  [[buffer(0)]],
    device DecodedSplat         *dst_splats [[buffer(1)]],
    constant DecodeUniforms     &u          [[buffer(2)]],
    uint                         gid        [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= u.splat_count) { return; }

    // POSITION (vec3)
    uint p_stride = comp_stride(u.positions);
    uint p_base   = u.positions.byte_offset + i * 3u * p_stride;
    float px = decode_component(src_bytes, u.positions, p_base + 0u * p_stride, 0u);
    float py = decode_component(src_bytes, u.positions, p_base + 1u * p_stride, 1u);
    float pz = decode_component(src_bytes, u.positions, p_base + 2u * p_stride, 2u);

    // ROTATION (vec4)
    uint r_stride = comp_stride(u.rotations);
    uint r_base   = u.rotations.byte_offset + i * 4u * r_stride;
    float rx = decode_component(src_bytes, u.rotations, r_base + 0u * r_stride, 0u);
    float ry = decode_component(src_bytes, u.rotations, r_base + 1u * r_stride, 1u);
    float rz = decode_component(src_bytes, u.rotations, r_base + 2u * r_stride, 2u);
    float rw = decode_component(src_bytes, u.rotations, r_base + 3u * r_stride, 3u);

    // SCALE (vec3)
    uint s_stride = comp_stride(u.scales);
    uint s_base   = u.scales.byte_offset + i * 3u * s_stride;
    float sx = decode_component(src_bytes, u.scales, s_base + 0u * s_stride, 0u);
    float sy = decode_component(src_bytes, u.scales, s_base + 1u * s_stride, 1u);
    float sz = decode_component(src_bytes, u.scales, s_base + 2u * s_stride, 2u);

    // OPACITY (scalar)
    uint o_stride = comp_stride(u.opacities);
    float opacity = decode_component(src_bytes, u.opacities,
                                     u.opacities.byte_offset + i * o_stride, 0u);

    // COLOR_DC (vec3)
    uint c_stride = comp_stride(u.color_dc);
    uint c_base   = u.color_dc.byte_offset + i * 3u * c_stride;
    float cr = decode_component(src_bytes, u.color_dc, c_base + 0u * c_stride, 0u);
    float cg = decode_component(src_bytes, u.color_dc, c_base + 1u * c_stride, 1u);
    float cb = decode_component(src_bytes, u.color_dc, c_base + 2u * c_stride, 2u);

    DecodedSplat out_s;
    out_s.pos   = float4(px, py, pz, opacity);
    out_s.scale = float4(sx, sy, sz, 0.0f);
    out_s.rot   = float4(rx, ry, rz, rw);
    out_s.color = float4(cr, cg, cb, 1.0f);
    dst_splats[i] = out_s;
}
