//
// QATPlyDequant.metal — Metal compute kernels for the QAT-PLY v1 wire
// format. One thread per output (row × channel). The int4 kernel reads
// each row's per-anchor scale into thread-local storage; the int8
// kernel broadcasts per-channel scales via a small constant buffer.
//
// SPDX-License-Identifier: MIT
//

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// int8 + per-channel scale.
//
//   q       : N * C bytes, row-major signed int8 columns (one per anchor)
//   scale   : C floats, per-channel
//   out     : N * C floats
//   dims.x  : N (n_rows / n_anchors)
//   dims.y  : C (n_channels)
//
// Grid dispatch should be (N, C, 1). The kernel keeps all reads coalesced
// along the channel axis; for the typical 32-channel f_anchor_feat case
// each SIMD lane lands on one channel and pulls scale[c] once.
// -----------------------------------------------------------------------
kernel void qat_dequant_int8(
    device const char  *q       [[ buffer(0) ]],
    device const float *scale   [[ buffer(1) ]],
    device float       *out     [[ buffer(2) ]],
    constant uint2     &dims    [[ buffer(3) ]],
    uint2              gid      [[ thread_position_in_grid ]]
)
{
    const uint N = dims.x;
    const uint C = dims.y;
    if (gid.x >= N || gid.y >= C) return;

    const uint idx = gid.x * C + gid.y;
    const int q_i  = (int)q[idx];
    out[idx] = (float)q_i * scale[gid.y];
}

// -----------------------------------------------------------------------
// int4 packed (two nibbles per byte) + per-anchor scale.
//
//   packed  : N * B bytes, where B = ceil(C / 2). byte_idx = c / 2,
//             nibble = (c%2==0) ? (b & 0x0F) : ((b >> 4) & 0x0F)
//   scale   : N floats, one per anchor
//   out     : N * C floats
//   dims.x  : N
//   dims.y  : C
//
// On-disk nibbles are unsigned-shifted: signed_q = (int)nibble - 8.
// -----------------------------------------------------------------------
kernel void qat_dequant_int4_packed(
    device const uchar *packed  [[ buffer(0) ]],
    device const float *scale   [[ buffer(1) ]],
    device float       *out     [[ buffer(2) ]],
    constant uint2     &dims    [[ buffer(3) ]],
    uint2              gid      [[ thread_position_in_grid ]]
)
{
    const uint N = dims.x;
    const uint C = dims.y;
    if (gid.x >= N || gid.y >= C) return;

    const uint B = (C + 1u) >> 1u;
    const uint row = gid.x;
    const uint c   = gid.y;
    const uint byte_idx = c >> 1u;

    const uchar byte_val = packed[row * B + byte_idx];
    const uint nibble = ((c & 1u) == 0u)
        ? (uint)(byte_val & 0x0Fu)
        : (uint)((byte_val >> 4) & 0x0Fu);
    const int signed_q = (int)nibble - 8;

    out[row * C + c] = (float)signed_q * scale[row];
}
