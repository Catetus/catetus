// SPDX-License-Identifier: Apache-2.0
//
// ProjectGather.metal — port of the project pass from
// `packages/viewer/src/webgpu/decode.wgsl` (the `cs_project` entry point)
// PLUS a `cs_gather` kernel that applies the sorted index buffer.
//
// `cs_project` consumes the canonical DecodedSplat records (output of
// SplatDecode.metal), projects each splat into clip space, builds the screen-
// space 2x2 covariance via cov3d → world-rotation → focal-divide, and emits
// (per-instance vertex data, sort key, identity index). Math mirrors
// `cs_project` in decode.wgsl byte-for-byte; the only deltas are MSL types
// (mat4x4<f32> → float4x4) and matrix indexing convention (Metal float4x4 is
// column-major, identical to WGSL).
//
// `cs_gather` takes the radix-sorted index buffer and gathers the per-instance
// records into back-to-front draw order. The renderer uses the result as a
// pre-permuted instance buffer; the unsorted `cs_project` output is the input.
//
// Initial pass keeps project + gather as two separate kernels; a fused
// variant ("ProjectSortGather") is the planned follow-up.

#include <metal_stdlib>
using namespace metal;

struct DecodedSplat {
    float4 pos;     // xyz=position, w=opacity
    float4 scale;   // xyz=scale, w=reserved
    float4 rot;     // quaternion (x,y,z,w)
    float4 color;   // rgb=DC SH, a=reserved
};

struct ProjectUniforms {
    float4x4 view;
    float4x4 view_proj;
    float2   viewport;
    float2   focal;
    uint     splat_count;
    uint     _pad;
    uint2    _pad2;
};

// Per-instance vertex-buffer record (must match FLOATS_PER_INSTANCE=12).
struct Instance {
    float4 clip_pos;
    float4 cov;
    float4 color;
};

// Build the upper-triangular 3D covariance Σ = R · diag(s²) · Rᵀ.
// Returns (σxx, σxy, σxz, σyy, σyz, σzz) packed into 2× float3.
struct Cov3 { float3 a; float3 b; };
static inline Cov3 cov3d(float3 scale, float4 q) {
    float n  = max(length(q), 1e-8f);
    float4 qn = q / n;
    float x = qn.x, y = qn.y, z = qn.z, w = qn.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    // Column-major rotation matrix entries.
    float r00 = 1.0f - 2.0f*(yy + zz);
    float r10 = 2.0f*(xy + wz);
    float r20 = 2.0f*(xz - wy);
    float r01 = 2.0f*(xy - wz);
    float r11 = 1.0f - 2.0f*(xx + zz);
    float r21 = 2.0f*(yz + wx);
    float r02 = 2.0f*(xz + wy);
    float r12 = 2.0f*(yz - wx);
    float r22 = 1.0f - 2.0f*(xx + yy);
    float sx = scale.x, sy = scale.y, sz = scale.z;
    float m00 = r00*sx, m10 = r10*sx, m20 = r20*sx;
    float m01 = r01*sy, m11 = r11*sy, m21 = r21*sy;
    float m02 = r02*sz, m12 = r12*sz, m22 = r22*sz;
    Cov3 r;
    r.a = float3(m00*m00 + m01*m01 + m02*m02,
                 m00*m10 + m01*m11 + m02*m12,
                 m00*m20 + m01*m21 + m02*m22);
    r.b = float3(m10*m10 + m11*m11 + m12*m12,
                 m10*m20 + m11*m21 + m12*m22,
                 m20*m20 + m21*m21 + m22*m22);
    return r;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void cs_project(
    const device DecodedSplat   *splats   [[buffer(0)]],
    device Instance             *inst_out [[buffer(1)]],
    device uint                 *keys_out [[buffer(2)]],
    device uint                 *idx_out  [[buffer(3)]],
    constant ProjectUniforms    &pu       [[buffer(4)]],
    uint                         gid      [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= pu.splat_count) { return; }

    DecodedSplat s = splats[i];
    float3 pos    = s.pos.xyz;
    float  opacity = s.pos.w;

    // Clip space.
    float4 clip = pu.view_proj * float4(pos, 1.0f);
    float invW  = (fabs(clip.w) > 1e-12f) ? (1.0f / clip.w) : 1.0f;
    float3 ndc  = float3(clip.x * invW, clip.y * invW, clip.z * invW);

    // View-space depth — row 2 of column-major view matrix dot (pos, 1).
    // Metal float4x4 is column-major: pu.view[col][row] == pu.view.columns[col][row].
    float vz = pu.view[0][2] * pos.x + pu.view[1][2] * pos.y
             + pu.view[2][2] * pos.z + pu.view[3][2];
    float depth  = -vz;
    bool  behind = (depth <= 0.0f);

    // 3D covariance → 2D screen covariance.
    Cov3 V = cov3d(s.scale.xyz, s.rot);
    // World→view rotation rows.
    float3 w0 = float3(pu.view[0][0], pu.view[1][0], pu.view[2][0]);
    float3 w1 = float3(pu.view[0][1], pu.view[1][1], pu.view[2][1]);

    float3 a0 = float3(
        w0.x * V.a.x + w0.y * V.a.y + w0.z * V.a.z,
        w0.x * V.a.y + w0.y * V.b.x + w0.z * V.b.y,
        w0.x * V.a.z + w0.y * V.b.y + w0.z * V.b.z);
    float3 a1 = float3(
        w1.x * V.a.x + w1.y * V.a.y + w1.z * V.a.z,
        w1.x * V.a.y + w1.y * V.b.x + w1.z * V.b.y,
        w1.x * V.a.z + w1.y * V.b.y + w1.z * V.b.z);
    float vxx = dot(a0, w0);
    float vxy = dot(a0, w1);
    float vyy = dot(a1, w1);

    float z   = max(fabs(depth), 1e-4f);
    float jx  = pu.focal.x / z;
    float jy  = pu.focal.y / z;
    float reg = 0.3f;
    float c00 = jx * jx * vxx + reg;
    float c01 = jx * jy * vxy;
    float c11 = jy * jy * vyy + reg;

    // 3σ radius from largest eigenvalue of the 2x2.
    float half_trace = 0.5f * (c00 + c11);
    float det        = max(c00 * c11 - c01 * c01, 0.0f);
    float term       = sqrt(max(half_trace * half_trace - det, 0.0f));
    float lambda_max = half_trace + term;
    float radius     = 3.0f * sqrt(max(lambda_max, 0.0f));
    if (behind) { radius = 0.0f; }

    Instance inst;
    float zc = clamp(ndc.z, 0.0f, 1.0f);
    if (behind) {
        inst.clip_pos = float4(2.0f, 2.0f, 1.0f, 1.0f); // off-screen + radius=0 kills it
        inst.cov      = float4(1.0f, 0.0f, 1.0f, 0.0f);
    } else {
        inst.clip_pos = float4(ndc.xy, zc, clip.w);
        inst.cov      = float4(c00, c01, c11, radius);
    }
    inst.color = float4(s.color.rgb, opacity);
    inst_out[i] = inst;

    // Sort key: bigger view-space depth = drawn first (back-to-front). Radix
    // sort ascends; we invert the float→u32 bitcast for descending semantics.
    float dpos = max(depth, 0.0f);
    uint  kd   = as_type<uint>(dpos);
    keys_out[i] = 0xffffffffu - kd;
    idx_out [i] = i;
}

// ---------------------------------------------------------------------------
// cs_gather — reorder the unsorted Instance buffer using the sorted index
// buffer produced by RadixSort.metal. The renderer consumes the result as a
// pre-permuted draw stream so the vertex shader can stay branch-free.
// ---------------------------------------------------------------------------
struct GatherUniforms {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void cs_gather(
    const device Instance        *inst_in    [[buffer(0)]],
    device Instance              *inst_out   [[buffer(1)]],
    const device uint            *sorted_idx [[buffer(2)]],
    constant GatherUniforms      &u          [[buffer(3)]],
    uint                          gid        [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= u.count) { return; }
    uint src = sorted_idx[i];
    inst_out[i] = inst_in[src];
}
