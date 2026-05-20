// SplatPointSprite.metal — Phase-1 point-sprite renderer.
//
// Vertex stage: builds a screen-space quad per splat, sized by clip-space
// `gl_PointSize` heuristic = scale * focal / depth. No anisotropy yet — that
// arrives with the 2D covariance kernel in the follow-up PR.
// Fragment stage: round soft alpha falloff weighted by splat opacity.

#include <metal_stdlib>
using namespace metal;

struct SplatVertex {
    float3 position;
    float4 rotation;
    float3 scale;
    float  opacity;
    float3 color;
};

struct VOut {
    float4 position [[position]];
    float2 uv;
    float3 color;
    float  opacity;
};

constant float2 kQuad[4] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
};

vertex VOut splat_point_vertex(uint vid [[vertex_id]],
                               uint iid [[instance_id]],
                               const device SplatVertex *splats [[buffer(0)]],
                               constant float4x4 &viewProj [[buffer(1)]],
                               const device uint *sortIndices [[buffer(2)]])
{
    // `sortIndices` is the back-to-front order produced by `ctmv_sort_by_depth`
    // (or the GPU radix-sort kernel, when wired in). We index the splat buffer
    // through it so the rasterizer sees splats in the correct alpha-compositing
    // order. Buffer is guaranteed to be at least `instanceCount` u32s.
    uint splatIdx = sortIndices[iid];
    SplatVertex s = splats[splatIdx];
    float4 clip = viewProj * float4(s.position, 1.0);
    float pxRadius = clamp(s.scale.x * 200.0 / max(clip.w, 0.001), 1.0, 64.0);
    float2 ndcOffset = kQuad[vid] * pxRadius / 800.0; // 800 ≈ half screen, refined later
    VOut out;
    out.position = float4(clip.xy + ndcOffset * clip.w, clip.z, clip.w);
    out.uv = kQuad[vid];
    out.color = s.color;
    out.opacity = s.opacity;
    return out;
}

fragment float4 splat_point_fragment(VOut in [[stage_in]])
{
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();
    float a = exp(-4.0 * r2) * in.opacity;
    return float4(in.color * a, a);
}
