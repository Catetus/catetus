// ProjectCovariance.metal — STUB.
//
// Status: PENDING. Port of the screen-space 2D covariance projection from
// `packages/viewer/src/renderer/math.ts::projectCovariance2D`. Until this
// kernel lands the Phase-1 renderer treats every splat as an isotropic quad
// sized by `scale.x` only.

#include <metal_stdlib>
using namespace metal;

kernel void project_covariance_stub(uint tid [[thread_position_in_grid]])
{
    (void)tid;
}
