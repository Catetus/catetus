// RadixSort.metal — STUB.
//
// Status: PENDING. The full port of
// `packages/viewer/src/webgpu/radix_sort.wgsl` lives in the follow-up PR.
// For now the CPU `sfmv_sort_by_depth` ABI handles ordering off the GPU.
//
// Once ported, this file will contain the three-kernel histogram /
// prefix-scan / scatter pipeline that produces the index buffer the
// point-sprite stage consumes.

#include <metal_stdlib>
using namespace metal;

kernel void radix_sort_stub(uint tid [[thread_position_in_grid]])
{
    (void)tid;
}
